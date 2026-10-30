# SPDX-License-Identifier: Apache-2.0
"""QwenImage offline DynamicEngine search and approved mixed-precision loading."""

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

import torch

from vllm_omni_metal.quantization.engine import DynamicEngine
from vllm_omni_metal.quantization.linear import build_profile
from vllm_omni_metal.quantization.runtime import load_profile

PROFILE_LIMIT = 4 * 1024**2


def _profile_bytes(value):
    def plain(item, depth=0):
        if depth > 32:
            raise ValueError("profile嵌套过深")
        if item is None or type(item) in (str, bool, int):
            return
        if type(item) is float and math.isfinite(item):
            return
        if type(item) is list:
            for child in item:
                plain(child, depth + 1)
            return
        if type(item) is dict and all(type(key) is str for key in item):
            for child in item.values():
                plain(child, depth + 1)
            return
        raise ValueError("profile必须为有限plain JSON")

    plain(value)
    if type(value) is not dict:
        raise ValueError("profile必须为JSON对象")
    payload = (
        json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    if len(payload) > PROFILE_LIMIT:
        raise ValueError("profile超过4MiB边界")
    return payload


def _publish_profile(path, profile):
    """同目录完成写入，再以原子hard-link提交；已有目标绝不覆盖。"""
    payload = _profile_bytes(profile)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=target.parent, prefix=".dynamic-profile-", delete=False
        ) as output:
            temporary = Path(output.name)
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.link(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def dense_view(module):
    """临时普通 Linear 视图，不分配或复制完整浮点权重，不进入模型树。"""
    if type(module) is not torch.nn.Linear:
        from vllm.model_executor.layers.linear import (
            RowParallelLinear,
            UnquantizedLinearMethod,
        )

        if (
            type(module) is not RowParallelLinear
            or module.tp_size != 1
            or module.tp_rank != 0
            or type(module.quant_method) is not UnquantizedLinearMethod
        ):
            raise ValueError("仅接受普通Linear或单设备未量化RowParallelLinear，拒绝融合/并行投影")
        if type(module.return_bias) is not bool or type(module.skip_bias_add) is not bool:
            raise ValueError("未知vLLM Linear返回合同")
    if module.bias is not None:
        raise ValueError("QwenImage选择性down仅支持官方无bias投影")
    if module._forward_hooks or module._forward_pre_hooks or module._backward_hooks:
        raise ValueError("不能丢弃投影hooks")
    view = torch.nn.Linear.__new__(torch.nn.Linear)
    torch.nn.Module.__init__(view)
    view.in_features = module.weight.shape[1]
    view.out_features = module.weight.shape[0]
    view.weight = module.weight
    view.register_parameter("bias", None)
    return view


class DynamicProjection(torch.nn.Module):
    """Preserve the bias-free vLLM tensor/tuple contract without float aliases."""

    def __init__(self, source, quantized):
        super().__init__()
        self.return_bias = getattr(source, "return_bias", False)
        self.skip_bias_add = getattr(source, "skip_bias_add", False)
        self.quantized = quantized

    def forward(self, hidden):
        result = self.quantized(hidden)
        return (result, None) if self.return_bias else result


def pixel_errors(reference, candidate):
    import numpy as np

    if reference.shape != candidate.shape or reference.dtype != candidate.dtype:
        raise ValueError("整图dtype/shape不匹配")
    a, b = reference.astype(np.float64), candidate.astype(np.float64)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("整图含非有限数据")
    delta = b - a
    return {
        "relative_l2": float(np.linalg.norm(delta) / max(np.linalg.norm(a), 1e-12)),
        "scaled_max_abs": float(np.abs(delta).max() / max(np.abs(a).max(), 1)),
        "bitwise_equal": bool(np.array_equal(reference, candidate)),
    }


def approval(calibration):
    if type(calibration) is not dict or calibration.get("policy") != "qwen-whole-image-dynamic-v1":
        raise ValueError("缺少QwenImage整图校准合同")
    samples = calibration.get("samples")
    quantized_modules = calibration.get("quantized_modules")
    if type(quantized_modules) is not int or quantized_modules < 0:
        raise ValueError("缺少实际量化投影数量")
    if (
        calibration.get("approved") is not True
        or calibration.get("lossy") is not True
        or calibration.get("threshold") != 0.01
        or type(samples) is not list
        or len(samples) != 4
        or any(type(sample) is not dict for sample in samples)
        or [s.get("quantized") for s in samples] != [False, True, True, False]
    ):
        raise ValueError("缺少完整float/candidate/candidate/float整图证据")
    for sample in samples:
        if (
            type(sample.get("calls")) is not int
            or sample["calls"] < (quantized_modules if sample["quantized"] else 0)
            or ((not sample["quantized"] or quantized_modules == 0) and sample["calls"] != 0)
            or type(sample.get("sha256")) is not str
            or len(sample["sha256"]) != 64
        ):
            raise ValueError("无效的整图实际调用或hash")
    if samples[0]["sha256"] != samples[3]["sha256"] or samples[1]["sha256"] != samples[2]["sha256"]:
        raise ValueError("同路径整图不能复现")
    if quantized_modules == 0 and samples[0]["sha256"] != samples[1]["sha256"]:
        raise ValueError("全源精度保留组合必须复现原精度整图")
    errors = calibration.get("errors", [])
    if type(errors) is not list or len(errors) != 2:
        raise ValueError("缺少所有正式候选整图误差")
    for error in errors:
        if type(error) is not dict:
            raise ValueError("整图误差必须为对象")
        for key in ("relative_l2", "scaled_max_abs"):
            value = error.get(key)
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or not 0 <= value <= 0.01
            ):
                raise ValueError("完整图片1%门禁拒绝候选，不发布profile")


def joint_approval(calibration, *, finalized=True):
    if type(calibration) is not dict or calibration.get("policy") != "qwen-whole-image-joint-v1":
        raise ValueError("旧量化/ANE批准不能拼接为联合批准")
    dynamic = dict(calibration, policy="qwen-whole-image-dynamic-v1")
    approval(dynamic)
    if calibration["quantized_modules"] <= 0:
        raise ValueError("联合批准必须实际执行量化down")
    for sample in calibration["samples"]:
        metrics = sample.get("snapshot", {}).get("ane_metrics", {})
        if sample["quantized"] and metrics.get("ane_chunks", 0) <= 0:
            raise ValueError("联合图片缺少实际ANE命中")
        if not sample["quantized"] and metrics.get("ane_chunks", 0) != 0:
            raise ValueError("原生基线不允许ANE命中")
    baseline = calibration.get("ane_baseline_seconds")
    candidates = calibration.get("ane_candidate_seconds")
    if (
        type(baseline) is not list
        or type(candidates) is not list
        or len(baseline) < 2
        or len(candidates) < 2
        or any(
            type(v) not in (int, float) or not math.isfinite(v) or v <= 0
            for v in baseline + candidates
        )
        or max(candidates) > min(baseline) * 0.97
    ):
        raise ValueError("ANE相对同量化GPU全pipeline至少3%收益门禁失败")
    if not calibration.get("ane_profiles") or any(
        r.get("chunks", 0) <= 0 for r in calibration["ane_profiles"]
    ):
        raise ValueError("联合批准缺少非零ANE projection profile")
    if finalized:
        lifecycle = calibration.get("lifecycle", {})
        if (
            lifecycle.get("released_repeatable") is not True
            or lifecycle.get("adapter_closed") is not True
            or lifecycle.get("engine_closed") is not True
            or lifecycle.get("supervisor_success") is not True
        ):
            raise ValueError("联合批准缺少释放复现/关闭/监督成功")


def validate_calibration(calibration, *, finalized=True):
    if calibration.get("policy") == "qwen-whole-image-joint-v1":
        joint_approval(calibration, finalized=finalized)
    else:
        approval(calibration)


class QwenDynamicAdapter:
    def __init__(self, owner):
        self.owner = owner
        self.root = owner.pipeline.transformer
        self.originals = {}
        self.candidates = {}
        self.hooks = []
        self.local = {}
        self.capture_pending = []
        self.capture_results = []
        self.search = None
        self.active = None
        self.scope = None
        self.status = "off"
        self.joint_profile = None
        selected = set(filter(None, os.environ.get("METAL_MODEL_OPTIMIZERS", "").split(",")))
        if selected - {"dynamic_quantization", "gpu_ops"}:
            raise ValueError("Qwen Omni 未实现所选 Metal provider")
        mode = os.environ.get("METAL_QUANTIZATION", "off")
        if mode not in ("off", "dynamic"):
            raise ValueError("METAL_QUANTIZATION必须为off或dynamic")
        if mode == "dynamic" and "dynamic_quantization" not in selected:
            raise ValueError("dynamic需要选择dynamic_quantization provider")
        if mode == "off":
            metadata = {"managed": True, "mode": "off", "approved": False}
            owner.pipeline._metal_quantization_metadata = metadata
            self.root._metal_quantization_metadata = metadata
            return
        path = os.environ.get("METAL_QUANTIZATION_PROFILE")
        if not path:
            raise ValueError("dynamic需要显式批准profile")
        self.scope = self._scope()
        profile = load_profile(path, scope=self.scope)
        approval(profile.get("calibration"))
        self._sources(profile.get("selected"))
        self._install(profile, require_approved=True)
        self.set_candidate(True)
        self.release()
        self.status = "approved dynamic mixed precision enabled"
        metadata = {
            "managed": True,
            "mode": "dynamic",
            "joint": False,
            "approved": True,
            "profile": profile,
            "profile_path": str(path),
            "scope": self.scope,
        }
        owner.pipeline._metal_quantization_metadata = metadata
        self.root._metal_quantization_metadata = metadata

    def _install(self, profile, *, require_approved):
        if require_approved:
            count = sum(item["bits"] in (4, 8) for item in profile["modules"].values())
            if profile["calibration"]["quantized_modules"] != count:
                raise ValueError("整图批准的量化数量与实际profile不匹配")
        # A temporary tree of zero-copy dense views lets the generic engine own
        # validation/packing, while the real tree keeps its vLLM return contract.
        temporary = torch.nn.Module()
        for name, view in self._views().items():
            parent = temporary
            parts = name.split(".")
            for part in parts[:-1]:
                if part not in parent._modules:
                    parent.add_module(part, torch.nn.Module())
                parent = parent._modules[part]
            parent.add_module(parts[-1], view)
        packed = DynamicEngine.apply(
            temporary, profile, scope=self.scope, require_approved=require_approved
        )
        replacements = {
            name: (
                DynamicProjection(source, packed[name])
                if profile["modules"][name]["bits"] in (4, 8)
                else source
            )
            for name, source in self.originals.items()
        }
        self.candidates = replacements
        self.search = profile

    def _scope(self):
        from importlib.metadata import version

        from vllm_omni_metal.quantization import engine, linear

        digest = hashlib.sha256(self.owner.scope.encode())
        from vllm_omni_metal.ane import build, cache, projection, runtime

        for path in (
            Path(__file__),
            Path(linear.__file__),
            Path(engine.__file__),
            Path(projection.__file__),
            Path(build.__file__),
            Path(cache.__file__),
            Path(runtime.__file__),
            Path(__file__).with_name("joint_adapter.py"),
            Path(__file__).with_name("prepare_quantization.py"),
            Path(__file__).with_name("prepare_ane.py"),
            Path(__file__).with_name("prepare.py"),
        ):
            digest.update(path.read_bytes())
        directory = Path(self.owner.pipeline.od_config.model).resolve(strict=True)
        # 完整checkpoint/config/tokenizer身份，不以mtime或只有所选权重代替模型绑定。
        files = sorted(
            p
            for p in directory.rglob("*")
            if p.is_file() and p.suffix in (".safetensors", ".json", ".jinja", ".txt", ".model")
        )
        if not any(p.suffix == ".safetensors" for p in files):
            raise ValueError("dynamic scope需要完整本地checkpoint")
        for path in files:
            digest.update(str(path.relative_to(directory)).encode())
            with path.open("rb") as source:
                while chunk := source.read(8 * 1024**2):
                    digest.update(chunk)
        digest.update(
            json.dumps(
                {
                    "omni": version("vllm-omni"),
                    "gpu": "auto" if "gpu_ops" in os.environ.get("METAL_MODEL_OPTIMIZERS", "").split(",") else "off",
                    "ane": "gate-projection-v1",
                    "steps": 8,
                    "cfg": 1.0,
                },
                sort_keys=True,
            ).encode()
        )
        return digest.hexdigest()

    def _sources(self, names):
        self.owner.idle()
        if self.originals or self.candidates:
            raise ValueError("候选已准备，不允许重复安装")
        if (
            type(names) is not list
            or not names
            or len(set(names)) != len(names)
            or (self.owner.group is not None and not hasattr(self.owner.group, "gates"))
        ):
            raise ValueError("需要非空唯一down列表；full-FFN ANE互斥，只允许gate-only controller")
        sources = {}
        for name in names:
            parts = name.split(".")
            if (
                len(parts) != 4
                or parts[0] != "transformer_blocks"
                or not parts[1].isdigit()
                or parts[2:] != ["img_mlp", "out"]
            ):
                raise ValueError("仅允许明确的img_mlp.out down候选")
            source = self.root.get_submodule(name)
            dense_view(source)
            sources[name] = source
        identities = {id(m) for m in sources.values()}
        chosen_storage = {m.weight.untyped_storage().data_ptr() for m in sources.values()}
        seen = set()
        for _, module in self.root.named_modules(remove_duplicate=False):
            if id(module) in identities:
                if id(module) in seen:
                    raise ValueError("不支持shared module alias")
                seen.add(id(module))
            for tensor in (*module.parameters(recurse=False), *module.buffers(recurse=False)):
                if tensor.numel() and tensor.untyped_storage().data_ptr() in chosen_storage:
                    if id(module) not in identities or tensor is not module.weight:
                        raise ValueError("不支持tied/shared权重或buffer alias")
        if len(chosen_storage) != len(sources):
            raise ValueError("不支持shared down权重")
        self.originals = sources

    def _views(self):
        return {name: dense_view(module) for name, module in self.originals.items()}

    def prepare(self, layers):
        names = [f"transformer_blocks.{i}.img_mlp.out" for i in layers]
        self._sources(names)
        self.scope = self._scope()
        self.capture_pending = list(names)
        self.capture_results = []
        self._prepare_capture_batch()
        return self.snapshot()

    def _prepare_capture_batch(self):
        # 每个批次仍运行完整模型；只保留一个候选层的四个完整激活。
        name = self.capture_pending[0]
        self.local = {name: []}
        captured_bytes = [0]
        calls = [0]

        def capture(module, inputs, output):
            number = calls[0]
            calls[0] += 1
            if number not in (0, 2, 4, 6):
                return
            tensor = inputs[0].detach()
            size = tensor.numel() * tensor.element_size()
            if captured_bytes[0] + size > 128 * 1024**2:
                raise ValueError("真实激活超过128MiB捕获边界")
            tensor = tensor.to(device="cpu", copy=True).contiguous()
            captured_bytes[0] += size
            self.local[name].append((number, tensor))

        self.hooks.append(self.originals[name].register_forward_hook(capture))
        self.status = "offline activation capture"

    def finish_capture(self):
        self.owner.idle()
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        if not self.local or any(len(items) != 4 for items in self.local.values()):
            raise ValueError("完整8步真实生成未捕获全部0/2/4/6激活")
        name = self.capture_pending[0]
        with torch.inference_mode():
            result = DynamicEngine.search({name: self._views()[name]}, self.local, scope=self.scope)
        self.local.clear()
        self.capture_results.append(result)
        self.capture_pending.pop(0)
        if self.capture_pending:
            self._prepare_capture_batch()
            return {"complete": False, "captured": name, "remaining": self.capture_pending[:]}
        audits, fingerprints, precisions = {}, {}, {}
        for batch in self.capture_results:
            audits.update(batch["audits"])
            fingerprints.update(batch["profile"]["calibration"]["captures"])
            precisions.update(batch["precisions"])
        calibration = {
            **result["profile"]["calibration"],
            "audits": audits,
            "captures": fingerprints,
            "precisions": precisions,
        }
        profile = build_profile(
            self._views(),
            list(self.originals),
            scope=self.scope,
            calibration=calibration,
            precisions=precisions,
        )
        with torch.inference_mode():
            self._install(profile, require_approved=False)
        self.capture_results.clear()
        return {
            "complete": True,
            "profile": profile,
            "audits": audits,
            "precisions": precisions,
            "local_pass": True,
        }

    def set_candidate(self, enabled):
        self.owner.idle()
        if self.candidates and not self.originals:
            raise ValueError("已释放浮点基线，不能再次切换路由")
        if self.hooks:
            raise ValueError("捕获未结束，不允许切换")
        if type(enabled) is not bool:
            raise ValueError("组合路由必须为bool")
        if enabled and not self.candidates:
            raise ValueError("尚未搜索候选")
        for path, source in self.originals.items():
            parent, _, child = path.rpartition(".")
            self.root.get_submodule(parent)._modules[child] = (
                self.candidates[path] if enabled else source
            )
        self.active = enabled
        for candidate in self.candidates.values():
            if isinstance(candidate, DynamicProjection):
                candidate.quantized.metrics.update(calls=0, rows=0)
        return self.snapshot()

    def profile(self, calibration):
        self.owner.idle()
        validate_calibration(calibration, finalized=False)
        if self.active is not True or not self.originals or self.search is None:
            raise ValueError("发布需要仍可指纹的原始权重和已执行组合")
        evidence = {**self.search["calibration"], **calibration}
        with torch.inference_mode():
            return build_profile(
                self._views(),
                list(self.originals),
                scope=self.scope,
                calibration=evidence,
                precisions=self.search["calibration"]["precisions"],
            )

    def release(self):
        import gc
        import weakref

        self.owner.idle()
        if self.active is not True or self.hooks:
            raise ValueError("需要健康已选组合才释放浮点基线")
        references = [
            (weakref.ref(source), weakref.ref(source.weight))
            for name, source in self.originals.items()
            if isinstance(self.candidates[name], DynamicProjection)
        ]
        self.originals.clear()
        gc.collect()
        if any(module() is not None or weight() is not None for module, weight in references):
            raise ValueError("量化投影原模块或浮点权重仍被引用")
        self.status = "dynamic active; quantized float baselines released"
        return self.snapshot()

    def close(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.local.clear()
        self.capture_pending.clear()
        self.capture_results.clear()
        self.originals.clear()
        self.candidates.clear()
        self.active = None
        self.status = "closed"

    def snapshot(self):
        modules = {
            name: {
                **module.quantized.metrics,
                "bits": module.quantized.bits,
                "source_bytes": module.quantized.source_bytes,
                "storage_bytes": module.quantized.storage_bytes,
            }
            if isinstance(module, DynamicProjection)
            else {
                "bits": module.weight.element_size() * 8,
                "calls": 0,
                "rows": 0,
                "retained_source_precision": True,
            }
            for name, module in self.candidates.items()
        }
        return {
            "status": self.status,
            "active": self.active,
            "scope": self.scope,
            "float_baseline_held": bool(self.originals),
            "modules": modules,
            "quantized_modules": sum(
                isinstance(m, DynamicProjection) for m in self.candidates.values()
            ),
            "calls": sum(m["calls"] for m in modules.values()) if self.active else 0,
        }
