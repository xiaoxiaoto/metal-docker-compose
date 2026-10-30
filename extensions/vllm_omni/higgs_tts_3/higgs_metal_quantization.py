# SPDX-License-Identifier: Apache-2.0
"""Higgs 私有 Code2Wav fc2 DynamicEngine；只在加载/离线空闲阶段构造。"""

import hashlib
import inspect
import json
import os
import weakref
from pathlib import Path

import torch

TEXTS = (
    "你好，欢迎使用语音合成。今天阳光明媚，我们一起去公园散步。",
    "The library opens at nine in the morning. Please return your books before Friday.",
)
POLICY = {
    "texts": list(TEXTS),
    "seed": 42,
    "max_new_tokens": 512,
    "sample_rate": 24000,
    "relative_l2_limit": 0.01,
    "scaled_max_abs_limit": 0.01,
    "selection": ["fc2"],
}


def digest_file(path):
    path = Path(path)
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024**2):
            digest.update(chunk)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError("指纹读取期间源文件改变")
    return digest.hexdigest()


def generation_config_revision(directory):
    """vLLM auto消费的可选输入；不存在也是必须绑定的语义状态。"""
    path = Path(directory) / "generation_config.json"
    if not path.exists() and not path.is_symlink():
        return {"present": False, "sha256": None}
    if not path.is_file():
        raise ValueError("generation_config.json不是可读取的普通配置文件")
    return {"present": True, "sha256": digest_file(path)}


def scope(model):
    from importlib.metadata import version
    from importlib.util import find_spec

    from vllm.model_executor.models import qwen2, qwen3
    from vllm_omni.model_executor.models.higgs_audio_v2 import higgs_audio_decoder
    from vllm_omni.model_executor.models.higgs_audio_v3 import higgs_audio_v3_code2wav

    from vllm_omni_metal.native import runner

    directory = Path(model._model_path).resolve(strict=True)
    paths = sorted(directory.glob("*.safetensors"))
    if not paths:
        raise ValueError("需要完整本地 safetensors，禁止联网或替代权重")
    files = [
        directory / name for name in ("config.json", "tokenizer.json", "tokenizer_config.json")
    ]
    index = directory / "model.safetensors.index.json"
    if index.is_file():
        files.append(index)
    record = {
        "directory": str(directory),
        "policy": POLICY,
        "weights_and_config": {p.name: digest_file(p) for p in [*files, *paths]},
        "generation_config": generation_config_revision(directory),
        "adapter": digest_file(__file__),
        "preparation": digest_file(find_spec("higgs_quantization_preparation").origin),
        "omni": version("vllm-omni"),
        "sources": {
            str(p): digest_file(p)
            for p in sorted(
                {
                    *Path(inspect.getfile(higgs_audio_v3_code2wav)).parent.glob("*.py"),
                    Path(inspect.getfile(higgs_audio_decoder)),
                    Path(inspect.getfile(qwen2)),
                    Path(inspect.getfile(qwen3)),
                    *Path(inspect.getfile(runner)).parent.glob("*.py"),
                }
            )
        },
        "deployment_sha256": digest_file(os.environ["HIGGS_DEPLOY_CONFIG"]),
        "dtype": str(model.fc2.weight.dtype),
    }
    encoded = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    return "higgs-code2wav-fc2-dynamic-v2:" + hashlib.sha256(encoded).hexdigest()


def require_dynamic_profile(profile):
    modules = profile.get("modules")
    calibration = profile.get("calibration", {})
    if (
        type(modules) is not dict
        or set(modules) != {"fc2"}
        or calibration.get("engine") != "DynamicEngine"
        or calibration.get("local_pass") is not True
    ):
        raise ValueError("Higgs 需要唯一fc2的真实DynamicEngine搜索profile")
    descriptor = modules["fc2"]
    bits = descriptor.get("bits")
    dtype = descriptor.get("dtype")
    if type(bits) is not int or not (
        (
            bits in (4, 8)
            and descriptor.get("mode") == "affine"
            and descriptor.get("group_size") == 64
            and dtype in ("torch.float16", "torch.bfloat16")
        )
        or (
            bits in (16, 32)
            and descriptor.get("mode") == "identity"
            and descriptor.get("group_size") is None
            and dtype in (("torch.float32",) if bits == 32 else ("torch.float16", "torch.bfloat16"))
        )
    ):
        raise ValueError("Higgs fc2精度必须为源FP16/BF16的W4/W8或原精度identity")


def approved(profile):
    require_dynamic_profile(profile)
    calibration = profile.get("calibration", {})
    records = calibration.get("samples")
    if (
        profile.get("selected") != ["fc2"]
        or calibration.get("approved") is not True
        or calibration.get("policy") != POLICY
        or type(records) is not list
        or len(records) != len(TEXTS)
    ):
        raise ValueError("Higgs profile 未通过完整生成审批或候选/输入策略不匹配")
    for text, sample in zip(TEXTS, records, strict=True):
        if (
            sample.get("text") != text
            or sample.get("seed_repeatable") is not True
            or sample.get("natural_stop") is not True
            or sample.get("released_matches") is not True
            or sample.get("same_shape") is not True
            or sample.get("same_dtype") is not True
            or type(sample.get("candidate_rows")) is not int
            or sample["candidate_rows"] <= 0
            or type(sample.get("candidate_calls")) is not int
            or sample["candidate_calls"] <= 0
        ):
            raise ValueError("Higgs profile 缺少完整生成/实际调用证据")
        for key in ("relative_l2", "scaled_max_abs"):
            value = sample.get(key)
            if type(value) not in (int, float) or not 0 <= value <= 0.01:
                raise ValueError("Higgs 最终波形超过1%质量门槛")
        for key in ("baseline_sha256", "candidate_sha256"):
            value = sample.get(key)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(c not in "0123456789abcdef" for c in value)
            ):
                raise ValueError("Higgs profile 缺少完整浮点输出哈希")


class CodecProjection(torch.nn.Module):
    """原 fc2 返回 Tensor；Talker vLLM tuple/skip_bias_add 路径完全不改。"""

    def __init__(self, quantized):
        super().__init__()
        self.quantized = quantized
        self.dtype = (
            quantized.weight.dtype if isinstance(quantized, torch.nn.Linear) else quantized.dtype
        )
        self.device = (
            quantized.weight.device if isinstance(quantized, torch.nn.Linear) else quantized.device
        )
        self.metrics = {"calls": 0, "rows": 0}

    def forward(self, value):
        # 只有量化 bridge 要求连续输入；identity 保留原布局与原生累加路径。
        if not isinstance(self.quantized, torch.nn.Linear):
            value = value.contiguous()
        result = self.quantized(value)
        self.metrics["calls"] += 1
        self.metrics["rows"] += value.numel() // value.shape[-1]
        if self.metrics["calls"] == 1:
            print(
                f"Higgs dynamic fc2真实调用：calls=1 rows={self.metrics['rows']}",
                flush=True,
            )
        return result


def install(model, profile, pipeline_scope, *, require_approved=True):
    from vllm_omni_metal.quantization.engine import DynamicEngine
    from vllm_omni_metal.quantization.linear import validate_profile

    if require_approved:
        approved(profile)
    if profile.get("selected") != ["fc2"]:
        raise ValueError("本适配仅支持 Code2Wav fc2，禁止全线性层量化")
    require_dynamic_profile(profile)
    validate_profile(profile, {"fc2": model.fc2}, scope=pipeline_scope)
    original = model.fc2
    try:
        with torch.inference_mode():
            result = DynamicEngine.apply(
                model, profile, scope=pipeline_scope, require_approved=require_approved
            )
        wrapped = CodecProjection(result["fc2"])
        model._modules["fc2"] = wrapped
    except BaseException:
        model._modules["fc2"] = original
        raise
    return wrapped


def finish_load(model):
    """由私有 Omni wheel 的已完成 codec loader 调用；请求中不校准。"""
    model._metal_fc2_dtype = model.fc2.weight.dtype
    model._metal_fc2_device = model.fc2.weight.device
    joint = os.environ.get("HIGGS_METAL_JOINT_CALIBRATION", "0") == "1"
    if joint:
        if type(model.fc2) is not torch.nn.Linear or model.fc2.weight.dtype not in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
        ):
            raise ValueError("Higgs joint requires untouched Code2Wav native fc2 identity")
        model._metal_fc2_source = (weakref.ref(model.fc2), weakref.ref(model.fc2.weight))
    mode = os.environ.get("METAL_QUANTIZATION", "off")
    if mode not in ("off", "dynamic"):
        raise ValueError("METAL_QUANTIZATION 只支持 off/dynamic")
    if mode == "dynamic" and "dynamic_quantization" not in {
        item.strip() for item in os.environ.get("METAL_MODEL_OPTIMIZERS", "").split(",")
    }:
        raise ValueError("dynamic requires the dynamic_quantization provider")
    if joint and mode != "off":
        raise ValueError(
            "Higgs joint keeps Code2Wav source identity; separate codec dynamic must be off"
        )
    model._metal_quantization_metadata = {
        "managed": True,
        "mode": mode,
        "approved": False,
    }
    model._higgs_quantization = {"enabled": False, "reason": "默认或显式off"}
    if mode == "off":
        return
    from vllm_omni_metal.quantization.runtime import load_profile

    path = os.environ.get("METAL_QUANTIZATION_PROFILE")
    if not path:
        raise ValueError("dynamic需要显式批准profile路径")
    with torch.inference_mode():
        selected_scope = scope(model)
        profile = load_profile(path, scope=selected_scope)
        install(model, profile, selected_scope)
    model._higgs_quantization = {
        "enabled": True,
        "profile_sha256": profile["sha256"],
        "selected": ["fc2"],
        "lossy": profile["modules"]["fc2"]["bits"] in (4, 8),
    }
    model._metal_quantization_metadata = {
        "managed": True,
        "mode": "dynamic",
        "approved": True,
        "profile": profile,
        "scope": selected_scope,
        "profile_path": str(path),
    }
    print(
        f"Higgs：加载批准的dynamic fc2 bits={profile['modules']['fc2']['bits']} "
        f"profile={profile['sha256']}",
        flush=True,
    )


class QuantizationPreparation:
    """仅离线 stage 1 的安全字符串 RPC，不加载另一份 Talker。"""

    def _quant_idle(self):
        model = self.get_model()
        if self.model_runner.requests:
            raise RuntimeError("离线切换要求全部请求已清理")
        return model

    def higgs_quant_prepare(self):
        model = self._quant_idle()
        if os.environ.get("METAL_QUANTIZATION", "off") != "off" or hasattr(
            model, "_higgs_benchmark"
        ):
            raise RuntimeError("需要独立 METAL_QUANTIZATION=off 校准 worker")
        original = model.fc2
        captures = []

        def capture(module, args):
            if len(captures) == 4:
                return
            value = args[0].detach()
            size = value.numel() * value.element_size()
            if size + sum(t.numel() * t.element_size() for _, t in captures) > 128 * 1024**2:
                raise ValueError("fc2真实激活超过128MiB上限")
            captures.append((len(captures), value.cpu().contiguous().clone()))

        model._higgs_benchmark = {
            "float": original,
            "scope": scope(model),
            "captures": captures,
            "capture_hook": original.register_forward_pre_hook(capture),
        }
        return {"supported": True, "result": {"capture_limit": 4}}

    def higgs_quant_search(self):
        import copy
        import weakref

        from vllm_omni_metal.quantization.engine import DynamicEngine

        model = self._quant_idle()
        state = model._higgs_benchmark
        state.pop("capture_hook").remove()
        if len(state["captures"]) != 4:
            raise ValueError("需要四个真实fc2激活；可来自多次完整请求，不是denoising步数")
        with torch.inference_mode():
            result = DynamicEngine.search(
                {"fc2": state["float"]}, {"fc2": state.pop("captures")}, scope=state["scope"]
            )
            profile = result["profile"]
            # Identity候选独立拥有原精度副本，才能真实验证原baseline释放。
            if profile["modules"]["fc2"]["bits"] in (16, 32):
                model._modules["fc2"] = copy.deepcopy(state["float"])
            try:
                candidate = install(model, profile, state["scope"], require_approved=False)
            finally:
                model._modules["fc2"] = state["float"]
        state.update(
            dynamic=candidate,
            profile=profile,
            baseline_ref=weakref.ref(state["float"]),
            baseline_weight_ref=weakref.ref(state["float"].weight),
        )
        return {"supported": True, "result": result}

    def higgs_quant_mode(self, mode):
        model = self._quant_idle()
        state = model._higgs_benchmark
        if mode not in ("float", "dynamic") or mode not in state:
            raise ValueError("未知路由或基线已释放")
        model._modules["fc2"] = state[mode]
        if "dynamic" in state:
            state["dynamic"].metrics.update(calls=0, rows=0)
        return {"supported": True, "result": {"mode": mode}}

    def higgs_quant_snapshot(self):
        model = self.get_model()
        state = getattr(model, "_higgs_benchmark", None)
        quantized = state.get("dynamic", model.fc2) if state else model.fc2
        metrics = (
            dict(quantized.metrics)
            if isinstance(quantized, CodecProjection)
            else {"calls": 0, "rows": 0}
        )
        return {
            "supported": True,
            "result": {
                "pending": len(self.model_runner.requests),
                "metrics": metrics,
                "deployment": model._higgs_quantization,
                "source_identity": {
                    "dtype": str(model.fc2.weight.dtype)
                    if type(model.fc2) is torch.nn.Linear
                    else None,
                    "identity": (
                        type(model.fc2) is torch.nn.Linear
                        and hasattr(model, "_metal_fc2_source")
                        and model._metal_fc2_source[0]() is model.fc2
                        and model._metal_fc2_source[1]() is model.fc2.weight
                        and model.fc2.weight.dtype == model._metal_fc2_dtype
                    ),
                },
            },
        }

    def higgs_quant_release(self):
        model = self._quant_idle()
        state = model._higgs_benchmark
        model._modules["fc2"] = state["dynamic"]
        del state["float"]
        import gc

        gc.collect()
        torch.mps.empty_cache()
        released = state["baseline_ref"]() is None and state["baseline_weight_ref"]() is None
        if not released:
            raise RuntimeError("原始fc2 module/weight仍被引用；不能批准")
        state["released"] = True
        return {"supported": True, "result": {"float_baseline_released": released}}

    def higgs_quant_approve(self, samples):
        """释放并复验后只重读0.5MiB候选源权重来重建摘要，不重载完整模型。"""
        from safetensors import safe_open

        from vllm_omni_metal.quantization.linear import build_profile, validate_profile

        model = self._quant_idle()
        state = model._higgs_benchmark
        if "float" in state or state.get("released") is not True:
            raise RuntimeError("必须先实际释放float基线并完成复验")
        directory = Path(model._model_path)
        tensors = {}
        prefix = "tied.embedding.modality_embeddings.0.model.fc2."
        for path in sorted(directory.glob("*.safetensors")):
            with safe_open(str(path), framework="pt", device="cpu") as checkpoint:
                for suffix in ("weight", "bias"):
                    key = prefix + suffix
                    if key in checkpoint.keys():
                        if suffix in tensors:
                            raise ValueError("checkpoint含重复fc2权重")
                        tensors[suffix] = checkpoint.get_tensor(key)
        if set(tensors) != {"weight", "bias"}:
            raise ValueError("缺少原始fc2权重，不发布profile")
        with torch.inference_mode():
            source = torch.nn.Linear(
                tensors["weight"].shape[1],
                tensors["weight"].shape[0],
                device=model._metal_fc2_device,
                dtype=model._metal_fc2_dtype,
            )
            source.weight.copy_(tensors["weight"])
            source.bias.copy_(tensors["bias"])
            validate_profile(state["profile"], {"fc2": source}, scope=state["scope"])
            result = build_profile(
                {"fc2": source},
                ["fc2"],
                scope=state["scope"],
                calibration={
                    **state["profile"]["calibration"],
                    "approved": True,
                    "policy": POLICY,
                    "samples": samples,
                    "lossy": state["profile"]["modules"]["fc2"]["bits"] in (4, 8),
                },
                precisions=state["profile"]["calibration"]["precisions"],
            )
            approved(result)
            validate_profile(result, {"fc2": source}, scope=state["scope"])
        return {"supported": True, "result": {"profile": result}}
