#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""隔离 worker 的固定 NAR W4/W8/BF16 诊断；不搜索、不批准、不改变部署。"""

import gc
import hashlib
import os
import re
import weakref
from pathlib import Path

from prepare_ane import AnePreparation, success
from prepare_quantization import NarDownAdapter, QuantizationPreparation, memory_snapshot

_LAYER_NAME = re.compile(r"nar_layers\.(0|[1-9][0-9]*)\.mlp\.down_proj")
_PRECISIONS = {"w4a16", "w8a16", "bf16"}


def validate_precisions(precisions, layer_count):
    """校验完整显式三精度计划，返回保持输入顺序的规范名称映射。"""
    if type(layer_count) is not int or layer_count < 1:
        raise ValueError("NAR层数必须为正整数")
    if type(precisions) is not dict or not 1 <= len(precisions) <= 4:
        raise ValueError("混合精度必须显式指定1–4个不同NAR down投影")
    plan = {}
    for name, precision in precisions.items():
        match = _LAYER_NAME.fullmatch(name) if type(name) is str else None
        if match is None or int(match[1]) >= layer_count:
            raise ValueError("混合精度键必须是范围内规范的nar_layers.<i>.mlp.down_proj")
        if type(precision) is not str or precision not in _PRECISIONS:
            raise ValueError("精度只能为w4a16、w8a16或bf16")
        plan[name] = precision
    if set(plan.values()) != _PRECISIONS:
        raise ValueError("完整混合诊断必须至少各有一个W4A16、W8A16和BF16")
    return plan


class MixedPreparation(QuantizationPreparation):
    """通过实例 worker_extension 接入现有 NAR hook，不修改模型或全局类。"""

    def _mixed_idle_model(self):
        model = AnePreparation._idle_model(self)
        if getattr(model, "_pending", None) is not None:
            raise RuntimeError("请求pending尚未清理，不能准备或切换混合精度")
        if os.environ.get("VLLM_OMNI_METAL_ANE") != "off":
            raise RuntimeError("混合精度诊断要求ANE显式off")
        return model

    def mixed_native_snapshot(self):
        """安装前原生FFN的独立参考快照，不声明混合适配器identity已验证。"""
        model = MixedPreparation._mixed_idle_model(self)
        group = model._metal_ffn
        if (
            group is None
            or isinstance(group, NarDownAdapter)
            or group._closed
            or group.disabled
            or group._benchmark
        ):
            raise RuntimeError("原生参考仅允许混合安装前的健康原始FFN组")
        return success(
            {
                "pending": 0,
                "mode": "native",
                "adapter_calls": 0,
                "quantized": {},
                "benchmark_held_float_bytes": 0,
                "memory": memory_snapshot(),
            }
        )

    def _mixed_adapter(self, model):
        adapter = model._metal_ffn
        if (
            not isinstance(adapter, NarDownAdapter)
            or adapter._closed
            or not hasattr(adapter, "precisions")
        ):
            raise RuntimeError("缺少健康的固定混合精度实例适配器")
        return adapter

    def _mixed_identity(self, model, adapter):
        retained = {
            name: (
                model.get_submodule(name) is module
                and model.get_submodule(name).weight is weight
                and model.nar_fused[int(name.split(".")[1])][7] is weight
            )
            for name, (module, weight) in adapter.retained_modules.items()
        }
        return {
            "retained_identity": bool(retained) and all(retained.values()),
            "retained_modules": retained,
        }

    def mixed_install(self, precisions, input_record):
        import torch

        model = MixedPreparation._mixed_idle_model(self)
        plan = validate_precisions(precisions, len(model.nar_layers))
        retained = {}
        for name, precision in plan.items():
            if precision == "bf16":
                module = model.get_submodule(name)
                if type(module) is not torch.nn.Linear or module.weight.dtype != torch.bfloat16:
                    raise ValueError("BF16 identity要求原模块已为普通BF16 Linear，不隐式转换")
                retained[name] = (module, module.weight)
        revision = {
            "input": input_record,
            "mixed_adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "precisions": plan,
            "no_sensitivity_search": True,
        }
        # 原安装器在构造健康候选后才接管；封套使既有scope同时绑定新源码和精度计划。
        requested = ",".join(name.split(".")[1] for name in plan)
        result = QuantizationPreparation.quant_install(self, requested, revision)["result"]
        adapter = model._metal_ffn
        adapter.capture = False
        adapter.captures.clear()
        adapter.capture_bytes = 0
        adapter.precisions = dict(plan)
        adapter.retained_modules = retained  # 只保留identity；量化源不放入这里。
        adapter.quantized_sources = {}
        result.update(
            precisions=dict(plan),
            no_sensitivity_search=True,
            approved=False,
            local_pass=None,
            **MixedPreparation._mixed_identity(self, model, adapter),
        )
        return success(result)

    def mixed_prepare(self):
        import torch

        from vllm_omni_metal.quantization.linear import apply_profile, build_profile

        model = MixedPreparation._mixed_idle_model(self)
        adapter = MixedPreparation._mixed_adapter(self, model)
        if adapter.quantized or adapter.baselines:
            raise RuntimeError("不能重复构造混合精度组")
        plan = validate_precisions(adapter.precisions, len(model.nar_layers))
        identity = MixedPreparation._mixed_identity(self, model, adapter)
        if not identity["retained_identity"]:
            raise RuntimeError("BF16 identity在准备前改变，拒绝替换")
        selected = list(plan)
        modules = {name: model.get_submodule(name) for name in selected}
        quantized_names = [name for name in selected if plan[name] != "bf16"]
        baselines = {name: modules[name] for name in quantized_names}
        sources = {
            name: {
                "module": weakref.ref(modules[name]),
                "weight": weakref.ref(modules[name].weight),
            }
            for name in quantized_names
        }
        slots = {}
        locations = {}
        for name in selected:
            index = int(name.split(".")[1])
            weights = model.nar_fused[index]
            if weights[7] is not modules[name].weight:
                raise RuntimeError("NAR融合down别名在安装后改变，拒绝替换")
            parent_name, _, child = name.rpartition(".")
            locations[name] = (model.get_submodule(parent_name), child)
            if name in baselines:
                slots[index] = weights
        with torch.inference_mode():
            profile = build_profile(
                modules,
                selected,
                scope=adapter.scope,
                precisions=plan,
                calibration={
                    "precisions": dict(plan),
                    "no_sensitivity_search": True,
                    "local_pass": None,
                    "approved": False,
                },
            )
            try:
                replacements = apply_profile(model, profile, scope=adapter.scope)
                identity = MixedPreparation._mixed_identity(self, model, adapter)
                if not identity["retained_identity"]:
                    raise RuntimeError("通用apply_profile未保持BF16模块/权重身份")
                # hook不消费量化层slot7，清理它才能避免隐藏的源浮点持有；identity不清。
                for index, weights in slots.items():
                    model.nar_fused[index] = (*weights[:7], None)
            except BaseException:
                # 通用API构造事务外再保护本地融合别名提交；失败仍为健康float。
                for name, (parent, child) in locations.items():
                    parent._modules[child] = modules[name]
                for index, weights in slots.items():
                    model.nar_fused[index] = weights
                adapter.mode = "float"
                raise
        adapter.baselines = baselines
        adapter.quantized = {name: replacements[name] for name in quantized_names}
        adapter.quantized_sources = sources
        adapter.mode = "float"
        return success(
            {
                "profile": profile,
                "selected": selected,
                "precisions": dict(plan),
                "local_pass": None,
                "no_sensitivity_search": True,
                "approved": False,
                "benchmark_retains_float": True,
                "benchmark_held_float_bytes": sum(
                    module.weight.numel() * module.weight.element_size()
                    for module in baselines.values()
                ),
                "source_bytes": sum(module.source_bytes for module in adapter.quantized.values()),
                "storage_bytes": sum(module.storage_bytes for module in adapter.quantized.values()),
                **identity,
            }
        )

    def quant_mode(self, mode):
        model = MixedPreparation._mixed_idle_model(self)
        adapter = MixedPreparation._mixed_adapter(self, model)
        if mode not in ("float", "mixed") or not adapter.quantized:
            raise ValueError("无已安装混合精度组或模式非法")
        if mode == "float" and not adapter.baselines:
            raise RuntimeError("量化项基准浮点已释放，不得伪装基线")
        identity = MixedPreparation._mixed_identity(self, model, adapter)
        if not identity["retained_identity"]:
            raise RuntimeError("BF16模块/权重identity已改变")
        adapter.mode = mode
        adapter.metrics.clear()
        for module in adapter.quantized.values():
            module.metrics.update(calls=0, rows=0)
        return success({"mode": mode, "precisions": dict(adapter.precisions), **identity})

    def quant_snapshot(self):
        # 快照保留父类pending观测能力，供请求结束后的idle轮询使用；不改变执行状态。
        if os.environ.get("VLLM_OMNI_METAL_ANE") != "off":
            raise RuntimeError("混合精度诊断要求ANE显式off")
        model = self.get_model()
        adapter = MixedPreparation._mixed_adapter(self, model)
        result = QuantizationPreparation.quant_snapshot(self)["result"]
        for name, module in adapter.quantized.items():
            result["quantized"][name]["bits"] = module.bits
        result.update(
            precisions=dict(adapter.precisions),
            no_sensitivity_search=True,
            approved=False,
            **MixedPreparation._mixed_identity(self, model, adapter),
        )
        return success(result)

    def quant_release_baseline(self):
        model = MixedPreparation._mixed_idle_model(self)
        adapter = MixedPreparation._mixed_adapter(self, model)
        if adapter.mode != "mixed" or not adapter.quantized:
            raise RuntimeError("仅已安装的mixed模式可释放量化项基准浮点")
        if not MixedPreparation._mixed_identity(self, model, adapter)["retained_identity"]:
            raise RuntimeError("BF16模块/权重identity已改变，拒绝释放")
        for name in adapter.quantized:
            if model.nar_fused[int(name.split(".")[1])][7] is not None:
                raise RuntimeError("量化项仍有融合浮点别名，拒绝声称释放")
        import mlx.core as mx
        import torch

        released = sum(
            module.weight.numel() * module.weight.element_size()
            for module in adapter.baselines.values()
        )
        adapter.baselines.clear()
        gc.collect()
        torch.mps.empty_cache()
        mx.clear_cache()
        source_released = {
            name: references["module"]() is None and references["weight"]() is None
            for name, references in adapter.quantized_sources.items()
        }
        if not source_released or not all(source_released.values()):
            raise RuntimeError(
                f"量化源浮点仍有引用，不能继续释放后生成：{source_released!r}；"
                f"已清基准引用字节={released}"
            )
        result = MixedPreparation.quant_snapshot(self)["result"]
        result.update(
            released_benchmark_float_bytes=released,
            quantized_source_released=source_released,
            all_quantized_sources_released=bool(source_released) and all(source_released.values()),
            independent_process=False,
            memory_claim="同worker释放量化项基准后的观测；保留BF16 identity，非独立冷启动或加载峰值验收",
        )
        return success(result)

    def quant_audit(self, count):
        raise RuntimeError("固定混合计划禁止敏感度搜索；只能调用mixed_prepare")

    def quant_apply(self, profile):
        raise RuntimeError("固定混合计划只能由mixed_prepare事务调用通用apply_profile")

    def quant_approve_profile(self, profile, completion):
        raise RuntimeError("混合精度仅供诊断，禁止批准profile")
