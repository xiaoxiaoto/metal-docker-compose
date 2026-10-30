#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""真实 YuE2 W8 校准或显式 W4/W8/BF16 诊断；交错 A/B，不改变服务。"""

import argparse
import asyncio
import copy
import gc
import hashlib
import inspect
import json
import os
import shutil
import sys
import tempfile
import time
import wave
import weakref
from pathlib import Path
from types import SimpleNamespace

import psutil
from prepare_ane import AnePreparation, rpc, success, supervise, write_json

CAPTURE_STEPS = (0, 8, 16, 24)
CAPTURE_BUDGET = 128 * 1024**2


def precision_object(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f"重复精度模块：{name}")
        result[name] = value
    return result


def require_w8_profile(profile):
    names, modules = profile.get("selected"), profile.get("modules")
    if (
        type(names) is not list
        or not names
        or any(type(name) is not str for name in names)
        or len(set(names)) != len(names)
        or type(modules) is not dict
        or set(modules) != set(names)
    ):
        raise ValueError("YuE2 W8A16 profile 选择/描述符不匹配")
    for descriptor in modules.values():
        if (
            type(descriptor) is not dict
            or type(descriptor.get("bits")) is not int
            or descriptor["bits"] != 8
            or descriptor.get("mode") != "affine"
            or type(descriptor.get("group_size")) is not int
            or descriptor["group_size"] != 64
        ):
            raise ValueError("YuE2 专用批准/执行仅支持 W8A16，拒绝 W4A16/BF16")


def weight_fingerprint(model, vae):
    result = {}
    for label, directory in (("model", model), ("vae", vae)):
        paths = sorted(Path(directory).glob("*.safetensors"))
        if not paths:
            raise FileNotFoundError(f"{label}: 缺少本地safetensors；禁止下载替代权重")
        entries = []
        for path in paths:
            before = path.stat()
            digest = hashlib.sha256()
            with path.open("rb") as source:
                while chunk := source.read(8 * 1024**2):
                    digest.update(chunk)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise RuntimeError("源权重在指纹读取期间改变")
            entries.append(
                {"path": str(path.resolve()), "bytes": after.st_size, "sha256": digest.hexdigest()}
            )
        result[label] = entries
    return result


def candidate_names(layer_count, requested):
    if requested == "auto":
        layers = sorted({0, layer_count // 3, 2 * layer_count // 3, layer_count - 1})
    else:
        layers = [int(value) for value in requested.split(",")]
        if len(layers) != len(set(layers)) or not 1 <= len(layers) <= 4:
            raise ValueError("显式候选必须为1–4个互异NAR层")
    if any(not 0 <= index < layer_count for index in layers):
        raise ValueError("NAR层索引越界")
    return [f"nar_layers.{index}.mlp.down_proj" for index in layers]


def compare_arrays(reference, candidate, *, pcm=True):
    import numpy as np

    result = {
        "same_shape": reference.shape == candidate.shape,
        "same_dtype": reference.dtype == candidate.dtype,
        "bitwise_identical": False,
        "pcm16_identical": False,
    }
    if not result["same_shape"]:
        return result
    a, b = reference.astype(np.float64), candidate.astype(np.float64)
    delta = b - a
    energy = float(np.square(a).sum())
    noise = float(np.square(delta).sum())
    result.update(
        bitwise_identical=bool(result["same_dtype"] and reference.tobytes() == candidate.tobytes()),
        pcm16_identical=bool(
            np.array_equal(
                np.rint(reference * 32767).astype("<i2"), np.rint(candidate * 32767).astype("<i2")
            )
        )
        if pcm
        else None,
        relative_l2=float(np.sqrt(noise / max(energy, 1e-24))),
        max_abs=float(np.abs(delta).max()),
        scaled_max_abs=float(np.abs(delta).max() / max(float(np.abs(a).max()), 1e-12)),
        snr_db=float(10 * np.log10(energy / noise)) if energy > 0 and noise > 0 else None,
        snr_infinite=noise == 0 and energy > 0,
    )
    return result


def select_ranked(audits, count):
    eligible = [
        entry
        for entry in audits
        if entry["captured_calls"] == len(CAPTURE_STEPS)
        and entry["relative_l2"] <= 0.01
        and entry["scaled_max_abs"] <= 0.01
    ]
    ranked = sorted(
        audits,
        key=lambda entry: (max(entry["relative_l2"], entry["scaled_max_abs"]), entry["name"]),
    )
    eligible_names = {entry["name"] for entry in eligible}
    selected = [entry["name"] for entry in ranked if entry["name"] in eligible_names][:count]
    return ranked, selected


class NarDownAdapter:
    """Instance-only bridge for the existing yue2mps2 FFN hook; no class/global patch.

    gate/up remains one native packed GEMM. Only independent down projections can be
    selected: q/k/v and gate/up views are inseparable and deliberately not candidates.
    """

    def __init__(self, model, names, scope):
        self.model = weakref.ref(model)
        self.names = names
        self.scope = scope
        self._closed = False
        self.mode = "float"
        self.capture = True
        self.captures = {name: [] for name in names}
        self.capture_bytes = 0
        self.seen = {name: 0 for name in names}
        self.baselines = {}
        self.quantized = {}
        self.metrics = {}

    def __call__(self, index, x, *, context=None):
        import torch
        import torch.nn.functional as functional

        if self._closed or torch.is_grad_enabled():
            raise RuntimeError("NAR适配器已关闭或不是无梯度推理")
        model = self.model()
        packed = model.nar_weights()[index][6]
        gate, up = functional.linear(x, packed).chunk(2, dim=-1)
        hidden = functional.silu(gate) * up
        name = f"nar_layers.{index}.mlp.down_proj"
        if self.capture and name in self.captures:
            step = self.seen[name]
            self.seen[name] += 1
            if step in CAPTURE_STEPS:
                size = hidden.numel() * hidden.element_size()
                if self.capture_bytes + size > CAPTURE_BUDGET:
                    raise MemoryError("真实激活捕获超过128MiB，拒绝缩小激活伪装校准")
                self.captures[name].append((step, hidden.detach().cpu()))
                self.capture_bytes += size
        module = (
            self.baselines[name]
            if self.mode == "float" and name in self.baselines
            else model.get_submodule(name)
        )
        result = module(hidden)
        self.metrics["calls"] = self.metrics.get("calls", 0) + 1
        return result

    def close(self):
        self.capture = False
        self.captures.clear()
        self.baselines.clear()
        self.quantized.clear()
        self._closed = True


def memory_snapshot():
    import mlx.core as mx
    import torch

    torch.mps.synchronize()
    mx.synchronize()
    return {
        "torch_current_bytes": int(torch.mps.current_allocated_memory()),
        "torch_driver_bytes": int(torch.mps.driver_allocated_memory()),
        "mlx_active_bytes": int(mx.get_active_memory()),
        "mlx_cache_bytes": int(mx.get_cache_memory()),
        "mlx_peak_bytes": int(mx.get_peak_memory()),
        "mlx_peak_scope": "entire-worker-lifetime; not per-request or per-route",
        "torch_peak_bytes": None,
        "torch_peak_note": "MPS allocator exposes current/driver usage, not a request peak",
        "worker_rss_bytes": psutil.Process().memory_info().rss,
        "available_bytes": psutil.virtual_memory().available,
    }


class QuantizationPreparation:
    """Official string-only worker_extension RPC, restricted to this isolated worker."""

    def quant_install(self, requested, input_record):
        import torch

        from vllm_omni_metal.ane.runtime import execution_scope

        model = AnePreparation._idle_model(self)
        previous = model._metal_ffn
        if os.environ.get("VLLM_OMNI_METAL_ANE") != "off" or previous is None:
            raise RuntimeError("需要ANE显式off且已加载yue2mps2健康组")
        if previous._closed or previous.disabled or previous._benchmark:
            raise RuntimeError("原FFN组不健康，不接管")
        names = candidate_names(len(model.nar_layers), requested)
        for name in names:
            index = int(name.split(".")[1])
            module = model.get_submodule(name)
            if (
                type(module) is not torch.nn.Linear
                or module.bias is not None
                or module.weight is not model.nar_weights()[index][7]
                or not module.weight.is_contiguous()
            ):
                raise ValueError("候选不是独立原生down权重，拒绝融合/共享拆分")
        source = Path(inspect.getfile(type(model)))
        revision = {
            "model_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "input": input_record,
            "candidates": names,
            "capture_steps": list(CAPTURE_STEPS),
        }
        scope = execution_scope(json.dumps(revision, sort_keys=True), model.vllm_config)
        candidate = NarDownAdapter(model, names, scope)
        AnePreparation._idle_model(self)
        model._metal_ffn = candidate
        previous.close()
        return success(
            {
                "candidates": names,
                "scope": scope,
                **revision,
                "execution_chain": "CachedNAR._velocity -> _metal_ffn -> packed gate/up -> nn.Linear down",
            }
        )

    def quant_environment(self):
        from importlib.metadata import version

        environment = AnePreparation.ane_environment(self)["result"]
        environment.update(
            mlx=version("mlx"),
            python=sys.executable,
            unsafe_serialization=os.environ.get("VLLM_ALLOW_INSECURE_SERIALIZATION", "0"),
        )
        return success(environment)

    def quant_snapshot(self):
        model = self.get_model()
        adapter = model._metal_ffn
        return success(
            {
                "pending": len(self.model_runner.requests)
                + len(model._states)
                + int(getattr(model, "_pending", None) is not None),
                "mode": adapter.mode,
                "adapter_calls": adapter.metrics.get("calls", 0),
                "quantized": {
                    name: {
                        "metrics": dict(module.metrics),
                        "storage_bytes": module.storage_bytes,
                        "source_bytes": module.source_bytes,
                    }
                    for name, module in adapter.quantized.items()
                },
                "benchmark_held_float_bytes": sum(
                    module.weight.numel() * module.weight.element_size()
                    for module in adapter.baselines.values()
                ),
                "memory": memory_snapshot(),
            }
        )

    def quant_audit(self, count):
        import torch

        from vllm_omni_metal.quantization.linear import QuantizedLinear, build_profile

        model = AnePreparation._idle_model(self)
        adapter = model._metal_ffn
        adapter.capture = False
        audits = []
        with torch.inference_mode():
            for name in adapter.names:
                module = model.get_submodule(name)
                quantized = QuantizedLinear.from_linear(module)
                errors = []
                for step, activation in adapter.captures[name]:
                    x = activation.to(device=module.weight.device)
                    expected, actual = module(x), quantized(x)
                    if not torch.isfinite(actual).all():
                        raise ValueError(f"{name}:量化输出非有限")
                    a = expected.float().cpu().numpy()
                    b = actual.float().cpu().numpy()
                    errors.append(
                        {"step": step, "shape": list(x.shape), **compare_arrays(a, b, pcm=False)}
                    )
                    del x, expected, actual
                if not errors:
                    raise RuntimeError(f"{name}:未捕获真实执行激活")
                audits.append(
                    {
                        "name": name,
                        "captured_calls": len(errors),
                        "actual_calls": adapter.seen[name],
                        "samples": errors,
                        "relative_l2": max(error["relative_l2"] for error in errors),
                        "scaled_max_abs": max(error["scaled_max_abs"] for error in errors),
                        "storage_bytes": quantized.storage_bytes,
                        "source_bytes": quantized.source_bytes,
                    }
                )
                del quantized
        ranking, eligible = select_ranked(audits, count)
        local_pass = len(eligible) == count
        # Locally rejected candidates are still measured explicitly for full evidence,
        # but can never be approved or saved as an approved profile.
        selected = eligible if local_pass else [entry["name"] for entry in ranking[:count]]
        adapter.captures.clear()
        adapter.capture_bytes = 0
        modules = {name: model.get_submodule(name) for name in adapter.names}
        with torch.inference_mode():
            profile = build_profile(
                modules,
                selected,
                scope=adapter.scope,
                calibration={
                    "ranking": ranking,
                    "capture_steps": list(CAPTURE_STEPS),
                    "local_error_limit": 0.01,
                    "local_pass": local_pass,
                    "approved": False,
                },
            )
        return success(
            {"ranking": ranking, "selected": selected, "profile": profile, "local_pass": local_pass}
        )

    def quant_apply(self, profile):
        import torch

        from vllm_omni_metal.quantization.linear import apply_profile, validate_profile

        model = AnePreparation._idle_model(self)
        adapter = model._metal_ffn
        if adapter.quantized:
            raise RuntimeError("不能重复替换量化组")
        require_w8_profile(profile)
        modules = {name: model.get_submodule(name) for name in adapter.names}
        with torch.inference_mode():
            names = validate_profile(profile, modules, scope=adapter.scope)
            baselines = {name: modules[name] for name in names}
            quantized = apply_profile(model, profile, scope=adapter.scope)
        adapter.baselines = baselines  # Benchmark owns these references, not QuantizedLinear.
        adapter.quantized = quantized
        # CachedNAR still unpacks slot 7, but the explicit FFN hook never consumes it.
        # Clearing it prevents a hidden persistent float copy during quantized-only use.
        for name in names:
            index = int(name.split(".")[1])
            weights = model.nar_fused[index]
            model.nar_fused[index] = (*weights[:7], None)
        return success(
            {
                "selected": names,
                "benchmark_retains_float": True,
                "source_bytes": sum(module.source_bytes for module in quantized.values()),
                "storage_bytes": sum(module.storage_bytes for module in quantized.values()),
            }
        )

    def quant_mode(self, mode):
        model = AnePreparation._idle_model(self)
        adapter = model._metal_ffn
        if mode not in ("float", "w8a16") or not adapter.quantized:
            raise ValueError("无已安装量化候选或模式非法")
        if mode == "float" and not adapter.baselines:
            raise RuntimeError("基准浮点已释放，不得伪装基线")
        adapter.mode = mode
        adapter.metrics.clear()
        for module in adapter.quantized.values():
            module.metrics.update(calls=0, rows=0)
        return success({"mode": mode})

    def quant_approve_profile(self, profile, completion):
        import torch

        from vllm_omni_metal.quantization.linear import build_profile, validate_profile

        model = AnePreparation._idle_model(self)
        adapter = model._metal_ffn
        if completion.get("approved") is not True or not adapter.baselines:
            raise RuntimeError("全流程未通过或基准浮点已释放，不能重绑定批准profile")
        require_w8_profile(profile)
        with torch.inference_mode():
            names = validate_profile(profile, adapter.baselines, scope=adapter.scope)
            if set(names) != set(adapter.quantized):
                raise ValueError("批准选择与实际量化模块不一致")
            calibration = copy.deepcopy(profile["calibration"])
            calibration.update(approved=True, full_pipeline=copy.deepcopy(completion))
            approved = build_profile(
                adapter.baselines, names, scope=adapter.scope, calibration=calibration
            )
            validate_profile(approved, adapter.baselines, scope=adapter.scope)
        return success({"profile": approved})

    def quant_release_baseline(self):
        import mlx.core as mx
        import torch

        model = AnePreparation._idle_model(self)
        adapter = model._metal_ffn
        if adapter.mode != "w8a16":
            raise RuntimeError("仅量化模式可释放基准浮点引用")
        released = sum(
            module.weight.numel() * module.weight.element_size()
            for module in adapter.baselines.values()
        )
        adapter.baselines.clear()
        gc.collect()
        torch.mps.empty_cache()
        mx.clear_cache()
        result = self.quant_snapshot()["result"]
        result.update(
            released_benchmark_float_bytes=released,
            independent_process=False,
            memory_claim="同worker释放所有选中down浮点引用后的观测；非独立冷启动净RSS验收",
        )
        return success(result)


async def worker(args):
    from importlib.metadata import version

    import numpy as np
    import torch
    import yaml
    from vllm_omni.entrypoints.async_omni import AsyncOmni
    from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest
    from vllm_omni.entrypoints.openai.tts_adapters.yue2 import Yue2Adapter

    if "yue2mps2" not in version("vllm-omni"):
        raise RuntimeError("必须使用现有yue2mps2私有overlay；本工具不安装/升级环境")
    if os.environ.get("VLLM_OMNI_METAL_ANE") != "off":
        raise RuntimeError("本次仅量化，ANE必须显式off")
    output = Path(args.output)
    deploy = yaml.safe_load(Path(args.deploy_config).read_text())
    if len(deploy["stages"]) != 1 or deploy.get("pipeline") != "yue2":
        raise ValueError("必须保留原生单阶段YuE2拓扑")
    mixed = args.precisions is not None
    route = "mixed" if mixed else "w8a16"
    deploy["stages"][0]["worker_extension_cls"] = (
        "mixed_quantization.MixedPreparation"
        if mixed
        else "prepare_quantization.QuantizationPreparation"
    )
    deploy_path = output / "deploy.yaml"
    deploy_path.write_text(yaml.safe_dump(deploy))
    engine, records, audio_by_mode = None, [], {}
    adapter_installed = False

    async def call(method, *values):
        return await rpc(engine, output, method, *values)

    async def idle():
        for _ in range(50):
            status = await call(
                "mixed_native_snapshot" if mixed and not adapter_installed else "quant_snapshot"
            )
            if status["pending"] == 0:
                return status
            await asyncio.sleep(0.1)
        raise RuntimeError("请求pending未清零")

    def array(value):
        if isinstance(value, (list, tuple)):
            if len(value) != 1:
                raise ValueError("必须单请求音频")
            return array(value[0])
        return (
            value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
        )

    try:
        sources = weight_fingerprint(args.model, args.vae)
        write_json(output / "source-weights.json", sources)
        engine = AsyncOmni(
            model=args.model,
            deploy_config=str(deploy_path),
            stage_init_timeout=360,
            init_timeout=420,
            log_stats=False,
            output_modalities=["audio"],
        )
        adapter = Yue2Adapter(SimpleNamespace(engine_client=engine))
        request = OpenAICreateSpeechRequest(
            model="yue2-quantization-calibration",
            input="[Verse]\n今晚星光照亮街道\n我们一起追逐微笑",
            instructions="Mandarin pop, warm lead vocal, piano and acoustic guitar, at 100 BPM",
            extra_params={"cot": "off"},
            response_format="wav",
            stream=False,
            seed=42,
            max_new_tokens=args.frames,
        )
        if adapter.validate(request) is not None:
            raise ValueError("上游拒绝真实请求")
        prepared = await adapter.build(
            request, copy.deepcopy(engine.default_sampling_params_list), False
        )
        params = adapter.apply_sampling_overrides(
            copy.deepcopy(engine.default_sampling_params_list),
            request,
            prepared.prompt,
            request_id="quant-prepare",
        )
        input_record = {
            "frames": args.frames,
            "seed": 42,
            "lyrics": request.input,
            "style": request.instructions,
            "prompt": prepared.prompt,
            "sampling": {
                "temperature": params[0].temperature,
                "top_p": params[0].top_p,
                "top_k": params[0].top_k,
                "max_tokens": params[0].max_tokens,
            },
            "extra_args": params[0].extra_args,
            "source_weights": sources,
        }
        write_json(output / "input.json", input_record)

        async def generate(mode, *, warmup=False, capture=False, released=False, initial=False):
            await idle()
            if not capture and not initial:
                await call("quant_mode", mode)
            index = len(records)
            write_json(
                output / "progress.json",
                {"request_started": time.time(), "index": index, "mode": mode, "capture": capture},
            )
            received = []
            started = time.perf_counter()
            async with asyncio.timeout(180):
                async for result in engine.generate(
                    copy.deepcopy(prepared.prompt),
                    request_id=f"quant-{index}",
                    sampling_params_list=copy.deepcopy(params),
                    output_modalities=["audio"],
                ):
                    if (
                        result.multimodal_output
                        and result.multimodal_output.get("audio") is not None
                    ):
                        received.append(result.multimodal_output)
            elapsed = time.perf_counter() - started
            write_json(output / "progress.json", {})
            status = await idle()
            if len(received) != 1:
                raise RuntimeError("没有唯一终端真实音频")
            audio = array(received[0]["audio"]).squeeze()
            sr = int(array(received[0].get("sr", 48000)).reshape(-1)[0])
            if audio.ndim != 2 or 2 not in audio.shape:
                raise ValueError("需要真实双声道音频")
            if audio.shape[0] == 2:
                audio = audio.T
            audio = np.ascontiguousarray(audio)
            if (
                sr != 48000
                or not np.isfinite(audio).all()
                or audio.std() <= 1e-6
                or len(audio) < 5 * sr
                or np.abs(audio).max() > 1
            ):
                raise ValueError("音频格式/有限值/非静音/时长/幅度检查失败")
            if mode == route and (
                not status["quantized"]
                or any(
                    entry["metrics"].get("calls", 0) <= 0 or entry["metrics"].get("rows", 0) <= 0
                    for entry in status["quantized"].values()
                )
            ):
                raise RuntimeError("量化候选没有真实调用，不能记录为量化验证")
            if mode == "float" and any(
                entry["metrics"].get("calls", 0) != 0 for entry in status["quantized"].values()
            ):
                raise RuntimeError("原精度路线意外命中量化forward")
            if mixed and adapter_installed and status.get("retained_identity") is not True:
                raise RuntimeError("BF16模块或参数身份改变")
            stem = f"{index}-{mode}"
            np.save(output / f"{stem}.npy", audio, allow_pickle=False)
            pcm = np.rint(audio * 32767).astype("<i2")
            wav_path = output / f"{stem}.wav"
            with wave.open(str(wav_path), "wb") as wav:
                wav.setnchannels(2)
                wav.setsampwidth(2)
                wav.setframerate(sr)
                wav.writeframes(pcm.tobytes())
            record = {
                "mode": mode,
                "warmup": warmup,
                "capture": capture,
                "baseline_released": released,
                "seconds": elapsed,
                "audio_seconds": len(audio) / sr,
                "rtf": elapsed / (len(audio) / sr),
                "shape": list(audio.shape),
                "dtype": str(audio.dtype),
                "sample_rate": sr,
                "seed": 42,
                "frames": args.frames,
                "npy": f"{stem}.npy",
                "wav": wav_path.name,
                "audio_sha256": hashlib.sha256(audio.tobytes()).hexdigest(),
                "npy_sha256": hashlib.sha256((output / f"{stem}.npy").read_bytes()).hexdigest(),
                "pcm16_sha256": hashlib.sha256(pcm.tobytes()).hexdigest(),
                "wav_sha256": hashlib.sha256(wav_path.read_bytes()).hexdigest(),
                **status,
            }
            records.append(record)
            write_json(output / "results.json", records)
            if not warmup and not released:
                audio_by_mode.setdefault(mode, []).append(audio)
            return record

        native_record = await generate("native", warmup=True, initial=True) if mixed else None
        installed = (
            await call("mixed_install", args.precisions, input_record)
            if mixed
            else await call("quant_install", args.candidate_layers, input_record)
        )
        adapter_installed = True
        write_json(output / "adapter.json", installed)
        environment = await call("quant_environment")
        environment["explicit_precisions"] = args.precisions
        environment["no_sensitivity_search"] = mixed
        write_json(output / "environment.json", environment)
        if mixed and not environment["torch_stream_attention_available"]:
            raise RuntimeError("本次混合精度对照要求已显式准备的Torch-stream launcher")

        reference_record = await generate("float", warmup=True, capture=not mixed, initial=mixed)
        native_comparison = None
        if mixed:
            native_comparison = compare_arrays(
                np.load(output / native_record["npy"], allow_pickle=False),
                np.load(output / reference_record["npy"], allow_pickle=False),
            )
            write_json(output / "native-reference-comparison.json", native_comparison)
            if not native_comparison["bitwise_identical"]:
                raise RuntimeError("本次hook原精度与独立原生参考不一致，停止量化对照")
        audit = (
            await call("mixed_prepare") if mixed else await call("quant_audit", args.select_count)
        )
        write_json(output / "calibration.json", audit)
        profile = audit["profile"]
        write_json(output / "candidate-profile.json", profile)
        if mixed:
            write_json(output / "applied.json", audit)
        else:
            write_json(output / "applied.json", await call("quant_apply", profile))
        await generate(route, warmup=True)
        for mode in ("float", route, route, "float"):
            await generate(mode)
        summaries = {}
        for mode in ("float", route):
            samples = [
                record for record in records if record["mode"] == mode and not record["warmup"]
            ]
            summaries[mode] = {
                "seconds": [record["seconds"] for record in samples],
                "median_seconds": float(np.median([record["seconds"] for record in samples])),
                "median_rtf": float(np.median([record["rtf"] for record in samples])),
                "seed_repeatable": len({record["audio_sha256"] for record in samples}) == 1,
                "sample_count": len(samples),
            }
        comparisons = [
            compare_arrays(a, b)
            for a, b in zip(audio_by_mode["float"], audio_by_mode[route], strict=True)
        ]
        same_work = all(item["same_shape"] for item in comparisons)
        lossless = all(item["bitwise_identical"] for item in comparisons)
        deterministic = all(summary["seed_repeatable"] for summary in summaries.values())
        speed_ratio = (
            summaries["float"]["median_seconds"] / summaries[route]["median_seconds"]
            if same_work
            else None
        )
        stable_speedup = (
            same_work
            and speed_ratio > 1.03
            and max(summaries[route]["seconds"]) < min(summaries["float"]["seconds"])
        )
        approved = bool(
            not mixed and audit["local_pass"] and lossless and deterministic and stable_speedup
        )
        completion = {
            "status": "approved" if approved else "rejected",
            "approved": approved,
            "lossless_definition": "原始返回浮点音频dtype/shape/bytes逐位一致；PCM相同不替代",
            "lossless": lossless,
            "seed_repeatable": deterministic,
            "same_workload": same_work,
            "local_activation_quality_pass": audit["local_pass"],
            "routes": summaries,
            "cross_route_audio": comparisons,
            f"float_over_{route}_speed_ratio": speed_ratio,
            "stable_speedup": bool(stable_speedup),
            "benchmark_retains_selected_float": True,
            "production_net_memory_reduction_claimed": False,
            "default_profile_published": False,
            "explicit_precisions": args.precisions,
            "no_sensitivity_search": mixed,
            "frame_budget_limited": True,
            "full_song_quality_validated": False,
            "native_reference_vs_float": native_comparison,
            "rejection_reasons": [
                reason
                for failed, reason in (
                    (audit["local_pass"] is False, "真实激活1%局部门槛未通过，仅诊断A/B"),
                    (mixed, "固定混合精度诊断不搜索敏感度、不发布默认批准"),
                    (not lossless, "原始浮点音频非逐位一致"),
                    (not deterministic, "同路径同seed不稳定"),
                    (not same_work, "音频形状/工作量不同"),
                    (not stable_speedup, "未通过稳定整流程3%加速门槛"),
                )
                if failed
            ],
        }
        # Save complete A/B evidence before the irreversible benchmark-reference release.
        write_json(output / "comparison.json", completion)
        approved_profile = (
            (await call("quant_approve_profile", profile, completion))["profile"]
            if completion["approved"]
            else None
        )
        await call("quant_mode", route)
        write_json(output / "quantized-only-memory.json", await call("quant_release_baseline"))
        released_record = await generate(route, warmup=True, released=True)
        completion["released_float_output_matches_candidate"] = released_record[
            "audio_sha256"
        ] == next(
            record["audio_sha256"]
            for record in records
            if record["mode"] == route and not record["warmup"]
        )
        if not completion["released_float_output_matches_candidate"]:
            completion["approved"] = False
            completion["status"] = "rejected"
            completion["rejection_reasons"].append("释放基准浮点后未复现候选输出")
        if completion["approved"]:
            write_json(output / "approved-profile.json", approved_profile)
            completion["approved_profile_saved"] = True
        write_json(output / "completed.json", completion)
    finally:
        if engine is not None:
            engine.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    cache = Path.home() / ".cache/modelscope/hub/models/m-a-p"
    parser.add_argument("--model", default=os.environ.get("YUE2_MODEL_DIR", str(cache / "YuE2-3B")))
    parser.add_argument("--vae", default=os.environ.get("YUE2_VAE_DIR", str(cache / "YuE2-Vae")))
    parser.add_argument("--frames", type=int, default=512)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--candidate-layers", default="auto", help="auto均匀4层，或1–4层索引逗号分隔；仅down"
    )
    selection.add_argument(
        "--precisions", help="显式JSON模块精度映射；必须包含W4/W8/BF16，不运行敏感度搜索"
    )
    parser.add_argument("--select-count", type=int, default=1, choices=(1, 2))
    parser.add_argument(
        "--deploy-config",
        default=str(Path(__file__).resolve().parents[3] / "composes/yue2-3b/metal-deploy.yaml"),
    )
    parser.add_argument("--output", help="新建的证据目录（可位于主仓库outputs）；必须为空/不存在")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--gpu-only", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.frames != 512:
        parser.error("真实验收固定512帧seed42，不能缩短请求")
    if os.environ.get("VLLM_ALLOW_INSECURE_SERIALIZATION", "0") not in ("0", ""):
        parser.error("禁止不安全pickle RPC")
    os.environ["VLLM_OMNI_METAL_ANE"] = "off"
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "0"
    os.environ["METAL_QUANTIZATION"] = "off"
    os.environ["METAL_QUANTIZATION_PROFILE"] = ""
    os.environ["METAL_MODEL_OPTIMIZERS"] = ""
    os.environ["METAL_ANE_PROFILE"] = ""
    os.environ["METAL_STREAMING_PROFILE"] = ""
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "composes/yue2-3b"))
    from prepare import validate_model

    args.model = str(validate_model(args.model, "yue2"))
    args.vae = str(validate_model(args.vae, "yue2_vae"))
    if args.precisions is not None:
        from mixed_quantization import validate_precisions

        if args.select_count != 1:
            parser.error("--precisions不接受--select-count搜索参数")
        try:
            args.precisions = validate_precisions(
                json.loads(args.precisions, object_pairs_hook=precision_object),
                json.loads((Path(args.model) / "config.json").read_text())["num_hidden_layers"],
            )
        except (ValueError, TypeError, KeyError) as error:
            parser.error(str(error))
    if args.worker:
        asyncio.run(worker(args))
        return
    available = psutil.virtual_memory().available
    directory = (
        Path(args.output).expanduser().resolve()
        if args.output
        else Path(tempfile.mkdtemp(prefix="yue2-w8a16-"))
    )
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError("证据目录必须为空，拒绝覆盖旧结果")
    directory.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(directory).free < 2 * 1024**3:
        raise OSError("证据磁盘至少保留2GiB")
    write_json(
        directory / "preflight.json",
        {
            "phase": "before-supervisor",
            "available_bytes": available,
            "required_available_bytes": 16 * 1024**3,
            "free_disk_bytes": shutil.disk_usage(directory).free,
            "precisions": args.precisions,
            "model": args.model,
            "vae": args.vae,
            "model_process_started": False,
            "startup_gate_pass": available >= 16 * 1024**3,
        },
    )
    if available < 16 * 1024**3:
        raise MemoryError(
            f"启动前可用RAM {available / 1024**3:.3f}GiB，小于16GiB；"
            f"未启动模型、不停止服务或提高水位。证据：{directory / 'preflight.json'}"
        )
    args.gpu_only, args.attention_ab = True, False
    supervise(
        args,
        script=Path(__file__).resolve(),
        directory=directory,
        extra_args=(
            ("--precisions", json.dumps(args.precisions))
            if args.precisions is not None
            else (
                "--candidate-layers",
                args.candidate_layers,
                "--select-count",
                str(args.select_count),
            )
        ),
        label="W4/W8/BF16固定混合精度真实A/B"
        if args.precisions is not None
        else "W8A16真实校准A/B",
    )


if __name__ == "__main__":
    main()
