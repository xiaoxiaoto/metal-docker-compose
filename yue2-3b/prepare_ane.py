#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""以真实 YuE2 请求离线校准 ANE，或显式测量 GPU（不批准 ANE）。"""

import argparse
import asyncio
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

import psutil
from prepare import validate_model


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def flatten(values):
    result = []
    for value in values:
        result.extend(flatten(value) if isinstance(value, list) else [value])
    return result


async def rpc(engine, output, method, *values):
    """Reject upstream error dictionaries before any downstream request can run."""
    replies = flatten(
        await engine.collective_rpc(
            method, args=values, timeout=600 if method == "ane_prepare" else 180, stage_ids=[0]
        )
    )
    if len(replies) != 1 or not isinstance(replies[0], dict):
        write_json(output / "rpc-error.json", {"method": method, "replies": replies})
        raise RuntimeError(f"{method}: 校准仅支持本机单 worker；原始回复已保存")
    reply = replies[0]
    if (
        reply.get("supported") is not True
        or reply.get("error")
        or reply.get("todo")
        or not isinstance(reply.get("result"), dict)
    ):
        write_json(output / "rpc-error.json", {"method": method, "reply": reply})
        raise RuntimeError(f"{method} failed: {reply!r}")
    return reply["result"]


def success(result):
    return {"supported": True, "result": result}


class AnePreparation:
    """String RPCs on this new isolated worker, never a callable/pickle or model registry patch."""

    def _idle_model(self):
        model = self.get_model()
        if self.model_runner.requests or model._states:
            raise RuntimeError("请求尚未清理，不能准备或切换离线算子")
        return model

    def _install_group(self, model, previous, candidate, *, require_pool):
        try:
            self._idle_model()
            if model._metal_ffn is not previous:
                raise RuntimeError("准备期间 FFN 组已改变，拒绝替换")
            if (
                candidate.disabled
                or candidate._closed
                or (require_pool and candidate._pool is None)
            ):
                raise RuntimeError("ANE 图加载或内存预算未通过，不发布校准")
        except BaseException:
            candidate.close()
            raise
        model._metal_ffn = candidate
        previous.close()

    def ane_prepare(self):
        import torch

        from vllm_omni_metal.ane.runtime import DenseSwiGLU
        from vllm_omni_metal.ane.build import prepare_group

        with torch.inference_mode():
            model = self._idle_model()
            group = model._metal_ffn
            if group is None or group.disabled or group._closed:
                raise RuntimeError("需要配套 yue2mps2 私有补丁和健康 MPS FFN 组")
            weights, scope = group.weights, group.scope
            prepare_group(weights, scope=scope)
            candidate = DenseSwiGLU(weights, scope=scope, precision="fp16", _benchmark=True)
            self._install_group(model, group, candidate, require_pool=True)
            return success({"layers": len(weights), "identity": candidate.identity})

    def ane_set_mode(self, chunks):
        model = self._idle_model()
        group = model._metal_ffn
        if (
            group is None
            or not group._benchmark
            or group.disabled
            or group._closed
            or type(chunks) is not int
            or chunks < 0
        ):
            raise RuntimeError("校准组不可用")
        group._benchmark_chunks = chunks
        group.metrics.clear()
        return success({"chunks": chunks})

    def ane_snapshot(self):
        model = self.get_model()
        group = model._metal_ffn
        if group is None:
            raise RuntimeError("缺少 MPS FFN 组")
        route = getattr(self, "_gpu_attention_state", {})
        return success(
            {
                "metrics": dict(group.metrics),
                "disabled": group.disabled,
                "benchmark": group._benchmark,
                "closed": group._closed,
                "pending": len(self.model_runner.requests) + len(model._states),
                "ane_mode": os.environ.get("VLLM_OMNI_METAL_ANE", "auto"),
                "attention_route": route.get("route"),
                "attention_forward_calls": route.get("forward_calls", 0),
                "torch_stream_forward_calls": route.get("torch_calls", 0),
                "checks": [
                    {"layer": i, "rows": r, "chunks": c, "context": context, **errors}
                    for (i, r, c, context), errors in group.checks.items()
                ],
            }
        )

    def ane_approve(self, rows, baseline, candidates):
        from vllm_omni_metal.ane.runtime import DenseSwiGLU
        from vllm_omni_metal.ane.build import publish_profile

        model = self._idle_model()
        previous = model._metal_ffn
        if previous is None or not previous._benchmark or previous.disabled or previous._closed:
            raise RuntimeError("校准组不可用")
        contexts = {key[3] for key in model._metal_ffn.checks if key[1] == rows}
        if len(contexts) != 1:
            raise ValueError("完整流水线的实际前缀形状不一致，拒绝批准")
        record = publish_profile(
            model._metal_ffn, rows, baseline, dict(candidates), context=contexts.pop()
        )
        # Exercise the default constructor/file reader without closing a healthy group first.
        candidate = DenseSwiGLU(previous.weights, scope=previous.scope, precision="fp16")
        self._install_group(model, previous, candidate, require_pool=record["chunks"] > 0)
        return success(record)

    def gpu_attention_route(self, route):
        """Isolated, idle-only benchmark switch; never changes model methods or ANE policy."""
        from vllm_omni_metal.ops import metal_attention, mps_attention

        model = self._idle_model()
        group = model._metal_ffn
        if (
            os.environ.get("VLLM_OMNI_METAL_ANE") != "off"
            or group is None
            or group._benchmark
            or group.disabled
            or group._closed
            or route not in ("current", "torch-stream", "mlx-bridge")
        ):
            raise RuntimeError("仅显式 GPU-only 的健康空闲组允许切换 attention")
        if not hasattr(self, "_gpu_attention_original"):
            self._gpu_attention_original = (
                mps_attention.get_mps_ops,
                metal_attention.paged_attention,
            )
        factory, attention = self._gpu_attention_original
        native = factory() if route != "mlx-bridge" else None
        if route == "torch-stream" and native is None:
            raise RuntimeError("Torch-stream launcher 未命中；不能计为优化")
        state = {
            "route": "torch-stream" if native is not None else "mlx-bridge",
            "forward_calls": 0,
            "torch_calls": 0,
        }

        class CountedNative:
            def forward(self, *args, **kwargs):
                result = native.forward(*args, **kwargs)
                state["torch_calls"] += 1
                return result

        counted = CountedNative() if native is not None else None

        def counted_attention(*args, **kwargs):
            result = attention(*args, **kwargs)
            state["forward_calls"] += 1
            return result

        mps_attention.get_mps_ops = lambda: counted
        metal_attention.paged_attention = counted_attention
        self._gpu_attention_state = state
        group.metrics.clear()
        return success({"route": state["route"]})

    def ane_environment(self):
        from dataclasses import asdict
        from importlib.metadata import version

        import torch

        from vllm_omni_metal.ops import mps_attention, aot_kernels

        model = self.get_model()
        config = model.vllm_config

        def artifact(path):
            return {
                "path": str(path),
                "exists": path.is_file(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None,
            }

        implementations = sorted(
            {
                f"{type(module.impl).__module__}.{type(module.impl).__qualname__}"
                for module in model.modules()
                if hasattr(module, "impl")
            }
        )
        return success(
            {
                "torch": str(torch.__version__),
                "omni": version("vllm-omni"),
                "ane_mode": os.environ.get("VLLM_OMNI_METAL_ANE", "auto"),
                "dtype": str(config.model_config.dtype),
                "parameter_dtypes": sorted({str(p.dtype) for p in model.parameters()}),
                "vae_dtypes": sorted({str(p.dtype) for p in model._vae.parameters()}),
                "attention_backend": str(config.attention_config.backend),
                "attention_implementations": implementations,
                "torch_stream_attention_available": mps_attention.get_mps_ops() is not None,
                "torch_stream_launcher": artifact(mps_attention._artifact_path()),
                "elementwise": {
                    str(dtype): artifact(aot_kernels._artifact_path(dtype))
                    for dtype in (torch.float32, torch.float16, torch.bfloat16)
                },
                "custom_ops": config.compilation_config.custom_ops,
                "ir_priorities": asdict(config.kernel_config.ir_op_priority),
                "ffn_scope": model._metal_ffn.scope,
                "max_model_len": config.model_config.max_model_len,
                "gpu_memory_utilization": config.cache_config.gpu_memory_utilization,
                "kv_cache_memory_bytes": config.cache_config.kv_cache_memory_bytes,
            }
        )


async def worker(args):
    from importlib.metadata import version
    from types import SimpleNamespace

    import numpy as np
    import torch
    import yaml
    from vllm_omni.entrypoints.async_omni import AsyncOmni
    from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest
    from vllm_omni.entrypoints.openai.tts_adapters.yue2 import Yue2Adapter

    if "yue2mps2" not in version("vllm-omni"):
        raise RuntimeError("请先用配套 prepare_runtime.py 构建新的私有 overlay")
    output = Path(args.output)
    profile = yaml.safe_load(Path(args.deploy_config).read_text())
    if len(profile["stages"]) != 1:
        raise ValueError("校准必须保留原生单阶段拓扑")
    # Only instrumentation changes; keep the selected service's resource/attention settings.
    profile["stages"][0]["worker_extension_cls"] = "prepare_ane.AnePreparation"
    deploy = output / "deploy.yaml"
    deploy.write_text(yaml.safe_dump(profile))
    engine = None
    records = []

    async def call(method, *values):
        return await rpc(engine, output, method, *values)

    async def idle_snapshot():
        for _ in range(50):
            status = await call("ane_snapshot")
            if status["pending"] == 0:
                return status
            await asyncio.sleep(0.1)
        raise RuntimeError("校准请求清理超时")

    def array(value):
        if isinstance(value, (list, tuple)):
            if len(value) != 1:
                raise ValueError("需要单请求音频")
            return array(value[0])
        return (
            value.detach().float().cpu().numpy()
            if isinstance(value, torch.Tensor)
            else np.asarray(value)
        )

    try:
        engine = AsyncOmni(
            model=args.model,
            deploy_config=str(deploy),
            stage_init_timeout=360,
            init_timeout=420,
            log_stats=False,
            output_modalities=["audio"],
        )
        write_json(output / "environment.json", await call("ane_environment"))
        if args.gpu_only:
            if os.environ.get("VLLM_OMNI_METAL_ANE") != "off":
                raise RuntimeError("GPU-only 子进程必须显式设置 VLLM_OMNI_METAL_ANE=off")
        else:
            write_json(output / "prepared.json", await call("ane_prepare"))
        adapter = Yue2Adapter(SimpleNamespace(engine_client=engine))
        request = OpenAICreateSpeechRequest(
            model="yue2-ane-calibration",
            input="[Verse]\n今晚星光照亮街道\n我们一起追逐微笑",
            instructions="Mandarin pop, warm lead vocal, piano and acoustic guitar, at 100 BPM",
            extra_params={"cot": "off"},
            response_format="wav",
            stream=False,
            seed=42,
            max_new_tokens=args.frames,
        )
        if adapter.validate(request) is not None:
            raise ValueError("上游拒绝校准请求")
        prepared = await adapter.build(
            request, copy.deepcopy(engine.default_sampling_params_list), False
        )
        params = adapter.apply_sampling_overrides(
            copy.deepcopy(engine.default_sampling_params_list),
            request,
            prepared.prompt,
            request_id="ane-prepare",
        )
        write_json(
            output / "input.json",
            {
                "frames": args.frames,
                "seed": 42,
                "prompt": prepared.prompt,
                "lyrics": request.input,
                "style": request.instructions,
                "sampling": {
                    "temperature": params[0].temperature,
                    "top_p": params[0].top_p,
                    "top_k": params[0].top_k,
                    "max_tokens": params[0].max_tokens,
                },
                "extra_args": params[0].extra_args,
            },
        )

        route_audio = {}

        async def generate(chunks, warmup=False, default=False, route=None):
            await idle_snapshot()
            if route is not None:
                await call("gpu_attention_route", route)
            elif not default:
                await call("ane_set_mode", chunks)
            index = len(records)
            write_json(
                output / "progress.json",
                {
                    "request_started": time.time(),
                    "index": index,
                    "chunks": chunks,
                    "default": default,
                    "route": route,
                },
            )
            received = []
            started = time.perf_counter()
            async with asyncio.timeout(180):
                async for result in engine.generate(
                    copy.deepcopy(prepared.prompt),
                    request_id=f"ane-{index}",
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
            status = await idle_snapshot()
            if len(received) != 1 or status["disabled"] or status["closed"]:
                raise RuntimeError("没有有效终端音频或 FFN 数值/设备检查失败，不批准该方案")
            if args.gpu_only and (
                status["benchmark"]
                or status["ane_mode"] != "off"
                or status["metrics"].get("ane_chunks", 0)
                or status["attention_forward_calls"] <= 0
                or (
                    status["attention_route"] == "torch-stream"
                    and status["torch_stream_forward_calls"] <= 0
                )
            ):
                raise RuntimeError("GPU-only 未命中声明的 attention 路径或意外使用 ANE")
            audio = array(received[0]["audio"]).squeeze()
            sr = int(array(received[0].get("sr", 48000)).reshape(-1)[0])
            if audio.ndim != 2 or 2 not in audio.shape:
                raise ValueError("需要真实双声道音频")
            if audio.shape[0] == 2:
                audio = audio.T
            if (
                sr != 48000
                or not np.isfinite(audio).all()
                or audio.std() <= 1e-6
                or len(audio) < 5 * sr
                or np.abs(audio).max() > 1
            ):
                raise ValueError("音频格式/有限值/非静音/幅度范围检查失败")
            if chunks and status["metrics"].get("ane_chunks", 0) == 0:
                raise RuntimeError("协同候选未真正调用 CoreML chunk，不能计为 ANE 加速")
            label = status["attention_route"] if args.gpu_only else f"chunks{chunks}"
            wav_path = output / f"{index}-{label}.wav"
            with wave.open(str(wav_path), "wb") as wav:
                wav.setnchannels(2)
                wav.setsampwidth(2)
                wav.setframerate(sr)
                wav.writeframes(np.rint(audio * 32767).astype("<i2").tobytes())
            record = {
                "chunks": chunks,
                "warmup": warmup,
                "default": default,
                "seconds": elapsed,
                "seed": 42,
                "frames": args.frames,
                "wav": wav_path.name,
                "audio_sha256": hashlib.sha256(audio.tobytes()).hexdigest(),
                "audio_seconds": len(audio) / sr,
                **status,
            }
            records.append(record)
            write_json(output / "results.json", records)
            if args.gpu_only and not warmup:
                route_audio.setdefault(status["attention_route"], audio.copy())
            return record

        if args.gpu_only:
            routes = ["mlx-bridge", "torch-stream"] if args.attention_ab else ["current"]
            for route in routes:
                await generate(0, warmup=True, route=route)
            for route in [*routes, *reversed(routes)]:
                await generate(0, route=route)
            measured = [r for r in records if not r["warmup"]]
            summaries = {}
            for route in route_audio:
                samples = [r for r in measured if r["attention_route"] == route]
                if len(samples) < 2 or len({r["audio_sha256"] for r in samples}) != 1:
                    raise RuntimeError("GPU 同路径同seed不能复现；不发布完成记录")
                summaries[route] = {
                    "seconds": [r["seconds"] for r in samples],
                    "median_seconds": float(np.median([r["seconds"] for r in samples])),
                    "audio_sha256": samples[0]["audio_sha256"],
                }
            completion = {
                "mode": "gpu-only",
                "attention_ab": args.attention_ab,
                "seed": 42,
                "frames": args.frames,
                "seed_repeatable_per_route": True,
                "routes": summaries,
                "ane_prepared": False,
                "ane_approved": False,
            }
            if args.attention_ab:
                baseline = route_audio["mlx-bridge"]
                candidate = route_audio["torch-stream"]
                comparison = {"same_shape": baseline.shape == candidate.shape}
                if comparison["same_shape"]:
                    delta = candidate.astype(np.float64) - baseline.astype(np.float64)
                    norm = float(np.linalg.norm(baseline.astype(np.float64)))
                    comparison.update(
                        {
                            "identical": bool(np.array_equal(baseline, candidate)),
                            "relative_l2": float(np.linalg.norm(delta) / max(norm, 1e-12)),
                            "max_abs": float(np.abs(delta).max()),
                        }
                    )
                completion["cross_route_audio"] = comparison
                completion["torch_stream_latency_change_fraction"] = (
                    1
                    - summaries["torch-stream"]["median_seconds"]
                    / summaries["mlx-bridge"]["median_seconds"]
                )
            write_json(output / "completed.json", completion)
            return

        modes = list(range((args.frames + 1) // 256 + 1))
        for mode in modes:
            await generate(mode, warmup=True)
        for mode in [*modes, *reversed(modes)]:
            await generate(mode)
        measured = [r for r in records if not r["warmup"]]
        for mode in modes:
            if len({r["audio_sha256"] for r in measured if r["chunks"] == mode}) != 1:
                raise RuntimeError("同模式同seed不能复现，不批准默认配置")
        baseline = [r["seconds"] for r in measured if r["chunks"] == 0]
        candidates = [
            (mode, [r["seconds"] for r in measured if r["chunks"] == mode]) for mode in modes[1:]
        ]
        record = await call("ane_approve", args.frames + 2, baseline, candidates)
        write_json(output / "approved.json", record)
        first = await generate(record["chunks"], default=True)
        second = await generate(record["chunks"], default=True)
        expected = next(r["audio_sha256"] for r in measured if r["chunks"] == record["chunks"])
        if (
            first["benchmark"]
            or second["benchmark"]
            or first["audio_sha256"] != expected
            or second["audio_sha256"] != expected
        ):
            raise RuntimeError("默认 auto 未复现已校准输出；保留证据，需人工检查配置")
        write_json(output / "completed.json", {"approved": record, "default_seed_repeatable": True})
    finally:
        if engine is not None:
            engine.shutdown()


def supervise(args, *, script=None, directory=None, extra_args=(), label=None):
    if psutil.virtual_memory().available < 16 * 1024**3:
        raise MemoryError("完整校准要求启动前至少16GiB可用RAM；不停止其他服务或提高水位")
    directory = Path(directory) if directory is not None else Path(tempfile.mkdtemp(prefix="yue2-ane-"))
    env = dict(
        os.environ,
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        YUE2_VAE=args.vae,
        VLLM_WORKER_MULTIPROC_METHOD="spawn",
        OMP_NUM_THREADS="2",
    )
    if args.gpu_only:
        env["VLLM_OMNI_METAL_ANE"] = "off"
    env["PYTHONPATH"] = str(Path(__file__).parent) + os.pathsep + env.get("PYTHONPATH", "")
    command = [
        sys.executable,
        str(script or Path(__file__).resolve()),
        "--worker",
        "--output",
        str(directory),
        "--model",
        args.model,
        "--vae",
        args.vae,
        "--frames",
        str(args.frames),
        "--deploy-config",
        args.deploy_config,
    ]
    if args.gpu_only:
        command.append("--gpu-only")
    if args.attention_ab:
        command.append("--attention-ab")
    command.extend(extra_args)
    label = label or (
        "GPU-only attention A/B"
        if args.attention_ab
        else "GPU-only"
        if args.gpu_only
        else "ANE 校准"
    )
    print(f"{label} 证据目录：{directory}", flush=True)
    started = time.monotonic()
    reason = None
    peak_rss = 0
    minimum_available = psutil.virtual_memory().available
    with (directory / "engine.log").open("w") as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            while process.poll() is None:
                root = psutil.Process(process.pid)
                rss = 0
                for child in [root, *root.children(recursive=True)]:
                    try:
                        rss += child.memory_info().rss
                    except psutil.NoSuchProcess:
                        pass
                available = psutil.virtual_memory().available
                peak_rss = max(peak_rss, rss)
                minimum_available = min(minimum_available, available)
                progress = directory / "progress.json"
                request = json.loads(progress.read_text()) if progress.exists() else {}
                if available < 4 * 1024**3 or rss > 14 * 1024**3:
                    reason = "RAM/RSS预算已达停止线"
                elif shutil.disk_usage(directory).free < 2 * 1024**3:
                    reason = "磁盘可用空间低于2GiB"
                elif time.monotonic() - started > 900:
                    reason = "离线阶段超过900秒"
                elif (
                    request.get("request_started")
                    and time.time() - request["request_started"] > 180
                ):
                    reason = "单请求超过180秒"
                if reason:
                    raise RuntimeError(reason)
                time.sleep(1)
        finally:
            if process.poll() is None:
                try:
                    descendants = psutil.Process(process.pid).children(recursive=True)
                except psutil.NoSuchProcess:
                    descendants = []
                for child in reversed(descendants):
                    try:
                        child.terminate()
                    except psutil.NoSuchProcess:
                        pass
                process.terminate()
                _, alive = psutil.wait_procs(descendants, timeout=8)
                for child in alive:
                    try:
                        child.kill()
                    except psutil.NoSuchProcess:
                        pass
                try:
                    process.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            write_json(
                directory / "supervisor.json",
                {
                    "returncode": process.returncode,
                    "stop_reason": reason,
                    "seconds": time.monotonic() - started,
                    "peak_rss_bytes": peak_rss,
                    "minimum_available_bytes": minimum_available,
                },
            )
    if process.returncode:
        raise RuntimeError(f"{label} 失败；未重启服务。检查 {directory / 'engine.log'}")
    if not (directory / "completed.json").is_file():
        raise RuntimeError(f"{label} 未发布完整验收记录：{directory}")
    print(f"{label} 完成；当前服务不变。证据：{directory}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    cache = Path.home() / ".cache/modelscope/hub/models/m-a-p"
    parser.add_argument("--model", default=os.environ.get("YUE2_MODEL_DIR", str(cache / "YuE2-3B")))
    parser.add_argument("--vae", default=os.environ.get("YUE2_VAE_DIR", str(cache / "YuE2-Vae")))
    parser.add_argument("--frames", type=int, default=512)
    parser.add_argument(
        "--gpu-only",
        action="store_true",
        help="显式关闭 ANE，以512帧 seed42 请求独立 warmup + 两次完整 GPU 计时",
    )
    parser.add_argument(
        "--attention-ab",
        action="store_true",
        help="仅 GPU-only：同 worker 比较旧 MLX bridge 和当前 Torch-stream attention",
    )
    parser.add_argument(
        "--deploy-config", default=str(Path(__file__).with_name("metal-deploy.yaml"))
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--output", help=argparse.SUPPRESS)
    args = parser.parse_args()
    # shortcut: bounded short-clip calibration only, expand after long-NAR memory profiling.
    if not 256 <= args.frames <= 512:
        parser.error("本工具仅校准256–512帧；其他形状保持GPU，不改变模型长歌能力")
    if args.attention_ab and not args.gpu_only:
        parser.error("--attention-ab 仅允许与 --gpu-only 一起使用")
    if args.gpu_only:
        if args.frames != 512:
            parser.error("GPU-only 固定512帧，不能通过缩短请求宣称完成")
        os.environ["VLLM_OMNI_METAL_ANE"] = "off"
    elif os.environ.get("VLLM_OMNI_METAL_ANE", "auto") != "auto":
        parser.error("ANE 校准要求 VLLM_OMNI_METAL_ANE=auto；独立GPU测量请用 --gpu-only")
    args.model = str(validate_model(args.model, "yue2"))
    args.vae = str(validate_model(args.vae, "yue2_vae"))
    if args.worker:
        asyncio.run(worker(args))
    else:
        supervise(args)


if __name__ == "__main__":
    main()
