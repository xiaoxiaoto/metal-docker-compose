#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""完整原生双语TTS交错A/B：Code2Wav量化或Talker ANE+DynamicEngine联合审批。"""

import argparse
import asyncio
import copy
import hashlib
import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path
from types import SimpleNamespace

import psutil


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def publish(path, value):
    """同目录原子 no-replace 发布；不覆盖任何既有批准产物。"""
    path = Path(path)
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as file:
        temporary = Path(file.name)
        file.write(payload)
        file.flush()
        os.fsync(file.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def comparison(a, b):
    import numpy as np

    result = {"same_shape": a.shape == b.shape, "same_dtype": a.dtype == b.dtype}
    if not result["same_shape"]:
        return result
    x, y = a.astype(np.float64), b.astype(np.float64)
    delta = y - x
    result.update(
        relative_l2=float(np.linalg.norm(delta) / max(np.linalg.norm(x), 1e-12)),
        scaled_max_abs=float(np.abs(delta).max() / max(np.abs(x).max(), 1e-12)),
        max_abs=float(np.abs(delta).max()),
        bitwise_identical=a.dtype == b.dtype and a.tobytes() == b.tobytes(),
        pcm16_identical=bool(
            np.array_equal(np.rint(a * 32767).astype("<i2"), np.rint(b * 32767).astype("<i2"))
        ),
    )
    return result


async def worker(args):
    from importlib.metadata import version

    import numpy as np
    import torch
    import yaml
    from higgs_metal_quantization import POLICY, TEXTS

    joint = getattr(args, "joint", False)
    if joint:
        from higgs_joint import POLICY, TEXTS
    from vllm_omni.entrypoints.async_omni import AsyncOmni
    from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest
    from vllm_omni.entrypoints.openai.speech_usage import SpeechOutputTokenCounter
    from vllm_omni.entrypoints.openai.tts_adapters.higgs_audio_v3 import (
        HiggsAudioV3Adapter,
    )
    from vllm_omni.entrypoints.utils import coerce_param_message_types

    if "higgsdynamicv2" not in version("vllm-omni"):
        raise RuntimeError("必须使用 Higgs 私有正常wheel；禁止共享环境中校准")
    resource = Path(importlib.util.find_spec("higgs_quantization_preparation").origin).resolve()
    if (
        not resource.is_relative_to(Path(sys.prefix).resolve())
        or hashlib.sha256(resource.read_bytes()).digest()
        != hashlib.sha256(Path(__file__).read_bytes()).digest()
    ):
        raise RuntimeError("离线CLI与owned私有wheel身份资源不同，需重新准备runtime")
    directory = Path(args.output)
    deploy = yaml.safe_load(Path(args.deploy_config).read_text())
    if [s["stage_id"] for s in deploy["stages"]] != [0, 1] or deploy.get("async_chunk") is not True:
        raise ValueError("必须保留现有原生双阶段 async chunk 拓扑")
    deploy["stages"][1]["worker_extension_cls"] = "higgs_metal_quantization.QuantizationPreparation"
    if joint:
        deploy["stages"][0]["worker_extension_cls"] = "higgs_joint.JointPreparation"
    deploy_path = directory / "deploy.yaml"
    deploy_path.write_text(yaml.safe_dump(deploy))
    save(directory / "input.json", POLICY)
    engine, records = None, []

    async def rpc(method, *values):
        def flatten(value):
            if isinstance(value, list):
                return [item for child in value for item in flatten(child)]
            return [value]

        replies = flatten(
            await engine.collective_rpc(
                method.replace("higgs_quant_", "higgs_joint_") if joint else method,
                args=values,
                timeout=180,
                stage_ids=[0] if joint else [1],
            )
        )
        if (
            len(replies) != 1
            or not isinstance(replies[0], dict)
            or replies[0].get("supported") is not True
            or replies[0].get("error")
            or not isinstance(replies[0].get("result"), dict)
        ):
            save(directory / "rpc-error.json", {"method": method, "replies": replies})
            raise RuntimeError(f"{method}: worker RPC 失败，不能继续")
        return replies[0]["result"]

    async def idle():
        for _ in range(50):
            state = await rpc("higgs_quant_snapshot")
            if state["pending"] == 0:
                return state
            await asyncio.sleep(0.1)
        raise RuntimeError("请求pending未清零")

    def array(value):
        if isinstance(value, (tuple, list)):
            return np.concatenate([array(v).reshape(-1) for v in value])
        return (
            value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
        )

    try:
        engine = AsyncOmni(
            model=args.model,
            deploy_config=str(deploy_path),
            stage_init_timeout=360,
            init_timeout=420,
            log_stats=False,
            output_modalities=["audio"],
        )
        adapter = HiggsAudioV3Adapter(SimpleNamespace(engine_client=engine))
        await rpc("higgs_quant_prepare")
        codec_evidence = None
        if joint:
            from higgs_joint import validate_codec_identity

            async def codec_snapshot():
                replies = await engine.collective_rpc(
                    "higgs_quant_snapshot", args=(), timeout=180, stage_ids=[1]
                )
                while isinstance(replies, list) and len(replies) == 1:
                    replies = replies[0]
                if not isinstance(replies, dict) or replies.get("supported") is not True:
                    raise RuntimeError("missing actual codec identity RPC evidence")
                evidence = replies["result"]["source_identity"]
                validate_codec_identity(evidence)
                return evidence

            codec_evidence = await codec_snapshot()
            save(directory / "codec-identity.json", codec_evidence)

        async def generate(text, mode, *, warmup=False, released=False):
            await idle()
            await rpc("higgs_quant_mode", mode)
            request = OpenAICreateSpeechRequest(
                model="higgs-audio-v3-tts-4b",
                input=text,
                response_format="wav",
                stream=False,
                seed=42,
                max_new_tokens=512,
            )
            params = coerce_param_message_types(
                copy.deepcopy(engine.default_sampling_params_list), False
            )
            prepared = await adapter.build(request, params, False)
            params = adapter.apply_sampling_overrides(params, request, prepared.prompt)
            for param in params:
                param.seed = 42
            index = len(records)
            save(directory / "progress.json", {"request_started": time.time(), "index": index})
            audio_output = None
            usage = SpeechOutputTokenCounter()
            started = time.perf_counter()
            async with asyncio.timeout(180):
                async for result in engine.generate(
                    copy.deepcopy(prepared.prompt),
                    request_id=f"higgs-quant-{index}",
                    sampling_params_list=params,
                    output_modalities=["audio"],
                ):
                    usage.observe(result)
                    mm = getattr(result, "multimodal_output", None)
                    if not mm:
                        for completion in getattr(result, "outputs", ()) or ():
                            mm = getattr(completion, "multimodal_output", None)
                            if mm:
                                break
                    if mm and mm.get("audio") is not None:
                        audio_output = mm  # FINAL_ONLY，最后累积波形；不重复拼接累计输出。
            elapsed = time.perf_counter() - started
            save(directory / "progress.json", {})
            state = await idle()
            if joint and await codec_snapshot() != codec_evidence:
                raise RuntimeError(
                    "Code2Wav native module/weight/dtype identity changed during generation"
                )
            if audio_output is None:
                raise ValueError("未收到完整音频")
            audio = np.ascontiguousarray(array(audio_output["audio"]).squeeze())
            stem = f"{index}-{mode}"
            np.save(directory / f"{stem}.npy", audio, allow_pickle=False)
            save(
                directory / f"{stem}-output.json",
                {
                    "text": text,
                    "mode": mode,
                    "stage0_finish_reason": usage.stage0_finish_reason,
                    "shape": list(audio.shape),
                    "dtype": str(audio.dtype),
                    **state,
                },
            )
            sr_value = audio_output.get("sr")
            if sr_value is None:
                raise ValueError("音频缺少实际采样率")
            sr = int(array(sr_value).reshape(-1)[-1])
            if (
                sr != 24000
                or audio.ndim != 1
                or len(audio) < sr
                or not np.isfinite(audio).all()
                or audio.std() <= 1e-6
                or np.abs(audio).max() > 1
            ):
                raise ValueError("真实音频采样率/单声道/长度/幅度/非静音门禁失败")
            natural = usage.stage0_finish_reason == "stop"
            if not natural:
                raise ValueError(
                    f"Talker非自然结束：{usage.stage0_finish_reason!r}；不得审批截断波形"
                )
            if mode == "dynamic" and (
                state["metrics"]["calls"] <= 0 or state["metrics"]["rows"] <= 0
            ):
                raise RuntimeError("未实际调用dynamic候选fc2（包括原精度保留）")
            if (
                joint
                and mode in ("quantized", "dynamic")
                and (
                    state["metrics"]["quantized_calls"] <= 0
                    or (mode == "dynamic" and state["metrics"]["ane_chunks"] <= 0)
                )
            ):
                raise RuntimeError("zero actual ANE/quantization hits; short inputs are not padded")
            pcm = np.rint(audio * 32767).astype("<i2")
            with wave.open(str(directory / f"{stem}.wav"), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(sr)
                wav.writeframes(pcm.tobytes())
            record = {
                "text": text,
                "mode": mode,
                "warmup": warmup,
                "released": released,
                "seconds": elapsed,
                "sample_rate": sr,
                "shape": list(audio.shape),
                "dtype": str(audio.dtype),
                "audio_seconds": len(audio) / sr,
                "natural_stop": natural,
                "stage0_finish_reason": usage.stage0_finish_reason,
                "audio_sha256": hashlib.sha256(audio.tobytes()).hexdigest(),
                "pcm16_sha256": hashlib.sha256(pcm.tobytes()).hexdigest(),
                "npy": f"{stem}.npy",
                "wav": f"{stem}.wav",
                **state,
            }
            records.append(record)
            save(directory / "results.json", records)
            return audio, record

        # 捕获的是实际fc2调用序号，可跨完整TTS请求；不要求四次denoising。
        for text in (*TEXTS, *TEXTS):
            await generate(text, "float", warmup=True)
        installation = await rpc("higgs_quant_search")
        save(directory / "adapter.json", installation)
        save(directory / "candidate-profile.json", installation["profile"])
        lossy = any(d["bits"] in (4, 8) for d in installation["profile"]["modules"].values())

        samples, candidates = [], []
        timings = {"quantized": [0.0, 0.0], "dynamic": [0.0, 0.0]}
        if joint:
            for text in TEXTS:
                for mode in ("float", "quantized", "dynamic"):
                    await generate(text, mode, warmup=True)
            await rpc("higgs_joint_audited")
        for text in TEXTS:
            if not joint:
                for mode in ("float", "dynamic"):
                    await generate(text, mode, warmup=True)
            outputs = {"float": [], "dynamic": []}
            measured = {"float": [], "dynamic": []}
            for mode in ("float", "dynamic", "dynamic", "float"):
                value, record = await generate(text, mode)
                outputs[mode].append(value)
                measured[mode].append(record)
            metrics = comparison(outputs["float"][0], outputs["dynamic"][0])
            repeatable = all(
                values[0].dtype == values[1].dtype
                and values[0].shape == values[1].shape
                and values[0].tobytes() == values[1].tobytes()
                for values in outputs.values()
            )
            sample = {
                "text": text,
                "seed_repeatable": repeatable,
                "natural_stop": True,
                "candidate_calls": sum(r["metrics"]["calls"] for r in measured["dynamic"]),
                "candidate_rows": sum(r["metrics"]["rows"] for r in measured["dynamic"]),
                "baseline_sha256": measured["float"][0]["audio_sha256"],
                "candidate_sha256": measured["dynamic"][0]["audio_sha256"],
                **metrics,
            }
            if joint:
                sample.update(
                    ane_chunks=sum(r["metrics"]["ane_chunks"] for r in measured["dynamic"]),
                    quantized_calls=sum(
                        r["metrics"]["quantized_calls"] for r in measured["dynamic"]
                    ),
                )
            samples.append(sample)
            candidates.append(outputs["dynamic"][0])
            save(directory / "comparison.json", {"samples": samples, "lossy": lossy})
        quality = all(
            s["same_shape"]
            and s["same_dtype"]
            and s["seed_repeatable"]
            and s["relative_l2"] <= 0.01
            and s["scaled_max_abs"] <= 0.01
            for s in samples
        )
        if not quality:
            save(
                directory / "completed.json",
                {
                    "approved": False,
                    "status": "rejected",
                    "reason": "最终波形1%或同seed确定性门禁未通过",
                    "samples": samples,
                },
            )
            return
        if joint:
            for sample, text, expected in zip(samples, TEXTS, candidates, strict=True):
                timed = {"quantized": [], "dynamic": []}
                for mode in ("quantized", "dynamic", "dynamic", "quantized"):
                    value, record = await generate(text, mode)
                    slot = len(timed[mode])
                    timings[mode][slot] += record["seconds"]
                    timed[mode].append(value)
                if any(
                    values[0].dtype != values[1].dtype
                    or values[0].shape != values[1].shape
                    or values[0].tobytes() != values[1].tobytes()
                    for values in timed.values()
                ) or any(
                    value.dtype != expected.dtype
                    or value.shape != expected.shape
                    or value.tobytes() != expected.tobytes()
                    for value in timed["dynamic"]
                ):
                    raise ValueError("ANE增益计时路线未复现同seed完整波形")
                metrics = comparison(timed["quantized"][0], timed["dynamic"][0])
                if (
                    not metrics["same_shape"]
                    or not metrics["same_dtype"]
                    or metrics["relative_l2"] > 0.01
                    or metrics["scaled_max_abs"] > 0.01
                ):
                    raise ValueError("ANE相对同一动态量化GPU基线的完整波形1%门禁未通过")
                sample.update(
                    ane_relative_l2=metrics["relative_l2"],
                    ane_scaled_max_abs=metrics["scaled_max_abs"],
                    ane_reference_sha256=hashlib.sha256(
                        timed["quantized"][0].tobytes()
                    ).hexdigest(),
                )
            save(directory / "comparison.json", {"samples": samples, "lossy": lossy})
            staged = await rpc("higgs_joint_stage", timings["quantized"], timings["dynamic"])
            save(directory / "staged-ane-policies.json", staged)
        await rpc("higgs_quant_release")
        for sample, text, expected in zip(samples, TEXTS, candidates, strict=True):
            actual, _ = await generate(text, "dynamic", released=True)
            if (
                actual.dtype != expected.dtype
                or actual.shape != expected.shape
                or actual.tobytes() != expected.tobytes()
            ):
                raise ValueError("释放float基线后未复现量化输出，不发布profile")
            sample["released_matches"] = True
        release = (
            await rpc("higgs_quant_approve", samples, codec_evidence)
            if joint
            else await rpc("higgs_quant_approve", samples)
        )
        engine.shutdown()
        engine = None
        save(
            directory / "completed.json",
            {
                "approved": True,
                "status": "approved",
                "lossy": lossy,
                "released_float_reverified": True,
                "samples": samples,
                "speedup_claimed": False,
                "listening_asr_accepted": False,
            },
        )
        save(directory / "reviewed-profile.json", release["profile"])
        if joint:
            save(
                directory / "completed.json",
                {
                    "approved": True,
                    "status": "approved",
                    "joint": True,
                    "released_float_reverified": True,
                    "engine_shutdown": True,
                    "samples": samples,
                    "bilingual_seconds": timings,
                    "speedup_claimed": True,
                    "listening_asr_accepted": False,
                },
            )
    except BaseException as error:
        save(
            directory / "completed.json",
            {"approved": False, "status": "failed", "error": f"{type(error).__name__}: {error}"},
        )
        raise
    finally:
        if engine is not None:
            try:
                if joint:
                    await rpc("higgs_joint_close")
            finally:
                engine.shutdown()


def supervise(args):
    directory = Path(args.output)
    if psutil.virtual_memory().available < 16 * 1024**3:
        raise MemoryError("启动需至少16GiB可用RAM；不停止其它服务、不提高水位")
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError("证据目录必须为空，禁止覆盖历史结果")
    directory.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(directory).free < 2 * 1024**3:
        raise OSError("磁盘需至少保留2GiB")
    command = [
        sys.executable,
        "-I",
        str(Path(__file__).resolve()),
        "--worker",
        "--model",
        args.model,
        "--output",
        str(directory),
        "--deploy-config",
        args.deploy_config,
    ]
    if getattr(args, "joint", False):
        command.append("--joint")
    environment = dict(
        os.environ,
        HIGGS_METAL_QUANTIZATION="off",
        VLLM_OMNI_METAL_QUANTIZATION="off",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        VLLM_WORKER_MULTIPROC_METHOD="spawn",
        VLLM_ALLOW_INSECURE_SERIALIZATION="0",
        OMP_NUM_THREADS="2",
        HIGGS_METAL_QUANTIZATION_DEPLOY_CONFIG=args.deploy_config,
        HIGGS_METAL_JOINT="off",
        HIGGS_METAL_JOINT_CALIBRATION="1" if getattr(args, "joint", False) else "0",
        VLLM_OMNI_METAL_ANE="auto" if getattr(args, "joint", False) else "off",
    )
    started, peak, minimum, reason = time.monotonic(), 0, psutil.virtual_memory().available, None
    with (directory / "engine.log").open("w") as log:
        process = subprocess.Popen(
            command, env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            while process.poll() is None:
                try:
                    root = psutil.Process(process.pid)
                    rss = sum(
                        p.memory_info().rss
                        for p in [root, *root.children(recursive=True)]
                        if p.is_running()
                    )
                except psutil.NoSuchProcess:
                    rss = 0
                available = psutil.virtual_memory().available
                peak, minimum = max(peak, rss), min(minimum, available)
                progress = directory / "progress.json"
                current = json.loads(progress.read_text()) if progress.exists() else {}
                if available < 4 * 1024**3 or rss > 14 * 1024**3:
                    reason = "运行available<4GiB或进程树RSS>14GiB"
                elif shutil.disk_usage(directory).free < 2 * 1024**3:
                    reason = "磁盘低于2GiB"
                elif time.monotonic() - started > 3600:
                    reason = "离线总阶段超过3600秒"
                elif (
                    current.get("request_started")
                    and time.time() - current["request_started"] > 180
                ):
                    reason = "单请求超过180秒"
                if reason:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                    break
                time.sleep(1)
            process.wait()
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            save(
                directory / "supervisor.json",
                {
                    "returncode": process.returncode,
                    "reason": reason,
                    "peak_tree_rss_bytes": peak,
                    "minimum_available_bytes": minimum,
                    "seconds": time.monotonic() - started,
                },
            )
    if reason or process.returncode:
        raise RuntimeError(f"校准失败：{reason or process.returncode}，原始证据：{directory}")
    completed = json.loads((directory / "completed.json").read_text())
    if completed.get("approved") is True:
        profile = json.loads((directory / "reviewed-profile.json").read_text())
        if getattr(args, "joint", False):
            from higgs_joint import approved

            from vllm_omni_metal.ane.projection import publish_staged_projection_policy
            from vllm_omni_metal.quantization.linear import _encoded

            if completed.get("engine_shutdown") is not True:
                raise RuntimeError("joint engine did not confirm shutdown")
            profile["calibration"]["joint"].update(
                engine_shutdown=True,
                supervisor_success=True,
                supervisor_sha256=hashlib.sha256(
                    (directory / "supervisor.json").read_bytes()
                ).hexdigest(),
            )
            profile.pop("sha256")
            profile["sha256"] = hashlib.sha256(_encoded(profile)).hexdigest()
            approved(profile)
            for record in profile["calibration"]["joint"]["projections"].values():
                publish_staged_projection_policy(record["staged_policy"])
        publish(directory / "approved-profile.json", profile)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=str(Path.home() / ".cache/modelscope/hub/models/bosonai/higgs-audio-v3-tts-4b"),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--deploy-config", default=str(Path(__file__).with_name("metal-deploy.yaml"))
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--joint",
        action="store_true",
        help="Real Talker gate-up ANE + DynamicEngine down joint approval",
    )
    args = parser.parse_args()
    args.deploy_config = str(Path(args.deploy_config).expanduser().resolve(strict=True))
    if os.environ.get("VLLM_ALLOW_INSECURE_SERIALIZATION", "0") not in ("", "0"):
        parser.error("禁止不安全pickle RPC")
    spec = importlib.util.spec_from_file_location(
        "higgs_prepare", Path(__file__).with_name("prepare.py")
    )
    prepare = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prepare)
    args.model = str(Path(args.model).expanduser().resolve())
    prepare.validate_model(args.model)
    args.output = str(Path(args.output).expanduser().resolve())
    if args.worker:
        asyncio.run(worker(args))
    else:
        supervise(args)


if __name__ == "__main__":
    main()
