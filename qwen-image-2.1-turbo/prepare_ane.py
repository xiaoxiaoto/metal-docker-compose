#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""真实 QwenImage GPU 算子 A/B 或显式离线 GPU+ANE 校准；不启动 HTTP 服务。"""

import argparse
import asyncio
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parent


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.chmod(0o600)
    temporary.replace(path)


def one_reply(value):
    while isinstance(value, list) and len(value) == 1:
        value = value[0]
    if not isinstance(value, dict) or value.get("error") or value.get("supported") is False:
        raise RuntimeError(f"diffusion RPC failed: {value!r}")
    return value


async def worker(args):
    from importlib.metadata import distribution, version

    import numpy as np
    import torch
    from PIL import Image
    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.diffusion.diffusion_engine import DiffusionEngine
    from vllm_omni.diffusion.request import OmniDiffusionRequest
    from vllm_omni.diffusion.utils.image_output import extract_images_from_outputs
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    from vllm_omni_metal.ops.aot_kernels import (
        _artifact_path,
        _diffusion_artifact_path,
    )

    output = Path(args.output)
    if "qwenmetal" not in version("vllm-omni"):
        raise RuntimeError("必须使用 prepare_runtime.py 的 QwenImage 私有 Omni 环境")
    import importlib.util

    spec = importlib.util.spec_from_file_location("qwen_local_prepare", ROOT / "prepare.py")
    local_prepare = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(local_prepare)
    validate_model = local_prepare.validate_model

    validate_model(args.model)
    config = OmniDiffusionConfig.from_kwargs(
        model=args.model,
        dtype="bfloat16",
        diffusion_load_format="default",
        distributed_executor_backend="uni",
        diffusion_attention_backend="TORCH_SDPA",
        enforce_eager=True,
        vae_use_slicing=True,
        vae_use_tiling=True,
        worker_extension_cls="vllm_omni.diffusion.models.qwen_image_21.metal_adapter.QwenMetalPreparation",
    )
    config.enrich_config()
    environment = {
        "versions": {
            p: str(version(p))
            for p in ("torch", "vllm", "vllm-metal", "vllm-omni", "vllm-omni-metal")
        },
        "python": sys.executable,
        "model": args.model,
        "omni_source": json.loads(distribution("vllm-omni").read_text("direct_url.json")),
        "size": args.size,
        "seed": args.seed,
        "prompt": args.prompt,
        "image": args.image,
        "layers": args.layers,
        "mode": args.mode,
        "aot_silu": _artifact_path(torch.bfloat16).is_file(),
        "aot_rotary": _diffusion_artifact_path(torch.bfloat16).is_file(),
    }
    write_json(output / "environment.json", environment)
    write_json(output / "progress.json", {"phase": "load", "started": time.time()})
    engine = None
    records = []
    pixels = {}
    try:
        engine = DiffusionEngine(config)

        def rpc(method, *values):
            reply = one_reply(engine.collective_rpc(method, timeout=600, args=values))
            if reply.get("busy") or reply.get("closed"):
                raise RuntimeError("pipeline not idle after RPC")
            return reply

        write_json(output / "loaded.json", rpc("qwen_metal_snapshot"))

        async def generate(gpu, chunks, *, warmup=False, default=False):
            if not default:
                rpc("qwen_metal_set_mode", gpu, chunks)
            number = len(records)
            request_id = f"qwen-metal-{number}"
            prompt = {"prompt": args.prompt, "modalities": ["image"]}
            if args.image:
                with Image.open(args.image) as image:
                    prompt["multi_modal_data"] = {"image": [image.convert("RGBA")]}
            width, height = (int(value) for value in args.size.split("x"))
            request = OmniDiffusionRequest(
                prompt=prompt,
                sampling_params=OmniDiffusionSamplingParams(
                    seed=args.seed,
                    generator_device="cpu",
                    num_inference_steps=8,
                    true_cfg_scale=1.0,
                    width=width,
                    height=height,
                ),
                request_id=request_id,
            )
            write_json(
                output / "progress.json",
                {"phase": "generate", "request_started": time.time(), "gpu": gpu, "chunks": chunks},
            )
            started = time.perf_counter()
            batches = []
            async for batch in engine.step_streaming(request):
                batches = batch
            torch.mps.synchronize()
            seconds = time.perf_counter() - started
            images = extract_images_from_outputs(batches)
            if len(images) != 1 or images[0].size != (width, height):
                raise RuntimeError("真实推理未返回匹配尺寸的单张图片")
            image = images[0]
            array = np.asarray(image)
            if image.mode not in ("RGB", "RGBA") or array[..., :3].std() <= 1:
                raise RuntimeError("真实推理返回无效或单色图片")
            filename = output / f"sample-{number}.png"
            image.save(filename)
            snapshot = rpc("qwen_metal_snapshot")
            if snapshot["parameter_device"] != "mps:0" and snapshot["parameter_device"] != "mps":
                raise RuntimeError("模型未在 MPS 执行")
            record = dict(
                snapshot,
                seconds=seconds,
                warmup=warmup,
                default=default,
                requested_chunks=chunks,
                image=str(filename),
                sha256=hashlib.sha256(array.tobytes()).hexdigest(),
                size=list(image.size),
                mode=image.mode,
            )
            records.append(record)
            pixels[number] = array.copy()
            write_json(output / "samples.json", records)
            if snapshot["ane_disabled"]:
                raise RuntimeError("ANE live-input gate rejected the candidate; no approval")
            if chunks and snapshot["ane_metrics"].get("ane_chunks", 0) == 0:
                raise RuntimeError("ANE candidate did not execute actual chunks")
            return number

        async def timed(gpu, chunks, **kwargs):
            return await asyncio.wait_for(generate(gpu, chunks, **kwargs), timeout=1800)

        if args.mode == "gpu-ab":
            await timed(False, 0, warmup=True)
            await timed(True, 0, warmup=True)
            for enabled in (False, True, True, False):
                await timed(enabled, 0)
            routes = {}
            for enabled in (False, True):
                selected = [
                    i
                    for i, item in enumerate(records)
                    if not item["warmup"] and item["gpu_enabled"] == enabled
                ]
                if len({records[i]["sha256"] for i in selected}) != 1:
                    raise RuntimeError("同路径同 seed 输出不能复现")
                routes[str(enabled)] = {
                    "seconds": [records[i]["seconds"] for i in selected],
                    "sha256": records[selected[0]]["sha256"],
                }
            reference = next(
                i
                for i, item in enumerate(records)
                if not item["warmup"] and not item["gpu_enabled"]
            )
            candidate = next(
                i for i, item in enumerate(records) if not item["warmup"] and item["gpu_enabled"]
            )
            a, b = pixels[reference].astype(np.float64), pixels[candidate].astype(np.float64)
            delta = b - a
            errors = {
                "relative_l2": float(np.linalg.norm(delta) / max(np.linalg.norm(a), 1e-12)),
                "scaled_max_abs": float(np.abs(delta).max() / max(np.abs(a).max(), 1)),
            }
            result = {
                "mode": "gpu-ab",
                "routes": routes,
                "pixel_errors": errors,
                "ane_approved": False,
            }
            write_json(output / "comparison.json", result)
            if errors["relative_l2"] > 0.01 or errors["scaled_max_abs"] > 0.01:
                raise RuntimeError("GPU operator whole-image 1% comparison failed")
            write_json(output / "completed.json", result)
            return

        write_json(output / "progress.json", {"phase": "ane_prepare", "started": time.time()})
        write_json(output / "prepared.json", rpc("qwen_metal_prepare", args.layers))
        await timed(True, 0, warmup=True)
        await timed(True, args.chunks, warmup=True)
        write_json(output / "audited.json", rpc("qwen_metal_finish_audit"))
        for chunks in (0, args.chunks, args.chunks, 0):
            await timed(True, chunks)
        baseline = [
            item["seconds"]
            for item in records
            if not item["warmup"] and not item["requested_chunks"]
        ]
        candidates = {
            args.chunks: [
                item["seconds"]
                for item in records
                if not item["warmup"] and item["requested_chunks"]
            ]
        }
        measured = [i for i, item in enumerate(records) if not item["warmup"]]
        for chunks in (0, args.chunks):
            if (
                len(
                    {
                        records[i]["sha256"]
                        for i in measured
                        if records[i]["requested_chunks"] == chunks
                    }
                )
                != 1
            ):
                raise RuntimeError("同 ANE 模式同 seed 输出不能复现")
        reference = next(i for i in measured if not records[i]["requested_chunks"])
        candidate = next(i for i in measured if records[i]["requested_chunks"])
        a, b = pixels[reference].astype(np.float64), pixels[candidate].astype(np.float64)
        errors = {
            "relative_l2": float(np.linalg.norm(b - a) / max(np.linalg.norm(a), 1e-12)),
            "scaled_max_abs": float(np.abs(b - a).max() / max(np.abs(a).max(), 1)),
        }
        write_json(output / "pixel-errors.json", errors)
        if max(errors.values()) > 0.01:
            raise RuntimeError("ANE 整图误差超过1%，不发布profile")
        shapes = records[candidate]["shapes"]
        approvals = engine.collective_rpc(
            "qwen_metal_approve", timeout=600, args=(shapes, baseline, list(candidates.items()))
        )
        while (
            isinstance(approvals, list) and len(approvals) == 1 and isinstance(approvals[0], list)
        ):
            approvals = approvals[0]
        if (
            not isinstance(approvals, list)
            or not approvals
            or any("chunks" not in r for r in approvals)
        ):
            raise RuntimeError(f"ANE approval RPC failed: {approvals!r}")
        choices = {r["chunks"] for r in approvals}
        if len(choices) != 1:
            raise RuntimeError("请求内不同形状的批准模式不一致，不能冒充完整对照")
        selected_chunks = choices.pop()
        write_json(output / "approved.json", approvals)
        # Default auto may correctly select GPU when the measured ANE path is slower.
        first = await timed(True, selected_chunks, default=True)
        second = await timed(True, selected_chunks, default=True)
        if records[first]["ane_benchmark"] or records[second]["ane_benchmark"]:
            raise RuntimeError("默认 profile reader 未切换成功")
        expected = records[candidate if selected_chunks else reference]["sha256"]
        if records[first]["sha256"] != expected or records[second]["sha256"] != expected:
            raise RuntimeError("默认 auto 未复现已校准路径的同seed输出")
        write_json(
            output / "completed.json",
            {
                "mode": "ane",
                "approvals": approvals,
                "pixel_errors": errors,
                "default_repeatable": True,
            },
        )
    finally:
        if engine is not None:
            engine.close()
        write_json(output / "closed.json", {"engine_closed": True})


def supervise(args, *, worker_script=None):
    if psutil.virtual_memory().available < 32 * 1024**3:
        raise MemoryError(
            "QwenImage 校准要求启动前至少32GiB available；请先停止本模型，不自动停止其他服务"
        )
    if shutil.disk_usage(args.output).free < 4 * 1024**3:
        raise MemoryError("校准至少保留4GiB磁盘；不自动清理用户数据")
    env = dict(
        os.environ,
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        PYTHONDONTWRITEBYTECODE="1",
        PYTORCH_ENABLE_MPS_FALLBACK="0",
        VLLM_WORKER_MULTIPROC_METHOD="spawn",
        OMP_NUM_THREADS="2",
        QWEN_IMAGE_METAL_ANE_LAYERS="",
        QWEN_IMAGE_METAL_GPU="auto",
        QWEN_IMAGE_METAL_QUANTIZATION="off",
        VLLM_OMNI_METAL_QUANTIZATION="off",
        VLLM_OMNI_METAL_ANE="off",
    )
    if args.mode == "gpu-ab":
        env["VLLM_OMNI_METAL_ANE"] = "off"
    command = [
        sys.executable,
        "-I",
        str(worker_script or Path(__file__).resolve()),
        "--worker",
        "--output",
        args.output,
        "--model",
        args.model,
        "--size",
        args.size,
        "--seed",
        str(args.seed),
        "--prompt",
        args.prompt,
        "--mode",
        args.mode,
        "--layers",
        ",".join(map(str, args.layers)),
        "--chunks",
        str(args.chunks),
    ]
    if args.image:
        command.extend(["--image", args.image])
    print(f"真实校准证据目录：{args.output}", flush=True)
    started = time.monotonic()
    reason = None
    peak_rss = 0
    minimum_available = psutil.virtual_memory().available
    with (Path(args.output) / "engine.log").open("w") as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
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
                    continue
                available = psutil.virtual_memory().available
                peak_rss, minimum_available = max(peak_rss, rss), min(minimum_available, available)
                if available < 4 * 1024**3 or rss > 42 * 1024**3:
                    reason = "RAM/RSS预算达到停止线"
                elif shutil.disk_usage(args.output).free < 2 * 1024**3:
                    reason = "磁盘保留低于2GiB"
                elif time.monotonic() - started > 3600:
                    reason = "校准超过3600秒"
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
                    child.kill()
                try:
                    process.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            write_json(
                Path(args.output) / "supervisor.json",
                {
                    "returncode": process.returncode,
                    "stop_reason": reason,
                    "seconds": time.monotonic() - started,
                    "peak_rss_bytes": peak_rss,
                    "minimum_available_bytes": minimum_available,
                },
            )
    if process.returncode or not (Path(args.output) / "completed.json").is_file():
        raise RuntimeError(
            f"真实校准未通过；查看 {Path(args.output) / 'engine.log'}，未启用未经批准的ANE"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("gpu-ab", "ane"), default="ane")
    parser.add_argument(
        "--model",
        default=str(Path.home() / ".cache/modelscope/hub/models/Qwen/Qwen-Image-2.1-Turbo"),
    )
    parser.add_argument("--size", default="512x512")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", default="一只橘猫坐在窗台上，窗外是蓝天，阳光温暖，细节清晰。")
    parser.add_argument("--image", help="存在的真实条件图片；不自动下载")
    parser.add_argument(
        "--layers",
        default="0",
        help="显式校准的 transformer 层，逗号分隔；默认仅第0层，不代表全部32层",
    )
    parser.add_argument("--chunks", type=int, default=1)
    parser.add_argument("--output", required=True, help="部署目录外的新证据目录")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.layers = [int(value) for value in args.layers.split(",")]
    args.model = str(Path(args.model).resolve())
    if args.chunks < 1 or not args.layers or len(set(args.layers)) != len(args.layers):
        parser.error("需要正整数chunks与不重复layers")
    if args.size not in ("512x512", "768x768", "1024x1024"):
        parser.error("离线工具仅接受已界定的512/768/1024方形请求，不能冒充2K验收")
    if args.image and not Path(args.image).is_file():
        parser.error("--image 必须是存在的实际图片")
    output = Path(args.output).resolve()
    if output.is_relative_to(ROOT):
        parser.error("证据目录必须位于部署目录外")
    if args.worker:
        asyncio.run(worker(args))
    else:
        output.mkdir(parents=True, exist_ok=False)
        supervise(args)
        if args.mode == "ane":
            from vllm_omni_metal.ane.projection import publish_staged_projection_policy

            approvals = json.loads((output / "approved.json").read_text())
            policies = {record["identity"]: record["staged_policy"] for record in approvals}
            for policy in policies.values():
                publish_staged_projection_policy(policy)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
