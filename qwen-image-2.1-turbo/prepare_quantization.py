#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""真实完整图DynamicEngine混合精度校准；沿用32GiB启动/监督门禁。"""

import argparse
import asyncio
import hashlib
import importlib.util
import json
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sidecar(name):
    spec = importlib.util.spec_from_file_location(f"qwen_local_{name}", ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


async def worker(args):
    from importlib.metadata import version

    import numpy as np
    import torch
    from PIL import Image
    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.diffusion.diffusion_engine import DiffusionEngine
    from vllm_omni.diffusion.models.qwen_image_21.quantization_adapter import (
        approval,
        joint_approval,
        pixel_errors,
    )
    from vllm_omni.diffusion.request import OmniDiffusionRequest
    from vllm_omni.diffusion.utils.image_output import extract_images_from_outputs
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    helpers = sidecar("prepare_ane")
    write_json = helpers.write_json
    output = Path(args.output)
    if "qwenmetal" not in version("vllm-omni"):
        raise RuntimeError("必须使用含DynamicEngine适配的QwenImage私有wheel")
    sidecar("prepare").validate_model(args.model)
    write_json(
        output / "environment.json",
        {
            "python": sys.executable,
            "versions": {
                p: str(version(p))
                for p in ("torch", "vllm", "vllm-metal", "vllm-omni", "vllm-omni-metal", "mlx")
            },
            "model": args.model,
            "prompt": args.prompt,
            "seed": args.seed,
            "size": args.size,
            "image": args.image,
            "image_sha256": None
            if not args.image
            else hashlib.sha256(Path(args.image).read_bytes()).hexdigest(),
            "layers": args.layers,
            "steps": 8,
            "cfg": 1.0,
            "ane": "gate-projection" if args.mode == "joint" else "off",
            "gpu": "auto",
            "lossy": True,
            "threshold": 0.01,
        },
    )
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
    engine = None
    records = []
    arrays = {}
    try:
        write_json(output / "progress.json", {"phase": "load", "started": time.time()})
        engine = DiffusionEngine(config)

        def rpc(method, *values):
            reply = helpers.one_reply(engine.collective_rpc(method, timeout=600, args=values))
            if reply.get("busy") or (reply.get("closed") and method != "qwen_metal_close"):
                raise RuntimeError("DynamicEngine RPC需要健康空闲pipeline")
            return reply

        async def generate(quantized, *, warmup=False, ane=0):
            number = len(records)
            write_json(
                output / "progress.json",
                {
                    "phase": "generate",
                    "number": number,
                    "quantized": quantized,
                    "started": time.time(),
                },
            )
            width, height = map(int, args.size.split("x"))
            prompt = {"prompt": args.prompt, "modalities": ["image"]}
            if args.image:
                with Image.open(args.image) as image:
                    prompt["multi_modal_data"] = {"image": [image.convert("RGBA")]}
            request = OmniDiffusionRequest(
                prompt=prompt,
                request_id=f"qwen-dynamic-{number}",
                sampling_params=OmniDiffusionSamplingParams(
                    seed=args.seed,
                    generator_device="cpu",
                    num_inference_steps=8,
                    true_cfg_scale=1.0,
                    width=width,
                    height=height,
                ),
            )
            started = time.perf_counter()
            batches = []
            async for batch in engine.step_streaming(request):
                batches = batch
            torch.mps.synchronize()
            images = extract_images_from_outputs(batches)
            seconds = time.perf_counter() - started
            if (
                len(images) != 1
                or images[0].size != (width, height)
                or images[0].mode not in ("RGB", "RGBA")
            ):
                raise RuntimeError("真实整图输出尺寸/模式无效")
            array = np.asarray(images[0])
            if not np.isfinite(array).all() or array[..., :3].std() <= 1:
                raise RuntimeError("真实整图无效或单色")
            images[0].save(output / f"sample-{number}.png")
            np.save(output / f"sample-{number}.npy", array, allow_pickle=False)
            snapshot = rpc("qwen_metal_snapshot")
            if snapshot["parameter_device"] not in ("mps", "mps:0"):
                raise RuntimeError("校准必须使用MPS实际完整pipeline")
            if args.mode != "joint" and snapshot["layers"]:
                raise RuntimeError("纯量化校准不能绑定ANE")
            hits = snapshot["ane_metrics"].get("ane_chunks", 0)
            if (ane > 0 and (hits <= 0 or snapshot["ane_disabled"])) or (ane == 0 and hits != 0):
                raise RuntimeError("实际ANE路由与请求不符")
            route = snapshot["quantization"]
            calls = route["calls"]
            expected = route["quantized_modules"] if quantized else 0
            if calls < expected or (expected == 0 and calls != 0):
                raise RuntimeError("实际混合精度调用与请求路由不符")
            if quantized and any(
                item["bits"] in (4, 8) and item["calls"] <= 0 for item in route["modules"].values()
            ):
                raise RuntimeError("组合中存在未真实执行的量化投影")
            record = {
                "quantized": quantized,
                "warmup": warmup,
                "seconds": seconds,
                "calls": calls,
                "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "snapshot": snapshot,
            }
            records.append(record)
            arrays[number] = array.copy()
            write_json(output / "samples.json", records)
            return number

        async def timed(*values, **kwargs):
            return await asyncio.wait_for(generate(*values, **kwargs), timeout=1800)

        write_json(output / "loaded.json", rpc("qwen_metal_snapshot"))
        native = await timed(False, warmup=True)
        write_json(output / "prepared.json", rpc("qwen_dynamic_prepare", args.layers))
        for batch in range(len(args.layers)):
            captured = await timed(False, warmup=True)
            if records[native]["sha256"] != records[captured]["sha256"]:
                raise RuntimeError("捕获hook改变原精度整图输出")
            capture = rpc("qwen_dynamic_finish_capture")
            write_json(output / f"activation-batch-{batch}.json", capture)
            if capture["complete"] != (batch == len(args.layers) - 1):
                raise RuntimeError("完整候选层分批捕获状态不一致")
        write_json(output / "activation-calibration.json", capture)
        selected = list(capture["precisions"])
        if args.mode == "joint":
            if not any(p in ("w4a16", "w8a16") for p in capture["precisions"].values()):
                raise RuntimeError("联合校准必须至少实际量化一个down")
            write_json(output / "ane-prepared.json", rpc("qwen_metal_prepare", args.layers))
            rpc("qwen_metal_set_mode", True, 0)
        rpc("qwen_dynamic_set_candidate", False)
        await timed(False, warmup=True)
        rpc("qwen_dynamic_set_candidate", True)
        await timed(True, warmup=True)
        gpu_quantized = []
        if args.mode == "joint":
            for _ in range(2):
                rpc("qwen_dynamic_set_candidate", True)
                rpc("qwen_metal_set_mode", True, 0)
                gpu_quantized.append(await timed(True))
            rpc("qwen_metal_set_mode", True, args.chunks)
            await timed(True, warmup=True, ane=args.chunks)
            rpc("qwen_metal_finish_audit")
        formal = []
        for quantized in (False, True, True, False):
            rpc("qwen_dynamic_set_candidate", quantized)
            chunks = args.chunks if quantized and args.mode == "joint" else 0
            if args.mode == "joint":
                rpc("qwen_metal_set_mode", True, chunks)
            formal.append(await timed(quantized, ane=chunks))
        reference = arrays[formal[0]]
        errors = [pixel_errors(reference, arrays[i]) for i in formal if records[i]["quantized"]]
        calibration = {
            "policy": "qwen-whole-image-joint-v1"
            if args.mode == "joint"
            else "qwen-whole-image-dynamic-v1",
            "quantized_modules": sum(
                p in ("w4a16", "w8a16") for p in capture["precisions"].values()
            ),
            "approved": False,
            "lossy": True,
            "threshold": 0.01,
            "errors": errors,
            "samples": [records[i] for i in formal],
            "input": json.loads((output / "environment.json").read_text()),
            "activation_calibration": capture,
            "selection": "所有指定down的DynamicEngine W4/W8/source组合；非全层搜索",
        }
        write_json(output / "comparison.json", calibration)
        approved = dict(calibration, approved=True)
        if args.mode == "joint":
            rpc("qwen_dynamic_set_candidate", True)
            rpc("qwen_metal_set_mode", True, args.chunks)
            # Re-audit the selected shape/context immediately before staging its
            # subordinate ANE policy. Only the supervisor publishes joint approval.
            await timed(True, warmup=True, ane=args.chunks)
            audit = rpc("qwen_metal_finish_audit")
            baseline = [records[i]["seconds"] for i in gpu_quantized]
            candidates = [records[i]["seconds"] for i in formal if records[i]["quantized"]]
            if max(candidates) > min(baseline) * 0.97:
                raise RuntimeError("联合路径未比同量化GPU全pipeline稳定快至少3%")
            profiles = rpc(
                "qwen_joint_approve", audit["shapes"], baseline, [(args.chunks, candidates)]
            )
            approved.update(
                ane_baseline_seconds=baseline,
                ane_candidate_seconds=candidates,
                ane_profiles=profiles["records"],
            )
            joint_approval(approved, finalized=False)
        else:
            approval(approved)
        rpc("qwen_dynamic_set_candidate", True)
        profile = rpc("qwen_dynamic_profile", approved)
        write_json(output / "released.json", rpc("qwen_dynamic_release"))
        released = await timed(True, ane=args.chunks if args.mode == "joint" else 0)
        if records[released]["sha256"] != records[formal[1]]["sha256"]:
            raise RuntimeError("释放浮点基线后整图无法复现，不发布profile")
        closed = rpc("qwen_metal_close")
        if not closed["closed"] or closed["quantization"]["status"] != "closed":
            raise RuntimeError("私有执行器关闭检查失败")
        if args.mode == "joint" and not closed["ane_disabled"]:
            raise RuntimeError("ANE projection未关闭")
        write_json(output / "adapter-closed.json", closed)
    except Exception as error:
        comparison = output / "comparison.json"
        rejected = json.loads(comparison.read_text()) if comparison.is_file() else {}
        rejected.update(approved=False, rejection_reason=str(error))
        write_json(comparison, rejected)
        raise
    finally:
        if engine is not None:
            engine.close()
        write_json(output / "closed.json", {"engine_closed": True})
    write_json(output / "comparison.json", approved)
    write_json(output / "candidate-profile.json", profile)
    write_json(
        output / "completed.json",
        {
            "selected": selected,
            "scope": profile["scope"],
            "precisions": capture["precisions"],
            "quantized_modules": approved["quantized_modules"],
            "errors": errors,
            "lossy": True,
            "released_output_sha256": records[released]["sha256"],
            "profile_sha256": profile["sha256"],
            "http_validated": False,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=str(Path.home() / ".cache/modelscope/hub/models/Qwen/Qwen-Image-2.1-Turbo"),
    )
    parser.add_argument("--size", choices=("512x512", "768x768", "1024x1024"), default="512x512")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", default="一只橘猫坐在窗台上，窗外是蓝天，阳光温暖，细节清晰。")
    parser.add_argument("--image")
    parser.add_argument("--layers", default="0,10,21,31", help="有限down候选；不代表全层搜索")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--profile", default=str(Path.home() / ".cache/qwen-image-metal/dynamic-approved.json")
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--mode", choices=("gpu-ab", "joint"), default="gpu-ab")
    parser.add_argument("--chunks", type=int, default=1)
    args = parser.parse_args()
    args.layers = [int(i) for i in args.layers.split(",")]
    if (
        not args.layers
        or any(i < 0 for i in args.layers)
        or len(set(args.layers)) != len(args.layers)
    ):
        parser.error("layers必须为唯一非负索引")
    if args.chunks < 1:
        parser.error("chunks必须为正整数")
    args.model = str(Path(args.model).resolve())
    if args.image:
        args.image = str(Path(args.image).resolve(strict=True))
    output = Path(args.output).resolve()
    if output.is_relative_to(ROOT):
        parser.error("证据目录必须位于部署目录外")
    args.output = str(output)
    if args.worker:
        if (
            os.environ.get("QWEN_IMAGE_METAL_QUANTIZATION") != "off"
            or os.environ.get("VLLM_OMNI_METAL_QUANTIZATION") != "off"
            or os.environ.get("VLLM_OMNI_METAL_ANE") != "off"
        ):
            raise RuntimeError("worker必须由监督器启动，量化/ANE初始off")
        asyncio.run(worker(args))
    else:
        output.mkdir(parents=True, exist_ok=False)
        helpers = sidecar("prepare_ane")
        try:
            helpers.supervise(args, worker_script=Path(__file__).resolve())
        except Exception as error:
            comparison = output / "comparison.json"
            rejected = json.loads(comparison.read_text()) if comparison.is_file() else {}
            rejected.update(approved=False, rejection_reason=str(error))
            helpers.write_json(comparison, rejected)
            raise
        if not (output / "closed.json").is_file() or not (output / "adapter-closed.json").is_file():
            raise RuntimeError("worker未正常关闭，不发布profile")
        from vllm_omni.diffusion.models.qwen_image_21.quantization_adapter import (
            _publish_profile,
        )

        from vllm_omni_metal.quantization.runtime import load_profile

        completed = json.loads((output / "completed.json").read_text())
        profile = load_profile(output / "candidate-profile.json", scope=completed["scope"])
        from vllm_omni.diffusion.models.qwen_image_21.quantization_adapter import (
            validate_calibration,
        )

        validate_calibration(profile["calibration"], finalized=False)
        supervisor = json.loads((output / "supervisor.json").read_text())
        if supervisor["returncode"] != 0 or supervisor["stop_reason"] is not None:
            raise RuntimeError("监督未成功，不发布联合profile")
        if args.mode == "joint":
            closed = json.loads((output / "closed.json").read_text())
            adapter_closed = json.loads((output / "adapter-closed.json").read_text())
            released = json.loads((output / "released.json").read_text())
            samples = json.loads((output / "samples.json").read_text())
            expected = profile["calibration"]["samples"][1]["sha256"]
            if (
                closed.get("engine_closed") is not True
                or adapter_closed.get("closed") is not True
                or released.get("float_baseline_held") is not False
                or completed["released_output_sha256"] != expected
                or samples[-1]["sha256"] != expected
            ):
                raise RuntimeError("联合生命周期证据不完整，不发布")
            profile["calibration"]["lifecycle"] = {
                "released_repeatable": True,
                "adapter_closed": True,
                "engine_closed": True,
                "supervisor_success": True,
            }
            validate_calibration(profile["calibration"])
            payload = {k: v for k, v in profile.items() if k != "sha256"}
            profile["sha256"] = hashlib.sha256(
                json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
            ).hexdigest()
            from vllm_omni_metal.ane.projection import publish_staged_projection_policy

            policies = {
                r["identity"]: r["staged_policy"] for r in profile["calibration"]["ane_profiles"]
            }
            for policy in policies.values():
                publish_staged_projection_policy(policy)
        target = Path(args.profile).expanduser().resolve()
        _publish_profile(target, profile)
        helpers.write_json(
            output / "published.json", {"path": str(target), "sha256": profile["sha256"]}
        )
        print(f"已通过真实整图1%门槛（有损）；profile：{target}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
