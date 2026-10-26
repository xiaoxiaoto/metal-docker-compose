#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""真实 SeedVR2 7B 图片与五帧视频 DynamicEngine 离线校准；不下载、不管理服务。"""

import argparse
import gc
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import weakref
from pathlib import Path

GIB = 1024**3
WEIGHT_BYTES = 16480583960 + 501324814


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as file:
        temporary = Path(file.name)
        json.dump(value, file, ensure_ascii=False, sort_keys=True, allow_nan=False, indent=2)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)


def publish_json(path, value):
    """同目录原子 no-replace；并发批准/用户既有文件均不能被覆盖。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as file:
            temporary = Path(file.name)
            json.dump(value, file, ensure_ascii=False, sort_keys=True, allow_nan=False, indent=2)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            raise FileExistsError(f"批准路径已有产物，拒绝覆盖；本次证据仍保留：{path}") from None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def projection_error(reference, actual, *, budget=128 * 1024**2):
    """遍历全部投影输出、分块归约；不捕获输入，不做有利子采样。"""
    if (
        reference.shape != actual.shape
        or reference.dtype != actual.dtype
        or reference.device != actual.device
        or not reference.is_contiguous()
        or not actual.is_contiguous()
        or reference.numel() == 0
    ):
        raise ValueError("真实局部投影输出 shape/dtype/device/连续性无效")
    persistent = actual.numel() * actual.element_size()
    # 实际候选输出加最多四份FP32分块scratch，总计保持原128MiB预算。
    chunk_size = (budget - persistent) // (4 * 4)
    if chunk_size <= 0:
        raise MemoryError("单次完整投影输出已占满128MiB；拒绝子采样")
    a_flat, b_flat = reference.view(-1), actual.view(-1)
    energy = noise = peak = maximum = 0.0
    for start in range(0, reference.numel(), chunk_size):
        a = a_flat[start : start + chunk_size].float()
        b = b_flat[start : start + chunk_size].float()
        delta = b - a
        energy += float(a.square().sum().item())
        peak = max(peak, float(a.abs().amax().item()))
        maximum = max(maximum, float(delta.abs().amax().item()))
        noise += float(delta.square().sum().item())
        del a, b, delta
    if not all(math.isfinite(value) for value in (energy, noise, peak, maximum)):
        raise ValueError("真实完整局部投影含非有限值或归约溢出")
    return {
        "relative_l2": math.sqrt(noise / max(energy, 1e-24)),
        "scaled_max_abs": maximum / max(peak, 1e-12),
    }


def compare(reference, actual):
    import numpy as np

    if reference.shape != actual.shape or reference.dtype != actual.dtype or not actual.size:
        raise ValueError("完整输出 shape/dtype 不匹配或为空")
    a, b = reference.astype(np.float64), actual.astype(np.float64)
    if not np.isfinite(a).all() or not np.isfinite(b).all() or np.ptp(a) == 0 or np.ptp(b) == 0:
        raise ValueError("真实完整输出非有限或为单色")
    delta = b - a
    return {
        "relative_l2": float(np.linalg.norm(delta.ravel()) / max(np.linalg.norm(a.ravel()), 1e-12)),
        "scaled_max_abs": float(np.abs(delta).max() / max(np.abs(a).max(), 1e-12)),
        "bitwise_identical": reference.tobytes() == actual.tobytes(),
    }


def qualified(error):
    return error["relative_l2"] <= 0.01 and error["scaled_max_abs"] <= 0.01


def resource_gate(output, *, startup=False, process=None):
    import psutil

    available = psutil.virtual_memory().available
    minimum = WEIGHT_BYTES + 10 * GIB if startup else 4 * GIB
    rss = 0
    if process is not None:
        for item in [process, *process.children(recursive=True)]:
            try:
                rss += item.memory_info().rss
            except psutil.NoSuchProcess:
                pass
    free = shutil.disk_usage(output).free
    snapshot = {
        "available_bytes": available,
        "process_tree_rss_bytes": rss,
        "disk_free_bytes": free,
        "minimum_available_bytes": minimum,
        "maximum_rss_bytes": WEIGHT_BYTES + 10 * GIB,
    }
    if available < minimum or free < 2 * GIB or rss > WEIGHT_BYTES + 10 * GIB:
        raise MemoryError(f"SeedVR2 安全资源门禁拒绝：{snapshot}")
    return snapshot


def stage(args, name):
    write_json(args.output / "stage.json", {"stage": name, "started": time.time()})


def capture_candidate(name, original, workloads, run, baseline, directory, *, label, scope):
    """Capture each complete workload once, then search with bounded disk-backed inputs."""
    import torch

    from vllm_omni_metal.quantization.capture import DiskCaptures
    from vllm_omni_metal.quantization.engine import DynamicEngine

    captures = DiskCaptures(
        {name: original},
        Path(directory).absolute(),
        max_bytes=128 * 1024**2,
        max_disk_bytes=GIB,
    )
    holder = {"step": None}
    recorded = set()
    hook = None

    def capture(module, inputs):
        step = holder["step"]
        if step in recorded:
            return
        (activation,) = inputs
        if activation.dtype != torch.float16:
            raise ValueError("完整捕获必须保持原始 FP16 dtype")
        captures.record(name, step, activation)
        recorded.add(step)

    try:
        hook = original.register_forward_pre_hook(capture)
        for step, workload in enumerate(workloads):
            holder["step"] = step
            unchanged, _ = run(workload, label)
            if unchanged.tobytes() != baseline[workload[:2]].tobytes():
                raise ValueError("捕获改变原始完整 FP16 基线输出")
        hook.remove()
        hook = None
        return DynamicEngine.search({name: original}, captures, scope=scope)
    finally:
        if hook is not None:
            hook.remove()
        try:
            write_json(Path(directory).with_suffix(".json"), captures.summary())
        finally:
            captures.close()


def worker(args):
    import importlib.metadata

    import numpy as np
    import torch
    from PIL import Image
    from safetensors import safe_open
    from vllm_omni.diffusion.config import set_current_diffusion_config
    from vllm_omni.diffusion.data import AttentionConfig, OmniDiffusionConfig
    from vllm_omni.diffusion.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm_omni.diffusion.forward_context import set_forward_context
    from vllm_omni.diffusion.models.seedvr2 import quantization as adapter
    from vllm_omni.diffusion.models.seedvr2.pipeline_seedvr2 import (
        SeedVR2Pipeline,
        prepare_request,
    )
    from vllm_omni.diffusion.models.seedvr2.weight_manifest import (
        MANIFEST,
        check_index,
        validate_files,
    )
    from vllm_omni.diffusion.request import OmniDiffusionRequest
    from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    from vllm_omni_metal.quantization.engine import DynamicEngine
    from vllm_omni_metal.quantization.linear import (
        QuantizedLinear,
        _untied,
        build_profile,
        validate_profile,
    )

    if importlib.metadata.version("vllm-omni") != "0.32.0.dev0+cpu.seedvr2mps2":
        raise ValueError("必须使用本目录准备的 seedvr2mps2 私有非 editable wheel")
    root = Path(sys.prefix).resolve()
    if not Path(adapter.__file__).resolve().is_relative_to(root) or not (root / ".ready").is_file():
        raise ValueError("校准必须由拥有 ready 标记的私有 overlay Python -I 执行")
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != adapter.digest(
        Path(adapter.__file__).with_name("calibration.py")
    ):
        raise ValueError("校准器与私有 wheel 指纹不同；重新准备 overlay")
    if not torch.backends.mps.is_available():
        raise ValueError("需要真实 MPS，不允许 CPU fallback")
    if torch.mps.recommended_max_memory() < WEIGHT_BYTES + 4 * GIB:
        raise MemoryError("Metal 推荐预算不足权重加 4 GiB，不调整水位")
    os.environ["SEEDVR2_METAL_QUANTIZATION"] = "off"
    os.environ["VLLM_OMNI_METAL_QUANTIZATION"] = "off"
    stage(args, "validating-real-weights-and-inputs")
    if not check_index(args.model).is_file():
        raise ValueError("缺少原始7B model_index.json；先用本目录安全准备器建立索引")
    validate_files(args.model)
    image = Image.open(args.image)
    image.load()
    config = OmniDiffusionConfig(
        model=str(args.model),
        model_class_name="SeedVR2Pipeline",
        dtype=torch.float16,
        enforce_eager=True,
        vae_use_tiling=True,
        distributed_executor_backend="uni",
        diffusion_attention_config=AttentionConfig(default="TORCH_SDPA"),
    )
    workloads = []
    for kind in ("image", "video"):
        for seed in adapter.SEEDS:
            params = OmniDiffusionSamplingParams(
                seed=seed,
                num_inference_steps=1,
                guidance_scale=1,
                height=args.height,
                width=args.width,
                num_outputs_per_prompt=1,
            )
            media = {"image": [image.copy()]} if kind == "image" else {"video": str(args.video)}
            request = OmniDiffusionRequest(
                prompt={"prompt": " ", "multi_modal_data": media},
                sampling_params=params,
                request_id=f"{kind}-{seed}",
            )
            prepare_request(request)
            prepared = request.prepared_layout
            if prepared.frame_count != (1 if kind == "image" else 5):
                raise ValueError("真实输入必须是一张图片和最小非单帧合法 5 帧 CFR 视频；不裁剪替代")
            h, w = prepared.frames.shape[1:3]
            if args.height <= h or args.width <= w:
                raise ValueError("校准必须真实超分：输出宽高均大于原始输入；不自动缩图")
            workloads.append((kind, seed, request))
    resource_gate(args.output, startup=True)
    stage(args, "loading-fp16-baseline")
    init_distributed_environment(
        world_size=1,
        rank=0,
        local_rank=0,
        backend="gloo",
        distributed_init_method="file://" + str(args.output / "distributed-init"),
    )
    initialize_model_parallel(data_parallel_size=1, sequence_parallel_size=1, backend="gloo")
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float16)
        with set_current_diffusion_config(config), torch.device("mps"), torch.inference_mode():
            pipeline = SeedVR2Pipeline(od_config=config)

            def weights():
                for name in MANIFEST:
                    prefix = "transformer." if name.startswith("diffusion_models/") else "vae."
                    with safe_open(args.model / name, framework="pt", device="cpu") as source:
                        for key in source.keys():
                            yield prefix + key, source.get_tensor(key)

            pipeline.load_weights(weights())
    finally:
        torch.set_default_dtype(previous)
    pipeline.eval()
    pipeline.vae.use_tiling = True
    bound_scope = adapter.scope(pipeline)
    from vllm_omni.diffusion.models.seedvr2 import joint

    joint_scope = joint.execution_scope(pipeline, bound_scope=bound_scope) if args.joint else None
    write_json(
        args.output / "environment.json",
        {
            "scope": bound_scope,
            "python": sys.executable,
            "torch": str(torch.__version__),
            "omni": importlib.metadata.version("vllm-omni"),
            "width": args.width,
            "height": args.height,
            "image": {"path": str(args.image), "sha256": adapter.digest(args.image)},
            "video": {"path": str(args.video), "sha256": adapter.digest(args.video)},
            "original_weight_bytes": WEIGHT_BYTES,
            "lossy": True,
        },
    )
    candidates = adapter.modules(pipeline)
    _untied(candidates, list(candidates), root=pipeline)

    def run(workload, label, quantized=None):
        kind, seed, request = workload
        stage(args, f"{label}-{kind}-{seed}")
        resource_gate(args.output)
        request.sampling_params.generator = torch.Generator(device="mps").manual_seed(seed)
        calls = dict.fromkeys(quantized or {}, 0)

        def count(module, inputs, output, *, name):
            calls[name] += 1

        from functools import partial

        handles = [
            module.register_forward_hook(partial(count, name=name))
            for name, module in (quantized or {}).items()
        ]
        torch.mps.synchronize()
        started = time.perf_counter()
        try:
            with torch.inference_mode(), set_forward_context(omni_diffusion_config=config):
                result = pipeline(DiffusionRequestBatch([request]))
        finally:
            for handle in handles:
                handle.remove()
        torch.mps.synchronize()
        if len(result) != 1 or result[0].error is not None:
            raise ValueError("完整 pipeline 未返回单个成功输出")
        payload = result[0].output["payload"]
        value = payload["image" if kind == "image" else "video"]
        array = value.detach().cpu().numpy()
        run.last_seconds = time.perf_counter() - started
        expected = (
            (1, args.height, args.width, 3)
            if kind == "image"
            else (1, 5, args.height, args.width, 3)
        )
        if array.shape != expected or array.dtype != np.uint8:
            raise ValueError(f"完整原生 RGB 输出合同不匹配：{array.shape}, {array.dtype}")
        if kind == "video":
            fps = result[0].output["metadata"].get("video", {}).get("fps")
            if fps != request.prepared_layout.source.fps:
                raise ValueError("完整视频未保留源 fps")
        np.save(args.output / f"{label}-{kind}-{seed}.npy", array, allow_pickle=False)
        frames = array.reshape(-1, args.height, args.width, 3)
        for index, frame in enumerate(frames):
            Image.fromarray(frame).save(args.output / f"{label}-{kind}-{seed}-{index}.png")
        if any(count <= 0 for count in calls.values()):
            raise ValueError("完整 pipeline 未命中所选量化 forward")
        return array, calls

    baseline = {}
    for workload in workloads:
        key = workload[:2]
        baseline[key], _ = run(workload, "baseline")
    for workload in reversed(workloads):
        repeat, _ = run(workload, "baseline-repeat")
        if repeat.tobytes() != baseline[workload[:2]].tobytes():
            raise ValueError("原始 FP16 同 seed 完整输出不稳定，拒绝发布")
    audits = []
    successful = []
    with torch.inference_mode():
        for index, (name, original) in enumerate(candidates.items()):
            stage(args, f"candidate-{index}-capture")
            search = capture_candidate(
                name,
                original,
                workloads,
                run,
                baseline,
                args.output / f"capture-{index}",
                label=f"candidate-{index}-capture",
                scope=bound_scope,
            )
            precision = search["precisions"][name]
            bits = {"w4a16": 4, "w8a16": 8, "fp16": 16}[precision]
            quantized = original if bits == 16 else QuantizedLinear.from_linear(original, bits=bits)
            local = []
            workload_holder = {"workload": None}

            def audit_projection(
                module,
                inputs,
                output,
                *,
                bits=bits,
                quantized=quantized,
                workload_holder=workload_holder,
                local=local,
            ):
                (x,) = inputs
                actual = output if bits == 16 else quantized(x)
                error = projection_error(output, actual)
                kind, seed = workload_holder["workload"]
                local.append(
                    {
                        **error,
                        "kind": kind,
                        "seed": seed,
                        "calls": 1,
                        "shape": list(x.shape),
                        "output_shape": list(output.shape),
                    }
                )

            hook = original.register_forward_hook(audit_projection)
            try:
                for workload in workloads:
                    workload_holder["workload"] = workload[:2]
                    unchanged, _ = run(workload, f"candidate-{index}-local")
                    if unchanged.tobytes() != baseline[workload[:2]].tobytes():
                        raise ValueError("局部审计改变完整基线输出")
            finally:
                hook.remove()
                del audit_projection
            audit = {
                "name": name,
                "audited_calls": len(local),
                "local": local,
                "samples": [],
                "search": search["profile"]["calibration"],
                "precision": precision,
            }
            audits.append(audit)
            covered = {(entry["kind"], entry["seed"]) for entry in local if entry["calls"] > 0}
            if covered != {workload[:2] for workload in workloads} or not all(
                qualified(error) for error in local
            ):
                del quantized
                continue
            parent = pipeline.get_submodule(name.rpartition(".")[0])
            parent._modules["proj_out"] = quantized
            try:
                for workload in workloads:
                    actual, calls = run(workload, f"candidate-{index}", {name: quantized})
                    repeat, _ = run(workload, f"candidate-{index}-repeat", {name: quantized})
                    error = compare(baseline[workload[:2]], actual)
                    sample = {
                        **error,
                        "kind": workload[0],
                        "seed": workload[1],
                        "frames": 1 if workload[0] == "image" else 5,
                        "calls": calls,
                        "repeat_identical": repeat.tobytes() == actual.tobytes(),
                        "input_sha256": adapter.digest(
                            args.image if workload[0] == "image" else args.video
                        ),
                        "baseline_sha256": hashlib.sha256(
                            baseline[workload[:2]].tobytes()
                        ).hexdigest(),
                        "candidate_sha256": hashlib.sha256(actual.tobytes()).hexdigest(),
                    }
                    audit["samples"].append(sample)
                if (not args.joint or bits in (4, 8)) and all(
                    qualified(s) and s["repeat_identical"] for s in audit["samples"]
                ):
                    successful.append(
                        (
                            max(
                                max(s["relative_l2"], s["scaled_max_abs"]) for s in audit["samples"]
                            ),
                            index,
                            name,
                        )
                    )
            finally:
                parent._modules["proj_out"] = original
            write_json(args.output / "candidates.json", audits)
            del quantized
    del original, candidates
    gc.collect()
    write_json(args.output / "candidates.json", audits)
    if not successful:
        write_json(
            args.output / "result.json", {"status": "rejected", "lossy": True, "candidates": audits}
        )
        raise ValueError(
            "所有有限候选未通过真实局部激活及完整图片/视频 1% 门禁；不发布批准 profile"
        )
    _, selected_index, selected_name = min(successful)
    selected_audit = audits[selected_index]
    with torch.inference_mode():
        metadata = {
            "adapter": adapter.ADAPTER_VERSION,
            "status": "approved",
            "lossy": selected_audit["precision"] != "fp16",
            **selected_audit["search"],
            "approved": True,
            "samples": selected_audit["samples"],
            "local": selected_audit["local"],
            "audited_calls": selected_audit["audited_calls"],
        }
        # 最终证据尚未释放前不将该临时元数据传给生产消费者，也不写批准文件。
        candidates = adapter.modules(pipeline)
        profile = build_profile(
            candidates,
            [selected_name],
            scope=bound_scope,
            calibration=metadata,
            precisions={selected_name: selected_audit["precision"]},
        )
        validate_profile(profile, candidates, scope=bound_scope)

        original_ref = weakref.ref(candidates[selected_name])
        weight_ref = weakref.ref(candidates[selected_name].weight)
        replacements = DynamicEngine.apply(pipeline, profile, scope=bound_scope)
        candidates.clear()
        # Identity intentionally retains the source; quantized paths must release it.
        if selected_audit["precision"] == "fp16":
            replacements = {selected_name: pipeline.get_submodule(selected_name)}
    gc.collect()
    torch.mps.empty_cache()
    if selected_audit["precision"] != "fp16" and (
        original_ref() is not None or weight_ref() is not None
    ):
        raise ValueError("所选原 FP16 module/weight 未释放；不发布批准")
    for workload in reversed(workloads):
        actual, calls = run(workload, "released-baseline", replacements)
        sample = next(s for s in metadata["samples"] if (s["kind"], s["seed"]) == workload[:2])
        sample["released_baseline_identical"] = (
            hashlib.sha256(actual.tobytes()).hexdigest() == sample["candidate_sha256"]
        )
        sample["released_calls"] = calls
        if not sample["released_baseline_identical"]:
            raise ValueError("释放所选浮点基线后的完整输出不一致；不发布")
    # 临时 profile 的所选浮点 descriptor 已在替换前验证；仅更新校准元数据后重建摘要。
    # build_profile 必须使用原 FP16 权重，因此从固定 safetensors 仅流式重读所选投影。
    name = selected_name.removeprefix("transformer.")
    with safe_open(
        args.model / "diffusion_models/seedvr2_7b_fp16.safetensors", framework="pt", device="cpu"
    ) as source:
        with torch.device("mps"), torch.inference_mode():
            descriptor = profile["modules"][selected_name]
            linear = torch.nn.Linear(
                descriptor["shape"][1],
                descriptor["shape"][0],
                bias=True,
                dtype=torch.float16,
                device="meta",
            )
            linear.weight = torch.nn.Parameter(
                source.get_tensor(name + ".weight").to("mps"), requires_grad=False
            )
            linear.bias = torch.nn.Parameter(
                source.get_tensor(name + ".bias").to("mps"), requires_grad=False
            )
            profile = build_profile(
                {selected_name: linear},
                [selected_name],
                scope=bound_scope,
                calibration=metadata,
                precisions={selected_name: selected_audit["precision"]},
            )
            validate_profile(profile, {selected_name: linear}, scope=bound_scope)
    del linear
    adapter.validate_approval(profile)
    if args.joint:
        from vllm_omni.diffusion.models.seedvr2.joint_calibration import calibrate

        profile = calibrate(
            pipeline, profile, workloads, baseline, run, replacements, args, bound_scope=joint_scope
        )
    else:
        from vllm_omni.diffusion.distributed.parallel_state import (
            destroy_distributed_environment,
            destroy_model_parallel,
        )

        destroy_model_parallel()
        destroy_distributed_environment()
    write_json(args.output / "approved.json", profile)
    write_json(
        args.output / "result.json",
        {
            "status": "approved",
            "lossy": selected_audit["precision"] != "fp16",
            "selected": [selected_name],
            "profile_sha256": profile["sha256"],
            "candidates": audits,
        },
    )
    stage(args, "completed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=Path, default=Path.home() / ".cache/modelscope/hub/models/Comfy-Org/SeedVR2"
    )
    parser.add_argument("--image", required=True, type=Path, help="真实原始图片，不自动生成/缩图")
    parser.add_argument(
        "--video", required=True, type=Path, help="真实最小合法5帧CFR视频，不自动裁剪"
    )
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True, help="新的证据目录，拒绝混用历史产物")
    parser.add_argument(
        "--profile",
        type=Path,
        default=Path.home() / "Library/Application Support/seedvr2/dynamic-approved.json",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--joint",
        action="store_true",
        help="calibrate real ANE proj_in + GPU GELU + DynamicEngine proj_out",
    )
    args = parser.parse_args()
    args.output = args.output.expanduser().resolve()
    for field in ("model", "image", "video", "profile"):
        path = getattr(args, field).expanduser()
        setattr(args, field, path.absolute() if field == "profile" else path.resolve())
    if args.worker:
        worker(args)
        return 0
    if args.output.exists():
        raise ValueError("证据目录已存在；拒绝覆盖或借用历史结果")
    # 在启动子进程/资源预留之前，仅用本目录现有的安全 header/hash 检查拒绝缺失权重。
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "seedvr2_weight_manifest", Path(__file__).with_name("prepare.py")
    )
    manifest = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(manifest)
    if not manifest.check_index(args.model).is_file():
        raise ValueError("缺少原始7B model_index.json；先用本目录安全准备器建立索引")
    manifest.validate_files(args.model)
    if any(not path.is_file() for path in (args.image, args.video)):
        raise FileNotFoundError("必须先具备真实图片与5帧视频；不会下载或生成输入")
    args.output.mkdir(parents=True)
    initial = resource_gate(args.output, startup=True)
    import psutil

    environment = dict(
        os.environ,
        SEEDVR2_METAL_QUANTIZATION="off",
        VLLM_OMNI_METAL_QUANTIZATION="off",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        PYTORCH_ENABLE_MPS_FALLBACK="0",
        VLLM_WORKER_MULTIPROC_METHOD="spawn",
        VLLM_PLUGINS="metal,vllm_omni_register_models",
    )
    command = [sys.executable, "-I", str(Path(__file__).resolve()), *sys.argv[1:], "--worker"]
    history = [initial]
    started = time.monotonic()
    with (args.output / "worker.log").open("w") as log:
        child = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
        try:
            process = psutil.Process(child.pid)
            while child.poll() is None:
                history.append(resource_gate(args.output, process=process))
                elapsed = time.monotonic() - started
                current = args.output / "stage.json"
                if elapsed > 7200 or (
                    current.exists()
                    and time.time() - json.loads(current.read_text())["started"] > 1800
                ):
                    raise TimeoutError("完整校准总计7200秒/单阶段1800秒门禁拒绝")
                time.sleep(1)
            if child.returncode:
                raise RuntimeError(
                    f"真实校准失败，exit={child.returncode}；见 {args.output / 'worker.log'}"
                )
        except BaseException:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
            raise
        finally:
            write_json(
                args.output / "supervisor.json",
                {
                    "resource_samples": history,
                    "elapsed_seconds": time.monotonic() - started,
                    "returncode": child.returncode,
                },
            )
    resource_gate(args.output)
    # 父监督器确认子进程零退出后才提交；超时/资源拒绝不能留下新默认批准文件。
    from vllm_omni_metal.quantization.runtime import _constant, _pairs

    approved = json.loads(
        (args.output / "approved.json").read_text(),
        object_pairs_hook=_pairs,
        parse_constant=_constant,
    )
    if args.joint:
        from vllm_omni.diffusion.models.seedvr2 import joint

        joint.validate(approved, scope=approved["scope"])
        from vllm_omni_metal.ane.projection import publish_staged_projection_policy

        publish_staged_projection_policy(approved["projection"]["staged_policy"])
    publish_json(args.profile, approved)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"SeedVR2 校准拒绝：{error}", file=sys.stderr)
        raise SystemExit(1) from error
