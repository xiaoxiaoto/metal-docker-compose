# SPDX-License-Identifier: Apache-2.0
"""SeedVR2 7B 独立 GELU(tanh) video down 投影；仅离线/加载时替换。"""

import hashlib
import json
import math
import os
from pathlib import Path

import torch

ADAPTER_VERSION = "seedvr2-dynamic-v2"
CANDIDATES = tuple(f"transformer.blocks.{index}.mlp.vid.proj_out" for index in (0, 12, 24, 35))
SEEDS = (7723, 7724)


def digest(path):
    path = Path(path)
    before = path.stat()
    result = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024**2), b""):
            result.update(chunk)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("指纹读取期间文件改变")
    return result.hexdigest()


def modules(pipeline):
    """7B 的实际 proj_out 返回 Tensor；不包装 vLLM tuple/skip_bias_add Linear。"""
    result = {}
    if getattr(pipeline, "variant", None) != "7b-comfy-fp16":
        raise ValueError("动态量化只接受私有 7B FP16 pipeline")
    for name in CANDIDATES:
        parent = pipeline.get_submodule(name.rpartition(".")[0])
        projection = pipeline.get_submodule(name)
        if (
            type(parent.act) is not torch.nn.GELU
            or parent.act.approximate != "tanh"
            or type(projection) is not torch.nn.Linear
            or projection.bias is None
            or projection.weight.dtype != torch.float16
        ):
            raise ValueError("7B 普通带 bias GELU(tanh) 投影合同不匹配")
        result[name] = projection
    return result


def scope(pipeline):
    from . import config as seed_config
    from .weight_manifest import MANIFEST, check_index, validate_files

    config = pipeline.od_config
    if (
        config.dtype != torch.float16
        or not config.enforce_eager
        or config.quantization_config is not None
    ):
        raise ValueError("保持 FP16 eager 和原始 checkpoint；拒绝 config 量化")
    if not config.vae_use_tiling:
        raise ValueError("批准 scope 要求保持原部署 VAE tiling")
    directory = Path(config.model).expanduser().resolve()
    if not check_index(directory).is_file():
        raise ValueError("缺少固定7B model_index.json")
    validate_files(directory)  # 完整发布 SHA256，不把局部投影哈希冒充完整权重绑定。
    root = Path(__file__).resolve().parent
    payload = {
        "adapter": ADAPTER_VERSION,
        "weights": {
            name: {"path": str(directory / name), "bytes": size, "sha256": sha}
            for name, (size, sha) in MANIFEST.items()
        },
        "index": digest(directory / "model_index.json"),
        "source": {file.name: digest(file) for file in sorted(root.glob("*.py"))},
        "policy": {
            "dtype": "float16",
            "eager": True,
            "tiling": True,
            "frame_pixels": pipeline.frame_pixels,
            "clip_pixels": pipeline.clip_pixels,
            "max_frames": seed_config.VLLM_OMNI_SEEDVR2_MAX_FRAMES,
            "seeds": list(SEEDS),
            "candidates": list(CANDIDATES),
            "final_relative_l2": 0.01,
            "final_scaled_max_abs": 0.01,
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return ADAPTER_VERSION + ":" + hashlib.sha256(encoded).hexdigest()


def validate_approval(profile):
    """局部激活不足以批准；必须两种真实完整输出、两个 seed 和释放基线后的复验。"""
    if type(profile) is not dict or type(profile.get("calibration")) is not dict:
        raise ValueError("批准 profile 与校准元数据必须是 JSON 对象")
    selected = profile.get("selected")
    metadata = profile.get("calibration", {})
    if (
        type(selected) is not list
        or not selected
        or len(set(selected)) != len(selected)
        or any(name not in CANDIDATES for name in selected)
        or metadata.get("adapter") != ADAPTER_VERSION
        or metadata.get("status") != "approved"
        or metadata.get("engine") != "DynamicEngine"
        or metadata.get("approved") is not True
    ):
        raise ValueError("缺少 DynamicEngine 真实完整输出批准记录")
    descriptors = profile.get("modules")
    if type(descriptors) is not dict or set(descriptors) != set(selected):
        raise ValueError("动态批准 profile 缺少投影描述符")
    for descriptor in descriptors.values():
        if (
            type(descriptor) is not dict
            or descriptor.get("dtype") != "torch.float16"
            or not (
                (
                    descriptor.get("bits") in (4, 8)
                    and descriptor.get("mode") == "affine"
                    and descriptor.get("group_size") == 64
                )
                or (
                    descriptor.get("bits") == 16
                    and descriptor.get("mode") == "identity"
                    and descriptor.get("group_size") is None
                )
            )
        ):
            raise ValueError("SeedVR2 动态候选必须保留源 FP16 激活/identity")
    if metadata.get("thresholds") != {"relative_l2": 0.01, "scaled_max_abs": 0.01}:
        raise ValueError("DynamicEngine 必须使用固定 1% 双门槛")
    for name in selected:
        captures = metadata.get("captures", {}).get(name)
        audit = metadata.get("audits", {}).get(name, {})
        precision = metadata.get("precisions", {}).get(name)
        expected = {4: "w4a16", 8: "w8a16", 16: "fp16"}[descriptors[name]["bits"]]
        if (
            precision != expected
            or audit.get("selected") != expected
            or audit.get("candidates", {}).get(expected, {}).get("passed") is not True
            or type(captures) is not list
            or len(captures) != 4
            or {entry.get("step") for entry in captures} != {0, 1, 2, 3}
            or any(entry.get("dtype") != "torch.float16" for entry in captures)
        ):
            raise ValueError("缺少匹配源 FP16 的四步 DynamicEngine 搜索证据")
    local = metadata.get("local")
    if type(local) is not list or len(local) < 4 or metadata.get("audited_calls") != len(local):
        raise ValueError("缺少四个真实完整图的局部激活校准")
    covered = set()
    for error in local:
        if (
            type(error) is not dict
            or type(error.get("kind")) is not str
            or type(error.get("seed")) is not int
            or type(error.get("calls")) is not int
            or error["calls"] <= 0
        ):
            raise ValueError("局部激活真实模态/seed/调用记录无效")
        key = (error["kind"], error["seed"])
        if key not in {(kind, seed) for kind in ("image", "video") for seed in SEEDS}:
            raise ValueError("局部激活样本不属于完整校准工作量")
        covered.add(key)
        for field in ("shape", "output_shape"):
            shape = error.get(field)
            if (
                type(shape) is not list
                or not shape
                or any(type(n) is not int or n <= 0 for n in shape)
            ):
                raise ValueError("局部完整激活 shape 证据缺失")
        for field in ("relative_l2", "scaled_max_abs"):
            value = error.get(field)
            if (
                type(value) not in (float, int)
                or not math.isfinite(value)
                or not 0 <= value <= 0.01
            ):
                raise ValueError("真实局部激活未通过 1% 门禁")
    if covered != {(kind, seed) for kind in ("image", "video") for seed in SEEDS}:
        raise ValueError("局部激活未覆盖两种模态和两个seed")
    samples = metadata.get("samples")
    if type(samples) is not list or len(samples) != 4:
        raise ValueError("批准要求图片/五帧视频各两个固定 seed")
    seen = set()
    for sample in samples:
        if type(sample) is not dict:
            raise ValueError("完整输出证据格式无效")
        if type(sample.get("kind")) is not str or type(sample.get("seed")) is not int:
            raise ValueError("完整输出模态/seed 格式无效")
        key = (sample.get("kind"), sample.get("seed"))
        if key in seen or key not in {
            (kind, seed) for kind in ("image", "video") for seed in SEEDS
        }:
            raise ValueError("完整输出样本重复或不完整")
        seen.add(key)
        expected_frames = 1 if key[0] == "image" else 5
        if (
            sample.get("frames") != expected_frames
            or sample.get("repeat_identical") is not True
            or sample.get("released_baseline_identical") is not True
            or type(sample.get("released_calls")) is not dict
            or any(
                type(sample["released_calls"].get(name)) is not int
                or sample["released_calls"][name] <= 0
                for name in selected
            )
            or type(sample.get("calls")) is not dict
            or any(
                type(sample["calls"].get(name)) is not int or sample["calls"][name] <= 0
                for name in selected
            )
        ):
            raise ValueError("完整输出/真实量化调用/释放基线后复验不完整")
        for field in ("relative_l2", "scaled_max_abs"):
            value = sample.get(field)
            if (
                type(value) not in (float, int)
                or not math.isfinite(value)
                or not 0 <= value <= 0.01
            ):
                raise ValueError("完整超分输出未通过 1% 误差门禁")
        for field in ("input_sha256", "baseline_sha256", "candidate_sha256"):
            value = sample.get(field)
            if (
                type(value) is not str
                or len(value) != 64
                or any(c not in "0123456789abcdef" for c in value)
            ):
                raise ValueError("完整输出或真实输入 SHA256 缺失")


def apply(pipeline, profile, *, bound_scope):
    from vllm_omni_metal.quantization.engine import DynamicEngine
    from vllm_omni_metal.quantization.linear import validate_profile

    candidates = modules(pipeline)
    validate_approval(profile)
    validate_profile(profile, candidates, scope=bound_scope)
    return DynamicEngine.apply(pipeline, profile, scope=bound_scope)


def consume_profile(pipeline):
    """加载时只消费批准文件；默认 off，请求阶段不搜索。"""
    mode = os.environ.get(
        "SEEDVR2_METAL_QUANTIZATION",
        os.environ.get("VLLM_OMNI_METAL_QUANTIZATION", "off"),
    )
    if mode not in ("off", "dynamic", "joint"):
        raise ValueError("SEEDVR2_METAL_QUANTIZATION 必须是 off/dynamic/joint")
    pipeline.seedvr2_quantization = {"enabled": False, "reason": "explicit-off", "selected": []}
    pipeline._metal_quantization_metadata = {
        "managed": True,
        "mode": mode,
        "approved": False,
    }
    if mode == "off":
        return
    if mode == "joint":
        from .joint import consume

        path = os.environ.get("SEEDVR2_METAL_QUANTIZATION_PROFILE")
        if not path:
            raise ValueError("joint requires an explicit approved profile")
        with torch.inference_mode():
            consume(pipeline, path)
        return
    path = os.environ.get(
        "SEEDVR2_METAL_QUANTIZATION_PROFILE",
        os.environ.get("VLLM_OMNI_METAL_QUANTIZATION_PROFILE"),
    )
    if not path:
        raise ValueError("dynamic 缺少批准 profile 路径")
    from vllm_omni_metal.quantization.runtime import load_profile

    bound_scope = scope(pipeline)
    profile = load_profile(path, scope=bound_scope)
    with torch.inference_mode():
        replacements = apply(pipeline, profile, bound_scope=bound_scope)
    pipeline._metal_quantization_metadata = {
        "managed": True,
        "mode": "dynamic",
        "approved": True,
        "profile": profile,
        "profile_path": str(path),
        "scope": bound_scope,
        "selected": list(profile["selected"]),
    }
    pipeline.seedvr2_quantization = {
        "enabled": True,
        "reason": "approved-dynamic",
        "selected": list(profile["selected"]),
        "profile": str(path),
        "quantized": list(replacements),
    }
    print(f"SeedVR2 DynamicEngine 已加载批准 profile：{profile['selected']}", flush=True)
