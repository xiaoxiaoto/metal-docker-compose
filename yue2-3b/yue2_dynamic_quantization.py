# SPDX-License-Identifier: Apache-2.0
"""Private YuE2 dynamic-precision loading; never enables a service or searches online."""

import hashlib
import inspect
import json
import math
import os
import weakref
from pathlib import Path

import torch
import torch.nn.functional as F

DEFAULT_PROFILE = Path.home() / "Library/Application Support/yue2-3b/dynamic-profile.json"


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def file_sha256(path):
    path = Path(path)
    before = path.stat()
    result = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024**2):
            result.update(chunk)
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise ValueError("source changed while hashing")
    return result.hexdigest()


def validate_completion(completion, *, selected):
    """Recompute approval from actual complete route records, not approval flags."""
    try:
        records = completion["records"]
        expected = [
            ("native", True, False),
            ("float", True, True),
            ("dynamic", True, False),
            ("float", False, False),
            ("dynamic", False, False),
            ("dynamic", False, False),
            ("float", False, False),
        ]
        if [(r["mode"], r["warmup"], r["capture"]) for r in records] != expected:
            raise ValueError("incomplete native/warmup/interleaved A/B evidence")
        for record in records:
            if (
                record["pending"] != 0
                or record["seed"] != 42
                or record["frames"] != 512
                or record["sample_rate"] != 48000
                or record["shape"][-1] != 2
                or not math.isfinite(record["seconds"])
                or record["seconds"] <= 0
            ):
                raise ValueError("invalid request workload or cleanup")
            if record["mode"] == "dynamic":
                quantized = record["quantized"]
                if set(quantized) != set(selected) or any(
                    q["metrics"].get("calls", 0) <= 0 or q["metrics"].get("rows", 0) <= 0
                    for q in quantized.values()
                ):
                    raise ValueError("missing actual quantized execution")
            elif any(q["metrics"].get("calls", 0) for q in record["quantized"].values()):
                raise ValueError("floating route executed quantized modules")
        native, hooked = records[:2]
        if any(native[k] != hooked[k] for k in ("shape", "dtype", "audio_sha256")):
            raise ValueError("native reference and floating hook differ")
        floating = [r for r in records if r["mode"] == "float" and not r["warmup"]]
        dynamic = [r for r in records if r["mode"] == "dynamic" and not r["warmup"]]
        if (
            len({(r["dtype"], tuple(r["shape"]), r["audio_sha256"]) for r in floating + dynamic})
            != 1
        ):
            raise ValueError("final audio is not bitwise identical and repeatable")
        a, b = sorted(r["seconds"] for r in floating), sorted(r["seconds"] for r in dynamic)
        if not (sum(a) / sum(b) > 1.03 and max(b) < min(a)):
            raise ValueError("stable full-pipeline 3% speed gate failed")
        release = completion["release"]
        if (
            release["kind"] != "expected-post-release-assertion"
            or release["audio_sha256"] != dynamic[0]["audio_sha256"]
            or release["pending"] != 0
            or release["benchmark_held_float_bytes"] != 0
            or release["reproduced"] is not True
            or release["float_weakrefs_released"] is not True
        ):
            raise ValueError("floating-reference release was not reproduced")
        if (
            completion["local_pass"] is not True
            or completion["approved"] is not True
            or not selected
        ):
            raise ValueError("no passing quantized candidate")
    except (KeyError, TypeError, IndexError) as error:
        raise ValueError("malformed full-pipeline approval evidence") from error
    if completion.get("publication_requirements") != {
        "engine_shutdown": True,
        "supervisor_success": True,
        "release_assertion_verified": True,
    }:
        raise ValueError("release, shutdown and supervisor publication requirements are missing")
    return completion


def validate_search(profile):
    """Require the original complete local sensitivity decision, including BF16."""
    try:
        calibration = profile["calibration"]
        if calibration["local_pass"] is not True or calibration["thresholds"] != {
            "relative_l2": 0.01,
            "scaled_max_abs": 0.01,
        }:
            raise ValueError("local sensitivity thresholds changed")
        names = profile["selected"]
        if any(set(calibration[key]) != set(names) for key in ("audits", "captures", "precisions")):
            raise ValueError("incomplete local sensitivity evidence")
        for name in names:
            captures = calibration["captures"][name]
            if [item["step"] for item in captures] != [0, 8, 16, 24]:
                raise ValueError("missing four real capture steps")
            for item in captures:
                if (
                    item["dtype"] != "torch.bfloat16"
                    or not item["shape"]
                    or any(type(n) is not int or n <= 0 for n in item["shape"])
                    or item["shape"][-1] != profile["modules"][name]["shape"][-1]
                    or len(item["sha256"]) != 64
                ):
                    raise ValueError("invalid activation content fingerprint")
            audit = calibration["audits"][name]
            precision = calibration["precisions"][name]
            candidates = audit["candidates"]
            if set(candidates) != {"bf16", "w4a16", "w8a16"} or audit["selected"] != precision:
                raise ValueError("incomplete independent W4/W8/BF16 audit")
            bits = {"bf16": 16, "w4a16": 4, "w8a16": 8}[precision]
            if profile["modules"][name]["bits"] != bits:
                raise ValueError("search precision and selected descriptor differ")
            for candidate in candidates.values():
                if "reason" in candidate:
                    if candidate["passed"] is not False:
                        raise ValueError("unmeasured candidate cannot pass")
                    continue
                if [s["step"] for s in candidate["steps"]] != [0, 8, 16, 24]:
                    raise ValueError("incomplete candidate outputs")
                l2 = max(s["relative_l2"] for s in candidate["steps"])
                peak = max(s["scaled_max_abs"] for s in candidate["steps"])
                if (
                    candidate["relative_l2"] != l2
                    or candidate["scaled_max_abs"] != peak
                    or not all(math.isfinite(v) and v >= 0 for v in (l2, peak))
                    or candidate["passed"] is not (l2 <= 0.01 and peak <= 0.01)
                ):
                    raise ValueError("local output quality evidence differs")
            eligible = [p for p, a in candidates.items() if a["passed"]]
            chosen = min(
                eligible,
                key=lambda p: (
                    candidates[p]["storage_bytes"],
                    {"bf16": 0, "w4a16": 1, "w8a16": 2}[p],
                ),
            )
            if chosen != precision:
                raise ValueError("selected precision is not minimum eligible storage")
    except (KeyError, TypeError, IndexError) as error:
        raise ValueError("malformed local sensitivity evidence") from error


def _idle(model, previous):
    if (
        model._states
        or getattr(model, "_pending", None) is not None
        or model._metal_ffn is not previous
    ):
        raise RuntimeError("dynamic installation requires an unchanged, idle model")


def healthy_group(model):
    from vllm_omni_metal.ane.runtime import DenseSwiGLU

    previous = model._metal_ffn
    _idle(model, previous)
    if (
        os.environ.get("VLLM_OMNI_METAL_ANE") != "off"
        or type(previous) is not DenseSwiGLU
        or previous._closed
        or previous.disabled
        or previous._benchmark
        or previous._pool is not None
        or previous._bridge is not None
    ):
        raise RuntimeError("requires healthy resource-free ANE-off DenseSwiGLU")
    weights = model.nar_fused
    if (
        type(weights) is not list
        or len(weights) != len(model.nar_layers)
        or len(previous.weights) != len(weights)
        or any(
            packed is not w[6] or down is not w[7]
            for (packed, down), w in zip(previous.weights, weights, strict=True)
        )
    ):
        raise ValueError("original fused weight identities differ")
    return previous


def verify_sources(model, sources):
    roots = {
        "model": Path(model.vllm_config.model_config.model).expanduser().resolve(),
        "vae": Path(os.environ["YUE2_VAE"]).expanduser().resolve(),
    }
    if set(sources) != set(roots):
        raise ValueError("incomplete checkpoint identity")
    for label, root in roots.items():
        actual = sorted(root.glob("*.safetensors"))
        entries = sources[label]
        if not actual or [str(p.resolve()) for p in actual] != [e["path"] for e in entries]:
            raise ValueError("source weight set mismatch")
        for path, entry in zip(actual, entries, strict=True):
            if path.stat().st_size != entry["bytes"] or file_sha256(path) != entry["sha256"]:
                raise ValueError("source checkpoint content mismatch")


def reject_float_aliases(model, tensor, index):
    identity = (tensor.device, tensor.untyped_storage().data_ptr())

    def contains(value, seen):
        if isinstance(value, torch.Tensor):
            return (
                value.numel() > 0 and (value.device, value.untyped_storage().data_ptr()) == identity
            )
        if id(value) in seen:
            return False
        seen.add(id(value))
        if isinstance(value, dict):
            return any(contains(v, seen) for v in value.values())
        if isinstance(value, (tuple, list)):
            return any(contains(v, seen) for v in value)
        if isinstance(value, torch.nn.Module):
            return any(contains(v, seen) for v in value.parameters())
        return False

    for layer, weights in enumerate(model.nar_fused):
        for slot, value in enumerate(weights):
            if (layer, slot) != (index, 7) and contains(value, set()):
                raise ValueError("unexpected fused floating alias")
    for module in model.modules():
        for name, value in vars(module).items():
            if name in {"_parameters", "_buffers", "_modules"} or (
                module is model and name in {"nar_fused", "_metal_ffn"}
            ):
                continue
            if contains(value, set()):
                raise ValueError("unregistered floating weight alias")


class DynamicNarAdapter:
    """Packed native gate/up; root down projection is the sole execution authority."""

    def __init__(self, model, scope):
        self._model = weakref.ref(model)
        self.scope = scope
        self._closed = False
        self.metrics = {"calls": 0}

    def __call__(self, index, x, *, context=None):
        if self._closed or torch.is_grad_enabled():
            raise RuntimeError("dynamic adapter is closed or gradients enabled")
        model = self._model()
        if model is None:
            raise RuntimeError("dynamic adapter model no longer exists")
        if type(index) is not int or not 0 <= index < len(model.nar_fused):
            raise ValueError("invalid NAR layer")
        gate, up = F.linear(x, model.nar_fused[index][6]).chunk(2, dim=-1)
        result = model.nar_layers[index].mlp.down_proj(F.silu(gate) * up)
        self.metrics["calls"] += 1
        return result

    def close(self):
        self._closed = True


def install(model, *, scope):
    mode = os.environ.get(
        "YUE2_METAL_QUANTIZATION",
        os.environ.get("VLLM_OMNI_METAL_QUANTIZATION", "off"),
    )
    if mode == "off":
        model._metal_quantization_metadata = {
            "managed": True,
            "mode": "off",
            "approved": False,
        }
        return None
    if mode != "dynamic":
        raise ValueError("YUE2_METAL_QUANTIZATION must be dynamic or off")
    from vllm_omni_metal.quantization.engine import DynamicEngine
    from vllm_omni_metal.quantization.linear import validate_profile
    from vllm_omni_metal.quantization.runtime import load_profile

    previous = healthy_group(model)
    if scope != previous.scope:
        raise ValueError("native FFN scope differs")
    path = os.environ.get(
        "YUE2_METAL_QUANTIZATION_PROFILE",
        os.environ.get("VLLM_OMNI_METAL_QUANTIZATION_PROFILE", str(DEFAULT_PROFILE)),
    )
    profile = load_profile(path, scope=scope)
    calibration = profile["calibration"]
    names = profile["selected"]
    quantized = [n for n in names if profile["modules"][n]["bits"] != 16]
    validate_completion(calibration["full_pipeline"], selected=quantized)
    if calibration.get("engine") != "DynamicEngine" or calibration.get("approved") is not True:
        raise ValueError("only freshly approved DynamicEngine profiles are accepted")
    if (
        calibration["adapter_sha256"] != file_sha256(__file__)
        or calibration["preparation_sha256"]
        != file_sha256(Path(__file__).with_name("prepare_dynamic_quantization.py"))
        or calibration["model_source_sha256"] != file_sha256(inspect.getfile(type(model)))
    ):
        raise ValueError("model/adapter/preparation implementation changed")
    verify_sources(model, calibration["source_weights"])
    modules = {}
    if not 1 <= len(names) <= 4:
        raise ValueError("expected at most four measured down candidates")
    for name in names:
        parts = name.split(".")
        if len(parts) != 4 or parts[0] != "nar_layers" or parts[2:] != ["mlp", "down_proj"]:
            raise ValueError("only independent NAR down projections are allowed")
        index = int(parts[1])
        if str(index) != parts[1] or not 0 <= index < len(model.nar_layers):
            raise ValueError("invalid candidate layer")
        module = model.get_submodule(name)
        if (
            type(module) is not torch.nn.Linear
            or module.bias is not None
            or module.weight is not model.nar_fused[index][7]
        ):
            raise ValueError("candidate is not the original independent down projection")
        modules[name] = module
        if name in quantized:
            reject_float_aliases(model, module.weight, index)
    candidate = DynamicNarAdapter(model, scope)
    with torch.inference_mode():
        validate_profile(profile, modules, scope=scope)
        validate_search(profile)
        _idle(model, previous)
        replacements = DynamicEngine.apply(model, profile, scope=scope)
    previous.close()
    previous.weights = ()
    for name in quantized:
        index = int(name.split(".")[1])
        model.nar_fused[index] = (*model.nar_fused[index][:7], None)
    model._metal_ffn = candidate
    model._metal_quantization_metadata = {
        "managed": True,
        "mode": "dynamic",
        "approved": True,
        "profile": profile,
        "source_bytes": sum(replacements[n].source_bytes for n in quantized),
        "storage_bytes": sum(replacements[n].storage_bytes for n in quantized),
    }
    return candidate
