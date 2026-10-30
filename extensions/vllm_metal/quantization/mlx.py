# SPDX-License-Identifier: Apache-2.0
"""Explicit native MLX dense PTQ using the shared DynamicEngine policy.

Search is offline. Loading requires a task-approved profile. Inference stays in
MLX: the Torch bridge is used only to audit source storage and numerical error.
"""

import hashlib
import os
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from vllm_omni_metal.quantization import engine, linear
from vllm_omni_metal.quantization import runtime as admission

_PRECISIONS = {mx.float16: "fp16", mx.bfloat16: "bf16", mx.float32: "fp32"}


def seal_profile(profile):
    """Hash plain metadata; sealing never grants task-quality approval."""
    payload = {key: value for key, value in profile.items() if key != "sha256"}
    profile["sha256"] = hashlib.sha256(linear._encoded(payload)).hexdigest()
    return profile


def _environment():
    return {
        "runtime": "mlx",
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "base": linear._environment(),
        "mlx_module_sha256": hashlib.sha256(
            (Path(nn.__file__).parent / "layers/linear.py").read_bytes()
        ).hexdigest(),
    }


def _host(array):
    from vllm_metal.tensor_bridge import mlx_to_torch

    return mlx_to_torch(array, device="cpu")


def _tensor_identity(value):
    descriptor = {"shape": list(value.shape), "dtype": str(value.dtype)}
    digest = hashlib.sha256(linear._encoded(descriptor))
    admission._tensor_bytes(digest, _host(value))
    return {**descriptor, "sha256": digest.hexdigest()}


def _ordinary(module):
    if type(module) is not nn.Linear:
        raise ValueError("native quantization requires ordinary unquantized MLX Linear")
    weight = module.weight
    if weight.ndim != 2 or min(weight.shape) <= 0 or weight.dtype not in _PRECISIONS:
        raise ValueError("MLX Linear requires nonempty FP16/BF16/FP32 weight")
    if set(module) != ({"weight", "bias"} if "bias" in module else {"weight"}):
        raise ValueError("MLX Linear has extra state requiring an explicit adapter")
    if "bias" in module and (
        module.bias.shape != (weight.shape[0],) or module.bias.dtype != weight.dtype
    ):
        raise ValueError("MLX Linear bias must match weight dtype and output width")
    for value in module.values():
        if not bool(mx.all(mx.isfinite(value))):
            raise ValueError("MLX Linear source must be finite")
    return weight


class NativeQuantizedLinear(nn.Module):
    """Group64 affine projection with no floating weight or per-call bridge."""

    metal_requires_eager = True

    def __init__(self, source, *, bits):
        super().__init__()
        weight = _ordinary(source)
        linear._bits(bits)
        if weight.dtype == mx.float32 or weight.shape[1] % linear.GROUP_SIZE:
            raise ValueError("native W4/W8 requires group64 FP16/BF16; no implicit cast")
        self.input_dims = weight.shape[1]
        self.output_dims = weight.shape[0]
        self.source_dtype = weight.dtype
        self.bits = bits
        if not mx.metal.is_available():
            raise RuntimeError("native quantization requires an available Metal GPU")
        with mx.stream(mx.gpu):
            self.weight, self.scales, self.biases = mx.quantize(
                weight, group_size=linear.GROUP_SIZE, bits=bits, mode="affine"
            )
            if "bias" in source:
                self.bias = mx.array(source.bias)
            mx.eval(self.parameters())
            if not bool(mx.all(mx.isfinite(self.scales)) & mx.all(mx.isfinite(self.biases))):
                raise ValueError(
                    "native quantized affine data overflowed; source remains unchanged"
                )
        self.train(source.training)
        self.freeze()

    @property
    def storage_bytes(self):
        return sum(value.nbytes for value in self.values() if isinstance(value, mx.array))

    def __call__(self, x):
        if (
            not isinstance(x, mx.array)
            or x.dtype != self.source_dtype
            or x.ndim < 1
            or x.size == 0
            or x.shape[-1] != self.input_dims
        ):
            raise ValueError("native quantized input must match source dtype/width and be nonempty")
        if (
            self.weight.dtype != mx.uint32
            or self.scales.dtype != self.source_dtype
            or self.biases.dtype != self.source_dtype
            or ("bias" in self and self.bias.dtype != self.source_dtype)
        ):
            raise ValueError("native quantized storage dtype changed; reload approved source")
        with mx.stream(mx.gpu):
            result = mx.quantized_matmul(
                x,
                self.weight,
                self.scales,
                self.biases,
                transpose=True,
                group_size=linear.GROUP_SIZE,
                bits=self.bits,
                mode="affine",
            )
            if "bias" in self:
                result = result + self.bias
            finite = mx.all(mx.isfinite(x)) & mx.all(mx.isfinite(result))
            mx.eval(result, finite)
            if not bool(finite):
                raise ValueError("native quantized input and output must be finite")
            return result


def _walk(root):
    """Walk actual dict/list module locations, keeping aliases for validation."""
    modules, arrays, slots, structure = {}, {}, {}, {}

    def visit(value, path, parent, key, ancestors):
        if isinstance(value, mx.array):
            arrays[path] = value
            return
        if not isinstance(value, (dict, list, tuple)):
            structure[path] = {"value": admission._config_identity(value)}
            return
        if id(value) in ancestors:
            raise ValueError("cyclic MLX module state needs an explicit adapter")
        ancestors = ancestors | {id(value)}
        structure[path] = {
            "container": admission._type_name(value),
            "keys": (
                [key for key in value if type(key) is str and not key.startswith("_")]
                if isinstance(value, dict)
                else len(value)
            ),
        }
        if isinstance(value, nn.Module):
            modules[path] = value
            if parent is not None:
                slots[path] = (parent, key)
        items = enumerate(value) if isinstance(value, (list, tuple)) else value.items()
        for name, child in items:
            if isinstance(name, str) and name.startswith("_"):
                continue
            if isinstance(value, dict) and type(name) is not str:
                raise ValueError("MLX dictionary state keys must be strings")
            if not isinstance(name, (str, int)) or "." in str(name):
                raise ValueError("MLX state keys require unambiguous dotted paths")
            visit(child, f"{path}.{name}" if path else str(name), value, name, ancestors)

    visit(root, "", None, None, set())
    return modules, arrays, slots, structure


class MLXDenseAdapter:
    """Explicit ordinary Linear replacement at concrete MLX tree locations."""

    def __init__(self, root):
        if not isinstance(root, nn.Module):
            raise NotImplementedError(
                "MLX facades require metal_quantization_scope/apply_metal_quantization_profile hooks"
            )
        self.root = root

    def scope(self, config=None):
        modules, arrays, _, structure = _walk(self.root)
        identity = {
            "runtime": "mlx",
            "config": admission._config_identity(config),
            "modules": {
                name: {
                    "code": admission._code_identity(type(module)),
                    "training": module.training,
                    "attributes": {
                        key: admission._config_identity(value)
                        for key, value in vars(module).items()
                        if not key.startswith("_")
                    },
                }
                for name, module in modules.items()
            },
            "state": {name: _tensor_identity(value) for name, value in arrays.items()},
            "structure": structure,
        }
        return "mlx:" + hashlib.sha256(linear._encoded(identity)).hexdigest()

    def _selection(self, selected):
        if type(selected) is not list or not selected or len(set(selected)) != len(selected):
            raise ValueError("selected must contain distinct nonempty MLX module paths")
        modules, arrays, slots, _ = _walk(self.root)
        chosen_ids = set()
        for name in selected:
            linear._name(name)
            if name not in slots:
                raise ValueError("selected MLX projection requires a writable dict/list location")
            parent = slots[name][0]
            if not (
                type(parent) in (dict, list)
                or (
                    isinstance(parent, nn.Module)
                    and type(parent).__setitem__ is dict.__setitem__
                    and type(parent).__getitem__ is dict.__getitem__
                )
            ):
                raise ValueError("selected MLX projection requires a builtin writable setter")
            _ordinary(modules[name])
            chosen_ids.add(id(modules[name]))
        if any(
            sum(id(module) == identity for module in modules.values()) != 1
            for identity in chosen_ids
        ):
            raise ValueError("selected shared/aliased MLX modules are unsupported")
        # MLX views can use distinct array objects over overlapping storage.
        # CPU DLPack exposes their actual byte ranges without copying weights.
        intervals = []
        for name, array in arrays.items():
            host = _host(array)
            if not host.numel():
                continue
            start = host.data_ptr()
            end = (
                start
                + (
                    1
                    + sum(
                        (size - 1) * stride
                        for size, stride in zip(host.shape, host.stride(), strict=True)
                    )
                )
                * host.element_size()
            )
            chosen = any(name in (f"{path}.weight", f"{path}.bias") for path in selected)
            intervals.append((start, end, chosen))
        all_end = selected_end = -1
        for start, end, chosen in sorted(intervals):
            if start < selected_end or (chosen and start < all_end):
                raise ValueError("selected shared/aliased MLX parameter storage is unsupported")
            all_end = max(all_end, end)
            if chosen:
                selected_end = max(selected_end, end)
        return {name: modules[name] for name in selected}, slots

    def apply(self, profile, *, scope, require_approved=True):
        if type(require_approved) is not bool:
            raise ValueError("require_approved must be boolean")
        admission._profile_header(profile, scope)
        if linear._encoded(profile["environment"]) != linear._encoded(_environment()):
            raise ValueError("native MLX profile source environment mismatch")
        calibration = profile["calibration"]
        if (
            type(calibration) is not dict
            or calibration.get("engine") != "DynamicEngine"
            or calibration.get("implementation_sha256") != engine._implementation_sha256()
        ):
            raise ValueError("native MLX profile requires matching DynamicEngine calibration")
        if type(calibration.get("approved")) is not bool or (
            require_approved and not calibration["approved"]
        ):
            raise ValueError("native MLX profile must be task approved")
        selected = profile["selected"]
        modules, slots = self._selection(selected)
        precisions = calibration.get("precisions")
        if (
            type(precisions) is not dict
            or set(precisions) != set(selected)
            or set(profile["modules"]) != set(selected)
        ):
            raise ValueError("profile must describe exactly the selected MLX projections")
        replacements = {}
        for name, source in modules.items():
            precision = precisions[name]
            if precision not in ("w4a16", "w8a16", _PRECISIONS[source.weight.dtype]):
                raise ValueError("native MLX profile cannot cast source dtype")
            if profile["modules"][name] != _descriptor(source, precision):
                raise ValueError("native MLX source descriptor mismatch")
            replacements[name] = source
        # Construct every packed candidate before making any tree mutation.
        for name, source in modules.items():
            precision = precisions[name]
            if precision in ("w4a16", "w8a16"):
                replacements[name] = NativeQuantizedLinear(
                    source, bits=4 if precision == "w4a16" else 8
                )
        for name, replacement in replacements.items():
            parent, key = slots[name]
            if type(parent) is list:
                list.__setitem__(parent, key, replacement)
            else:
                dict.__setitem__(parent, key, replacement)
        return replacements


def _descriptor(source, precision):
    return {
        "precision": precision,
        "weight": _tensor_identity(source.weight),
        "bias": _tensor_identity(source.bias) if "bias" in source else None,
    }


class MLXDynamicEngine:
    """Native reference and packed trials, shared dual-1% policy and selection."""

    @staticmethod
    def search(adapter, captures, *, scope, max_capture_bytes=64 * 1024**2):
        linear._scope(scope)
        if type(captures) is not dict or not captures:
            raise ValueError("native search requires explicit captured MLX activations")
        if type(max_capture_bytes) is not int or max_capture_bytes <= 0:
            raise ValueError("max_capture_bytes must be a positive bounded activation budget")
        capture_bytes = 0
        for samples in captures.values():
            if type(samples) is not list:
                raise ValueError("captures must contain lists of native activation steps")
            for sample in samples:
                if type(sample) not in (list, tuple) or len(sample) != 2:
                    raise ValueError("captures must contain (step, native activation) pairs")
                if isinstance(sample[1], mx.array):
                    capture_bytes += sample[1].nbytes
        if capture_bytes > max_capture_bytes:
            raise ValueError("native captures exceed the offline activation budget")
        modules, _ = adapter._selection(list(captures))
        audits, precisions, fingerprints = {}, {}, {}
        common_steps = None
        for name, source in modules.items():
            samples = captures[name]
            if type(samples) is not list or len(samples) < 4:
                raise ValueError("native search requires at least four distinct captured steps")
            steps = set()
            for step, value in samples:
                if type(step) is not int or step < 0 or step in steps:
                    raise ValueError("captured steps must be distinct nonnegative integers")
                steps.add(step)
                if (
                    not isinstance(value, mx.array)
                    or value.dtype != source.weight.dtype
                    or value.ndim < 1
                    or not value.size
                    or value.shape[-1] != source.weight.shape[1]
                    or not bool(mx.all(mx.isfinite(value)))
                ):
                    raise ValueError("captures must be finite nonempty native source dtype/width")
            if common_steps is not None and steps != common_steps:
                raise ValueError("native projections must use the same captured steps")
            common_steps = steps
            source_precision = _PRECISIONS[source.weight.dtype]
            source_bytes = sum(value.nbytes for value in source.values())
            references = [_host(source(value)).clone() for _, value in samples]
            candidates = {
                source_precision: engine._audit(
                    source_bytes,
                    [
                        {"step": step, **engine._errors(ref, ref)}
                        for (step, _), ref in zip(samples, references, strict=True)
                    ],
                )
            }
            for precision, bits in (("w4a16", 4), ("w8a16", 8)):
                if source_precision == "fp32" or source.weight.shape[1] % linear.GROUP_SIZE:
                    candidates[precision] = {
                        "passed": False,
                        "storage_bytes": None,
                        "reason": "group64 FP16/BF16 required; no implicit cast",
                    }
                    continue
                candidate = NativeQuantizedLinear(source, bits=bits)
                errors = [
                    {"step": step, **engine._errors(ref, _host(candidate(value)))}
                    for (step, value), ref in zip(samples, references, strict=True)
                ]
                candidates[precision] = engine._audit(candidate.storage_bytes, errors)
                del candidate
            chosen = engine._select(candidates)
            precisions[name] = chosen
            audits[name] = {
                "source_bytes": source_bytes,
                "candidates": candidates,
                "selected": chosen,
            }
            fingerprints[name] = [
                {"step": step, **_tensor_identity(value)} for step, value in samples
            ]
        profile = seal_profile(
            {
                "schema": 2,
                "scope": scope,
                "environment": _environment(),
                "selected": list(modules),
                "modules": {
                    name: _descriptor(source, precisions[name]) for name, source in modules.items()
                },
                "calibration": {
                    "engine": "DynamicEngine",
                    "implementation_sha256": engine._implementation_sha256(),
                    "approved": False,
                    "thresholds": {
                        "relative_l2": engine._THRESHOLD,
                        "scaled_max_abs": engine._THRESHOLD,
                    },
                    "precisions": precisions,
                    "audits": audits,
                    "captures": fingerprints,
                    "local_pass": True,
                },
            }
        )
        return {"profile": profile, "audits": audits, "precisions": precisions, "local_pass": True}


def configure_vllm_model(model, config=None, *, runtime="mlx"):
    """Opt-in vLLM provider; deployment-global settings do not activate it by themselves."""
    mode = os.environ.get("METAL_QUANTIZATION", "off")
    if mode not in ("off", "dynamic"):
        raise ValueError("METAL_QUANTIZATION must be off or dynamic")
    managed = getattr(model, "_metal_native_quantization_metadata", None)
    if managed is not None and mode == "off":
        raise ValueError("installed native quantization cannot be disabled in place; reload source")
    if mode == "off":
        return {"mode": "off", "approved": False, "runtime": runtime}
    if runtime != "mlx":
        raise NotImplementedError("native quantization provider requires the MLX runtime")
    if getattr(config, "lora_config", None) is not None:
        raise NotImplementedError("native dynamic quantization with LoRA is not supported")
    parallel = getattr(config, "parallel_config", None)
    if any(
        getattr(parallel, name, 1) != 1
        for name in ("tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size")
    ):
        raise NotImplementedError("native dynamic quantization requires a single-device execution")
    if getattr(config, "speculative_config", None) is not None:
        raise NotImplementedError(
            "native dynamic quantization with speculative decoding is not supported"
        )
    if getattr(getattr(config, "model_config", None), "quantization", None) is not None:
        raise NotImplementedError("native dynamic quantization requires an unquantized source")
    path = os.environ.get("METAL_QUANTIZATION_PROFILE")
    if not path:
        raise ValueError("METAL_QUANTIZATION_PROFILE is required for dynamic mode")
    if managed is not None:
        if managed.get("profile_path") != path:
            raise ValueError("native quantization already installed from a different profile")
        profile = admission.load_profile(path, scope=managed["scope"])
        if profile["sha256"] != managed["profile_sha256"]:
            raise ValueError("native quantization profile changed after installation")
        return managed
    adapter = MLXDenseAdapter(model)
    scope = adapter.scope(config)
    profile = admission.load_profile(path, scope=scope)
    applied = adapter.apply(profile, scope=scope)
    metadata = {
        "mode": "dynamic",
        "approved": True,
        "runtime": "mlx",
        "scope": scope,
        "profile_path": path,
        "profile_sha256": profile["sha256"],
        "selected": list(applied),
    }
    object.__setattr__(model, "_metal_native_quantization_metadata", metadata)
    return metadata
