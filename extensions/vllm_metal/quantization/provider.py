# SPDX-License-Identifier: Apache-2.0
"""One explicit provider for native MLX and ordinary Torch/MPS projections."""

import os


def configure_vllm_model(model, config=None, *, runtime="mlx"):
    if runtime == "mlx":
        from .mlx import configure_vllm_model as configure

        return configure(model, config, runtime=runtime)
    if runtime != "mps":
        raise NotImplementedError("dynamic quantization requires MLX or Torch MPS")
    mode = os.environ.get("METAL_QUANTIZATION", "off")
    if mode not in ("off", "dynamic"):
        raise ValueError("METAL_QUANTIZATION must be off or dynamic")
    managed = getattr(model, "_metal_extension_quantization", None)
    if managed is not None:
        raise ValueError("quantization is already installed; reload the unmodified model")
    if mode == "off":
        return {"mode": "off", "approved": False, "runtime": runtime}
    if getattr(config, "lora_config", None) is not None:
        raise NotImplementedError("dynamic quantization does not support LoRA")
    if getattr(config, "speculative_config", None) is not None:
        raise NotImplementedError("dynamic quantization does not support speculative decoding")
    parallel = getattr(config, "parallel_config", None)
    if any(getattr(parallel, field, 1) != 1 for field in (
        "tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size",
    )):
        raise NotImplementedError("dynamic quantization requires one device")
    if getattr(getattr(config, "model_config", None), "quantization", None) is not None:
        raise NotImplementedError("dynamic quantization requires an unquantized source")
    path = os.environ.get("METAL_QUANTIZATION_PROFILE")
    if not path:
        raise ValueError("METAL_QUANTIZATION_PROFILE is required for dynamic mode")
    from vllm_omni_metal.quantization import runtime as admission
    from vllm_omni_metal.quantization.adapter import DenseModelAdapter

    scope = admission.model_scope(model, config, runtime="torch")
    profile = admission.load_profile(path, scope=scope)
    admission._approved(profile)
    applied = DenseModelAdapter(model).apply(profile, scope=scope)
    if set(applied) != set(profile["selected"]):
        raise RuntimeError("quantization did not install every approved projection")
    metadata = {"mode": "dynamic", "approved": True, "runtime": runtime,
                "scope": scope, "profile_path": path, "selected": list(applied),
                "profile_sha256": profile["sha256"]}
    object.__setattr__(model, "_metal_extension_quantization", metadata)
    return metadata
