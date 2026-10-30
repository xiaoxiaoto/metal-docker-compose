# SPDX-License-Identifier: Apache-2.0
"""Explicit Metal GPU operators and opt-in MPS model integration.

MLX models retain their native paged attention, normalization, RoPE and compiled
MLP paths. Diffusion operators below are explicit APIs, not model rewrites.
Metal libraries must be prepared offline with vllm_omni_metal.ops.aot_kernels.
"""

import torch


def _mps(*values):
    import torch

    for value in values:
        if value is not None and (not isinstance(value, torch.Tensor) or value.device.type != "mps"):
            raise ValueError("GPU operations require MPS tensors")


def rms_norm(x, weight, eps):
    _mps(x, weight)
    from vllm_omni_metal.diffusion.ops import rms_norm as operation

    return operation(x, weight, eps)


def add_rms_norm(x, residual, weight, eps):
    _mps(x, residual, weight)
    from vllm_omni_metal.ops import aot_kernels
    from vllm_metal.pytorch_backend.normalization import fused_add_rms_norm

    if aot_kernels._supports_add_rms_norm_epsilon(eps) and aot_kernels.supports_add_rms_norm(x, residual, weight):
        return aot_kernels.add_rms_norm(x, residual, weight, eps)
    return fused_add_rms_norm(x, residual, weight, eps)


def silu_and_mul(x):
    _mps(x)
    from vllm_omni_metal.ops import aot_kernels
    import torch.nn.functional as F

    if aot_kernels.supports_silu(x):
        return aot_kernels.silu_and_mul(x)
    width = x.shape[-1] // 2
    return F.silu(x[..., :width]) * x[..., width:]


def silu_mul(gate, up):
    _mps(gate, up)
    import torch.nn.functional as F
    from vllm_omni_metal.ops import aot_kernels

    if aot_kernels.supports_separate_silu(gate, up):
        return aot_kernels.separate_silu_mul(gate, up)
    return F.silu(gate) * up


def cached_neox_rope(x, cos, sin, rotary_dim):
    _mps(x, cos, sin)
    import torch
    from vllm_omni_metal.ops import aot_kernels

    if (x.ndim != 3 or not x.is_contiguous() or x.requires_grad
        or x.is_neg() or not 0 < x.numel() < 2**31
        or x.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        or not 0 < rotary_dim <= x.shape[-1] <= 1024 or rotary_dim % 2
        or cos.shape != (x.shape[0], rotary_dim // 2) or sin.shape != cos.shape
        or cos.device != x.device or sin.device != x.device
        or cos.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        or sin.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        or cos.requires_grad or sin.requires_grad or aot_kernels._library(x.dtype) is None):
        raise ValueError("unsupported cached NeoX input or missing AOT artifact")
    return aot_kernels.cached_neox_rope(x, cos, sin, rotary_dim)


def complex_rotary(x, freqs):
    _mps(x, freqs)
    from vllm_omni_metal.diffusion.ops import rotary

    return rotary(x, freqs)


def gated_residual_rms_adaln(residual, branch, gate, weight, shift, scale, eps):
    _mps(residual, branch, gate, weight, shift, scale)
    from vllm_omni_metal.diffusion.ops import gated_residual_rms_adaln as operation

    return operation(residual, branch, gate, weight, shift, scale, eps)


def qk_norm_rotary(query, key, query_weight, key_weight, freqs, query_eps=1e-6, key_eps=1e-6):
    _mps(query, key, query_weight, key_weight, freqs)
    from vllm_omni_metal.diffusion.ops import qk_norm_rotary as operation

    return operation(query, key, query_weight, key_weight, freqs, query_eps, key_eps)


def register_mps_ops(config):
    """Register OOT implementations before model construction, only when selected."""
    from vllm_metal import envs
    if "gpu_ops" not in envs.METAL_MODEL_OPTIMIZERS:
        return
    from vllm.model_executor.custom_op import op_registry_oot
    from vllm.model_executor.layers.activation import SiluAndMul
    from vllm_metal.pytorch_backend.rotary import MPSRotaryEmbedding
    from vllm_omni_metal.ops import aot_kernels

    # Resolve existing artifacts during initialization, never invoke a compiler.
    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        aot_kernels._library(dtype)

    if "SiluAndMul" not in op_registry_oot:
        @SiluAndMul.register_oot
        class GPUSiluAndMul(SiluAndMul):
            def forward_oot(self, x):
                if aot_kernels.supports_silu(x):
                    return aot_kernels.silu_and_mul(x)
                return self.forward_native(x)

        GPUSiluAndMul.name = SiluAndMul.name

    # The base MPS provider remains the fallback for full NeoX and unsupported
    # inputs. Do not replace a separately registered custom provider.
    if op_registry_oot.get("RotaryEmbedding") is MPSRotaryEmbedding:
        class GPURotaryEmbedding(MPSRotaryEmbedding):
            def forward_oot(self, positions, query, key=None):
                tensors = (query,) if key is None else (query, key)
                supported = (
                    self.is_neox_style
                    and 0 < self.rotary_dim <= self.head_size <= 1024
                    and self.rotary_dim % 2 == 0
                    and positions.device == query.device
                    and positions.dtype in (torch.int32, torch.int64)
                    and positions.numel() > 0
                    and all(
                        t.device.type == "mps" and t.device == query.device
                        and t.dtype == query.dtype
                        and t.dtype in (torch.float16, torch.bfloat16, torch.float32)
                        and not t.requires_grad and t.ndim in (2, 3)
                        and t.is_contiguous() and not t.is_neg()
                        and 0 < t.numel() < 2**31
                        and t.numel() % (positions.numel() * self.head_size) == 0
                        for t in tensors
                    )
                )
                if not supported or aot_kernels._library(query.dtype) is None:
                    return super().forward_native(positions, query, key)
                # Full low-precision Q/K is already fused by the base provider.
                if (key is not None and self.rotary_dim == self.head_size
                    and query.dtype in (torch.float16, torch.bfloat16)
                    and positions.dtype == torch.int64):
                    return super().forward_native(positions, query, key)
                cache = self._match_cos_sin_cache_dtype(query)
                selected = cache.index_select(0, positions.flatten())
                cos, sin = (table.float().contiguous() for table in selected.chunk(2, dim=-1))
                rotated = [
                    aot_kernels.cached_neox_rope(
                        t.view(positions.numel(), -1, self.head_size), cos, sin, self.rotary_dim
                    ).reshape(t.shape)
                    for t in tensors
                ]
                return rotated[0], rotated[1] if key is not None else None

        GPURotaryEmbedding.name = MPSRotaryEmbedding.name
        op_registry_oot["RotaryEmbedding"] = GPURotaryEmbedding

def install(model, config, *, runtime="mlx"):
    """Install explicitly selected integrations; keep metadata free of tensors."""
    if runtime == "mlx":
        return {"runtime": "mlx", "native": ["paged_attention", "normalization", "rope", "compiled_mlp"], "diffusion": "explicit_api"}
    if runtime != "mps":
        raise ValueError(f"unsupported GPU runtime: {runtime}")
    from vllm_omni_metal.ops import aot_kernels

    return {
        "runtime": "mps",
        "elementwise_aot": {
            name: aot_kernels._library(dtype) is not None
            for name, dtype in (
                ("float16", torch.float16),
                ("bfloat16", torch.bfloat16),
                ("float32", torch.float32),
            )
        },
        "diffusion": "explicit_api",
    }
