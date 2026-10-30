# SPDX-License-Identifier: Apache-2.0
"""Explicit MLX/ANE bridge; all graph compilation is an offline operation.

The native callbacks execute the caller's MLX operators. DLPack handoffs are
synchronized and outputs copied so neither framework can donate a live buffer.
"""
import hashlib
from pathlib import Path
import mlx.core as mx
import torch
import threading

from vllm_metal.tensor_bridge import mlx_to_torch, torch_to_mlx
from vllm_omni_metal.ane.projection import DenseProjection as TorchProjection
from vllm_omni_metal.ane.runtime import DenseSwiGLU as TorchDense, TiledDenseSwiGLU as TorchTiled
from ..quantization.mlx import _tensor_identity
from vllm_omni_metal.quantization.linear import _encoded


def _snapshot(weights):
    return tuple(tuple(_tensor_identity(value) for value in layer) for layer in weights)


def _check_sources(weights, snapshot):
    if weights is None:
        raise RuntimeError("ANE bridge is closed")
    if _snapshot(weights) != snapshot:
        raise RuntimeError("ANE source weights changed after binding")

def ffn_scope(scope, weights, *, kind):
    """Bind explicit MLX FFN contracts to code, caller scope and actual weights."""
    if (not isinstance(scope, str) or not scope.startswith("mlx-ane:")
            or len(scope) != len("mlx-ane:") + 64 or kind not in ("packed", "tiled")):
        raise ValueError("bounded FFN scope and packed/tiled kind are required")
    identity = {"scope": scope, "kind": kind, "weights": _snapshot(weights),
                "bridge": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    return "mlx-ane-ffn:" + hashlib.sha256(_encoded(identity)).hexdigest()


def prepare_ffn(weights, *, scope, kind, contract):
    """Prepare immutable graphs offline; this never publishes an approval."""
    if contract != "bias-free-silu-gate-times-up-then-down-v1":
        raise ValueError("an explicit bias-free SwiGLU mathematical contract is required")
    from vllm_omni_metal.ane.build import prepare_group, prepare_tiled_group
    weights = tuple(tuple(layer) for layer in weights)
    identity = ffn_scope(scope, weights, kind=kind)
    tensors = tuple(tuple(_torch(value) for value in layer) for layer in weights)
    return (prepare_group if kind == "packed" else prepare_tiled_group)(tensors, scope=identity)

def publish_ffn(session, rows, baseline, candidates, *, context=None):
    """Explicitly approve and publish after real full-task measurements.

    This is an offline, caller-authorized operation. The shared upstream
    publisher enforces the 1% live-input audit and 3% pipeline gate.
    """
    _check_sources(session._sources, session._snapshot)
    from vllm_omni_metal.ane.build import publish_profile, publish_tiled_profile
    from vllm_omni_metal.ane.cache import select_profile
    if select_profile(rows, baseline, candidates)["chunks"] != session.core._benchmark_chunks:
        raise ValueError("the selected FFN split must be the frozen audited allocation")
    if isinstance(session, DenseSwiGLU):
        if not session._audited:
            raise RuntimeError("finish_audit must precede FFN publication")
        return publish_profile(session.core, rows, baseline, candidates, context=context)
    if isinstance(session, TiledDenseSwiGLU):
        if session.core._audit:
            raise RuntimeError("finish_audit must precede FFN publication")
        return publish_tiled_profile(session.core, rows, baseline, candidates, context=context)
    raise TypeError("an MLX FFN calibration session is required")




def _torch(value):
    if not isinstance(value, mx.array):
        raise TypeError("ANE bridge requires an MLX array")
    mx.eval(value)
    mx.synchronize()
    return mlx_to_torch(value, device="mps").contiguous().clone()


def _mlx(value):
    result = torch_to_mlx(value, copy=True)
    mx.eval(result)
    mx.synchronize()
    return result


def _native(callback):
    def invoke(value):
        result = callback(_mlx(value))
        if not isinstance(result, mx.array):
            raise TypeError("native callback must return an MLX array")
        return _torch(result)
    return invoke


class DenseProjection:
    """Projection with immutable owned Torch weights and a real MLX baseline."""
    metal_requires_eager = True

    def __init__(self, weight, bias=None, *, scope, benchmark=False, policy=None):
        self._sources = ((weight,) if bias is None else (weight, bias),)
        self._snapshot = _snapshot(self._sources)
        self.weights = (_torch(weight), None if bias is None else _torch(bias))
        with torch.inference_mode():
            self.core = TorchProjection(*self.weights, scope=scope, precision="fp16",
                                        _benchmark=benchmark, _staged_policy=policy)

    def __call__(self, x, *, native, context=None):
        _check_sources(self._sources, self._snapshot)
        with torch.inference_mode():
            return _mlx(self.core(_torch(x), native=_native(native), context=context))

    def set_benchmark_chunks(self, chunks):
        self.core.set_benchmark_chunks(chunks)

    def finish_audit(self):
        self.core.finish_audit()

    def close(self):
        self.core.close()
        self.core.weight = self.core.bias = None
        self.weights = self._sources = None


class TiledDenseSwiGLU:
    """Explicit separate gate/up/down SwiGLU mathematical contract.

    Caller guarantees native(x) equals down(silu(gate(x))*up(x)), with no
    biases or other operations. An audited full-output gate is still required.
    """
    metal_requires_eager = True

    def __init__(self, weights, *, scope, contract, benchmark=False):
        if contract != "bias-free-silu-gate-times-up-then-down-v1":
            raise ValueError("an explicit bias-free SwiGLU mathematical contract is required")
        self._sources = tuple(tuple(layer) for layer in weights)
        self._snapshot = _snapshot(self._sources)
        self.weights = tuple(tuple(_torch(value) for value in layer) for layer in self._sources)
        self._lock = threading.Lock()
        identity = ffn_scope(scope, self._sources, kind="tiled")
        with torch.inference_mode():
            self.core = TorchTiled(self.weights, scope=identity, precision="fp16", _benchmark=benchmark)
        self._first_tiles = {}
        if self.core._group is not None:
            offset = 0
            for plan in self.core.plans:
                for tile in range(len(plan)):
                    packed = self.core._group.weights[offset + tile][0]
                    self._first_tiles[id(packed)] = tile == 0
                offset += len(plan)

    def __call__(self, index, x, *, native, context=None):
        _check_sources(self._sources, self._snapshot)
        callback = _native(native)
        with self._lock, torch.inference_mode():
            group = self.core._group
            if group is not None:
                # The first tile supplies the original whole-FFN GPU rows;
                # later tiles contribute zero GPU rows, while all ANE tile
                # contributions are accumulated by the shared tiled core.
                def tile_native(value, packed, down):
                    if self._first_tiles[id(packed)]:
                        if x.ndim == 3:
                            return callback(value.unsqueeze(0)).squeeze(0)
                        return callback(value)
                    return torch.zeros_like(value)
                group.native = tile_native
            try:
                return _mlx(self.core(index, _torch(x), native=callback, context=context))
            finally:
                if group is not None:
                    del group.native

    def set_benchmark_chunks(self, chunks):
        if not self.core._benchmark or not self.core._audit or type(chunks) is not int or chunks < 0:
            raise ValueError("offline chunk count must be a nonnegative integer")
        self.core._benchmark_chunks = chunks

    def finish_audit(self):
        self.core.finish_audit()

    def close(self):
        self.core.close()
        self.core.weights = ()
        self.weights = self._sources = None


class _PackedCore(TorchDense):
    def native(self, x, packed, down):
        return self.callback(x)


class DenseSwiGLU:
    """Explicit bias-free packed gate/up then down SwiGLU contract."""
    metal_requires_eager = True

    def __init__(self, weights, *, scope, contract, benchmark=False):
        if contract != "bias-free-silu-gate-times-up-then-down-v1":
            raise ValueError("an explicit bias-free SwiGLU mathematical contract is required")
        self._sources = tuple(tuple(pair) for pair in weights)
        self._snapshot = _snapshot(self._sources)
        self.weights = tuple(tuple(_torch(value) for value in pair) for pair in self._sources)
        self._lock = threading.Lock()
        self._audited = False
        identity = ffn_scope(scope, self._sources, kind="packed")
        with torch.inference_mode():
            self.core = _PackedCore(self.weights, scope=identity, precision="fp16", _benchmark=benchmark)

    def __call__(self, index, x, *, native, context=None):
        _check_sources(self._sources, self._snapshot)
        if self._audited and (index, x.shape[0], self.core._benchmark_chunks, context) not in self.core.checks:
            raise RuntimeError("timed execution requires an audited FFN shape")
        with self._lock, torch.inference_mode():
            self.core.callback = _native(native)
            try:
                return _mlx(self.core(index, _torch(x), context=context))
            finally:
                self.core.callback = None

    def set_benchmark_chunks(self, chunks):
        if self._audited or not self.core._benchmark or type(chunks) is not int or chunks < 0:
            raise ValueError("offline chunk count must be a nonnegative integer")
        self.core._benchmark_chunks = chunks

    def finish_audit(self):
        core = self.core
        if (not core._benchmark or core.disabled or core._closed or core._pool is None
                or core._benchmark_chunks <= 0 or any(
                    not any(key[0] == index and key[2] == core._benchmark_chunks
                            for key in core.checks)
                    for index in range(len(core.weights)))):
            raise RuntimeError("every packed FFN requires a live-input audit")
        self._audited = True

    def close(self):
        self.core.close()
        self.core.weights = ()
        self.weights = self._sources = None
