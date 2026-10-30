# SPDX-License-Identifier: Apache-2.0
"""Offline MLX export and inference-only, disk-backed MLX dense projections.

The artifact layout and validation are shared with the vendored Omni streaming
format in streaming.py (Apache-2.0). Export happens only offline, after the
source model has been loaded; this is not an out-of-core checkpoint loader.
"""

import hashlib
import os
import threading
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import torch

from . import streaming as disk
from ..quantization.mlx import MLXDenseAdapter, _host, _ordinary
from .streaming import load_profile as load_profile
from .streaming import write_profile as write_profile


@mx.custom_function
def _inference_input(x):
    return x


@_inference_input.vjp
def _reject_vjp(*args):
    raise RuntimeError("MLX streaming cannot be differentiated")


@_inference_input.jvp
def _reject_jvp(*args):
    raise RuntimeError("MLX streaming cannot be differentiated")



def prepare_mlx_linear(linear, directory, *, max_chunk_bytes=disk._DEFAULT_LIMIT):
    """Export an ordinary MLX linear to a new private directory, offline only."""
    _ordinary(linear)
    weight = _host(linear.weight).contiguous()
    bias = _host(linear.bias).contiguous() if "bias" in linear else None
    source = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=bias is not None, device="meta")
    source.weight = torch.nn.Parameter(weight, requires_grad=False)
    if bias is not None:
        source.bias = torch.nn.Parameter(bias, requires_grad=False)
    disk.prepare_linear(source, directory, max_chunk_bytes=max_chunk_bytes)


def _source_matches(source, manifest):
    """Bind every artifact byte, including bias, to the loaded source."""
    _ordinary(source)
    weight = _host(source.weight).contiguous()
    bias = _host(source.bias).contiguous() if "bias" in source else None
    if (manifest["dtype"] != str(weight.dtype).removeprefix("torch.")
            or manifest["in_features"] != weight.shape[1]
            or manifest["out_features"] != weight.shape[0]
            or manifest["bias"] != (bias is not None)):
        raise ValueError("streaming artifact does not describe source Linear")
    for chunk in manifest["chunks"]:
        start, end = chunk["row_start"], chunk["row_end"]
        digest = hashlib.sha256(disk._tensor_bytes(weight[start:end]))
        if bias is not None:
            digest.update(disk._tensor_bytes(bias[start:end]))
        if digest.hexdigest() != chunk["sha256"]:
            raise ValueError("streaming artifact source weight identity mismatch")


class MLXStreamedLinear(nn.Module):
    """Native MLX GPU computation, with no resident floating projection weight."""
    metal_requires_eager = True

    def __init__(self, directory, *, max_weight_bytes=disk._DEFAULT_LIMIT):
        super().__init__()
        disk._positive_int(max_weight_bytes, "max_weight_bytes")
        if not mx.metal.is_available():
            raise RuntimeError("MLX streaming requires an available Metal GPU")
        object.__setattr__(self, "_lock", threading.Lock())
        object.__setattr__(self, "_directory_fd", None)
        parent, name = disk._open_parent(directory)
        try:
            fd = os.open(name, os.O_RDONLY | disk._DIRECTORY | disk._NOFOLLOW, dir_fd=parent)
        finally:
            os.close(parent)
        try:
            disk._check_stat(os.fstat(fd), directory=True)
            manifest, digest = disk._load_manifest(fd, max_weight_bytes)
        except BaseException:
            os.close(fd)
            raise
        object.__setattr__(self, "_directory_fd", fd)
        object.__setattr__(self, "_manifest", manifest)
        object.__setattr__(self, "_manifest_hash", digest)
        object.__setattr__(self, "_max_weight_bytes", max_weight_bytes)
        self._set_training_mode(False)

    @property
    def state(self):
        raise RuntimeError("MLXStreamedLinear cannot expose a state dictionary")

    def _set_training_mode(self, mode):
        if mode:
            raise RuntimeError("MLXStreamedLinear is inference-only")
        return super()._set_training_mode(False)

    def update(self, *args, **kwargs):
        raise RuntimeError("MLXStreamedLinear cannot update or load state")

    def load_weights(self, *args, **kwargs):
        raise RuntimeError("MLXStreamedLinear cannot load state")

    def save_weights(self, *args, **kwargs):
        raise RuntimeError("MLXStreamedLinear cannot save state")

    def set_dtype(self, *args, **kwargs):
        raise RuntimeError("MLXStreamedLinear cannot be cast")

    def __getstate__(self):
        raise RuntimeError("MLXStreamedLinear cannot be serialized")

    def close(self):
        with self._lock:
            if self._directory_fd is not None:
                os.close(self._directory_fd)
                object.__setattr__(self, "_directory_fd", None)

    def __del__(self):
        fd = getattr(self, "_directory_fd", None)
        if fd is not None:
            os.close(fd)
            object.__setattr__(self, "_directory_fd", None)

    def __call__(self, x):
        with self._lock:
            if self._directory_fd is None:
                raise RuntimeError("MLXStreamedLinear is closed")
            if self.training:
                raise RuntimeError("MLXStreamedLinear is inference-only")
            disk._check_stat(os.fstat(self._directory_fd), directory=True)
            manifest, digest = disk._load_manifest(self._directory_fd, self._max_weight_bytes)
            if manifest != self._manifest or digest != self._manifest_hash:
                raise ValueError("artifact manifest changed")
            dtype = {"float16": mx.float16, "bfloat16": mx.bfloat16, "float32": mx.float32}[manifest["dtype"]]
            if (not isinstance(x, mx.array) or x.dtype != dtype or x.ndim < 1
                    or x.size == 0 or x.shape[-1] != manifest["in_features"]):
                raise ValueError("MLX streaming input must match source dtype and width and be nonempty")
            x = _inference_input(x)
            outputs = []
            with mx.stream(mx.gpu):
                for chunk in manifest["chunks"]:
                    fd = disk._open_file(self._directory_fd, chunk["file"])
                    try:
                        if os.fstat(fd).st_size != chunk["bytes"]:
                            raise ValueError("chunk size changed")
                        data = disk._read_bounded(fd, chunk["bytes"])
                    finally:
                        os.close(fd)
                    if hashlib.sha256(data).hexdigest() != chunk["sha256"]:
                        raise ValueError("chunk checksum mismatch")
                    rows = chunk["row_end"] - chunk["row_start"]
                    # BF16 is transferred as raw 16-bit storage without conversion.
                    npdtype = np.float32 if dtype == mx.float32 else np.uint16
                    flat = mx.array(np.frombuffer(data, dtype=npdtype))
                    if dtype != mx.float32:
                        flat = flat.view(dtype)
                    n = rows * manifest["in_features"]
                    weight = flat[:n].reshape(rows, manifest["in_features"])
                    if manifest["bias"]:
                        result = mx.addmm(flat[n:], x, weight.T)
                    else:
                        result = x @ weight.T
                    mx.eval(result)
                    mx.synchronize()
                    outputs.append(result)
                    del data, flat, weight, result
                result = mx.concatenate(outputs, axis=-1) if len(outputs) > 1 else outputs[0]
                mx.eval(result)
                mx.synchronize()
                return result


def make_profile(model, config, artifacts, *, max_weight_bytes=disk._DEFAULT_LIMIT):
    """Build reproducible plain metadata after offline export; never grants approval.

    artifacts maps exact MLX module paths to absolute artifact-directory paths.
    Model, configuration and all source tensor identities are bound by scope.
    """
    disk._positive_int(max_weight_bytes, "max_weight_bytes")
    if type(artifacts) is not dict or not artifacts or any(type(k) is not str or type(v) is not str for k, v in artifacts.items()):
        raise ValueError("artifacts must map selected module paths to absolute directory strings")
    adapter = MLXDenseAdapter(model)
    selected = sorted(artifacts)
    sources, _ = adapter._selection(selected)
    for name in selected:
        runner = MLXStreamedLinear(Path(artifacts[name]), max_weight_bytes=max_weight_bytes)
        try:
            _source_matches(sources[name], runner._manifest)
        finally:
            runner.close()
    return {"schema": 1, "runtime": "mlx", "scope": adapter.scope(config),
            "max_weight_bytes": max_weight_bytes, "artifacts": {k: artifacts[k] for k in selected}}


