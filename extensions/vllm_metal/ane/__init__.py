# SPDX-License-Identifier: Apache-2.0
"""Explicit, offline-calibrated MLX linear ANE integration.

Candidate profiles are unapproved; approval is an external task-quality decision.
No profile field is interpreted as a Python module, path to code, or expression.
"""
import hashlib
import json
import os
import stat
from pathlib import Path

import mlx.nn as nn

from vllm_omni_metal.ane import cache
from vllm_omni_metal.ane.projection import prepare_projection as _prepare
from vllm_omni_metal.ane.projection import publish_projection_profile
from vllm_omni_metal.quantization import linear

from .. import lifecycle
from ..quantization import mlx as mlx_adapter
from .bridge import (
    DenseProjection, DenseSwiGLU, TiledDenseSwiGLU, _torch, prepare_ffn, publish_ffn,
)

from . import torch as torch_adapter

# Existing Torch entry points remain available for explicit MPS integrations.
from vllm_omni_metal.ane.projection import DenseProjection as TorchDenseProjection
from vllm_omni_metal.ane.runtime import DenseSwiGLU as TorchDenseSwiGLU
from vllm_omni_metal.ane.runtime import TiledDenseSwiGLU as TorchTiledDenseSwiGLU
from vllm_omni_metal.ane.build import prepare_group, prepare_tiled_group, publish_profile, publish_tiled_profile


def _scope(model, config):
    identity = {
        "model": mlx_adapter.MLXDenseAdapter(model).scope(config),
        "environment": mlx_adapter._environment(),
        "implementation": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "bridge": hashlib.sha256(Path(__file__).with_name("bridge.py").read_bytes()).hexdigest(),
    }
    return "mlx-ane:" + hashlib.sha256(linear._encoded(identity)).hexdigest()


def source_scope(model, config):
    """Bind explicit FFN calibration to the actual MLX model and configuration."""
    return _scope(model, config)


def _descriptor(source):
    mlx_adapter._ordinary(source)
    return {"weight": mlx_adapter._tensor_identity(source.weight),
            "bias": mlx_adapter._tensor_identity(source.bias) if "bias" in source else None}


def _group_scope(scope, name):
    return scope + ":" + name


def prepare_projection(model, config, name):
    """Build one graph offline and return a live calibration session.

    Call session(x, native=source), set_benchmark_chunks('auto' or integer),
    finish_audit(), then candidate_projection with full-pipeline timings. Close
    the session after calibration. build_tools must have been run separately.
    """
    adapter = mlx_adapter.MLXDenseAdapter(model)
    sources, _ = adapter._selection([name])
    source = sources[name]
    scope = _scope(model, config)
    descriptor = _descriptor(source)
    weight = _torch(source.weight)
    bias = _torch(source.bias) if "bias" in source else None
    _prepare(weight, bias, scope=_group_scope(scope, name))
    session = DenseProjection(source.weight, source.bias if "bias" in source else None,
                              scope=_group_scope(scope, name), benchmark=True)
    if session.core._pool is None:
        session.close()
        raise RuntimeError("ANE calibration graph could not be loaded")
    return session, {"scope": scope, "name": name, "source": descriptor}


def candidate_projection(session, binding, rows, baseline, candidates, *, context=None):
    """Return an unapproved candidate with real audit and whole-task timings."""
    core = session.core
    if core._audit:
        raise RuntimeError("finish_audit must freeze the live-input audit first")
    record = publish_projection_profile(core, rows, baseline, candidates,
                                        context=context, publish=False)
    if not record["chunks"]:
        raise ValueError("candidate did not meet the 3% whole-pipeline gate")
    policy = {"schema": cache.SCHEMA, "group": core.identity,
              "device": core._bridge.device, "kind": "linear-bias-v1",
              "profiles": {cache.profile_label(rows, context): record}}
    return {"schema": 1, "approved": False, "scope": binding["scope"],
            "selected": {binding["name"]: {"source": binding["source"], "policy": policy}}}


def _read_profile(path):
    if not os.path.isabs(path):
        raise ValueError("METAL_ANE_PROFILE must be an absolute path")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode) or st.st_uid != os.getuid() or stat.S_IMODE(st.st_mode) != 0o600 or st.st_size > 1024**2:
            raise ValueError("ANE profile must be an owned regular 0600 file at most 1 MiB")
        def pairs(items):
            result = {}
            for key, value in items:
                if key in result:
                    raise ValueError("duplicate ANE profile key")
                result[key] = value
            return result
        with os.fdopen(fd, "r", encoding="utf-8") as handle:
            fd = -1
            profile = json.load(handle, object_pairs_hook=pairs,
                                parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
    finally:
        if fd != -1:
            os.close(fd)
    if type(profile) is not dict or set(profile) != {"schema", "approved", "scope", "selected"} or type(profile["schema"]) is not int or profile["schema"] != 1 or profile["approved"] is not True or type(profile["selected"]) is not dict or not profile["selected"]:
        raise ValueError("ANE profile must contain an explicit task approval")
    return profile


class _Linear(nn.Module):
    metal_requires_eager = True

    def __init__(self, source, projection, descriptor):
        super().__init__()
        self._source = source
        self._projection = projection
        self._descriptor = descriptor
        self.train(source.training)

    def __call__(self, x):
        if self._source is None:
            raise RuntimeError("ANE projection is closed")
        if _descriptor(self._source) != self._descriptor:
            raise RuntimeError("ANE source projection changed after binding")
        return self._projection(x, native=self._source)

    def close(self):
        self._projection.close()
        self._source = None


def apply(model, config, *, runtime="mlx"):
    """Install approved, unshared ordinary MLX or Torch/MPS Linear projections."""

    if runtime not in ("mlx", "mps"):
        raise NotImplementedError("automatic ANE replacement requires MLX or Torch MPS")
    path = os.environ.get("METAL_ANE_PROFILE")
    if not path:
        raise ValueError("explicit ANE selection requires METAL_ANE_PROFILE")
    profile = _read_profile(path)
    if runtime == "mps":
        return torch_adapter.apply(model, config, profile)
    scope = _scope(model, config)
    if profile["scope"] != scope:
        raise ValueError("ANE profile model/config/source scope mismatch")
    adapter = mlx_adapter.MLXDenseAdapter(model)
    sources, slots = adapter._selection(list(profile["selected"]))
    replacements = {}
    try:
        for name, source in sources.items():
            entry = profile["selected"][name]
            if type(entry) is not dict or set(entry) != {"source", "policy"} or entry["source"] != _descriptor(source):
                raise ValueError("ANE source descriptor mismatch")
            projection = DenseProjection(source.weight, source.bias if "bias" in source else None,
                                         scope=_group_scope(scope, name), policy=entry["policy"])
            if projection.core._pool is None or projection.core.disabled:
                projection.close()
                raise RuntimeError("approved ANE projection is unavailable")
            if not any(context is None and chunks for (_, context), chunks in projection.core.profiles.items()):
                projection.close()
                raise ValueError("automatic MLX Linear requires an approved context-free shape")
            replacements[name] = _Linear(source, projection, entry["source"])
    except BaseException:
        for wrapper in replacements.values():
            wrapper._projection.close()
        raise
    register_resource = lifecycle.register_resource
    for name, wrapper in replacements.items():
        parent, key = slots[name]
        if type(parent) is list:
            list.__setitem__(parent, key, wrapper)
        else:
            dict.__setitem__(parent, key, wrapper)
        register_resource(model, wrapper)
    return {"runtime": runtime, "enabled": True, "projections": list(replacements), "scope": scope}
