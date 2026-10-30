# SPDX-License-Identifier: Apache-2.0
"""Ordinary Torch/MPS linear integration for explicitly approved ANE profiles."""
import hashlib
from pathlib import Path

import torch
from vllm_omni_metal.ane.projection import DenseProjection, prepare_projection as _prepare
from vllm_omni_metal.quantization import linear, runtime as admission


def source_scope(model, config):
    if callable(getattr(model, "metal_quantization_scope", None)):
        raise ValueError("custom model scope hooks cannot establish an ordinary ANE source identity")
    if admission._components(model):
        raise ValueError("Torch pipeline components require an explicit ANE adapter")
    identity = {
        "model": admission.model_scope(model, config, runtime="torch"),
        "implementation": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "provider": hashlib.sha256(Path(__file__).with_name("__init__.py").read_bytes()).hexdigest(),
        "environment": linear._environment(),
    }
    return "mps-ane:" + hashlib.sha256(linear._encoded(identity)).hexdigest()


def _tensor(value):
    if value is None:
        return None
    digest = hashlib.sha256()
    admission._tensor_bytes(digest, value)
    return {"shape": list(value.shape), "dtype": str(value.dtype),
            "device": str(value.device), "sha256": digest.hexdigest()}


def _descriptor(source):
    if type(source) is not torch.nn.Linear or set(source._parameters) != {"weight", "bias"} or source._buffers or source._modules:
        raise ValueError("ANE requires an ordinary Torch Linear with no extra state")
    if (source._forward_hooks or source._forward_pre_hooks or source._backward_hooks
            or "forward" in source.__dict__):
        raise ValueError("ANE cannot replace a projection with hooks or a custom forward")
    if source.weight.device.type != "mps" or (source.bias is not None and source.bias.device != source.weight.device):
        raise ValueError("ANE requires MPS source weights")
    return {"weight": _tensor(source.weight), "bias": _tensor(source.bias)}


def _selection(model, names):
    if not isinstance(model, torch.nn.Module) or type(names) is not list or not names or len(names) != len(set(names)):
        raise ValueError("distinct named Torch projections on a Module are required")
    modules = dict(model.named_modules(remove_duplicate=False))
    all_modules = list(model.named_modules(remove_duplicate=False))
    tensors = list(model.named_parameters(remove_duplicate=False)) + list(model.named_buffers(remove_duplicate=False))
    result = {}
    for name in names:
        if type(name) is not str or not name or name not in modules:
            raise ValueError("selected projection must have a writable module path")
        module = modules[name]
        _descriptor(module)
        if sum(value is module for _, value in all_modules) != 1:
            raise ValueError("shared Torch modules are unsupported")
        parent_name, _, child = name.rpartition(".")
        parent = modules[parent_name]
        if type(parent._modules) is not dict or parent._modules.get(child) is not module:
            raise ValueError("selected module requires an ordinary registered slot")
        for parameter in (module.weight, module.bias):
            if parameter is None:
                continue
            storage = parameter.untyped_storage().data_ptr()
            if sum(value.untyped_storage().data_ptr() == storage for _, value in tensors) != 1:
                raise ValueError("shared Torch parameter storage is unsupported")
        result[name] = (module, parent, child)
    return result


class Session:
    """Live offline calibration session over an immutable original MPS linear."""
    def __init__(self, source, *, scope, benchmark=False, policy=None):
        self.source = source
        self.descriptor = _descriptor(source)
        with torch.inference_mode():
            self.core = DenseProjection(source.weight, source.bias, scope=scope,
                                        precision="fp16", _benchmark=benchmark,
                                        _staged_policy=policy)

    def __call__(self, x, *, native=None, context=None):
        if self.source is None:
            raise RuntimeError("ANE projection is closed")
        if _descriptor(self.source) != self.descriptor:
            raise RuntimeError("ANE source projection changed after binding")
        if native is not None and native is not self.source:
            raise ValueError("Torch calibration must use the original Linear as native baseline")
        with torch.inference_mode():
            return self.core(x, native=self.source if native is None else native, context=context)

    def set_benchmark_chunks(self, chunks):
        self.core.set_benchmark_chunks(chunks)

    def finish_audit(self):
        self.core.finish_audit()

    def close(self):
        self.core.close()
        self.core.weight = self.core.bias = None
        self.source = None


def prepare_projection(model, config, name):
    source, _, _ = _selection(model, [name])[name]
    scope = source_scope(model, config)
    descriptor = _descriptor(source)
    with torch.inference_mode():
        _prepare(source.weight, source.bias, scope=scope + ":" + name)
    session = Session(source, scope=scope + ":" + name, benchmark=True)
    if session.core._pool is None:
        session.close()
        raise RuntimeError("ANE calibration graph could not be loaded")
    return session, {"scope": scope, "name": name, "source": descriptor}


class _Linear(torch.nn.Module):
    metal_requires_eager = True

    def __init__(self, session):
        super().__init__()
        # Keep the original module out of this wrapper's registered module tree.
        object.__setattr__(self, "_session", session)
        self.training = session.source.training

    def forward(self, x):
        return self._session(x)

    def close(self):
        self._session.close()


def apply(model, config, profile):
    scope = source_scope(model, config)
    if profile["scope"] != scope:
        raise ValueError("ANE profile model/config/source scope mismatch")
    selected = _selection(model, list(profile["selected"]))
    replacements = {}
    try:
        for name, (source, parent, child) in selected.items():
            entry = profile["selected"][name]
            if type(entry) is not dict or set(entry) != {"source", "policy"} or entry["source"] != _descriptor(source):
                raise ValueError("ANE source descriptor mismatch")
            session = Session(source, scope=scope + ":" + name, policy=entry["policy"])
            if session.core._pool is None or session.core.disabled or not any(
                context is None and chunks for (_, context), chunks in session.core.profiles.items()
            ):
                session.close()
                raise RuntimeError("approved context-free ANE projection is unavailable")
            replacements[name] = _Linear(session)
    except BaseException:
        for wrapper in replacements.values():
            wrapper.close()
        raise
    from ..lifecycle import register_resource
    for name, wrapper in replacements.items():
        _, parent, child = selected[name]
        parent._modules[child] = wrapper
        register_resource(model, wrapper)
    return {"runtime": "mps", "enabled": True, "projections": list(replacements), "scope": scope}
