# SPDX-License-Identifier: Apache-2.0
"""Private SeedVR2 biased FP16 projection / GPU GELU / DynamicEngine contract."""

import hashlib
import json
import math
import os
import stat
from pathlib import Path

import torch

VERSION = "seedvr2-joint-v1"


def sha(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def execution_scope(pipeline, *, bound_scope=None):
    from vllm_omni_metal.ane import projection
    from vllm_omni_metal.quantization.engine import _implementation_sha256

    from . import quantization

    return (
        VERSION
        + ":"
        + sha(
            {
                "model": bound_scope if bound_scope is not None else quantization.scope(pipeline),
                "projection": quantization.digest(projection.__file__),
                "quantization": _implementation_sha256(),
                "joint": quantization.digest(__file__),
                "calibrator": quantization.digest(Path(__file__).with_name("joint_calibration.py")),
                "precision": "fp16",
                "activation": "GELU(tanh)",
                "bias": True,
                "chunks": 1,
                "contexts": [1, 5],
                "minimum_gain": 0.03,
            }
        )
    )


class JointProjection(torch.nn.Module):
    def __init__(self, original, *, scope, benchmark=False, staged_policy=None):
        super().__init__()
        from vllm_omni_metal.ane.projection import DenseProjection

        if (
            type(original) is not torch.nn.Linear
            or original.bias is None
            or original.weight.dtype != torch.float16
        ):
            raise ValueError("joint requires original FP16 biased proj_in")
        self.original = original
        self.projection = DenseProjection(
            original.weight,
            original.bias,
            scope=scope,
            precision="fp16",
            _benchmark=benchmark,
            _staged_policy=staged_policy,
        )
        self.context = None
        self.records = set()
        self.benchmark = benchmark

    def forward(self, x):
        rows = x.numel() // x.shape[-1]
        before = self.projection.metrics["ane_chunks"]
        result = self.projection(x, native=self.original, context=self.context)
        self.records.add((rows, self.context))
        require_hit = not self.benchmark or self.projection._benchmark_chunks > 0
        if require_hit and (
            self.projection.disabled or self.projection.metrics["ane_chunks"] <= before
        ):
            raise RuntimeError(
                "joint execution requires a real nonzero ANE hit; fallback forbidden"
            )
        return result

    def close(self):
        self.projection.close()


def validate(bundle, *, scope):
    from vllm_omni_metal.ane.cache import select_profile

    from .quantization import validate_approval

    if type(bundle) is not dict or set(bundle) != {
        "schema",
        "scope",
        "quantization",
        "projection",
        "samples",
        "lifecycle",
        "sha256",
    }:
        raise ValueError("expected complete joint approval")
    from vllm_omni_metal.quantization.runtime import _profile_header

    _profile_header(bundle["quantization"], bundle["quantization"].get("scope"))
    if (
        bundle["schema"] != VERSION
        or bundle["scope"] != scope
        or bundle["sha256"] != sha({k: v for k, v in bundle.items() if k != "sha256"})
    ):
        raise ValueError("joint scope/content mismatch")
    validate_approval(bundle["quantization"])
    selected = bundle["quantization"]["selected"]
    if len(selected) != 1 or bundle["quantization"]["modules"][selected[0]]["bits"] not in (4, 8):
        raise ValueError("joint approval requires actual quantization")
    p = bundle["projection"]
    if (
        type(p) is not dict
        or set(p) != {"name", "identity", "profiles", "staged_policy"}
        or p["name"] != selected[0].replace("proj_out", "proj_in")
        or not p["identity"]
    ):
        raise ValueError("joint projection identity missing")
    if bundle["lifecycle"] != {
        "float_down_released": True,
        "projection_closed": True,
        "engine": "stateless-DynamicEngine",
        "distributed_shutdown": True,
    }:
        raise ValueError("joint lifecycle incomplete")
    samples = bundle["samples"]
    if (
        type(samples) is not list
        or len(samples) != 4
        or {(s["kind"], s["seed"]) for s in samples}
        != {(k, s) for k in ("image", "video") for s in (7723, 7724)}
    ):
        raise ValueError("joint requires four complete workloads")
    for sample in samples:
        if (
            sample["repeat_identical"] is not True
            or sample["released_identical"] is not True
            or sample["ane_chunks"] <= 0
            or sample["quant_calls"] <= 0
        ):
            raise ValueError("joint real execution/repeat/release evidence missing")
        for key in ("relative_l2", "scaled_max_abs"):
            value = sample[key]
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or not 0 <= value <= 0.01
            ):
                raise ValueError("joint complete output fails fixed 1% gate")
        for key in ("baseline_sha256", "candidate_sha256", "input_sha256"):
            value = sample[key]
            if (
                type(value) is not str
                or len(value) != 64
                or any(c not in "0123456789abcdef" for c in value)
            ):
                raise ValueError("joint workload digest missing")
        if not sample["rows"] or len(sample["rows"]) != len(set(sample["rows"])):
            raise ValueError("joint requires nonempty unique activation rows")
        for rows in sample["rows"]:
            if (
                select_profile(rows, sample["baseline_seconds"], {1: sample["candidate_seconds"]})[
                    "chunks"
                ]
                != 1
            ):
                raise ValueError("joint requires at least 3% ANE whole-pipeline benefit")
    expected = {(r, 1 if s["kind"] == "image" else 5) for s in samples for r in s["rows"]}
    records = p["profiles"]
    if {(r["rows"], r["context"]) for r in records} != expected:
        raise ValueError("joint projection profiles do not cover complete workloads")
    if len(records) != len(expected):
        raise ValueError("duplicate joint projection profiles")
    from vllm_omni_metal.ane.projection import _profiles

    inputs = bundle["quantization"]["modules"][selected[0]]["shape"][0]
    profiles, _ = _profiles({f"{r['rows']}:{r['context']}": r["record"] for r in records}, inputs)
    if any(chunks != 1 for chunks in profiles.values()):
        raise ValueError("zero ANE split cannot approve joint execution")
    policy = p["staged_policy"]
    if (
        type(policy) is not dict
        or set(policy) != {"schema", "group", "device", "kind", "profiles"}
        or type(policy["schema"]) is not int
        or policy["schema"] != 1
        or policy["group"] != p["identity"]
        or policy["kind"] != "linear-bias-v1"
        or type(policy["device"]) is not str
        or not policy["device"]
        or policy["profiles"] != {f"{r['rows']}:{r['context']}": r["record"] for r in records}
    ):
        raise ValueError("staged ANE policy differs from joint approval")


def load(path, *, scope):
    from vllm_omni_metal.quantization.runtime import _constant, _pairs

    path = Path(path).expanduser()
    if not path.is_absolute():
        raise ValueError("joint profile must be absolute")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) != 0o600
            or info.st_size > 4 * 1024**2
        ):
            raise ValueError("joint profile requires bounded owned0600 regular file")
        raw = stream.read(4 * 1024**2 + 1)
    if len(raw) > 4 * 1024**2:
        raise ValueError("joint profile too large")
    bundle = json.loads(raw, object_pairs_hook=_pairs, parse_constant=_constant)
    validate(bundle, scope=scope)
    return bundle


def consume(pipeline, path):
    from . import quantization

    bound_scope = quantization.scope(pipeline)
    scope = execution_scope(pipeline, bound_scope=bound_scope)
    bundle = load(path, scope=scope)
    from vllm_omni_metal.quantization.runtime import _profile_header

    _profile_header(bundle["quantization"], bound_scope)
    name = bundle["projection"]["name"]
    parent = pipeline.get_submodule(name.rpartition(".")[0])
    wrapper = JointProjection(parent.proj_in, scope=scope)
    try:
        if wrapper.projection.identity != bundle["projection"]["identity"]:
            raise ValueError("joint prepared graph identity mismatch")
        if wrapper.projection._bridge is None or wrapper.projection.disabled:
            raise ValueError("joint production bridge unavailable")
        expected = {(r["rows"], r["context"]): 1 for r in bundle["projection"]["profiles"]}
        if wrapper.projection.profiles != expected:
            raise ValueError("installed ANE profiles differ from joint approval")
        from vllm_omni_metal.ane.cache import cache_root
        from vllm_omni_metal.ane.projection import _policy

        policy = _policy(cache_root() / f"{wrapper.projection.identity}.json")
        approved = {
            f"{r['rows']}:{r['context']}": r["record"] for r in bundle["projection"]["profiles"]
        }
        if policy["profiles"] != approved:
            raise ValueError("ANE policy numerical/timing evidence differs from joint approval")
        quantization.apply(pipeline, bundle["quantization"], bound_scope=bound_scope)
        parent.proj_in = wrapper

        # Each request has a fixed frame context, set at the pipeline boundary.
        def context(module, args):
            batch = args[0]
            requests = batch.requests if hasattr(batch, "requests") else batch
            frames = {request.prepared_layout.frame_count for request in requests}
            if len(frames) != 1 or next(iter(frames)) not in (1, 5):
                raise ValueError("joint workload not approved")
            wrapper.context = next(iter(frames))

        pipeline._seedvr2_joint_context_hook = pipeline.register_forward_pre_hook(context)
        pipeline._seedvr2_joint = wrapper
        import weakref

        pipeline._seedvr2_joint_finalizer = weakref.finalize(pipeline, wrapper.close)
        pipeline._metal_quantization_metadata = {
            "managed": True,
            "mode": "dynamic",
            "approved": True,
            "profile": bundle["quantization"],
            "joint_scope": scope,
        }
        pipeline.seedvr2_quantization = {
            "enabled": True,
            "reason": "approved-joint",
            "selected": bundle["quantization"]["selected"],
        }
    except BaseException:
        wrapper.close()
        raise


def close(pipeline):
    wrapper = getattr(pipeline, "_seedvr2_joint", None)
    if wrapper is not None:
        pipeline._seedvr2_joint_context_hook.remove()
        wrapper.close()
        pipeline._seedvr2_joint_finalizer.detach()
        del pipeline._seedvr2_joint
