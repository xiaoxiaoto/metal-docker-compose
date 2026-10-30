# SPDX-License-Identifier: Apache-2.0
"""Fresh joint approval, invoked by the packaged full-workload calibrator only."""

import hashlib


def calibrate(pipeline, profile, workloads, baseline, run, replacements, args, *, bound_scope):
    from vllm_omni_metal.ane.cache import profile_label, select_profile
    from vllm_omni_metal.ane.projection import prepare_projection, publish_projection_profile

    from . import joint
    from .calibration import compare, qualified

    selected = profile["selected"][0]
    if profile["modules"][selected]["bits"] not in (4, 8):
        raise ValueError("joint requires a real W4/W8 down projection, not identity")
    if not bound_scope or not bound_scope.startswith(joint.VERSION + ":"):
        raise ValueError("joint scope must bind the original full model before quantization")
    scope = bound_scope
    name = selected.replace("proj_out", "proj_in")
    parent = pipeline.get_submodule(name.rpartition(".")[0])
    original = parent.proj_in
    prepare_projection(original.weight, original.bias, scope=scope)
    wrapper = joint.JointProjection(original, scope=scope, benchmark=True)
    parent.proj_in = wrapper
    records = {}
    samples = []
    outputs = {}
    try:
        # Audit actual complete activations before freezing numerical diagnostics.
        for workload in workloads:
            wrapper.context = 1 if workload[0] == "image" else 5
            wrapper.projection.set_benchmark_chunks(1)
            run(workload, "joint-audit-warmup", replacements)
        wrapper.projection.finish_audit()
        for workload in workloads:
            kind, seed = workload[:2]
            context = 1 if kind == "image" else 5
            wrapper.context = context
            times = {0: [], 1: []}
            values = {0: [], 1: []}
            # Independent warmups, then full float-split/candidate/candidate/float-split.
            for chunks in (0, 1):
                wrapper.projection.set_benchmark_chunks(chunks)
                run(workload, f"joint-warmup-{chunks}", replacements)
            wrapper.records.clear()
            before = wrapper.projection.metrics["ane_chunks"]
            quant_calls = 0
            for chunks in (0, 1, 1, 0):
                wrapper.projection.set_benchmark_chunks(chunks)
                actual, calls = run(
                    workload, f"joint-abba-{chunks}-{len(times[chunks])}", replacements
                )
                times[chunks].append(run.last_seconds)
                values[chunks].append(actual)
                quant_calls += calls[selected]
            rows = sorted(r for r, c in wrapper.records if c == context)
            if not rows or any(
                select_profile(r, times[0], {1: times[1]})["chunks"] != 1 for r in rows
            ):
                raise ValueError(
                    "ANE did not achieve stable >=3% complete pipeline benefit over identical quant-only path"
                )
            actual = values[1][0]
            error = compare(baseline[kind, seed], actual)
            repeat = actual.tobytes() == values[1][1].tobytes()
            hits = wrapper.projection.metrics["ane_chunks"] - before
            if not qualified(error) or not repeat or hits <= 0 or quant_calls <= 0:
                raise ValueError("joint complete output/repeat/real execution gate rejected")
            # Both timing references must still reproduce the quant-only candidate.
            expected = next(
                s
                for s in profile["calibration"]["samples"]
                if (s["kind"], s["seed"]) == (kind, seed)
            )["candidate_sha256"]
            if any(hashlib.sha256(v.tobytes()).hexdigest() != expected for v in values[0]):
                raise ValueError("joint zero-split changed the quant-only reference")
            for r in rows:
                record = publish_projection_profile(
                    wrapper.projection, r, times[0], {1: times[1]}, context=context, publish=False
                )
                records[r, context] = {"rows": r, "context": context, "record": record}
            outputs[kind, seed] = hashlib.sha256(actual.tobytes()).hexdigest()
            samples.append(
                {
                    **error,
                    "kind": kind,
                    "seed": seed,
                    "rows": rows,
                    "baseline_seconds": times[0],
                    "candidate_seconds": times[1],
                    "repeat_identical": repeat,
                    "ane_chunks": hits,
                    "quant_calls": quant_calls,
                    "baseline_sha256": hashlib.sha256(baseline[kind, seed].tobytes()).hexdigest(),
                    "candidate_sha256": outputs[kind, seed],
                    "input_sha256": next(
                        s
                        for s in profile["calibration"]["samples"]
                        if (s["kind"], s["seed"]) == (kind, seed)
                    )["input_sha256"],
                }
            )
        identity = wrapper.projection.identity
        staged_policy = {
            "schema": 1,
            "group": identity,
            "device": wrapper.projection._bridge.device,
            "kind": "linear-bias-v1",
            "profiles": {profile_label(r, c): item["record"] for (r, c), item in records.items()},
        }
    finally:
        parent.proj_in = original
        wrapper.close()
    if not wrapper.projection._closed:
        raise ValueError("ANE calibration bridge did not close")
    # Recreate production (not benchmark) bridge, with no retained down baseline.
    production = joint.JointProjection(original, scope=scope, staged_policy=staged_policy)
    parent.proj_in = production
    try:
        for workload in reversed(workloads):
            production.context = 1 if workload[0] == "image" else 5
            actual, calls = run(workload, "joint-released-production", replacements)
            sample = next(s for s in samples if (s["kind"], s["seed"]) == workload[:2])
            sample["released_identical"] = (
                hashlib.sha256(actual.tobytes()).hexdigest() == outputs[workload[:2]]
            )
            if not sample["released_identical"] or calls[selected] <= 0:
                raise ValueError("production bridge release/repeat gate rejected")
    finally:
        parent.proj_in = original
        production.close()
    if not production.projection._closed:
        raise ValueError("ANE production bridge did not close")
    from vllm_omni.diffusion.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
    )

    destroy_model_parallel()
    destroy_distributed_environment()
    bundle = {
        "schema": joint.VERSION,
        "scope": scope,
        "quantization": profile,
        "projection": {
            "name": name,
            "identity": identity,
            "profiles": list(records.values()),
            "staged_policy": staged_policy,
        },
        "samples": samples,
        "lifecycle": {
            "float_down_released": True,
            "projection_closed": True,
            "engine": "stateless-DynamicEngine",
            "distributed_shutdown": True,
        },
    }
    bundle["sha256"] = joint.sha(bundle)
    joint.validate(bundle, scope=scope)
    return bundle
