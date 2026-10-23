# SPDX-License-Identifier: Apache-2.0
"""Private, load-time Higgs Talker gate-up ANE + down DynamicEngine contract."""

import copy
import hashlib
import inspect
import json
import os
import weakref
from pathlib import Path

import torch
from higgs_metal_quantization import TEXTS, digest_file, generation_config_revision

LAYERS = (0, 11, 23, 35)
SELECTED = [f"model.layers.{i}.mlp.down_proj" for i in LAYERS]
POLICY = {
    "texts": list(TEXTS),
    "seed": 42,
    "max_new_tokens": 512,
    "sample_rate": 24000,
    "layers": list(LAYERS),
    "capture_steps": [0, 1, 2, 3],
    "relative_l2_limit": 0.01,
    "scaled_max_abs_limit": 0.01,
    "minimum_speedup": 0.03,
    "ane_chunks": 1,
    "speed_reference": "same-dynamic-down-native-gate-up",
    "code2wav": "source-native-identity",
    "selection": SELECTED,
}


def scope(model):
    from importlib.metadata import version
    from importlib.util import find_spec

    from vllm.model_executor.models import qwen3
    from vllm_omni.entrypoints.openai.tts_adapters import higgs_audio_v3 as tts_adapter
    from vllm_omni.model_executor.models.higgs_audio_v2 import higgs_audio_decoder
    from vllm_omni.model_executor.models.higgs_audio_v3 import higgs_audio_v3_talker
    from vllm_omni.model_executor.stage_input_processors import higgs_audio_v3 as processor

    from vllm_omni_metal.ane import projection
    from vllm_omni_metal.native import runner
    from vllm_omni_metal.quantization import adapter, engine, linear

    directory = Path(model.vllm_config.model_config.model).resolve(strict=True)
    paths = sorted(directory.glob("*.safetensors"))
    if not paths:
        raise ValueError("joint requires complete local source safetensors")
    files = [directory / n for n in ("config.json", "tokenizer.json", "tokenizer_config.json")]
    index = directory / "model.safetensors.index.json"
    if index.exists():
        files.append(index)
    sources = {
        Path(__file__),
        Path(find_spec("higgs_joint_preparation").origin),
        Path(find_spec("higgs_quantization_preparation").origin),
        Path(find_spec("higgs_metal_quantization").origin),
    }
    for module in (
        qwen3,
        higgs_audio_v3_talker,
        higgs_audio_decoder,
        processor,
        tts_adapter,
        projection,
        adapter,
        engine,
        linear,
        runner,
    ):
        sources.update(Path(inspect.getfile(module)).parent.glob("*.py"))
    record = {
        "policy": POLICY,
        "directory": str(directory),
        "weights": {p.name: digest_file(p) for p in [*files, *paths]},
        "generation_config": generation_config_revision(directory),
        "sources": {str(p): digest_file(p) for p in sorted(sources)},
        "deploy": digest_file(os.environ["HIGGS_METAL_QUANTIZATION_DEPLOY_CONFIG"]),
        "omni": version("vllm-omni"),
        "dtype": str(model.model.layers[0].mlp.gate_up_proj.weight.dtype),
    }
    return (
        "higgs-talker-joint-v1:"
        + hashlib.sha256(
            json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )


def projections(model):
    from vllm_omni_metal.quantization.adapter import _view

    if len(model.model.layers) != 36:
        raise ValueError("joint requires the complete 36-layer Qwen3 Talker")
    result = {}
    for i in LAYERS:
        mlp = model.model.layers[i].mlp
        gate = mlp.gate_up_proj
        view = _view(gate)
        down = _view(mlp.down_proj)
        if (
            tuple(view.weight.shape) != (19456, 2560)
            or tuple(down.weight.shape) != (2560, 9728)
            or view.weight.dtype != torch.bfloat16
            or down.weight.dtype != torch.bfloat16
        ):
            raise ValueError(
                "joint requires native BF16 packed gate_up19456x2560 and down2560x9728"
            )
        result[i] = gate
    return result


class GateProjection(torch.nn.Module):
    """Keep the packed projection and its vLLM return/skip-bias ABI intact."""

    def __init__(self, source, *, scope, benchmark=False, staged_policy=None):
        super().__init__()
        from vllm_omni_metal.ane.projection import DenseProjection

        self.source = source
        self.projection = DenseProjection(
            source.weight,
            None if source.skip_bias_add else source.bias,
            scope=scope,
            precision="fp16",
            _benchmark=benchmark,
            _staged_policy=staged_policy,
        )
        if self.projection._bridge is None:
            self.projection.close()
            raise RuntimeError("joint requires a prepared, approved, real ANE projection")
        if benchmark:
            self.projection.set_benchmark_chunks(1)
        self._resource_finalizer = weakref.finalize(self, self.projection.close)

    def forward(self, x):
        def native(value):
            output = self.source(value)
            return output[0] if self.source.return_bias else output

        p = self.projection
        rows = x.numel() // x.shape[-1]
        if not p._benchmark and rows > 256 and (rows, None) not in p.profiles:
            raise RuntimeError("joint prefill shape has no complete-pipeline approval")
        result = p(x, native=native)
        bias = self.source.bias if self.source.skip_bias_add else None
        return (result, bias) if self.source.return_bias else result


class CountProjection(torch.nn.Module):
    def __init__(self, source):
        super().__init__()
        self.source = source
        self.metrics = {"calls": 0, "rows": 0}

    def forward(self, x):
        self.metrics["calls"] += 1
        self.metrics["rows"] += x.numel() // x.shape[-1]
        return self.source(x)


def validate_codec_identity(record):
    if (
        type(record) is not dict
        or set(record) != {"dtype", "identity"}
        or record["identity"] is not True
        or record["dtype"] not in ("torch.float16", "torch.bfloat16", "torch.float32")
    ):
        raise ValueError(f"actual untouched Code2Wav native identity is required: {record!r}")


def approved(profile, *, require_shutdown=True):
    calibration = profile.get("calibration", {})
    joint = calibration.get("joint", {})
    codec = joint.get("codec_identity")
    if (
        profile.get("selected") != SELECTED
        or calibration.get("engine") != "DynamicEngine"
        or calibration.get("approved") is not True
        or calibration.get("local_pass") is not True
        or joint.get("policy") != POLICY
        or joint.get("float_released") is not True
        or joint.get("ane_closed") is not True
        or type(codec) is not dict
        or codec.get("identity") is not True
        or not any(profile["modules"][n]["bits"] in (4, 8) for n in SELECTED)
        or set(joint.get("projections", {})) != {str(i) for i in LAYERS}
    ):
        raise ValueError(
            "missing fresh complete Higgs joint approval; independent approvals cannot be composed"
        )
    validate_codec_identity(codec)
    if require_shutdown and (
        joint.get("engine_shutdown") is not True or joint.get("supervisor_success") is not True
    ):
        raise ValueError(
            "joint approval requires actual engine shutdown and successful resource supervisor"
        )
    from vllm_omni_metal.quantization.linear import _encoded

    candidate = {k: v for k, v in profile.items() if k != "sha256"}
    candidate["calibration"] = {k: v for k, v in calibration.items() if k != "joint"}
    candidate["calibration"]["approved"] = False
    if hashlib.sha256(_encoded(candidate)).hexdigest() != joint.get(
        "quantization_candidate_sha256"
    ):
        raise ValueError("joint approval does not bind this exact quantization search profile")
    for record in joint["projections"].values():
        if not record.get("identity") or not record.get("profiles"):
            raise ValueError("missing shape-bound actual ANE profiles")
        from vllm_omni_metal.ane.cache import validate_profiles
        from vllm_omni_metal.ane.projection import _passing

        validate_profiles(record["profiles"])
        for evidence in record["profiles"].values():
            if evidence.get("chunks") != 1 or not evidence.get("numerical_check"):
                raise ValueError("zero ANE split cannot approve joint execution")
            if not _passing(evidence["numerical_check"]):
                raise ValueError("joint ANE local full-output dual 1% gate failed")
    samples = joint.get("samples", [])
    if len(samples) != len(TEXTS):
        raise ValueError("joint needs complete bilingual output evidence")
    for text, sample in zip(TEXTS, samples, strict=True):
        if (
            sample.get("text") != text
            or any(
                sample.get(k) is not True
                for k in (
                    "seed_repeatable",
                    "natural_stop",
                    "released_matches",
                    "same_shape",
                    "same_dtype",
                )
            )
            or any(
                type(sample.get(k)) is not int or sample[k] <= 0
                for k in ("candidate_calls", "candidate_rows", "ane_chunks", "quantized_calls")
            )
            or any(
                type(sample.get(k)) not in (int, float) or not 0 <= sample[k] <= 0.01
                for k in ("relative_l2", "scaled_max_abs", "ane_relative_l2", "ane_scaled_max_abs")
            )
        ):
            raise ValueError("joint complete TTS quality/hit/repeat/release gate failed")
        for key in ("baseline_sha256", "candidate_sha256", "ane_reference_sha256"):
            value = sample.get(key)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(c not in "0123456789abcdef" for c in value)
            ):
                raise ValueError("joint requires complete raw floating output hashes")


def projection_scope(selected_scope, profile, layer):
    candidate = (
        profile["calibration"]
        .get("joint", {})
        .get("quantization_candidate_sha256", profile["sha256"])
    )
    return f"{selected_scope}:dynamic:{candidate}:gate:{layer}"


def install(model, profile, selected_scope, *, benchmark=False):
    from vllm_omni_metal.quantization.adapter import DenseModelAdapter

    if not benchmark:
        approved(profile)
    gates = projections(model)
    wrappers = {}
    original_down = {n: model.get_submodule(n) for n in SELECTED}
    try:
        for i, source in gates.items():
            wrappers[i] = GateProjection(
                source, scope=projection_scope(selected_scope, profile, i), benchmark=benchmark
            )
            if not benchmark:
                bound = profile["calibration"]["joint"]["projections"][str(i)]
                p = wrappers[i].projection
                from vllm_omni_metal.ane import cache
                from vllm_omni_metal.ane.projection import _policy

                policy = _policy(cache.cache_root() / f"{p.identity}.json")
                if p.identity != bound["identity"] or policy["profiles"] != bound["profiles"]:
                    raise ValueError("ANE policy differs from this complete joint approval")
        result = DenseModelAdapter(model).apply(
            profile, scope=selected_scope, require_approved=not benchmark
        )
        for i in LAYERS:
            mlp = model.model.layers[i].mlp
            mlp.gate_up_proj = wrappers[i]
            mlp.down_proj = CountProjection(result[f"model.layers.{i}.mlp.down_proj"])
    except BaseException:
        for i, wrapper in wrappers.items():
            wrapper.projection.close()
            model.model.layers[i].mlp.gate_up_proj = gates[i]
        for name, source in original_down.items():
            parent, _, child = name.rpartition(".")
            model.get_submodule(parent)._modules[child] = source
        raise
    return wrappers


def finish_load(model):
    mode = os.environ.get("HIGGS_METAL_JOINT", "off")
    model._metal_quantization_metadata = {"managed": True, "mode": "off", "approved": False}
    if mode == "off":
        return
    if mode != "joint":
        raise ValueError("HIGGS_METAL_JOINT only accepts off/joint")
    if os.environ.get("HIGGS_METAL_QUANTIZATION", "off") != "off":
        raise ValueError("joint owns Talker down; separate codec quantization must be off")
    if os.environ.get("VLLM_OMNI_METAL_ANE", "auto") != "auto":
        raise ValueError("joint requires actual ANE auto execution")
    from vllm_omni_metal.quantization.runtime import load_profile

    with torch.inference_mode():
        selected_scope = scope(model)
        profile = load_profile(os.environ["HIGGS_METAL_JOINT_PROFILE"], scope=selected_scope)
        model._higgs_joint_gates = install(model, profile, selected_scope)
    model._metal_quantization_metadata = {
        "managed": True,
        "mode": "dynamic",
        "approved": True,
        "profile": profile,
        "scope": selected_scope,
    }


class JointPreparation:
    """Safe stage-0 RPC extension. No alternate model loader or registry entry."""

    def _idle(self):
        if self.model_runner.requests:
            raise RuntimeError("joint transitions require no pending requests")
        return self.get_model()

    def higgs_joint_prepare(self):
        model = self._idle()
        if os.environ.get("HIGGS_METAL_JOINT", "off") != "off" or hasattr(model, "_joint"):
            raise RuntimeError("joint calibration requires a fresh disabled worker")
        from vllm_omni_metal.quantization.adapter import DenseModelAdapter

        projections(model)
        selected_scope = scope(model)
        capture = DenseModelAdapter(model).capture(
            SELECTED, steps=(0, 1, 2, 3), max_bytes=128 * 1024**2
        )
        capture.__enter__()
        model._joint = {
            "scope": selected_scope,
            "capture": capture,
            "float": {i: model.model.layers[i].mlp.down_proj for i in LAYERS},
            "gate": {i: model.model.layers[i].mlp.gate_up_proj for i in LAYERS},
        }
        return {"supported": True, "result": {"capture_steps": [0, 1, 2, 3]}}

    def higgs_joint_search(self):
        from vllm_omni_metal.ane.projection import prepare_projection
        from vllm_omni_metal.quantization.adapter import DenseModelAdapter

        model = self._idle()
        state = model._joint
        capture = state.pop("capture")
        capture.__exit__(None, None, None)
        with torch.inference_mode():
            result = DenseModelAdapter(model).search(capture.captures, scope=state["scope"])
            profile = result["profile"]
            if not any(profile["modules"][n]["bits"] in (4, 8) for n in SELECTED):
                raise ValueError(
                    "all down projections retained identity; no real joint quantization"
                )
            for i, gate in state["gate"].items():
                prepare_projection(
                    gate.weight,
                    None if gate.skip_bias_add else gate.bias,
                    scope=projection_scope(state["scope"], profile, i),
                )
            state["refs"] = [
                (weakref.ref(m), weakref.ref(m.weight)) for m in state["float"].values()
            ]
            for i in LAYERS:
                n = f"model.layers.{i}.mlp.down_proj"
                if profile["modules"][n]["bits"] == 16:
                    model.model.layers[i].mlp.down_proj = copy.deepcopy(state["float"][i])
            state["wrappers"] = install(model, profile, state["scope"], benchmark=True)
            state["dynamic"] = {i: model.model.layers[i].mlp.down_proj for i in LAYERS}
            state["profile"] = profile
        del capture
        self.higgs_joint_mode("float")
        return {"supported": True, "result": result}

    def higgs_joint_mode(self, mode):
        model = self._idle()
        state = model._joint
        route = "float" if mode == "float" else "dynamic"
        if mode not in ("float", "quantized", "dynamic") or route not in state:
            raise ValueError("unknown joint route or released baseline")
        for i in LAYERS:
            mlp = model.model.layers[i].mlp
            mlp.gate_up_proj = state["wrappers"][i] if mode == "dynamic" else state["gate"][i]
            mlp.down_proj = state[route][i]
            if "dynamic" in state:
                state["dynamic"][i].metrics.update(calls=0, rows=0)
                wrapper = state["wrappers"][i]
                wrapper._ane_start = wrapper.projection.metrics["ane_chunks"]
        return {"supported": True, "result": {"mode": mode}}

    def higgs_joint_snapshot(self):
        model = self.get_model()
        state = model._joint
        metrics = {"calls": 0, "rows": 0, "ane_chunks": 0, "quantized_calls": 0}
        for i, down in state.get("dynamic", {}).items():
            metrics["calls"] += down.metrics["calls"]
            metrics["rows"] += down.metrics["rows"]
            p = state["wrappers"][i].projection
            metrics["ane_chunks"] += p.metrics["ane_chunks"] - getattr(
                state["wrappers"][i], "_ane_start", 0
            )
            if state["profile"]["modules"][f"model.layers.{i}.mlp.down_proj"]["bits"] in (4, 8):
                metrics["quantized_calls"] += down.metrics["calls"]
        return {
            "supported": True,
            "result": {"pending": len(self.model_runner.requests), "metrics": metrics},
        }

    def higgs_joint_audited(self):
        for wrapper in self._idle()._joint["wrappers"].values():
            wrapper.projection.finish_audit()
        return {"supported": True, "result": {"audited": True}}

    def higgs_joint_stage(self, baseline, candidate):
        from vllm_omni_metal.ane import cache
        from vllm_omni_metal.ane.projection import publish_projection_profile

        model = self._idle()
        state = model._joint
        policies = {}
        replacements = {}
        try:
            for i, wrapper in state["wrappers"].items():
                p = wrapper.projection
                records = {}
                for rows, _chunks, context in p.checks:
                    record = publish_projection_profile(
                        p, rows, baseline, {1: candidate}, context=context, publish=False
                    )
                    if record["chunks"] != 1:
                        raise ValueError("whole bilingual TTS failed fixed 3% speed gate")
                    records[cache.profile_label(rows, context)] = record
                if not records:
                    raise ValueError("zero actual ANE hits; short inputs cannot approve joint mode")
                staged = {
                    "schema": 1,
                    "group": p.identity,
                    "device": p._bridge.device,
                    "kind": "linear-bias-v1",
                    "profiles": records,
                }
                policies[str(i)] = {
                    "identity": p.identity,
                    "profiles": records,
                    "staged_policy": staged,
                }
                p.close()
                replacements[i] = GateProjection(
                    state["gate"][i], scope=p.scope, staged_policy=staged
                )
        except BaseException:
            for wrapper in replacements.values():
                wrapper.projection.close()
            raise
        state["wrappers"] = replacements
        state["policies"] = policies
        self.higgs_joint_mode("dynamic")
        return {"supported": True, "result": {"policies": policies, "immutable_replay": True}}

    def higgs_joint_release(self):
        import gc

        model = self._idle()
        state = model._joint
        self.higgs_joint_mode("dynamic")
        del state["float"]
        gc.collect()
        torch.mps.empty_cache()
        if any(module() is not None or weight() is not None for module, weight in state["refs"]):
            raise RuntimeError("floating down baseline/weight still referenced")
        state["released"] = True
        return {"supported": True, "result": {"float_baseline_released": True}}

    def higgs_joint_approve(self, samples, codec_evidence):
        from vllm_omni_metal.quantization.linear import _encoded

        state = self._idle()._joint
        if not state.get("released"):
            raise RuntimeError("release and real output reproduction are required")
        validate_codec_identity(codec_evidence)
        if not state.get("policies"):
            raise ValueError("immutable staged-policy replay is required")
        for wrapper in state["wrappers"].values():
            wrapper.projection.close()
        profile = copy.deepcopy(state["profile"])
        profile["calibration"].update(
            approved=True,
            joint={
                "policy": POLICY,
                "samples": samples,
                "float_released": True,
                "ane_closed": all(w.projection._closed for w in state["wrappers"].values()),
                "codec_identity": copy.deepcopy(codec_evidence),
                "projections": state["policies"],
                "quantization_candidate_sha256": state["profile"]["sha256"],
            },
        )
        profile.pop("sha256")
        profile["sha256"] = hashlib.sha256(_encoded(profile)).hexdigest()
        approved(profile, require_shutdown=False)
        return {"supported": True, "result": {"profile": profile}}

    def higgs_joint_close(self):
        state = getattr(self.get_model(), "_joint", {})
        try:
            capture = state.pop("capture", None)
            if capture is not None:
                capture.close()
        finally:
            for wrapper in state.get("wrappers", {}).values():
                wrapper.projection.close()
        return {"supported": True, "result": {"closed": True}}
