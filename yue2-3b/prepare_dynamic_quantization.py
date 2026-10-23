#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Offline real YuE2 activation sensitivity search and full-pipeline approval."""

import argparse
import asyncio
import copy
import gc
import inspect
import json
import os
import shutil
import tempfile
import time
import wave
import weakref
from pathlib import Path
from types import SimpleNamespace

import psutil
from prepare import validate_model
from prepare_ane import AnePreparation, rpc, success, supervise, write_json
from prepare_quantization import (
    CAPTURE_STEPS,
    NarDownAdapter,
    QuantizationPreparation,
    candidate_names,
    compare_arrays,
    memory_snapshot,
    weight_fingerprint,
)
from yue2_dynamic_quantization import (
    digest,
    file_sha256,
    healthy_group,
    validate_completion,
    validate_search,
)


class DynamicPreparation(QuantizationPreparation):
    """Private isolated-worker RPCs; captures remain CPU tensors inside the worker."""

    def _dynamic_model(self):
        model = AnePreparation._idle_model(self)
        if getattr(model, "_pending", None) is not None:
            raise RuntimeError("pending model work prevents dynamic quantization changes")
        return model

    def dynamic_native_snapshot(self):
        model = self.get_model()
        return success(
            {
                "pending": len(self.model_runner.requests)
                + len(model._states)
                + int(getattr(model, "_pending", None) is not None),
                "quantized": {},
                "benchmark_held_float_bytes": 0,
                "memory": memory_snapshot(),
            }
        )

    def dynamic_install(self, requested, input_record):
        import torch

        model = self._dynamic_model()
        previous = healthy_group(model)
        names = candidate_names(len(model.nar_layers), requested)
        for name in names:
            index = int(name.split(".")[1])
            module = model.get_submodule(name)
            if (
                type(module) is not torch.nn.Linear
                or module.bias is not None
                or module.weight is not model.nar_fused[index][7]
                or module.weight.dtype != torch.bfloat16
                or module.weight.device.type != "mps"
            ):
                raise ValueError("candidate must be an original independent BF16 MPS down Linear")
        adapter = NarDownAdapter(model, names, previous.scope)
        adapter.revision = {
            "source_weights": input_record["source_weights"],
            "input_sha256": digest(input_record),
            "model_source_sha256": file_sha256(inspect.getfile(type(model))),
            "adapter_sha256": file_sha256(Path(__file__).with_name("yue2_dynamic_quantization.py")),
            "preparation_sha256": file_sha256(__file__),
        }
        # BF16 retention is checked by object identity through every full generation.
        adapter.retained = {}
        self._dynamic_model()
        model._metal_ffn = adapter
        previous.close()
        previous.weights = ()
        return success(
            {
                "candidates": names,
                "scope": adapter.scope,
                "capture_steps": list(CAPTURE_STEPS),
                **adapter.revision,
            }
        )

    def dynamic_search(self):
        import torch

        from vllm_omni_metal.quantization.engine import DynamicEngine
        from vllm_omni_metal.quantization.linear import build_profile

        model = self._dynamic_model()
        adapter = model._metal_ffn
        adapter.capture = False
        modules = {n: model.get_submodule(n) for n in adapter.names}
        with torch.inference_mode():
            result = DynamicEngine.search(modules, adapter.captures, scope=adapter.scope)
            calibration = {**result["profile"]["calibration"], **adapter.revision}
            result["profile"] = build_profile(
                modules,
                result["profile"]["selected"],
                scope=adapter.scope,
                calibration=calibration,
                precisions=result["precisions"],
            )
        adapter.captures.clear()
        adapter.capture_bytes = 0
        return success(result)

    def dynamic_apply(self, profile):
        import torch

        from vllm_omni_metal.quantization.engine import DynamicEngine

        model = self._dynamic_model()
        adapter = model._metal_ffn
        if adapter.quantized or adapter.baselines:
            raise RuntimeError("dynamic candidates already installed")
        if type(profile) is not dict or profile.get("selected") != adapter.names:
            raise ValueError("dynamic profile must cover exactly the captured NAR candidates")
        validate_search(profile)
        names = profile["selected"]
        for name in names:
            module = model.get_submodule(name)
            index = int(name.split(".")[1])
            if model.nar_fused[index][7] is not module.weight:
                raise ValueError("captured NAR fused down identity changed")
        baselines = {
            n: model.get_submodule(n) for n in names if profile["modules"][n]["bits"] != 16
        }
        retained = {
            n: (model.get_submodule(n), model.get_submodule(n).weight)
            for n in names
            if profile["modules"][n]["bits"] == 16
        }
        with torch.inference_mode():
            replacements = DynamicEngine.apply(
                model, profile, scope=adapter.scope, require_approved=False
            )
        adapter.baselines = baselines
        adapter.quantized = {n: replacements[n] for n in baselines}
        adapter.retained = retained
        adapter.float_references = [
            weakref.ref(value) for module in baselines.values() for value in (module, module.weight)
        ]
        for name in baselines:
            index = int(name.split(".")[1])
            model.nar_fused[index] = (*model.nar_fused[index][:7], None)
        return success(
            {
                "selected": names,
                "quantized": list(adapter.quantized),
                "benchmark_retains_float": bool(baselines),
            }
        )

    def quant_mode(self, mode):
        model = self._dynamic_model()
        adapter = model._metal_ffn
        if mode not in ("float", "dynamic"):
            raise ValueError("invalid benchmark route")
        if mode == "float" and adapter.quantized and not adapter.baselines:
            raise RuntimeError("floating references already released")
        adapter.mode = mode
        adapter.metrics.clear()
        for module in adapter.quantized.values():
            module.metrics.update(calls=0, rows=0)
        return success({"mode": mode})

    def quant_snapshot(self):
        result = super().quant_snapshot()["result"]
        model = self.get_model()
        adapter = model._metal_ffn
        for name, module in adapter.quantized.items():
            result["quantized"][name]["bits"] = module.bits
        result["retained_identity"] = all(
            model.get_submodule(n) is module and module.weight is weight
            for n, (module, weight) in adapter.retained.items()
        )
        result["float_weakrefs_released"] = all(
            ref() is None for ref in getattr(adapter, "float_references", ())
        )
        return success(result)

    def dynamic_approval_candidate(self, profile, completion):
        """Prepare metadata while originals exist; publish only after release and shutdown.

        The release assertion contains the expected hash, not invented timing or RSS.
        The coordinator must observe that exact assertion after removing every baseline.
        """
        import torch

        from vllm_omni_metal.quantization.linear import build_profile, validate_profile

        model = self._dynamic_model()
        adapter = model._metal_ffn
        validate_completion(completion, selected=list(adapter.quantized))
        originals = {
            n: adapter.baselines[n] if n in adapter.baselines else model.get_submodule(n)
            for n in profile["selected"]
        }
        with torch.inference_mode():
            validate_profile(profile, originals, scope=adapter.scope)
            approved = build_profile(
                originals,
                profile["selected"],
                scope=adapter.scope,
                calibration={
                    **profile["calibration"],
                    "approved": True,
                    "full_pipeline": completion,
                },
                precisions=profile["calibration"]["precisions"],
            )
        with torch.inference_mode():
            validate_profile(approved, originals, scope=adapter.scope)
        return success({"profile": approved})

    def quant_release_baseline(self):
        import mlx.core as mx
        import torch

        model = self._dynamic_model()
        adapter = model._metal_ffn
        if adapter.mode != "dynamic":
            raise RuntimeError("release requires dynamic route")
        released = sum(
            m.weight.numel() * m.weight.element_size() for m in adapter.baselines.values()
        )
        adapter.baselines.clear()
        gc.collect()
        torch.mps.empty_cache()
        mx.clear_cache()
        if not all(ref() is None for ref in getattr(adapter, "float_references", ())):
            raise RuntimeError("selected floating modules/weights retain unexpected aliases")
        result = self.quant_snapshot()["result"]
        result.update(
            released_benchmark_float_bytes=released,
            independent_process=False,
            memory_claim="same-worker observation, not independent cold-start peak RSS",
        )
        return success(result)


def summarize(records, comparisons, local_pass):
    """No speed or approval claim is made for a BF16-only search result."""
    selected = list(records[2]["quantized"])
    dynamic = next(r for r in records if r["mode"] == "dynamic" and not r["warmup"])
    completion = {
        "approved": True,
        "local_pass": local_pass,
        "records": records,
        "cross_route_audio": comparisons,
        "release": {
            "kind": "expected-post-release-assertion",
            "audio_sha256": dynamic["audio_sha256"],
            "pending": 0,
            "benchmark_held_float_bytes": 0,
            "float_weakrefs_released": True,
            "reproduced": True,
        },
        "publication_requirements": {
            "engine_shutdown": True,
            "supervisor_success": True,
            "release_assertion_verified": True,
        },
        "full_song_quality_validated": False,
        "frame_budget_limited": True,
        "production_net_memory_reduction_claimed": False,
    }
    if not selected:
        completion.update(approved=False, rejection_reason="no_quantized_candidate")
        return completion
    try:
        validate_completion(completion, selected=selected)
    except ValueError as error:
        completion.update(approved=False, rejection_reason=str(error))
    return completion


async def worker(args):
    from importlib.metadata import version

    import numpy as np
    import torch
    import yaml
    from vllm_omni.entrypoints.async_omni import AsyncOmni
    from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest
    from vllm_omni.entrypoints.openai.tts_adapters.yue2 import Yue2Adapter

    if "yue2mps2" not in version("vllm-omni"):
        raise RuntimeError(
            "requires existing patched YuE2 overlay; never installs weights or dependencies"
        )
    output = Path(args.output)
    deploy = yaml.safe_load(Path(args.deploy_config).read_text())
    if len(deploy["stages"]) != 1 or deploy.get("pipeline") != "yue2":
        raise ValueError("native single-stage YuE2 topology required")
    deploy["stages"][0]["worker_extension_cls"] = "prepare_dynamic_quantization.DynamicPreparation"
    deploy_path = output / "deploy.yaml"
    deploy_path.write_text(yaml.safe_dump(deploy))
    engine, records, arrays = None, [], {}
    installed = False
    pending_profile = None

    async def call(method, *values):
        return await rpc(engine, output, method, *values)

    async def idle():
        for _ in range(50):
            status = await call("quant_snapshot" if installed else "dynamic_native_snapshot")
            if status["pending"] == 0:
                return status
            await asyncio.sleep(0.1)
        raise RuntimeError("pending requests did not drain")

    def array(value):
        if isinstance(value, (list, tuple)):
            if len(value) != 1:
                raise ValueError("expected one audio result")
            return array(value[0])
        return (
            value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
        )

    try:
        sources = weight_fingerprint(args.model, args.vae)
        write_json(output / "source-weights.json", sources)
        engine = AsyncOmni(
            model=args.model,
            deploy_config=str(deploy_path),
            stage_init_timeout=360,
            init_timeout=420,
            log_stats=False,
            output_modalities=["audio"],
        )
        adapter = Yue2Adapter(SimpleNamespace(engine_client=engine))
        request = OpenAICreateSpeechRequest(
            model="yue2-dynamic-calibration",
            input="[Verse]\n今晚星光照亮街道\n我们一起追逐微笑",
            instructions="Mandarin pop, warm lead vocal, piano and acoustic guitar, at 100 BPM",
            extra_params={"cot": "off"},
            response_format="wav",
            stream=False,
            seed=42,
            max_new_tokens=512,
        )
        if adapter.validate(request) is not None:
            raise ValueError("native adapter rejected calibration request")
        prepared = await adapter.build(
            request, copy.deepcopy(engine.default_sampling_params_list), False
        )
        params = adapter.apply_sampling_overrides(
            copy.deepcopy(engine.default_sampling_params_list),
            request,
            prepared.prompt,
            request_id="dynamic-prepare",
        )
        input_record = {
            "frames": 512,
            "seed": 42,
            "lyrics": request.input,
            "style": request.instructions,
            "prompt": prepared.prompt,
            "sampling": {
                "temperature": params[0].temperature,
                "top_p": params[0].top_p,
                "top_k": params[0].top_k,
                "max_tokens": params[0].max_tokens,
            },
            "extra_args": params[0].extra_args,
            "source_weights": sources,
        }
        write_json(output / "input.json", input_record)

        async def generate(mode, *, warmup=False, capture=False, released=False):
            await idle()
            if installed and not capture:
                await call("quant_mode", mode)
            index = len(records)
            write_json(
                output / "progress.json",
                {"request_started": time.time(), "index": index, "mode": mode},
            )
            received = []
            started = time.perf_counter()
            async with asyncio.timeout(180):
                async for result in engine.generate(
                    copy.deepcopy(prepared.prompt),
                    request_id=f"dynamic-{index}",
                    sampling_params_list=copy.deepcopy(params),
                    output_modalities=["audio"],
                ):
                    if (
                        result.multimodal_output
                        and result.multimodal_output.get("audio") is not None
                    ):
                        received.append(result.multimodal_output)
            elapsed = time.perf_counter() - started
            write_json(output / "progress.json", {})
            status = await idle()
            if len(received) != 1:
                raise ValueError("missing unique terminal audio")
            audio = array(received[0]["audio"]).squeeze()
            sr = int(array(received[0].get("sr", 48000)).reshape(-1)[0])
            if audio.ndim != 2 or 2 not in audio.shape:
                raise ValueError("expected stereo audio")
            if audio.shape[0] == 2:
                audio = audio.T
            audio = np.ascontiguousarray(audio)
            if (
                sr != 48000
                or not np.isfinite(audio).all()
                or audio.std() <= 1e-6
                or len(audio) < 5 * sr
                or np.abs(audio).max() > 1
            ):
                raise ValueError("audio format/finite/non-silent/duration/amplitude gate failed")
            if status.get("retained_identity", True) is not True:
                raise ValueError("BF16 identity changed")
            import hashlib

            stem = f"{index}-{mode}"
            np.save(output / f"{stem}.npy", audio, allow_pickle=False)
            pcm = np.rint(audio * 32767).astype("<i2")
            with wave.open(str(output / f"{stem}.wav"), "wb") as wav:
                wav.setnchannels(2)
                wav.setsampwidth(2)
                wav.setframerate(sr)
                wav.writeframes(pcm.tobytes())
            record = {
                "mode": mode,
                "warmup": warmup,
                "capture": capture,
                "baseline_released": released,
                "seconds": elapsed,
                "shape": list(audio.shape),
                "dtype": str(audio.dtype),
                "audio_seconds": len(audio) / sr,
                "rtf": elapsed / (len(audio) / sr),
                "sample_rate": sr,
                "frames": 512,
                "seed": 42,
                "audio_sha256": hashlib.sha256(audio.tobytes()).hexdigest(),
                "npy": f"{stem}.npy",
                "npy_sha256": file_sha256(output / f"{stem}.npy"),
                "wav": f"{stem}.wav",
                "wav_sha256": file_sha256(output / f"{stem}.wav"),
                **status,
            }
            records.append(record)
            write_json(output / "results.json", records)
            arrays.setdefault(mode, []).append(audio)
            return record

        await generate("native", warmup=True)
        write_json(
            output / "adapter.json",
            await call("dynamic_install", args.candidate_layers, input_record),
        )
        installed = True
        write_json(output / "environment.json", await call("quant_environment"))
        await generate("float", warmup=True, capture=True)
        native_comparison = compare_arrays(arrays["native"][0], arrays["float"][0])
        write_json(output / "native-reference-comparison.json", native_comparison)
        if not native_comparison["bitwise_identical"]:
            raise ValueError(
                "native and floating-hook reference differ; no sensitivity search permitted"
            )
        audit = await call("dynamic_search")
        profile = audit["profile"]
        write_json(output / "calibration.json", audit)
        write_json(output / "candidate-profile.json", profile)
        write_json(output / "applied.json", await call("dynamic_apply", profile))
        await generate("dynamic", warmup=True)
        for mode in ("float", "dynamic", "dynamic", "float"):
            await generate(mode)
        comparisons = [
            compare_arrays(a, b)
            for a, b in zip(arrays["float"][1:], arrays["dynamic"][1:], strict=True)
        ]
        completion = summarize(copy.deepcopy(records), comparisons, audit["local_pass"])
        if completion["approved"]:
            pending_profile = (await call("dynamic_approval_candidate", profile, completion))[
                "profile"
            ]
        write_json(
            output / "comparison.json",
            {
                "local_and_combined_candidate_pass": completion["approved"],
                "records": completion["records"],
                "cross_route_audio": comparisons,
                "release_assertion": completion["release"],
                "release_observed": False,
                "engine_closed": False,
                "supervisor_finished": False,
            },
        )
        await call("quant_mode", "dynamic")
        released = await call("quant_release_baseline")
        write_json(output / "quantized-only-memory.json", released)
        actual = await generate("dynamic", warmup=True, released=True)
        expected = completion["release"]
        reproduced = (
            actual["audio_sha256"] == expected["audio_sha256"]
            and actual["pending"] == 0
            and actual["benchmark_held_float_bytes"] == 0
            and actual["float_weakrefs_released"] is True
        )
        if not reproduced:
            completion.update(
                approved=False, rejection_reason="post-release output did not reproduce"
            )
            pending_profile = None
        completion["release_observation"] = {
            "kind": "actual-post-release-observation",
            "audio_sha256": actual["audio_sha256"],
            "pending": actual["pending"],
            "benchmark_held_float_bytes": actual["benchmark_held_float_bytes"],
            "float_weakrefs_released": actual["float_weakrefs_released"],
            "reproduced": reproduced,
        }
    finally:
        if engine is not None:
            engine.shutdown()
    completion["engine_closed"] = True
    completion["supervisor_finished"] = False
    write_json(output / "completed.json", completion)
    # A worker-local candidate is not a production artifact until supervisor success.
    if pending_profile is not None:
        write_json(output / "pending-approved-profile.json", pending_profile)


def publish_after_supervision(directory):
    """Publish only observed release/shutdown and successful supervisor completion."""
    directory = Path(directory)
    supervisor = json.loads((directory / "supervisor.json").read_text())
    if supervisor.get("returncode") != 0 or supervisor.get("stop_reason") is not None:
        raise RuntimeError("supervisor did not complete successfully")
    completed_path = directory / "completed.json"
    completion = json.loads(completed_path.read_text())
    if completion.get("engine_closed") is not True:
        raise RuntimeError("engine shutdown was not observed")
    completion.update(supervisor_finished=True, supervisor=supervisor)
    write_json(completed_path, completion)
    pending = directory / "pending-approved-profile.json"
    if not pending.exists():
        return
    profile = json.loads(pending.read_text())
    measured = profile["calibration"]["full_pipeline"]
    if not completion["approved"] or measured["records"] != completion["records"]:
        raise ValueError("pending approval differs from complete observed A/B")
    expected, actual = measured["release"], completion["release_observation"]
    if actual.get("kind") != "actual-post-release-observation" or any(
        actual.get(key) != expected[key]
        for key in (
            "audio_sha256",
            "pending",
            "benchmark_held_float_bytes",
            "float_weakrefs_released",
            "reproduced",
        )
    ):
        raise ValueError("actual release differs from the content-bound expected assertion")
    quantized = [name for name in profile["selected"] if profile["modules"][name]["bits"] != 16]
    validate_completion(measured, selected=quantized)
    pending.chmod(0o600)
    pending.replace(directory / "approved-profile.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    cache = Path.home() / ".cache/modelscope/hub/models/m-a-p"
    parser.add_argument("--model", default=os.environ.get("YUE2_MODEL_DIR", str(cache / "YuE2-3B")))
    parser.add_argument("--vae", default=os.environ.get("YUE2_VAE_DIR", str(cache / "YuE2-Vae")))
    parser.add_argument("--frames", type=int, default=512)
    parser.add_argument("--candidate-layers", default="auto")
    parser.add_argument(
        "--deploy-config", default=str(Path(__file__).with_name("metal-deploy.yaml"))
    )
    parser.add_argument("--output")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--gpu-only", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.frames != 512:
        parser.error("calibration is fixed at 512 frames and seed42")
    if os.environ.get("VLLM_ALLOW_INSECURE_SERIALIZATION", "0") not in ("", "0"):
        parser.error("unsafe serialization forbidden")
    os.environ.update(
        VLLM_OMNI_METAL_ANE="off",
        VLLM_ALLOW_INSECURE_SERIALIZATION="0",
        YUE2_METAL_QUANTIZATION="off",
        VLLM_OMNI_METAL_QUANTIZATION="off",
    )
    args.model = str(validate_model(args.model, "yue2"))
    args.vae = str(validate_model(args.vae, "yue2_vae"))
    os.environ["YUE2_VAE"] = args.vae
    if args.worker:
        asyncio.run(worker(args))
        return
    directory = (
        Path(args.output).expanduser().resolve()
        if args.output
        else Path(tempfile.mkdtemp(prefix="yue2-dynamic-"))
    )
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError("evidence directory must be empty")
    directory.mkdir(parents=True, exist_ok=True)
    available, disk = psutil.virtual_memory().available, shutil.disk_usage(directory).free
    write_json(
        directory / "preflight.json",
        {
            "available_bytes": available,
            "required_available_bytes": 16 * 1024**3,
            "free_disk_bytes": disk,
            "required_disk_bytes": 2 * 1024**3,
            "model_process_started": False,
            "startup_gate_pass": available >= 16 * 1024**3 and disk >= 2 * 1024**3,
            "model": args.model,
            "vae": args.vae,
        },
    )
    if available < 16 * 1024**3 or disk < 2 * 1024**3:
        raise MemoryError(f"preflight resource gate refused model startup; evidence: {directory}")
    args.gpu_only, args.attention_ab = True, False
    supervise(
        args,
        script=Path(__file__).resolve(),
        directory=directory,
        extra_args=("--candidate-layers", args.candidate_layers),
        label="DynamicEngine real sensitivity and A/B",
    )
    publish_after_supervision(directory)


if __name__ == "__main__":
    main()
