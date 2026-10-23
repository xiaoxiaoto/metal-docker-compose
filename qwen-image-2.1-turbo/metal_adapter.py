# SPDX-License-Identifier: Apache-2.0
"""Private QwenImage operator binding; no model registry or sampling replacement."""

import hashlib
import json
import os
from pathlib import Path

import torch
from diffusers.models.normalization import RMSNorm

from vllm_omni_metal.diffusion import ops as diffusion_ops

from .joint_adapter import ProjectionGroup


def _enabled():
    mode = os.environ.get("QWEN_IMAGE_METAL_GPU", "off")
    if mode not in ("auto", "off"):
        raise ValueError("QWEN_IMAGE_METAL_GPU must be auto or off")
    return mode == "auto"


def silu_mul(gate, up, *, _validated=False):
    with diffusion_ops.use_optimizations(_enabled()):
        return diffusion_ops.silu_mul(gate, up, _validated=_validated)


def rotary(x, freqs):
    with diffusion_ops.use_optimizations(_enabled()):
        return diffusion_ops.rotary(x, freqs)


def mps_rotary(x, freqs):
    from vllm_omni_metal.ops.vpipe_kernels import supports_interleaved_rotary

    if not _enabled() or not supports_interleaved_rotary(x, freqs):
        return None
    return diffusion_ops.rotary(x, freqs)


class MetalRMSNorm(RMSNorm):
    def forward(self, hidden_states):
        if hidden_states.device.type != "mps":
            return super().forward(hidden_states)
        with diffusion_ops.use_optimizations(_enabled()):
            return diffusion_ops.rms_norm(hidden_states, self.weight, self.eps)


class QwenMetalAdapter:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.group = None
        self.indices = {}
        self.busy = False
        self.closed = False
        self.last_shapes = set()
        self.layers = []
        self.scope = self._scope()
        from .quantization_adapter import QwenDynamicAdapter

        self.quantization = QwenDynamicAdapter(self)
        selected = os.environ.get("QWEN_IMAGE_METAL_ANE_LAYERS", "")
        joint = self.quantization.joint_profile
        if joint is not None:
            layers = sorted(int(name.split(".")[1]) for name in joint["selected"])
            if selected and selected != ",".join(map(str, layers)):
                raise ValueError("joint ANE layers must match this complete approved profile")
            if not _enabled():
                raise ValueError("joint production requires the calibrated optimized GPU baseline")
            self.bind(layers, prepare=False)
            try:
                self.group.validate_records(joint["calibration"]["ane_profiles"])
            except BaseException:
                self.close()
                raise
        elif selected and self.quantization.active is not None:
            raise ValueError("不能拼接独立ANE与dynamic批准；请选择joint完整校准profile")
        if (
            selected
            and _enabled()
            and self.group is None
            and os.environ.get("VLLM_OMNI_METAL_ANE", "off") != "off"
        ):
            layers = (
                list(range(len(self.blocks)))
                if selected == "all"
                else [int(s) for s in selected.split(",")]
            )
            self.bind(layers, prepare=False)

    @property
    def blocks(self):
        return self.pipeline.transformer.transformer_blocks

    def _scope(self):
        digest = hashlib.sha256()
        for name in (
            "metal_adapter.py",
            "quantization_adapter.py",
            "joint_adapter.py",
            "prepare_quantization.py",
            "prepare_ane.py",
            "prepare.py",
            "qwen_image_21_transformer.py",
            "pipeline_qwen_image_21.py",
        ):
            digest.update(Path(__file__).with_name(name).read_bytes())
        from vllm_omni_metal.ops import vpipe_kernels

        for module in (diffusion_ops, vpipe_kernels):
            digest.update(Path(module.__file__).read_bytes())
        config = self.pipeline.od_config
        descriptor = {
            "dtype": str(next(self.pipeline.transformer.parameters()).dtype),
            "transformer": config.tf_model_config,
            "attention": str(config.diffusion_attention_config),
            "sampling_sigmas": self.pipeline.sample_sigmas,
            "vae_slicing": config.vae_use_slicing,
            "vae_tiling": config.vae_use_tiling,
            "gpu_operators": "auto",
            "aot_silu": vpipe_kernels._artifact_path(torch.bfloat16).is_file(),
            "aot_rotary": vpipe_kernels._diffusion_artifact_path(torch.bfloat16).is_file(),
        }
        digest.update(json.dumps(descriptor, sort_keys=True, default=str).encode())
        return digest.hexdigest()

    def idle(self):
        if self.closed or self.busy:
            raise RuntimeError("QwenImage operator switch requires a healthy idle pipeline")

    def begin(self, request):
        self.idle()
        sampling = request.sampling_params_list[0]
        self.request_shape = {
            "width": sampling.width,
            "height": sampling.height,
            "steps": sampling.num_inference_steps,
            "cfg": sampling.true_cfg_scale,
        }
        if self.group is not None:
            self.group.context(self.request_shape)
        self.busy = True

    def end(self):
        self.busy = False

    def bind(self, layers, *, prepare):
        self.idle()
        if self.group is not None:
            raise ValueError("ANE gate already bound")
        if (
            not layers
            or len(set(layers)) != len(layers)
            or any(type(i) is not int or not 0 <= i < len(self.blocks) for i in layers)
        ):
            raise ValueError("ANE layers must be distinct transformer block indices")
        if prepare:
            os.environ["VLLM_OMNI_METAL_ANE"] = "auto"
        # Only the unquantized gate is shared with ANE. In particular no access
        # to img_mlp.out.weight: DynamicEngine may already have released it.
        scope = self.quantization.scope or self.quantization._scope()
        self.group = ProjectionGroup(self.blocks, layers, scope=scope, prepare=prepare)
        self.layers = list(layers)
        return self.snapshot()

    def set_mode(self, gpu_enabled, chunks):
        self.idle()
        if type(gpu_enabled) is not bool or type(chunks) is not int or chunks < 0:
            raise ValueError("invalid operator benchmark mode")
        if chunks and not gpu_enabled:
            raise ValueError("ANE calibration requires its optimized GPU baseline")
        if chunks and (self.group is None or not self.group._benchmark or self.group.disabled):
            raise RuntimeError("ANE candidate requires successful offline preparation")
        os.environ["QWEN_IMAGE_METAL_GPU"] = "auto" if gpu_enabled else "off"
        diffusion_ops.reset_metrics()
        self.last_shapes.clear()
        if self.group is not None:
            self.group.reset(chunks)
        return self.snapshot()

    def approve(self, shapes, baseline, candidates):
        self.idle()
        if self.group is None or not _enabled():
            raise RuntimeError(
                "approval requires an optimized GPU baseline and a prepared ANE group"
            )
        records = self.group.approve(baseline, candidates)
        self.group.production()
        return records

    def snapshot(self):
        group = self.group
        checks = (
            []
            if group is None
            else [{"key": list(key), **value} for key, value in group.checks.items()]
        )
        return {
            "gpu_enabled": _enabled(),
            "quantization": self.quantization.snapshot(),
            "gpu_metrics": diffusion_ops.metrics_snapshot(),
            "layers": self.layers,
            "scope": self.scope,
            "busy": self.busy,
            "closed": self.closed,
            "shapes": []
            if group is None
            else [list(shape) for shape in sorted(set().union(*(g.shapes for g in group.gates)))],
            "ane_metrics": {} if group is None else dict(group.metrics),
            "ane_disabled": False if group is None else group.disabled,
            "ane_benchmark": False if group is None else group._benchmark,
            "ane_audit": False if group is None else group._benchmark and group._audit,
            "ane_checks": checks,
            "parameter_device": str(next(self.pipeline.transformer.parameters()).device),
        }

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.quantization.close()
        if self.group is not None:
            self.group.close()


class QwenMetalPreparation:
    """String RPC extension on the private diffusion worker."""

    def _qwen_adapter(self):
        adapter = self.model_runner.pipeline._metal_adapter
        adapter.idle()
        return adapter

    def qwen_metal_snapshot(self):
        return self._qwen_adapter().snapshot()

    def qwen_metal_prepare(self, layers):
        return self._qwen_adapter().bind(layers, prepare=True)

    def qwen_metal_set_mode(self, gpu_enabled, chunks):
        return self._qwen_adapter().set_mode(gpu_enabled, chunks)

    def qwen_metal_finish_audit(self):
        adapter = self._qwen_adapter()
        if adapter.group is None:
            raise RuntimeError("no ANE group to audit")
        adapter.group.finish_audit()
        return adapter.snapshot()

    def qwen_metal_approve(self, shapes, baseline, candidates):
        return self._qwen_adapter().approve(shapes, baseline, dict(candidates))

    def qwen_joint_approve(self, shapes, baseline, candidates):
        return {"records": self._qwen_adapter().approve(shapes, baseline, dict(candidates))}

    def qwen_dynamic_prepare(self, layers):
        return self._qwen_adapter().quantization.prepare(layers)

    def qwen_dynamic_finish_capture(self):
        return self._qwen_adapter().quantization.finish_capture()

    def qwen_dynamic_set_candidate(self, enabled):
        return self._qwen_adapter().quantization.set_candidate(enabled)

    def qwen_dynamic_profile(self, calibration):
        return self._qwen_adapter().quantization.profile(calibration)

    def qwen_metal_close(self):
        adapter = self._qwen_adapter()
        adapter.close()
        return adapter.snapshot()

    def qwen_dynamic_release(self):
        return self._qwen_adapter().quantization.release()
