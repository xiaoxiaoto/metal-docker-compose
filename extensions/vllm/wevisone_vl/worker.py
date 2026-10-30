"""vLLM worker RPCs for the fixed WeVisOne decoder."""

import hashlib
import os
from importlib.metadata import version
from pathlib import Path
from types import MethodType

import mlx.core as mx
import torch

from .pixel import decode, pixel_fusion_forward
from .upstream.config import Qwen3VLSegConfig
from .upstream.decoder import build_mask_decoder_from_qwen

_SIDECAR_SHA256 = "8528460481bd09d0780c53559b08194c7abbd29dff0d51783b619158acbe7b46"


class PixelWorkerExtension:
    def wevisone_initialize(self):
        if hasattr(self, "_wevisone_decoder"):
            raise RuntimeError("WeVisOne decoder already initialized")
        for package, expected in (("vllm", "0.32.0+cpu"), ("vllm-metal", "0.32.0"), ("mlx-vlm", "0.7.6")):
            if version(package) != expected:
                raise RuntimeError(f"unsupported {package} version: {version(package)}")
        if not torch.backends.mps.is_available() or not mx.metal.is_available():
            raise RuntimeError("WeVisOne requires MLX Metal and Torch MPS")
        if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
            raise RuntimeError("MPS fallback must be disabled")
        parallel = self.parallel_config
        if parallel.tensor_parallel_size != 1 or parallel.pipeline_parallel_size != 1 or parallel.world_size != 1:
            raise RuntimeError("WeVisOne requires a single worker")
        if self.lora_config is not None or self.speculative_config is not None:
            raise RuntimeError("LoRA and speculative decoding are unsupported")
        if self.model_config.dtype != torch.bfloat16 or self.model_config.quantization is not None:
            raise RuntimeError("WeVisOne requires an unquantized BF16 backbone")
        if os.environ.get("METAL_QUANTIZATION", "off") != "off" or os.environ.get("METAL_MODEL_OPTIMIZERS", "").strip():
            raise RuntimeError("Metal model optimizers are unsupported for pixel capture")
        model = self.model_runner.model
        processor = self.model_runner.tokenizer
        config = model.config
        vision = config.vision_config
        text = config.text_config
        if (config.model_type != "qwen3_vl" or text.hidden_size != 2560 or text.num_hidden_layers != 36
                or vision.hidden_size != 1024 or vision.depth != 24 or vision.spatial_merge_size != 2
                or list(model.vision_tower.deepstack_visual_indexes) != [5, 11, 17]
                or len(model.vision_tower.deepstack_merger_list) != 3
                or model.vision_tower.patch_embed.proj.weight.dtype != mx.bfloat16):
            raise RuntimeError("unsupported Qwen3-VL backbone configuration")
        tokenizer = processor.tokenizer
        if "<OBJ>" not in tokenizer.get_vocab():
            raise RuntimeError("tokenizer does not contain OBJ")
        obj_id = int(tokenizer.convert_tokens_to_ids("<OBJ>"))
        sidecar_path = Path(self.model_config.model) / "mask_decoder.pt"
        digest = hashlib.sha256()
        with sidecar_path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        if sidecar_path.stat().st_size != 38534741 or digest.hexdigest() != _SIDECAR_SHA256:
            raise RuntimeError("mask decoder sidecar identity mismatch")
        sidecar = torch.load(sidecar_path, map_location="cpu", weights_only=True)
        if not isinstance(sidecar, dict) or not isinstance(sidecar.get("state_dict"), dict):
            raise RuntimeError("invalid decoder sidecar")
        if sidecar.get("seg_token_id") is not None and int(sidecar["seg_token_id"]) != obj_id:
            raise RuntimeError("decoder OBJ token mismatch")
        cfg = Qwen3VLSegConfig.from_dict(sidecar.get("cfg"))
        if tuple(cfg.vision_feature_layers) != (5, 11, 17, 23) or int(sidecar.get("spatial_merge_size") or 2) != 2:
            raise RuntimeError("unsupported decoder feature configuration")
        decoder = build_mask_decoder_from_qwen(model, cfg)
        decoder.load_state_dict(sidecar["state_dict"], strict=True)
        # Adapt this private instance only; shared Torch functions and source stay unchanged.
        decoder.pixel_fusion.forward = MethodType(pixel_fusion_forward, decoder.pixel_fusion)
        decoder.to(device="mps", dtype=torch.float32).eval()
        torch.mps.synchronize()
        self._wevisone_obj_id = obj_id
        self._wevisone_cnn_size = int(cfg.cnn_image_size)
        self._wevisone_decoder = decoder
        return {"initialized": True}

    def wevisone_decode(self, payload):
        if not hasattr(self, "_wevisone_decoder"):
            raise RuntimeError("WeVisOne decoder was not initialized")
        return decode(self._wevisone_decoder, self.model_runner.model, self.model_runner.tokenizer,
                      self._wevisone_obj_id, self._wevisone_cnn_size, payload)
