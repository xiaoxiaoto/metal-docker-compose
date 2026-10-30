"""Reuse mlx-audio MOSS in vLLM Metal's existing one-shot STT runner."""

import numpy as np
import mlx.core as mx
import torch
from mlx_audio.stt.models.moss_transcribe_diarize import Model
from mlx_audio.stt.models.moss_transcribe_diarize.config import ModelConfig
from transformers import AutoTokenizer
from vllm_metal.stt.runtime import STTRuntimeAdapter
from vllm_metal.stt.serve import STTRequestInput, VLLMSTTRequestAdapter


MODEL_TYPE = "moss_transcribe_diarize"
_registered = False


def to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().float().numpy()
    return np.asarray(value)


class MossModel(Model):
    def __init__(self, config, dtype):
        if config.get("quantization") or config.get("quantization_config"):
            raise ValueError("MOSS Metal adapter requires unquantized weights")
        if config.get("audio_merge_size", 4) != 4:
            raise ValueError("mlx-audio MOSS currently requires audio_merge_size=4")
        super().__init__(ModelConfig.from_dict(config))
        self.serving_dtype = dtype

    def load_weights(self, weights, strict=True):
        # The shared STT loader requests strict=False; never accept missing MOSS weights.
        cast = [(name, value.astype(self.serving_dtype)) for name, value in weights]
        return super().load_weights(cast, strict=True)

    def create_runtime_adapter(self, model_path):
        return MossAdapter(self, model_path)


class MossAdapter(STTRuntimeAdapter):
    def __init__(self, model, model_path):
        super().__init__(model, model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
        self._eot = tokenizer.eos_token_id
        if not isinstance(self._eot, int):
            raise ValueError("MOSS tokenizer must have an EOS token")

    @property
    def transcriber(self):
        return self

    @property
    def eot_token(self):
        return self._eot

    def extract_audio_features(self, input_features):
        mel = to_numpy(input_features["input_features"])
        lengths = to_numpy(input_features["audio_feature_lengths"]).reshape(-1)
        counts = to_numpy(input_features["audio_chunk_counts"]).reshape(-1)
        if (mel.ndim != 3 or mel.shape[1] != self.model.config.audio_config.num_mel_bins
                or not 0 < mel.shape[2] <= 2 * self.model.config.audio_config.max_source_positions):
            raise ValueError("MOSS expects (chunks, mel_bins, frames) within Whisper's context")
        if (len(counts) != 1 or counts[0] != mel.shape[0] or mel.shape[0] == 0
                or len(lengths) != mel.shape[0]):
            raise ValueError("MOSS requires one audio and one length per chunk")
        max_tokens = min((mel.shape[2] + 1) // 2,
                         self.model.config.audio_config.max_source_positions) // 4
        if (not np.isfinite(mel).all() or not np.isfinite(lengths).all()
                or np.any(lengths != np.floor(lengths))
                or np.any(lengths <= 0) or np.any(lengths > max_tokens)):
            raise ValueError("Invalid MOSS mel values or chunk lengths")
        features = []
        # Encode chunks separately: long audio must not batch every Whisper chunk in memory.
        for chunk, length in zip(mel, lengths, strict=True):
            encoded = self.model.model.get_audio_features(
                mx.array(chunk.T[None], dtype=self.model.serving_dtype),
                mx.array([int(length)], dtype=mx.int32),
            )[0]
            mx.eval(encoded)
            features.append(encoded)
        result = mx.concatenate(features, axis=1)
        if not bool(mx.all(mx.isfinite(result)).item()):
            raise ValueError("MOSS encoder emitted non-finite features")
        return result

    def warm_up(self):
        self.extract_audio_features({
            "input_features": np.zeros((1, self.model.config.audio_config.num_mel_bins, 16)),
            "audio_feature_lengths": np.array([2]),
            "audio_chunk_counts": np.array([1]),
        })

    def decode_tokens(self, audio_features, prompt_token_ids, sampling):
        if not prompt_token_ids:
            raise ValueError("MOSS requires prompt_token_ids")
        mask = np.asarray(prompt_token_ids) == self.model.config.audio_token_id
        if int(mask.sum()) != audio_features.shape[1]:
            raise ValueError("MOSS audio placeholder count does not match encoded features")
        context_limit = self.model.config.text_config.max_position_embeddings
        budget = sampling.decode_budget(context_limit - len(prompt_token_ids))
        if budget <= 0:
            raise ValueError("MOSS request has no remaining decode budget")
        tokens = mx.array([prompt_token_ids], dtype=mx.int32)
        embeds = self.model.model.language_model.embed_tokens(tokens)
        embeds[:, mx.array(np.flatnonzero(mask), dtype=mx.int32), :] = audio_features.astype(embeds.dtype)
        cache = self.model.make_cache()
        logits = self.model(tokens, input_embeddings=embeds, cache=cache)
        generated = []
        stop_ids = set(sampling.sampling_params.stop_token_ids or ())
        for step in range(budget):
            if step:
                logits = self.model(mx.array([[generated[-1]]], dtype=mx.int32), cache=cache)
            mx.eval(logits)
            if not bool(mx.all(mx.isfinite(logits[:, -1, :])).item()):
                raise ValueError("MOSS decoder emitted non-finite logits")
            token = sampling.next_token(prompt_token_ids, generated, logits[:, -1, :])
            generated.append(token)
            if token in stop_ids or (token == self.eot_token and not sampling.sampling_params.ignore_eos):
                break
        # Do not fabricate EOS on budget exhaustion: vLLM must report length truncation.
        return generated


def register():
    """Process-local extension; never mutate shared Homebrew package files."""
    global _registered
    if _registered:
        return
    from vllm_metal.stt import detection, registry

    detection._STT_MODEL_TYPES = detection._STT_MODEL_TYPES | {MODEL_TYPE}
    registry._STT_MODEL_CONSTRUCTORS[MODEL_TYPE] = MossModel
    original = VLLMSTTRequestAdapter.from_vllm_request

    def from_request(cls, request):
        features = request.mm_features
        payload = features[0].data if features else None
        if payload is None or "audio_feature_lengths" not in payload:
            return original(request)
        if len(features) != 1 or features[0].modality != "audio":
            raise ValueError("MOSS STT accepts exactly one audio per request")
        values = {}
        for key in ("input_features", "audio_feature_lengths", "audio_chunk_counts"):
            field = payload.get(key)
            if field is None or field.data is None:
                raise ValueError(f"MOSS STT request is missing {key}")
            values[key] = field.data
        return STTRequestInput(
            request.req_id, tuple(request.prompt_token_ids or ()), values,
            request.sampling_params,
        )

    VLLMSTTRequestAdapter.from_vllm_request = classmethod(from_request)
    _registered = True
