# SPDX-License-Identifier: Apache-2.0
"""Native YuE2 ABC planning through the existing /v1/completions endpoint.

Run with the selected private overlay's Python. Imports are lazy and only load
native helpers/tokenizer, never an engine or checkpoint weights. ``build`` emits
JSON; ``extract`` parses a saved non-streaming response. Neither performs HTTP.

The installed CompletionRequest forwards vllm_xargs to SamplingParams.extra_args;
AsyncOmni.generate wraps bare SamplingParams as stage 0 and OmniRequestOutput
inherits RequestOutput. return_token_ids supplies choices[0].token_ids even when
skip_tokenizer_init makes choices[0].text empty. Decode only with qwen.tiktoken.
An explicit ABC_END stop reason also supports servers omitting the stop token.
Never append an end token to a length-capped or otherwise unfinished plan.
"""

from __future__ import annotations

import argparse
import importlib
import json
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace


@lru_cache(maxsize=1)
def _load_native(model_path: str):
    model = importlib.import_module("vllm_omni.model_executor.models.yue2.yue2")
    prompt = importlib.import_module("vllm_omni.tokenizers.yue2_prompt")
    tokenizers = importlib.import_module("vllm_omni.tokenizers.yue2_tokenizer")
    tokenizer = tokenizers.YuE2TextTokenizer(Path(model_path).expanduser() / "qwen.tiktoken")
    return SimpleNamespace(model=model, prompt=prompt, tokenizer=tokenizer)


def build_abc_request(model_path, sample: dict, *, model_name="yue2-3b") -> dict:
    """Build one native, seeded ABC request, with no synthetic/external score.

    sample contains style, lyrics, optional cot (full/melody), optional seed.
    Native sampling is performed by YuE2's sampler, not reimplemented here.
    Most ordinary sampling fields keep engine limits/metadata aligned. The native
    min-token control stays in extra_args: Metal rejects vLLM's logits processor.
    """
    if not isinstance(sample, dict):
        raise ValueError("sample must be an object")
    for key in ("style", "lyrics"):
        if not isinstance(sample.get(key), str) or not sample[key].strip():
            raise ValueError(f"sample.{key} must be a nonempty string")
    cot = sample.get("cot", "full")
    if cot not in ("full", "melody"):
        raise ValueError("ABC planning requires cot=full or melody")
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("model_name must be a nonempty string")
    native = _load_native(str(model_path))
    model, tokenizer = native.model, native.tokenizer
    seed = sample.get("seed", model.DEFAULT_SEED)
    if type(seed) is not int or not -(2**63) <= seed < 2**63:
        raise ValueError("sample.seed must be a signed 64-bit integer")
    prefix = native.prompt.abc_prefix_ids(tokenizer.encode, sample["style"], sample["lyrics"], cot)
    preset = model.ABC_SAMPLING
    # CompletionRequest's numeric/string extra-args union does not declare bool.
    # Integer 1 is JSON/msgpack-native and the model intentionally reads bool().
    extra = {
        model.KEY_PHASE: "abc",
        model.KEY_SEED: seed,
        model.KEY_TEMPERATURE: preset["temperature"],
        model.KEY_TOP_P: preset["top_p"],
        model.KEY_TOP_K: preset["top_k"],
        model.KEY_REPETITION_PENALTY: preset["repetition_penalty"],
        model.KEY_PENALTY_WINDOW: preset["penalty_window"],
        model.KEY_MIN_TOKENS: preset["min_tokens"],
        model.KEY_MAX_AUDIO_FRAMES: model.SEMANTIC_SAMPLING["max_tokens"],
        model.KEY_SKIP_SYNTHESIS: 1,
        model.KEY_PREFIX_IDS: list(prefix),
    }
    return {
        "model": model_name,
        "prompt": prefix,
        "add_special_tokens": False,
        "echo": False,
        "stream": False,
        "n": 1,
        "seed": seed,
        "max_tokens": preset["max_tokens"],
        "temperature": preset["temperature"],
        "top_p": preset["top_p"],
        "top_k": preset["top_k"],
        "repetition_penalty": preset["repetition_penalty"],
        "stop_token_ids": [model.ABC_END],
        "return_token_ids": True,
        "vllm_xargs": extra,
    }


def extract_abc(response: dict, model_path) -> dict:
    """Reject anything without an unambiguous natural ABC_END termination.

    generated_token_ids is exactly the returned sequence (possibly including its
    final ABC_END); abc_token_ids is the nonempty ordinary-text span, without END.
    finish_reason must be stop. If stop_reason is supplied it must be ABC_END;
    without it, a real trailing ABC_END is required. Contradictory stop metadata,
    length, explicit truncation, prompt echoes and non-text tokens are rejected.
    Response text is never trusted in place of checkpoint-native token decoding.
    naturally_ended attests token termination only, not valid ABC music syntax,
    musical structure, lyric alignment, completeness of arrangement, or quality.
    """
    if not isinstance(response, dict) or response.get("error") is not None:
        raise ValueError("expected a successful completion response object")
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise ValueError("expected exactly one completion choice")
    choice = choices[0]
    if response.get("truncated") or choice.get("truncated"):
        raise ValueError("ABC plan is explicitly truncated")
    finish_reason = choice.get("finish_reason")
    if finish_reason != "stop":
        raise ValueError(f"ABC plan did not naturally stop: {finish_reason!r}")
    generated = choice.get("token_ids")
    if (
        not isinstance(generated, list)
        or not generated
        or any(type(t) is not int for t in generated)
    ):
        raise ValueError("completion must return nonempty integer token_ids")
    native = _load_native(str(model_path))
    model = native.model
    has_end = generated[-1] == model.ABC_END
    stop_reason = choice.get("stop_reason")
    if stop_reason is not None:
        if type(stop_reason) is not int or stop_reason != model.ABC_END:
            raise ValueError("completion stop_reason does not match ABC_END")
    elif not has_end:
        raise ValueError("completion has no ABC_END termination evidence")
    text_ids = generated[:-1] if has_end else generated
    if not text_ids or any(not 0 <= t < model.EOD for t in text_ids):
        raise ValueError("ABC score must be nonempty and inside the ordinary text vocabulary")
    # Strict checks above intentionally precede the permissive native helper,
    # which can strip an ABC_START span. Echoed prompts must not pass as a score.
    abc_ids = native.prompt.abc_ids_from_generated(generated)
    abc = native.tokenizer.decode(abc_ids)
    if not isinstance(abc, str) or not abc.strip():
        raise ValueError("ABC score decodes to empty text")
    return {
        "abc": abc,
        "abc_token_ids": list(abc_ids),
        "generated_token_ids": list(generated),
        "finish_reason": finish_reason,
        "naturally_ended": True,
        "stop_reason": stop_reason,
        "end_token_in_output": has_end,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path", required=True, help="local checkpoint directory containing qwen.tiktoken"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="emit a /v1/completions JSON payload; no HTTP")
    build.add_argument("--sample", required=True, type=Path)
    build.add_argument("--model-name", default="yue2-3b")
    extract = subparsers.add_parser("extract", help="decode a saved completion response; no HTTP")
    extract.add_argument("--response", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "build":
        result = build_abc_request(
            args.model_path, json.loads(args.sample.read_text()), model_name=args.model_name
        )
    else:
        result = extract_abc(json.loads(args.response.read_text()), args.model_path)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2))


if __name__ == "__main__":
    main()
