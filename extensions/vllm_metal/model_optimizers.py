# SPDX-License-Identifier: Apache-2.0
"""Explicit load-time optimizers, independent of model and serving registries."""

import json
import os
from importlib.metadata import entry_points
from typing import Any

from vllm_metal import envs


def _audit_metadata(value: Any) -> dict[str, Any]:
    if type(value) is not dict:
        raise TypeError("optimizer metadata must be a plain JSON object")
    # Serialize first to reject cycles/nonfinite values before walking the tree.
    encoded = json.dumps(value, allow_nan=False)
    pending = [value]
    while pending:
        item = pending.pop()
        if type(item) is dict:
            if any(type(key) is not str for key in item):
                raise TypeError("optimizer metadata JSON keys must be strings")
            pending.extend(item.values())
        elif type(item) is list:
            pending.extend(item)
        elif item is not None and type(item) not in (str, bool, int, float):
            raise TypeError("optimizer metadata must contain only plain JSON values")
    # Do not retain provider-owned containers or references to model state.
    return json.loads(encoded)


def apply_model_optimizers(
    model: Any, config: Any, *, runtime: str = "mlx"
) -> list[dict[str, Any]]:
    """Run explicitly selected providers once during model loading.

    Providers mutate their model in place and own capability/quality validation
    and resource cleanup. Failed installation is fatal; no rollback is implied.
    """
    selected = envs.METAL_MODEL_OPTIMIZERS
    mode = os.environ.get("METAL_QUANTIZATION", "off")
    if mode not in ("off", "dynamic"):
        raise ValueError("METAL_QUANTIZATION must be off or dynamic")
    if set(selected) - {"dynamic_quantization", "gpu_ops", "ane", "streaming"}:
        raise ValueError("METAL_MODEL_OPTIMIZERS contains an unknown provider")
    for name, provider, required in (
        ("METAL_QUANTIZATION_PROFILE", "dynamic_quantization", mode == "dynamic"),
        ("METAL_ANE_PROFILE", "ane", "ane" in selected),
        ("METAL_STREAMING_PROFILE", "streaming", "streaming" in selected),
    ):
        path = os.environ.get(name, "")
        if path and not os.path.isabs(path):
            raise ValueError(f"{name} must be an absolute path")
        if required and (not path or provider not in selected):
            raise ValueError(f"{name} requires an absolute path and explicit {provider} selection")
        if path and name != "METAL_QUANTIZATION_PROFILE" and provider not in selected:
            raise ValueError(f"{name} requires explicit {provider} selection")
    if not selected:
        return []
    available = entry_points(group="vllm_metal.model_optimizers")
    resolved = []
    for name in selected:
        matches = [provider for provider in available if provider.name == name]
        if len(matches) != 1:
            raise ValueError(
                f"model optimizer {name!r} requires exactly one installed provider; "
                f"found {len(matches)}"
            )
        resolved.append(matches[0])
    providers = []
    for entry in resolved:
        provider = entry.load()
        if not callable(provider):
            raise TypeError(f"model optimizer {entry.name!r} must be callable")
        providers.append((entry.name, provider))
    try:
        return [
            {
                "name": name,
                "metadata": _audit_metadata(provider(model, config, runtime=runtime)),
            }
            for name, provider in providers
        ]
    except BaseException as failure:
        from .lifecycle import close_model_optimizers

        try:
            close_model_optimizers(model)
        except Exception as cleanup:
            failure.add_note(f"optimizer cleanup also failed: {cleanup}")
        raise
