# SPDX-License-Identifier: Apache-2.0
"""Source revisions reviewed for the private integration APIs used here."""

COMPATIBLE_SOURCES = {
    "vllm-metal": {
        "git": "https://github.com/vllm-project/vllm-metal.git",
        "rev": "6d6f2c37f8242f1ff1a336534adf6c2d2fe2936d",
    },
    "vllm-omni-metal": {
        "git": "https://github.com/xiaoxiaoto/vllm-omni-metal.git",
        "rev": "5774df7d720b3a8c6d32035729b42c31e480f346",
    },
}


def validate_sources(sources):
    """Reject source changes until integration compatibility is reviewed."""
    for name, expected in COMPATIBLE_SOURCES.items():
        if sources.get(name) != expected:
            raise ValueError(f"unreviewed source for {name}: expected {expected!r}")
