#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Offline complete bilingual Higgs ANE + DynamicEngine calibration (never starts services)."""

import hashlib
import importlib.util
import sys
from pathlib import Path


def main():
    resource = Path(importlib.util.find_spec("higgs_joint_preparation").origin).resolve()
    if (
        not resource.is_relative_to(Path(sys.prefix).resolve())
        or hashlib.sha256(resource.read_bytes()).digest()
        != hashlib.sha256(Path(__file__).read_bytes()).digest()
    ):
        raise RuntimeError("joint CLI differs from owned wheel resource; rebuild private runtime")
    spec = importlib.util.spec_from_file_location(
        "higgs_deployment_preparation", Path(__file__).with_name("prepare_quantization.py")
    )
    preparation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(preparation)
    if "--joint" not in sys.argv:
        sys.argv.append("--joint")
    preparation.main()


if __name__ == "__main__":
    main()
