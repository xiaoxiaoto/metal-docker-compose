#!/usr/bin/env python3
"""Validate the deployment's frozen backend cohort, including reused launchd services."""

import argparse
from importlib.metadata import PackageNotFoundError, distribution, version
import os
from pathlib import Path
import plistlib
import subprocess
import sys


EXPECTED = {
    "vllm": "0.32.0",
    "vllm-metal": "0.32.0",
    "vllm-omni": "0.32.0.dev0",
    "vllm-omni-metal": "0.32.0.dev0",
}


def check(omni=False):
    names = tuple(EXPECTED) if omni else ("vllm", "vllm-metal")
    actual = {}
    for name in names:
        try:
            actual[name] = version(name)
        except PackageNotFoundError:
            raise ValueError(f"缺少后端包：{name}") from None
    mismatches = [f"{name}={actual[name]}（需要 {EXPECTED[name]}）" for name in names
                  if actual[name].split("+", 1)[0] != EXPECTED[name]]
    if mismatches:
        raise ValueError("后端版本不匹配：" + "、".join(mismatches)
                         + "；请先停止共享环境的模型，再安装配套版本，不能混用或在运行中升级")
    if os.environ.get("VLLM_BACKEND_PYTHON"):
        base = Path(os.environ["VLLM_BACKEND_PYTHON"]).parent.parent.resolve()
        for name in ("vllm", "vllm-metal"):
            location = Path(distribution(name).locate_file("")).resolve()
            if not location.is_relative_to(base):
                raise ValueError(f"{name} 来自 {location}，不是统一后端 {base}；请停止旧服务后重启")
    return actual


def check_running(plist, omni=False):
    with Path(plist).open("rb") as file:
        config = plistlib.load(file)
    executable = Path(config["ProgramArguments"][0]).resolve(strict=True)
    python = executable.parent / "python"
    if not python.is_file():
        raise ValueError(f"无法确认运行服务的 Python 环境：{python}")
    # Resolve the service CLI, not whichever Python or Homebrew opt happens to be current.
    command = [str(python), str(Path(__file__).resolve())]
    if omni:
        command.append("--omni")
    subprocess.run(command, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--omni", action="store_true", help="同时校验 Omni 和 Omni Metal")
    parser.add_argument("--plist", type=Path, help="校验 launchd 服务使用的实际环境")
    args = parser.parse_args()
    try:
        if args.plist:
            check_running(args.plist, args.omni)
        else:
            actual = check(args.omni)
            print("后端版本校验通过：" + ", ".join(f"{name}={value}" for name, value in actual.items()),
                  file=sys.stderr)
    except (ValueError, KeyError, IndexError, OSError, subprocess.CalledProcessError) as error:
        print(f"版本检查失败：{error}", file=sys.stderr)
        sys.exit(1)
