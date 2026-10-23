#!/usr/bin/env python3
"""Install only the local adapter in a private overlay of the common backend."""

import hashlib
import subprocess
import sys
import sysconfig
import venv
from importlib.metadata import version
from pathlib import Path


def prepare():
    project = Path(__file__).resolve().parent
    subprocess.run(
        [sys.executable, "-I", str(project.parent / "check_versions.py"), "--omni"],
        check=True,
        stdout=sys.stderr,
    )
    versions = {name: version(name) for name in ("vllm", "vllm-metal", "mlx-audio")}
    if versions["vllm"].split("+")[0] != versions["vllm-metal"]:
        raise ValueError("vLLM and vLLM Metal versions must match")
    # The extension touches private STT hooks; do not silently adopt a different contract.
    if versions["vllm-metal"] != "0.31.0" or versions["mlx-audio"] not in ("0.5.8", "0.5.9"):
        raise ValueError(f"未经验证的依赖组合：{versions}；需要重新验证适配器，不能盲目升级")
    base = Path(sysconfig.get_path("purelib")).resolve()
    fingerprint = hashlib.sha256(
        (str(base) + str(versions) + sys.version).encode()
        + (project / "moss_metal.py").read_bytes()
        + (project / "pyproject.toml").read_bytes()
        + Path(__file__).read_bytes()
    ).hexdigest()[:16]
    runtime = (
        Path.home()
        / ".local/share/local-model-backend/overlays/moss-transcribe-diarize"
        / fingerprint
    )
    ready = runtime / ".ready"
    python = runtime / "bin/python"
    if ready.is_file() and ready.read_text() == fingerprint and (runtime / "bin/vllm").is_file():
        return runtime
    print(f"构建 MOSS 私有适配环境：{runtime}", file=sys.stderr)
    # Standalone macOS Python uses executable-relative dylibs; do not copy its binary.
    venv.create(runtime, with_pip=True, symlinks=True)
    site = (
        runtime
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    (site / "metal-base.pth").write_text(str(base) + "\n")
    subprocess.run(
        [
            str(python),
            "-I",
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-build-isolation",
            "--ignore-installed",
            str(project),
        ],
        check=True,
        stdout=sys.stderr,
    )
    subprocess.run([str(python), "-I", "-m", "pip", "check"], check=True, stdout=sys.stderr)
    ready.write_text(fingerprint)
    return runtime


if __name__ == "__main__":
    try:
        print(prepare())
    except Exception as error:
        print(f"私有 Metal 环境准备失败：{error}", file=sys.stderr)
        sys.exit(1)
