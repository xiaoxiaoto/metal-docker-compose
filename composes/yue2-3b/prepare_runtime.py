#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""为 YuE2 构建私有的 patched Omni；不改 Homebrew 或其他服务的安装文件。"""

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import tomllib
import venv
from importlib.metadata import distribution
from pathlib import Path


def run(*args, **kwargs):
    subprocess.run(args, check=True, stdout=sys.stderr, **kwargs)


PRIVATE_MODULES = (
    "yue2_dynamic_quantization",
    "prepare_dynamic_quantization",
    "prepare_quantization",
    "prepare_ane",
    "mixed_quantization",
)


def private_sources():
    extensions = Path(__file__).resolve().parents[2] / "extensions/vllm_omni/yue2_3b"
    return tuple(extensions / f"{name}.py" for name in PRIVATE_MODULES)


def selected_backend_wheel():
    selection = Path.home() / "Library/Application Support/yue2-3b/backend-wheel"
    value = os.environ.get("VLLM_OMNI_METAL_WHEEL")
    if value is None and selection.is_file():
        if selection.is_symlink() or selection.stat().st_uid != os.getuid():
            raise ValueError("YuE2 backend-wheel 必须是当前用户拥有的普通文件")
        value = selection.read_text().strip()
    if value is None:
        return None
    wheel = Path(value).expanduser()
    if not wheel.is_absolute() or not wheel.is_file() or wheel.suffix != ".whl":
        raise ValueError("VLLM_OMNI_METAL_WHEEL 必须指向现有的绝对 wheel 路径")
    if wheel.is_symlink() or wheel.stat().st_uid != os.getuid():
        raise ValueError("YuE2 后端 wheel 必须是当前用户拥有的普通文件")
    return wheel.resolve()


def package_private_module(checkout):
    """Declare YuE2 modules in Omni's existing setuptools wheel configuration."""
    checkout = Path(checkout)
    config = checkout / "pyproject.toml"
    text = config.read_text()
    setuptools = tomllib.loads(text)["tool"]["setuptools"]
    if "py-modules" in setuptools or text.count("[tool.setuptools]\n") != 1:
        raise ValueError("上游 setuptools 配置变化；不能安全声明 YuE2 私有模块")
    modules = json.dumps(list(PRIVATE_MODULES))
    config.write_text(
        text.replace(
            "[tool.setuptools]\n",
            f"[tool.setuptools]\npy-modules = {modules}\n",
            1,
        )
    )
    for source in private_sources():
        shutil.copyfile(source, checkout / source.name)


def runtime_fingerprint(patch, commit, base_site, backend_wheel=None):
    digest = hashlib.sha256(
        patch.read_bytes()
        + commit.encode()
        + str(base_site).encode()
        + sys.version.encode()
        + Path(__file__).read_bytes()
    )
    for path in private_sources():
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    if backend_wheel is not None:
        digest.update(str(backend_wheel).encode())
        digest.update(backend_wheel.read_bytes())
    backend = Path(distribution("vllm-omni-metal").locate_file("vllm_omni_metal"))
    if not (backend / "__init__.py").is_file():
        raise ValueError("统一后端必须提供已安装的非 editable Omni Metal wheel")
    for path in sorted(backend.rglob("*")):
        if (
            path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix != ".pyc"
            and path.name != ".DS_Store"
        ):
            digest.update(str(path.relative_to(backend)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def prepare_attention(python):
    # Isolated mode ignores a caller's PYTHONPATH/cwd, but retains the selected base .pth.
    # Probe first so an already matching artifact does not require a compiler or build budget.
    run(
        str(python),
        "-I",
        "-c",
        "from vllm_omni_metal.ops.mps_attention import get_mps_ops, build_mps_launcher; "
        "build_mps_launcher() if get_mps_ops() is None else None",
    )
    # A fresh process must really load the artifact, not retain a cached missing result.
    run(
        str(python),
        "-I",
        "-c",
        "from vllm_omni_metal.ops.mps_attention import get_mps_ops; "
        "assert get_mps_ops() is not None, 'Torch-stream attention launcher is not prepared'",
    )


def prepare():
    run(
        sys.executable,
        "-I",
        str(Path(__file__).resolve().parents[2] / "common_env" / "check_versions.py"),
        "--omni",
    )
    backend_wheel = selected_backend_wheel()
    patch = (
        Path(__file__).resolve().parents[2]
        / "extensions/vllm_omni/yue2_3b/yue2-mps.patch"
    )
    source = json.loads(distribution("vllm-omni").read_text("direct_url.json"))
    if source["url"] != "https://github.com/vllm-project/vllm-omni.git":
        raise ValueError("YuE2 必须从统一后端的官方 Omni 源码构建")
    commit = source["vcs_info"]["commit_id"]
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("统一后端的 Omni commit 无效")
    base_site = Path(sysconfig.get_path("purelib")).resolve()
    fingerprint = runtime_fingerprint(patch, commit, base_site, backend_wheel)
    runtime = Path.home() / ".local/share/local-model-backend/overlays/yue2-3b" / fingerprint
    ready = runtime / ".ready"
    python = runtime / "bin/python"
    complete = ready.is_file() and ready.read_text() == fingerprint and python.is_file()
    ready.unlink(missing_ok=True)
    if not complete:
        runtime.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        print(f"构建 YuE2 私有 Omni 环境：{runtime}", file=sys.stderr)
        # Standalone macOS Python uses executable-relative dylibs; do not copy its binary.
        venv.create(runtime, with_pip=False, symlinks=True)
        site = (
            runtime
            / "lib"
            / f"python{sys.version_info.major}.{sys.version_info.minor}"
            / "site-packages"
        )
        # Only inherit the selected Omni Metal environment, not arbitrary system packages.
        (site / "omni-metal-base.pth").write_text(str(base_site) + "\n")
        if backend_wheel is not None:
            run(
                str(python),
                "-I",
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--ignore-installed",
                str(backend_wheel),
            )
        with tempfile.TemporaryDirectory(prefix="yue2-omni-") as checkout:
            run("git", "-C", checkout, "init", "-q")
            run(
                "git",
                "-C",
                checkout,
                "fetch",
                "-q",
                "--depth",
                "1",
                "https://github.com/vllm-project/vllm-omni.git",
                commit,
            )
            run("git", "-C", checkout, "checkout", "-q", "--detach", "FETCH_HEAD")
            run("git", "-C", checkout, "apply", "--check", str(patch))
            run("git", "-C", checkout, "apply", str(patch))
            package_private_module(checkout)
            environment = dict(
                os.environ,
                VLLM_OMNI_TARGET_DEVICE="cpu",
                VLLM_OMNI_VERSION_OVERRIDE="0.32.0.dev0+cpu.yue2mps2",
            )
            # Install our wheel in the overlay only; never uninstall from the shared base.
            run(
                str(python),
                "-I",
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--ignore-installed",
                checkout,
                env=environment,
            )
    run(str(python), "-I", "-m", "pip", "check")
    run(
        str(python),
        "-I",
        "-c",
        "from pathlib import Path; import sys,vllm_omni; "
        "assert Path(vllm_omni.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()), "
        "'Omni was not loaded from the private environment'",
    )
    module_sha256 = hashlib.sha256(private_sources()[0].read_bytes()).hexdigest()
    run(
        str(python),
        "-I",
        "-c",
        "from pathlib import Path; from importlib.metadata import distribution; "
        "import base64,hashlib,sys,vllm_omni,yue2_dynamic_quantization; "
        "from vllm_omni_metal.quantization.linear import QuantizedLinear,validate_profile; "
        "from vllm_omni_metal.quantization.engine import DynamicEngine; "
        "from vllm_omni_metal.quantization.capture import CaptureSession; "
        "from vllm_omni_metal.quantization.adapter import DenseModelAdapter; "
        "from vllm_omni_metal.quantization.runtime import load_profile,configure_model_quantization; "
        "module=Path(yue2_dynamic_quantization.__file__).resolve(); "
        "assert module.parent == Path(vllm_omni.__file__).resolve().parent.parent, "
        "'YuE2 private module was not loaded from the Omni overlay'; "
        "assert module.is_relative_to(Path(sys.prefix).resolve()), "
        "'YuE2 private module escaped the overlay'; "
        "assert hashlib.sha256(module.read_bytes()).hexdigest() == sys.argv[1], "
        "'YuE2 private module source differs from the selected deployment'; "
        "entry=next(p for p in distribution('vllm-omni').files if str(p)==module.name); "
        "assert entry.hash is not None and entry.hash.mode=='sha256'; "
        "assert entry.hash.value==base64.urlsafe_b64encode(hashlib.sha256(module.read_bytes()).digest()).decode().rstrip('='); "
        "assert entry.size==module.stat().st_size",
        module_sha256,
    )
    run(
        str(python),
        "-I",
        "-c",
        "from pathlib import Path\n"
        "from importlib import import_module\n"
        "from importlib.metadata import distribution\n"
        "import base64, hashlib, sys, vllm_omni\n"
        "names = ('yue2_dynamic_quantization', 'prepare_dynamic_quantization', "
        "'prepare_quantization', 'prepare_ane', 'mixed_quantization')\n"
        "root = Path(vllm_omni.__file__).resolve().parent.parent\n"
        "entries = {str(p): p for p in distribution('vllm-omni').files}\n"
        "assert len(sys.argv) == len(names) + 1\n"
        "for name, expected in zip(names, sys.argv[1:]):\n"
        "    module = import_module(name)\n"
        "    source = Path(module.__file__).resolve()\n"
        "    data = source.read_bytes()\n"
        "    entry = entries[name + '.py']\n"
        "    assert source.parent == root and source.is_relative_to(Path(sys.prefix).resolve()), "
        "'YuE2 extension escaped the private Omni overlay: ' + name\n"
        "    assert hashlib.sha256(data).hexdigest() == expected, "
        "'YuE2 extension source differs from selected deployment: ' + name\n"
        "    assert entry.hash is not None and entry.hash.mode == 'sha256'\n"
        "    assert entry.hash.value == base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip('=')\n"
        "    assert entry.size == len(data)",
        *(hashlib.sha256(path.read_bytes()).hexdigest() for path in private_sources()),
    )
    if backend_wheel is not None:
        run(
            str(python),
            "-I",
            "-c",
            "from pathlib import Path; import sys,vllm_omni_metal; "
            "assert Path(vllm_omni_metal.__file__).resolve().is_relative_to("
            "Path(sys.prefix).resolve()), 'Metal wheel was not loaded from the YuE2 overlay'",
        )
    prepare_attention(python)
    if runtime_fingerprint(patch, commit, base_site, backend_wheel) != fingerprint:
        raise ValueError("后端源码在私有环境准备期间发生变化；未标记 ready，请重试")
    if backend_wheel is not None:
        selection = Path.home() / "Library/Application Support/yue2-3b/backend-wheel"
        selection.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with tempfile.NamedTemporaryFile(mode="w", dir=selection.parent, delete=False) as file:
            temporary = Path(file.name)
            file.write(str(backend_wheel) + "\n")
        try:
            os.replace(temporary, selection)
        finally:
            temporary.unlink(missing_ok=True)
    ready.write_text(fingerprint)
    return runtime


if __name__ == "__main__":
    try:
        print(prepare())
    except Exception as error:
        print(f"私有 Omni 环境准备失败：{error}", file=sys.stderr)
        sys.exit(1)
