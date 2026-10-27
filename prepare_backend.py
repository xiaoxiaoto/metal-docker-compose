#!/usr/bin/env python3
"""Rebuild the shared backend after safely stopping its dependent local models."""

import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.request import getproxies

from check_versions import EXPECTED

PROJECT = Path(__file__).resolve().parent
SOURCE = PROJECT.parent / "vllm-omni-metal"
CACHE = Path.home() / ".local/share/local-model-backend"
SERVICES = {
    "h2o": "com.local.h2o-lightning-vllm",
    "higgs": "com.local.higgs-tts-3-metal",
    "yue2": "com.local.yue2-3b-metal",
    "moss": "com.local.moss-transcribe-diarize-metal",
    "paddleocr": "com.local.paddleocr-vl-metal",
    "qwenimage": "com.local.qwen-image-2.1-turbo-metal",
    "seedvr2": "com.local.seedvr2-metal",
}


def inherit_system_proxies():
    """Pass macOS proxy settings to Git/pip/uv without overriding explicit env."""
    proxies = getproxies()
    for scheme in ("http", "https", "no"):
        key = f"{scheme}_proxy"
        if key not in os.environ and key.upper() not in os.environ and proxies.get(scheme):
            os.environ[key] = proxies[scheme]


def run(*command, **kwargs):
    subprocess.run(command, check=True, stdout=sys.stderr, **kwargs)


def remote_refs(repo, pattern):
    return subprocess.check_output(
        ["git", "ls-remote", "https://github.com/" + repo + ".git", pattern],
        text=True,
        timeout=60,
    )


def stable_version(refs):
    versions = [
        match.group(1)
        for line in refs.splitlines()
        if (match := re.search(r"\trefs/tags/v?(\d+\.\d+\.\d+)$", line))
    ]
    if not versions:
        raise ValueError("官方仓库没有可识别的稳定版本标签")
    return max(versions, key=lambda value: tuple(map(int, value.split("."))))


def official_versions():
    latest = {
        name: stable_version(remote_refs(repo, "refs/tags/*"))
        for name, repo in (("vllm", "vllm-project/vllm"), ("vllm-metal", "vllm-project/vllm-metal"))
    }
    if any(latest[name] != EXPECTED[name] for name in latest):
        raise ValueError(
            f"官方最新核心为 {latest}，当前后端只验证了 {EXPECTED}；"
            "必须先更新并验证兼容组合，不能混装或静默启动旧版本"
        )
    refs = remote_refs("vllm-project/vllm-omni", "refs/heads/main").splitlines()
    if len(refs) != 1 or not re.fullmatch(r"[0-9a-f]{40}\trefs/heads/main", refs[0]):
        raise ValueError("无法确定官方 Omni 最新 main commit")
    return dict(latest, omni_commit=refs[0].split()[0])


def fingerprint(latest):
    digest = hashlib.sha256(json.dumps(dict(latest, expected=EXPECTED), sort_keys=True).encode())
    digest.update(Path(__file__).read_bytes())
    files = [SOURCE / "pyproject.toml", SOURCE / "install.sh"]
    files += sorted(
        path
        for path in (SOURCE / "vllm_omni_metal").rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
        and path.name != ".DS_Store"
    )
    for path in files:
        digest.update(str(path.relative_to(SOURCE)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def stop_models():
    # Drain the common entry point before stopping any resident/external service.
    run(sys.executable, "-I", str(PROJECT / "lifecycle_control.py"), "stop")
    domain = f"gui/{os.getuid()}"
    for name in reversed(SERVICES):
        print(f"重建后端，先停止依赖模型：{name}", file=sys.stderr)
        run("bash", str(PROJECT / "stop.sh"), name)
    for label in SERVICES.values():
        state = subprocess.run(
            ["launchctl", "print", f"{domain}/{label}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if state.returncode == 0:
            raise ValueError(f"模型任务尚未退出，拒绝删除环境：{label}")


def remove_environments():
    """Delete only fingerprinted environments; never follow cache-directory symlinks."""
    if CACHE.is_symlink() or CACHE.stat().st_uid != os.getuid():
        raise ValueError(f"后端根目录路径或所有权无效：{CACHE}")
    root = CACHE.resolve()
    candidates = []
    for category in ("runtime", "overlays"):
        parent = CACHE / category
        if parent.is_symlink():
            raise ValueError(f"环境父目录不能为符号链接：{parent}")
        if not parent.exists():
            continue
        parents = [parent] if category == "runtime" else list(parent.iterdir())
        for directory in parents:
            if directory.is_symlink() or not directory.is_dir():
                raise ValueError(f"环境父目录无效：{directory}")
            for path in directory.iterdir():
                if not re.fullmatch(r"[0-9a-f]{16}", path.name):
                    continue
                if path.is_symlink() or not path.is_dir() or path.stat().st_uid != os.getuid():
                    raise ValueError(f"环境路径或所有权无效，拒绝删除：{path}")
                if not path.resolve().is_relative_to(root):
                    raise ValueError(f"环境路径越界，拒绝删除：{path}")
                candidates.append(path)
    current = CACHE / "current"
    if current.exists() and not current.is_symlink():
        raise ValueError(f"current 不是符号链接，拒绝删除：{current}")
    # A surviving worker must not lose its imported packages underneath it.
    processes = subprocess.check_output(["ps", "-axo", "pid=,command="], text=True)
    if any(str(CACHE / category) + "/" in processes for category in ("runtime", "overlays")):
        raise ValueError("仍有进程引用后端环境，拒绝删除；请检查残留 worker")
    current.unlink(missing_ok=True)
    for path in candidates:
        print(f"删除历史环境：{path}", file=sys.stderr)
        shutil.rmtree(path)


def install_plugin(runtime):
    # Build from a clean copy: preserve the checkout and exclude stale build artifacts/editable paths.
    with tempfile.TemporaryDirectory(prefix="omni-metal-wheel-") as temporary:
        build = Path(temporary) / "source"
        build.mkdir()
        for name in ("pyproject.toml", "README.md", "LICENSE"):
            shutil.copyfile(SOURCE / name, build / name)
        shutil.copytree(
            SOURCE / "vllm_omni_metal",
            build / "vllm_omni_metal",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"),
        )
        wheels = Path(temporary) / "wheels"
        python = str(runtime / "bin/python")
        run(python, "-I", "-m", "pip", "wheel", "--no-deps", str(build), "--wheel-dir", str(wheels))
        (wheel,) = wheels.glob("vllm_omni_metal-*.whl")
        run(python, "-I", "-m", "pip", "install", "--no-deps", "--force-reinstall", str(wheel))


def validate(runtime):
    python = str(runtime / "bin/python")
    run(python, "-I", str(PROJECT / "check_versions.py"), "--omni")
    run(python, "-I", "-m", "pip", "check")
    run(
        python,
        "-I",
        "-c",
        "import sys; from pathlib import Path; import vllm_omni_metal; "
        "assert Path(vllm_omni_metal.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()), "
        "'backend must be installed, not editable or inherited from another environment'; "
        "from vllm_omni_metal.ops.mps_attention import get_mps_ops; "
        "from vllm_omni_metal.quantization.linear import QuantizedLinear, validate_profile; "
        "from vllm_omni_metal.quantization.capture import CaptureSession; "
        "from vllm_omni_metal.quantization.adapter import DenseModelAdapter; "
        "from vllm_omni_metal.quantization.runtime import load_profile, configure_model_quantization; "
        "from vllm_omni_metal.ane.projection import DenseProjection, prepare_projection; "
        "assert get_mps_ops() is not None, 'Torch-stream attention launcher is not prepared'",
    )


def prepare():
    if not (SOURCE / "install.sh").is_file():
        raise ValueError(f"缺少用户指定的后端源码仓库：{SOURCE}")
    inherit_system_proxies()
    CACHE.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (CACHE / ".prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        latest = official_versions()
        print(
            f"官方版本预检：vLLM/Metal {latest['vllm']}，Omni main {latest['omni_commit']}；"
            "Omni/Omni Metal 使用配套开发版，不冒充稳定发布",
            file=sys.stderr,
        )
        key = fingerprint(latest)
        runtime = CACHE / "runtime" / key
        ready = runtime / ".ready"
        current = CACHE / "current"
        stop_models()
        remove_environments()
        runtime.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        print(f"从空目录重建兼容后端：{runtime}（不修改 Homebrew 或其他环境）", file=sys.stderr)
        run(
            "bash",
            str(SOURCE / "install.sh"),
            "--venv",
            str(runtime),
            "--with-mps-attention",
            env=dict(os.environ, VLLM_OMNI_COMMIT=latest["omni_commit"]),
        )
        install_plugin(runtime)
        validate(runtime)
        if fingerprint(latest) != key:
            raise ValueError("后端源码在构建/校验期间发生变化；未激活该环境，请重试")
        (runtime / "backend.json").write_text(
            json.dumps(dict(latest, expected=EXPECTED), indent=2) + "\n"
        )
        ready.write_text(key)
        temporary = CACHE / f".current-{os.getpid()}"
        try:
            temporary.symlink_to(runtime, target_is_directory=True)
            os.replace(temporary, current)
        finally:
            temporary.unlink(missing_ok=True)
        return runtime


if __name__ == "__main__":
    try:
        print(prepare())
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        print(f"后端预检/升级失败，不启动模型：{error}", file=sys.stderr)
        sys.exit(1)
