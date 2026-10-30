#!/usr/bin/env python3
"""从固定来源和显式补丁构建共享优化 wheel；不伪装官方 Git 安装来源。"""

import base64
import hashlib
import io
import json
import os
import stat
from pathlib import Path
import runpy
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import zipfile
from importlib.metadata import distribution

sys.dont_write_bytecode = True
PROJECT = Path(__file__).resolve().parent.parent
PATCHES = {
    "vllm-metal": PROJECT / "extensions/vllm_metal/patches/vllm-metal-model-optimizers.patch",
    "vllm-omni-metal": PROJECT / "extensions/vllm_omni_metal/vllm-omni-metal-scratch-budget.patch",
}
EXTENSION = "vllm-metal-extensions"
EXTENSION_SOURCE = PROJECT / "extensions/vllm_metal"
MANIFEST = "shared-optimizations.json"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sources():
    with (PROJECT / "pyproject.toml").open("rb") as file:
        return tomllib.load(file)["tool"]["uv"]["sources"]


def extension_sources():
    """Hash build inputs, excluding generated packaging artifacts."""
    result = {"pyproject.toml": sha256(PROJECT / "pyproject.toml")}
    for path in sorted(EXTENSION_SOURCE.rglob("*")):
        relative = path.relative_to(EXTENSION_SOURCE)
        if any(part in ("build", "dist", "__pycache__", ".ruff_cache", ".pytest_cache")
               or part.endswith(".egg-info")
               for part in relative.parts) or path.suffix == ".pyc":
            continue
        if path.is_symlink():
            raise ValueError(f"扩展源码不能包含符号链接：{relative}")
        if path.is_file():
            result[relative.as_posix()] = sha256(path)
    required = {
        "pyproject.toml", "__init__.py", "compatibility.py",
        "model_optimizers.py", "lifecycle.py",
        "ops/__init__.py", "ops/gpu.py",
        "ane/__init__.py", "ane/bridge.py", "ane/torch.py",
        "execution/__init__.py", "execution/streaming.py", "execution/mlx_streaming.py",
        "quantization/__init__.py", "quantization/mlx.py", "quantization/provider.py",
    }
    if not required <= result.keys():
        raise ValueError("扩展源码或功能子包缺失")
    return result


def verify_compatible_sources():
    contract = EXTENSION_SOURCE / "compatibility.py"
    runpy.run_path(str(contract))["validate_sources"](sources())


def verify_official_source(name, package):
    source = sources()[name]
    direct = json.loads(package.read_text("direct_url.json") or "{}")
    if (direct.get("url") != source["git"]
            or direct.get("vcs_info", {}).get("commit_id") != source["rev"]
            or direct.get("dir_info", {}).get("editable")):
        raise ValueError(f"需要固定官方 Git 来源的非 editable 包：{name}")
    verify_records(package)


def optimization_environment():
    """将启动时的公共 Metal 优化配置固化到 plist。"""
    mode = os.environ.get("METAL_QUANTIZATION", "off")
    if mode not in ("off", "dynamic"):
        raise ValueError("METAL_QUANTIZATION 必须为 off 或 dynamic")
    selected = os.environ.get("METAL_MODEL_OPTIMIZERS", "")
    names = selected.split(",") if selected else []
    if any(not name or name != name.strip() for name in names) or len(names) != len(set(names)):
        raise ValueError("METAL_MODEL_OPTIMIZERS 必须为无重复的逗号分隔名称")
    if set(names) - {"dynamic_quantization", "gpu_ops", "ane", "streaming"}:
        raise ValueError("METAL_MODEL_OPTIMIZERS 包含未知 provider")
    environment = {"METAL_MODEL_OPTIMIZERS": selected, "METAL_QUANTIZATION": mode}
    for name, provider, required in (
        ("METAL_QUANTIZATION_PROFILE", "dynamic_quantization", mode == "dynamic"),
        ("METAL_ANE_PROFILE", "ane", "ane" in names),
        ("METAL_STREAMING_PROFILE", "streaming", "streaming" in names),
    ):
        value = os.environ.get(name, "")
        if value and not Path(value).is_absolute():
            raise ValueError(f"{name} 必须为绝对路径")
        if required and (not value or provider not in names):
            raise ValueError(f"{name} 需要绝对路径及显式选择 {provider} provider")
        if value and name != "METAL_QUANTIZATION_PROFILE" and provider not in names:
            raise ValueError(f"{name} 需要显式选择 {provider} provider")
        environment[name] = value
    return environment


def run(*args, **kwargs):
    subprocess.run(args, check=True, stdout=sys.stderr, timeout=600, **kwargs)


def verify_records(package):
    """校验原 wheel 的发行文件；构建不能通过重写 RECORD 掩盖源码变化。"""
    root = Path(package.locate_file("")).resolve()
    prefix = package.metadata["Name"].replace("-", "_") + "/"
    files = {}
    for entry in package.files or ():
        name = str(entry)
        if not name.startswith(prefix) or name.endswith(".pyc"):
            continue
        path = Path(package.locate_file(entry)).resolve(strict=True)
        if not path.is_relative_to(root) or not entry.hash or entry.hash.mode != "sha256":
            raise ValueError(f"发行文件缺少有效 RECORD 或越界：{name}")
        digest = sha256(path)
        encoded = base64.urlsafe_b64encode(bytes.fromhex(digest)).decode().rstrip("=")
        if entry.hash.value != encoded or path.stat().st_size != entry.size:
            raise ValueError(f"发行文件被修改或损坏：{name}")
        files[name] = digest
    if not files:
        raise ValueError(f"发行包没有受 RECORD 保护的源码：{package.metadata['Name']}")
    return files


def verify_patched_package(name, package=None):
    """同时绑定固定来源、部署补丁、wheel 归档和实际安装文件。"""
    package = package or distribution(name)
    root = Path(package.locate_file("")).resolve()
    path = root.parents[2] / MANIFEST
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_size > 16 * 1024**2:
        raise ValueError("共享优化来源清单必须为当前用户持有的普通文件")
    data = json.loads(path.read_text())
    if data.get("schema") != 2 or set(data.get("packages", {})) != {*PATCHES, EXTENSION}:
        raise ValueError("共享优化来源清单格式不符")
    verify_compatible_sources()
    item = data["packages"][name]
    if name in PATCHES:
        if item["source"] != sources()[name] or item["patch_sha256"] != sha256(PATCHES[name]):
            raise ValueError(f"共享优化固定来源或补丁不符：{name}")
    elif name == EXTENSION:
        if item["source_files"] != extension_sources():
            raise ValueError("共享扩展源码或配置已变化，需要重新构建")
    else:
        raise ValueError(f"未知共享优化包：{name}")
    if data.get("preparer_sha256") != sha256(__file__):
        raise ValueError("共享优化准备器已变化，需要重新构建")
    direct = json.loads(package.read_text("direct_url.json") or "{}")
    if name == EXTENSION:
        if direct.get("dir_info", {}).get("editable") or direct.get("url") != PROJECT.as_uri():
            raise ValueError("共享扩展必须由根 pyproject.toml 非 editable 本地源码安装")
    else:
        if direct.get("dir_info", {}).get("editable") or not direct.get("url", "").startswith("file://"):
            raise ValueError(f"共享优化必须从显式非 editable wheel 安装：{name}")
        if direct.get("archive_info", {}).get("hashes", {}).get("sha256") != item["wheel_sha256"]:
            raise ValueError(f"共享优化 wheel 摘要不符：{name}")
    installed = verify_records(package)
    expected = item["files"]
    if not expected or any(installed.get(key) != value for key, value in expected.items()):
        raise ValueError(f"共享优化源码与构建清单不符：{name}")
    # prepare_native 只能额外添加原生文件，不能悄悄加入未构建的 Python 源码。
    actual_python = {path.relative_to(root).as_posix()
                     for path in (root / name.replace("-", "_")).rglob("*.py")}
    if any(key not in expected for key in actual_python):
        raise ValueError(f"共享优化出现未登记的 Python 模块：{name}")
    return item


def checkout_source(name, target):
    source = sources()[name]
    repository, commit = source["git"], source["rev"]
    checkout = PROJECT.parent / name
    if checkout.is_dir():
        origin = subprocess.check_output(
            ["git", "-C", str(checkout), "remote", "get-url", "origin"], text=True, timeout=10,
        ).strip().removesuffix("/").removesuffix(".git")
        if origin != repository.removesuffix("/").removesuffix(".git"):
            raise ValueError(f"相邻源码不是已确认来源：{name}")
        archive = subprocess.check_output(
            ["git", "-C", str(checkout), "archive", commit], timeout=60,
        )
        with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
            stream.extractall(target, filter="data")
    else:
        run("git", "init", "-q", str(target))
        run("git", "-C", str(target), "fetch", "-q", "--depth", "1", repository, commit)
        run("git", "-C", str(target), "checkout", "-q", "--detach", "FETCH_HEAD")
    run("git", "-C", str(target), "apply", "--check", str(PATCHES[name]))
    run("git", "-C", str(target), "apply", str(PATCHES[name]))


def build_wheel(name, target, wheels):
    checkout_source(name, target)
    run(sys.executable, "-I", "-m", "pip", "wheel", "--no-deps", "--no-build-isolation",
        str(target), "--wheel-dir", str(wheels))
    candidates = list(wheels.glob(name.replace("-", "_") + "-*.whl"))
    if len(candidates) != 1:
        raise ValueError(f"预期唯一共享优化 wheel：{name}")
    wheel = candidates[0]
    with zipfile.ZipFile(wheel) as archive:
        files = {}
        prefix = name.replace("-", "_") + "/"
        for entry in archive.namelist():
            if not entry.startswith(prefix) or entry.endswith("/"):
                continue
            path = target / entry
            digest = hashlib.sha256(archive.read(entry)).hexdigest()
            if not path.is_file() or sha256(path) != digest:
                raise ValueError(f"wheel 与固定补丁源码不符：{entry}")
            files[entry] = digest
        if not {str(path.relative_to(target)) for path in (target / prefix).rglob("*.py")} <= set(files):
            raise ValueError(f"共享优化 wheel 遗漏 Python 模块：{name}")
    return wheel, {"wheel_sha256": sha256(wheel), "files": files,
                   "source": sources()[name], "patch_sha256": sha256(PATCHES[name])}


def prepare():
    # 此入口只在根准备器新建且尚未激活的环境调用；拒绝修改当前共享环境。
    active = Path.home() / ".local/share/local-model-backend/current"
    if active.exists() and active.resolve() == Path(sys.prefix).resolve():
        raise ValueError("拒绝在 current 使用中的共享环境安装优化，请通过根准备器重建")
    verify_compatible_sources()
    for name in PATCHES:
        package = distribution(name)
        if not Path(package.locate_file("")).resolve().is_relative_to(Path(sys.prefix).resolve()):
            raise ValueError(f"原始源码包不属于本次隔离环境：{name}")
        verify_official_source(name, package)
    extension = distribution(EXTENSION)
    if not Path(extension.locate_file("")).resolve().is_relative_to(Path(sys.prefix).resolve()):
        raise ValueError("扩展必须由根项目安装到本次隔离环境")
    direct = json.loads(extension.read_text("direct_url.json") or "{}")
    if direct.get("dir_info", {}).get("editable") or direct.get("url") != PROJECT.as_uri():
        raise ValueError("扩展必须由根 pyproject.toml 非 editable 本地源码安装")
    extension_files = verify_records(extension)
    expected = {
        f"vllm_metal_extensions/{name}": digest
        for name, digest in extension_sources().items()
        if name.endswith(".py")
    }
    if extension_files != expected:
        raise ValueError("uv 安装的扩展与本地源码不符")
    manifest = Path(sys.prefix) / MANIFEST
    manifest.unlink(missing_ok=True)
    with tempfile.TemporaryDirectory(prefix="shared-metal-wheels-") as temporary:
        root = Path(temporary)
        wheels = []
        data = {EXTENSION: {"source_files": extension_sources(), "files": extension_files}}
        for name in PATCHES:
            target = root / name
            target.mkdir()
            wheel, item = build_wheel(name, target, root / "wheels")
            wheels.append(wheel)
            data[name] = item
        # 扩展已由 uv 锁定安装；此处只安装 Metal hook 补丁 wheel。
        run(sys.executable, "-I", "-m", "pip", "install", "--no-deps", "--force-reinstall",
            *(str(path) for path in wheels))
        manifest.write_text(json.dumps({"schema": 2, "preparer_sha256": sha256(__file__),
                                        "packages": data}, sort_keys=True, indent=2) + "\n")
        for name in (*PATCHES, EXTENSION):
            verify_patched_package(name)


if __name__ == "__main__":
    try:
        prepare()
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        print(f"共享优化构建或来源检查失败：{error}", file=sys.stderr)
        sys.exit(1)
