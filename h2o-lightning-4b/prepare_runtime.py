#!/usr/bin/env python3
"""将固定官方 Metal 源码补丁构建为 H2O 私有 wheel；不修改共享环境。"""

import base64
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig
import tarfile
import tempfile
import venv
import zipfile
from importlib.metadata import distribution, version

PROJECT = Path(__file__).resolve().parent
SOURCE = PROJECT.parent.parent / "vllm-metal"
REPOSITORY = "https://github.com/vllm-project/vllm-metal.git"
COMMIT = "6d6f2c37f8242f1ff1a336534adf6c2d2fe2936d"
METAL_VERSION = "0.32.0"
MLX_VERSION = "0.32.2"
PATCH = PROJECT / "patches/vllm-metal-bidirectional.patch"


def run(*args, **kwargs):
    subprocess.run(args, check=True, stdout=sys.stderr, **kwargs)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def installed_resources():
    """只复用已校验发行 wheel 的原生资源，不在启动/请求期编译。"""
    for name, expected in (("vllm-metal", METAL_VERSION), ("mlx", MLX_VERSION),
                           ("mlx-metal", MLX_VERSION), ("mlx-vlm", "0.7.6")):
        if version(name) != expected:
            raise ValueError(f"补丁只验证 {name}=={expected}，实际为 {version(name)}")
    package = distribution("vllm-metal")
    source = json.loads(package.read_text("direct_url.json") or "{}")
    if (
        source.get("url") != REPOSITORY
        or source.get("vcs_info", {}).get("commit_id") != COMMIT
        or source.get("dir_info", {}).get("editable")
    ):
        raise ValueError("需要共享后端从固定官方 v0.32.0 commit 构建的非 editable Metal")
    resources = {}
    for entry in package.files or ():
        relative = Path(str(entry))
        if not relative.parts or relative.parts[0] != "vllm_metal":
            continue
        if relative.suffix == ".pyc" or "__pycache__" in relative.parts:
            continue
        path = Path(package.locate_file(entry)).resolve()
        if not entry.hash or entry.hash.mode != "sha256" or not path.is_file():
            raise ValueError(f"官方 Metal RECORD 缺少有效 SHA256：{entry}")
        digest = sha256(path)
        encoded = base64.urlsafe_b64encode(bytes.fromhex(digest)).decode().rstrip("=")
        if entry.hash.value != encoded or path.stat().st_size != entry.size:
            raise ValueError(f"共享 Metal 发行文件被修改或损坏：{entry}")
        resources[str(relative)] = (path, digest)
    native = {name: data for name, data in resources.items() if
              Path(name).suffix in (".so", ".metallib") or
              Path(name).name in ("_paged_ops.mlx-version", "_mps_ops.torch-version")}
    if not any(Path(name).name.startswith("_paged_ops.") and name.endswith(".so") for name in native):
        raise ValueError("官方 wheel 缺少预构建 paged attention 扩展")
    if not any(name.endswith(".metallib") for name in native):
        raise ValueError("官方 wheel 缺少预构建 Metal shader libraries")
    return resources, native


def checkout_source(destination):
    """从固定 commit 导出干净副本；不覆盖或修改相邻源码工作区。"""
    if SOURCE.is_dir():
        origin = subprocess.check_output(
            ["git", "-C", str(SOURCE), "remote", "get-url", "origin"], text=True,
        ).strip().removesuffix("/")
        if origin.removesuffix(".git") != REPOSITORY.removesuffix(".git"):
            raise ValueError(f"相邻 vllm-metal 不是已确认的官方来源：{origin}")
        archive = subprocess.check_output(["git", "-C", str(SOURCE), "archive", COMMIT])
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            tar.extractall(destination, filter="data")
    else:
        run("git", "init", "-q", str(destination))
        run("git", "-C", str(destination), "fetch", "-q", "--depth", "1", REPOSITORY, COMMIT)
        run("git", "-C", str(destination), "checkout", "-q", "--detach", "FETCH_HEAD")
    run("git", "-C", str(destination), "apply", "--check", str(PATCH))
    run("git", "-C", str(destination), "apply", str(PATCH))


def hydrate_native(checkout, resources, native):
    # 补丁只修改 Python。原生源码必须与发行扩展的源码逐字一致，不能混入旧二进制。
    for name, (_path, expected) in resources.items():
        if Path(name).suffix not in (".cpp", ".h", ".metal", ".mm"):
            continue
        path = checkout / name
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"原生源码与发行 wheel 不匹配，需要独立重建而不是复用：{name}")
    for name, (path, _digest) in native.items():
        target = checkout / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)


def source_manifest(checkout, wheel):
    with zipfile.ZipFile(wheel) as archive:
        entries = {name for name in archive.namelist()
                   if name.startswith("vllm_metal/") and not name.endswith("/")}
        sources = {str(path.relative_to(checkout))
                   for path in (checkout / "vllm_metal").rglob("*.py")}
        if not sources.issubset(entries):
            raise ValueError("构建的 wheel 缺少 Metal Python 模块")
        result = {}
        for name in sorted(entries):
            path = checkout / name
            if not path.is_file() or hashlib.sha256(archive.read(name)).hexdigest() != sha256(path):
                raise ValueError(f"wheel 内容与补丁源码/预构建资源不符：{name}")
            result[name] = sha256(path)
        return result


def runtime_fingerprint(base_site, resources):
    digest = hashlib.sha256(
        COMMIT.encode() + PATCH.read_bytes() + Path(__file__).read_bytes()
        + str(base_site).encode() + sys.version.encode()
    )
    for name, (_path, file_hash) in sorted(resources.items()):
        digest.update(name.encode() + file_hash.encode())
    return digest.hexdigest()[:16]


def validate(runtime, manifest, base_site):
    python = str(runtime / "bin/python")
    environment = dict(os.environ)
    # 此变量只描述共享基础环境；overlay 分别校验继承的 core 与私有插件归属。
    environment.pop("VLLM_BACKEND_PYTHON", None)
    run(python, "-I", str(PROJECT.parent / "check_versions.py"), "--omni", env=environment)
    run(python, "-I", "-m", "pip", "check", env=environment)
    code = r'''
import base64,hashlib,json,sys
from pathlib import Path
from importlib.metadata import distribution
import vllm,vllm_metal
root=Path(sys.prefix).resolve()
base=Path(sys.argv[2]).resolve()
assert Path(vllm.__file__).resolve().is_relative_to(base), 'vLLM core escaped selected shared base'
assert Path(vllm_metal.__file__).resolve().is_relative_to(root), 'Metal is not installed in private overlay'
d=distribution('vllm-metal')
assert not json.loads(d.read_text('direct_url.json') or '{}').get('dir_info',{}).get('editable'), 'editable Metal forbidden'
expected=json.loads(Path(sys.argv[1]).read_text())
entries={str(p):p for p in d.files or ()}
for name,wanted in expected.items():
    p=Path(d.locate_file(name)).resolve()
    assert p.is_relative_to(root), name
    value=hashlib.sha256(p.read_bytes()).hexdigest()
    assert value==wanted, f'installed patched source/resource mismatch: {name}'
    entry=entries[name]
    assert entry.hash is not None and entry.hash.mode=='sha256', name
    assert entry.hash.value==base64.urlsafe_b64encode(bytes.fromhex(value)).decode().rstrip('='), name
    assert entry.size==p.stat().st_size, name
from vllm_metal.metal import get_ops
assert get_ops() is not None, 'prebuilt Metal paged attention not loadable'
'''
    run(python, "-I", "-c", code, str(manifest), str(base_site), env=environment)


def prepare():
    if sys.version_info[:2] != (3, 12) or sys.platform != "darwin":
        raise ValueError("补丁运行时需要共享后端的 macOS arm64 Python 3.12")
    run(sys.executable, "-I", str(PROJECT.parent / "check_versions.py"), "--omni")
    if not PATCH.is_file():
        raise ValueError(f"缺少已验证的 Metal 补丁：{PATCH}")
    resources, native = installed_resources()
    base_site = Path(sysconfig.get_path("purelib")).resolve()
    fingerprint = runtime_fingerprint(base_site, resources)
    parent = Path.home() / ".local/share/local-model-backend/overlays/h2o-lightning-4b"
    parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (parent / ".prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        runtime = parent / fingerprint
        ready = runtime / ".ready"
        manifest = runtime / "source-manifest.json"
        complete = (ready.is_file() and ready.read_text() == fingerprint and manifest.is_file()
                    and os.access(runtime / "bin/vllm", os.X_OK))
        ready.unlink(missing_ok=True)
        if not complete:
            print(f"构建 H2O 私有补丁式 vLLM Metal：{runtime}", file=sys.stderr)
            venv.create(runtime, with_pip=True, symlinks=True)
            site = runtime / "lib/python3.12/site-packages"
            (site / "metal-base.pth").write_text(str(base_site) + "\n")
            with tempfile.TemporaryDirectory(prefix="h2o-metal-wheel-") as temporary:
                build = Path(temporary) / "source"
                build.mkdir()
                checkout_source(build)
                hydrate_native(build, resources, native)
                wheels = Path(temporary) / "wheels"
                run(sys.executable, "-I", "-m", "pip", "wheel", "--no-deps", "--no-build-isolation",
                    str(build), "--wheel-dir", str(wheels))
                (wheel,) = wheels.glob("vllm_metal-*.whl")
                expected = source_manifest(build, wheel)
                run(str(runtime / "bin/python"), "-I", "-m", "pip", "install", "--no-deps",
                    "--ignore-installed", str(wheel))
                manifest.write_text(json.dumps(expected, sort_keys=True, indent=2) + "\n")
            # core 由 .pth 继承，生成标准 CLI 入口，确保 shebang 使用私有 Python。
            launcher = runtime / "bin/vllm"
            launcher.write_text(
                f"#!{runtime / 'bin/python'}\n"
                "import sys\nfrom vllm.entrypoints.cli.main import main\n"
                "if __name__ == '__main__':\n    sys.exit(main())\n"
            )
            launcher.chmod(0o755)
            (runtime / "backend.json").write_text(json.dumps({
                "source": REPOSITORY, "commit": COMMIT, "patch_sha256": sha256(PATCH),
                "base_site": str(base_site), "fingerprint": fingerprint,
            }, indent=2) + "\n")
        validate(runtime, manifest, base_site)
        if runtime_fingerprint(base_site, installed_resources()[0]) != fingerprint:
            raise ValueError("构建期间补丁或共享发行资源变化；不发布 ready，请重试")
        ready.write_text(fingerprint)
        return runtime


if __name__ == "__main__":
    try:
        print(prepare())
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        print(f"H2O 私有 Metal 准备失败：{error}", file=sys.stderr)
        sys.exit(1)
