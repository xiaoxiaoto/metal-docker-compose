#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""构建 Higgs 独占、正常 wheel 所有的 Omni overlay，不改共享安装。"""

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

PROJECT = Path(__file__).resolve().parent
MODULE = PROJECT / "higgs_metal_quantization.py"
BACKEND = PROJECT.parents[1] / "vllm-omni-metal"


def backend_sources():
    """严格沿后端自己的wheel资源声明选源，不复制旧build/editable产物。"""
    config = tomllib.loads((BACKEND / "pyproject.toml").read_text())
    package = BACKEND / "vllm_omni_metal"
    paths = {path for path in package.rglob("*.py") if "__pycache__" not in path.parts}
    for pattern in config["tool"]["setuptools"]["package-data"]["vllm_omni_metal"]:
        paths.update(path for path in package.glob(pattern) if path.is_file())
    paths.update(BACKEND / name for name in ("pyproject.toml", "README.md", "LICENSE"))
    if not paths or any(not path.is_file() for path in paths):
        raise ValueError("当前相邻后端源码/资源不完整，不能构建Higgs独占后端")
    return sorted(paths)


def install_backend(python):
    with tempfile.TemporaryDirectory(prefix="higgs-metal-wheel-") as temporary:
        checkout = Path(temporary) / "source"
        checkout.mkdir()
        for source in backend_sources():
            target = checkout / source.relative_to(BACKEND)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        wheels = Path(temporary) / "wheels"
        run(
            str(python),
            "-I",
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--wheel-dir",
            str(wheels),
            str(checkout),
        )
        (wheel,) = wheels.glob("vllm_omni_metal-*.whl")
        run(
            str(python), "-I", "-m", "pip", "install", "--no-deps", "--ignore-installed", str(wheel)
        )


def validate_backend(python):
    expected = {
        str(path.relative_to(BACKEND)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in backend_sources()
        if path.is_relative_to(BACKEND / "vllm_omni_metal")
    }
    run(
        str(python),
        "-I",
        "-c",
        """
import base64
import hashlib
import json
import sys
from pathlib import Path
from importlib.metadata import distribution
import vllm_omni_metal
from vllm_omni_metal.quantization.linear import QuantizedLinear, validate_profile
from vllm_omni_metal.quantization.capture import CaptureSession
from vllm_omni_metal.quantization.adapter import DenseModelAdapter
from vllm_omni_metal.quantization.runtime import load_profile, configure_model_quantization
from vllm_omni_metal.native import runner
dist = distribution('vllm-omni-metal')
root = Path(sys.prefix).resolve()
assert Path(vllm_omni_metal.__file__).resolve().is_relative_to(root)
records = {str(entry): entry for entry in dist.files or ()}
for name, expected in json.loads(sys.argv[1]).items():
    entry = records.get(name)
    assert entry is not None, f'后端资源不属于私有wheel RECORD: {name}'
    path = Path(dist.locate_file(entry)).resolve()
    assert path.is_relative_to(root), f'后端资源逃逸overlay: {name}'
    digest = hashlib.sha256(path.read_bytes()).digest()
    assert digest.hex() == expected, f'后端资源源码不匹配: {name}'
    assert entry.hash is not None and entry.hash.mode == 'sha256'
    assert entry.hash.value == base64.urlsafe_b64encode(digest).decode().rstrip('=')
    assert entry.size == path.stat().st_size
licenses = [entry for entry in dist.files or () if str(entry).endswith('/licenses/LICENSE')]
assert len(licenses) == 1, '后端wheel必须包含许可证'
license_path = Path(dist.locate_file(licenses[0])).resolve()
assert license_path.is_relative_to(root)
assert hashlib.sha256(license_path.read_bytes()).hexdigest() == sys.argv[2]
license_digest = hashlib.sha256(license_path.read_bytes()).digest()
assert licenses[0].hash is not None and licenses[0].hash.mode == 'sha256'
assert licenses[0].hash.value == base64.urlsafe_b64encode(license_digest).decode().rstrip('=')
assert licenses[0].size == license_path.stat().st_size
""",
        json.dumps(expected, sort_keys=True),
        hashlib.sha256((BACKEND / "LICENSE").read_bytes()).hexdigest(),
    )


def run(*args, **kwargs):
    subprocess.run(args, check=True, stdout=sys.stderr, **kwargs)


def fingerprint(commit, base):
    digest = hashlib.sha256((commit + str(base) + sys.version).encode())
    for file in (
        Path(__file__),
        MODULE,
        PROJECT / "prepare_quantization.py",
        PROJECT / "higgs_joint.py",
        PROJECT / "prepare_joint.py",
    ):
        digest.update(file.read_bytes())
    for package in ("vllm", "vllm-omni", "vllm-omni-metal", "torch", "mlx", "vllm-metal"):
        dist = distribution(package)
        digest.update((package + dist.version + str(dist.locate_file(""))).encode())
    for file in backend_sources():
        digest.update(str(file.relative_to(BACKEND)).encode())
        digest.update(file.read_bytes())
    return digest.hexdigest()[:16]


def package_private(checkout):
    config = checkout / "pyproject.toml"
    text = config.read_text()
    setuptools = tomllib.loads(text)["tool"]["setuptools"]
    if "py-modules" in setuptools or text.count("[tool.setuptools]\n") != 1:
        raise ValueError("上游打包合同变化，不能安全声明私有模块")
    config.write_text(
        text.replace(
            "[tool.setuptools]\n",
            '[tool.setuptools]\npy-modules = ["higgs_metal_quantization", "higgs_quantization_preparation", "higgs_joint", "higgs_joint_preparation"]\n',
            1,
        )
    )
    shutil.copyfile(MODULE, checkout / MODULE.name)
    shutil.copyfile(
        PROJECT / "prepare_quantization.py", checkout / "higgs_quantization_preparation.py"
    )
    shutil.copyfile(PROJECT / "higgs_joint.py", checkout / "higgs_joint.py")
    shutil.copyfile(PROJECT / "prepare_joint.py", checkout / "higgs_joint_preparation.py")
    source = checkout / "vllm_omni/model_executor/models/higgs_audio_v3/higgs_audio_v3_code2wav.py"
    text = source.read_text()
    # 两个真实 loader 均先完成 codec 安装，再运行独立加载钩子。
    marker = "        self._loaded = True\n"
    if (
        text.count(marker) != 2
        or text.count("self.fc2.weight.dtype") != 1
        or text.count("self.fc2.weight.device") != 1
    ):
        raise ValueError("上游 Code2Wav 加载/执行合同变化，拒绝生成不完整适配")
    text = text.replace(
        marker,
        marker
        + "        from higgs_metal_quantization import finish_load\n        finish_load(self)\n",
    )
    text = text.replace("self.fc2.weight.dtype", "self._metal_fc2_dtype")
    text = text.replace("self.fc2.weight.device", "self._metal_fc2_device")
    source.write_text(text)
    talker = checkout / "vllm_omni/model_executor/models/higgs_audio_v3/higgs_audio_v3_talker.py"
    text = talker.read_text()
    marker = "        return loaded_params\n"
    if text.count(marker) != 1:
        raise ValueError("上游Talker完成加载合同变化，拒绝私有hook")
    talker.write_text(
        text.replace(
            marker,
            "        from higgs_joint import finish_load as finish_joint_load\n"
            "        finish_joint_load(self)\n" + marker,
            1,
        )
    )


def prepare():
    run(sys.executable, "-I", str(PROJECT.parent / "check_versions.py"), "--omni")
    source = json.loads(distribution("vllm-omni").read_text("direct_url.json"))
    if source.get("url") != "https://github.com/vllm-project/vllm-omni.git":
        raise ValueError("仅允许从所选基础环境的官方 Omni 精确 commit 构建")
    commit = source["vcs_info"]["commit_id"]
    if not re.fullmatch("[0-9a-f]{40}", commit):
        raise ValueError("无效 Omni commit")
    base = Path(sysconfig.get_path("purelib")).resolve()
    selected = fingerprint(commit, base)
    root = Path.home() / ".local/share/local-model-backend/overlays/higgs-tts-3"
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    if root.is_symlink() or root.stat().st_uid != os.getuid():
        raise ValueError("Higgs overlay 必须是当前用户拥有的目录")
    runtime = root / selected
    if runtime.is_symlink() or (runtime.exists() and runtime.stat().st_uid != os.getuid()):
        raise ValueError("Higgs runtime 路径所有权无效")
    ready, python = runtime / ".ready", runtime / "bin/python"
    complete = ready.is_file() and ready.read_text() == selected and python.is_file()
    ready.unlink(missing_ok=True)
    if not complete:
        venv.create(runtime, with_pip=True, symlinks=True)
        site = (
            runtime
            / "lib"
            / f"python{sys.version_info.major}.{sys.version_info.minor}"
            / "site-packages"
        )
        # 明确复用此基础环境依赖；不声称完整重新安装所有依赖。
        (site / "omni-metal-base.pth").write_text(str(base) + "\n")
        # 旧冻结基础环境可继续提供依赖，但不能提供本次ops/native实现。
        install_backend(python)
        with tempfile.TemporaryDirectory(prefix="higgs-omni-") as temporary:
            checkout = Path(temporary)
            run("git", "-C", temporary, "init", "-q")
            run(
                "git",
                "-C",
                temporary,
                "fetch",
                "-q",
                "--depth",
                "1",
                "https://github.com/vllm-project/vllm-omni.git",
                commit,
            )
            run("git", "-C", temporary, "checkout", "-q", "--detach", "FETCH_HEAD")
            package_private(checkout)
            run(
                str(python),
                "-I",
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--ignore-installed",
                temporary,
                env=dict(
                    os.environ,
                    VLLM_OMNI_TARGET_DEVICE="cpu",
                    VLLM_OMNI_VERSION_OVERRIDE="0.32.0.dev0+cpu.higgsdynamicv2.higgsjointv1",
                ),
            )
    run(str(python), "-I", "-m", "pip", "check")
    validate_backend(python)
    run(
        str(python),
        "-I",
        "-c",
        "from pathlib import Path; import sys,hashlib,base64,vllm_omni,higgs_metal_quantization as m; "
        "from importlib.metadata import distribution; "
        "from vllm_omni_metal.quantization.linear import QuantizedLinear,validate_profile; "
        "p=Path(m.__file__).resolve(); "
        "assert p.is_relative_to(Path(sys.prefix).resolve()); "
        "assert Path(vllm_omni.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()); "
        "entries=[f for f in distribution('vllm-omni').files or () "
        "if Path(distribution('vllm-omni').locate_file(f)).resolve()==p]; "
        "assert len(entries)==1; entry=entries[0]; "
        "assert entry.hash is not None and entry.hash.mode=='sha256'; "
        "assert entry.hash.value==base64.urlsafe_b64encode(hashlib.sha256(p.read_bytes()).digest()).decode().rstrip('='); "
        "assert entry.size==p.stat().st_size; "
        "assert hashlib.sha256(p.read_bytes()).hexdigest()==sys.argv[1]",
        hashlib.sha256(MODULE.read_bytes()).hexdigest(),
    )
    run(
        str(python),
        "-I",
        "-c",
        "import importlib.util,hashlib,sys,base64; from pathlib import Path; "
        "from importlib.metadata import distribution; "
        "p=Path(importlib.util.find_spec('higgs_quantization_preparation').origin).resolve(); "
        "assert p.is_relative_to(Path(sys.prefix).resolve()); "
        "entries=[f for f in distribution('vllm-omni').files or () "
        "if Path(distribution('vllm-omni').locate_file(f)).resolve()==p]; "
        "assert len(entries)==1; e=entries[0]; "
        "assert e.size==p.stat().st_size and e.hash.mode=='sha256'; "
        "assert e.hash.value==base64.urlsafe_b64encode(hashlib.sha256(p.read_bytes()).digest()).decode().rstrip('='); "
        "assert hashlib.sha256(p.read_bytes()).hexdigest()==sys.argv[1]",
        hashlib.sha256((PROJECT / "prepare_quantization.py").read_bytes()).hexdigest(),
    )
    for name, source in (
        ("higgs_joint", PROJECT / "higgs_joint.py"),
        ("higgs_joint_preparation", PROJECT / "prepare_joint.py"),
    ):
        run(
            str(python),
            "-I",
            "-c",
            "import importlib.util,hashlib,sys,base64; from pathlib import Path; "
            "from importlib.metadata import distribution; "
            "p=Path(importlib.util.find_spec(sys.argv[1]).origin).resolve(); "
            "assert p.is_relative_to(Path(sys.prefix).resolve()); "
            "d=distribution('vllm-omni'); entries=[e for e in d.files or () "
            "if Path(d.locate_file(e)).resolve()==p]; assert len(entries)==1; e=entries[0]; "
            "digest=hashlib.sha256(p.read_bytes()).digest(); "
            "assert e.size==p.stat().st_size and e.hash.mode=='sha256'; "
            "assert e.hash.value==base64.urlsafe_b64encode(digest).decode().rstrip('='); "
            "assert digest.hex()==sys.argv[2]",
            name,
            hashlib.sha256(source.read_bytes()).hexdigest(),
        )
    run(
        str(python),
        "-I",
        "-c",
        """
import base64
import hashlib
import sys
from pathlib import Path
from importlib.metadata import distribution
dist = distribution('vllm-omni')
root = Path(sys.prefix).resolve()
records = {str(entry): entry for entry in dist.files or ()}
for filename, marker, count in (
    ('higgs_audio_v3_talker.py', '        finish_joint_load(self)\\n', 1),
    ('higgs_audio_v3_code2wav.py', '        finish_load(self)\\n', 2),
):
    name = 'vllm_omni/model_executor/models/higgs_audio_v3/' + filename
    entry = records[name]
    path = Path(dist.locate_file(entry)).resolve()
    assert path.is_relative_to(root)
    digest = hashlib.sha256(path.read_bytes()).digest()
    assert entry.hash is not None and entry.hash.mode == 'sha256'
    assert entry.hash.value == base64.urlsafe_b64encode(digest).decode().rstrip('=')
    assert entry.size == path.stat().st_size
    assert path.read_text().count(marker) == count, 'private load hook missing'
""",
    )
    run(
        str(python),
        "-I",
        "-c",
        "from vllm_omni_metal.ops.mps_attention import get_mps_ops,build_mps_launcher; "
        "build_mps_launcher() if get_mps_ops() is None else None",
    )
    run(
        str(python),
        "-I",
        "-c",
        "from vllm_omni_metal.ops.mps_attention import get_mps_ops; "
        "assert get_mps_ops() is not None, 'MPS attention launcher未准备'",
    )
    if fingerprint(commit, base) != selected:
        raise RuntimeError("准备期间输入改变，未发布ready")
    ready.write_text(selected)
    return runtime


if __name__ == "__main__":
    try:
        print(prepare())
    except Exception as error:
        print(f"Higgs 私有环境准备失败：{error}", file=sys.stderr)
        sys.exit(1)
