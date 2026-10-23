#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""为 SeedVR2 构建私有非 editable Omni wheel；此脚本不启动服务。"""

import hashlib
import json
import os
import subprocess
import sys
import sysconfig
import tempfile
import venv
from importlib.metadata import distribution
from pathlib import Path

ASSETS = {
    "quantization.py": "quantization.py",
    "prepare.py": "weight_manifest.py",
    "prepare_quantization.py": "calibration.py",
    "joint.py": "joint.py",
    "joint_calibration.py": "joint_calibration.py",
}
BASE_COMMIT = "9534efe802ef79ce52af31847fb144f63d259e90"
OMNI_URL = "https://github.com/vllm-project/vllm-omni.git"
VERSION = "0.32.0.dev0+cpu.seedvr2mps2"
BACKEND_SOURCE = Path(__file__).resolve().parents[2] / "vllm-omni-metal"


def backend_manifest():
    files = [BACKEND_SOURCE / name for name in ("pyproject.toml", "README.md", "LICENSE")]
    files += sorted(
        path
        for path in (BACKEND_SOURCE / "vllm_omni_metal").rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
        and path.name != ".DS_Store"
    )
    return {
        str(path.relative_to(BACKEND_SOURCE)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in files
    }


def install_backend(python, manifest):
    # 当前后端源码构建为本模型私有 wheel，不修改共享冻结环境。
    with tempfile.TemporaryDirectory(prefix="seedvr2-metal-wheel-") as temporary:
        build = Path(temporary) / "source"
        for relative, expected in manifest.items():
            target = build / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            content = (BACKEND_SOURCE / relative).read_bytes()
            if hashlib.sha256(content).hexdigest() != expected:
                raise ValueError("后端源码在私有 wheel 构建期间改变")
            target.write_bytes(content)
        wheels = Path(temporary) / "wheels"
        run(
            str(python),
            "-I",
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            str(build),
            "--wheel-dir",
            str(wheels),
        )
        (wheel,) = wheels.glob("vllm_omni_metal-*.whl")
        run(
            str(python), "-I", "-m", "pip", "install", "--no-deps", "--ignore-installed", str(wheel)
        )


def run(*args, **kwargs):
    subprocess.run(args, check=True, stdout=sys.stderr, **kwargs)


def base_source():
    backend = os.environ.get("VLLM_BACKEND_PYTHON")
    if (
        not backend
        or Path(sys.prefix).resolve() != Path(backend).absolute().parent.parent.resolve()
    ):
        raise ValueError("请使用共享 VLLM_BACKEND_PYTHON 调用 prepare_runtime.py")
    package = distribution("vllm-omni")
    source = json.loads(package.read_text("direct_url.json") or "null")
    if not isinstance(source, dict) or source.get("url") != OMNI_URL:
        raise ValueError("SeedVR2 必须从共享后端的官方 Omni 源码构建")
    if source.get("dir_info", {}).get("editable"):
        raise ValueError("SeedVR2 不接受 editable Omni 基础环境")
    commit = source.get("vcs_info", {}).get("commit_id")
    if commit != BASE_COMMIT:
        raise ValueError(f"SeedVR2 补丁仅适用于 Omni {BASE_COMMIT}，实际 {commit}；请重新适配补丁")
    return Path(sysconfig.get_path("purelib")).resolve()


def validate_runtime(runtime, provenance):
    python = runtime / "bin/python"
    if json.loads((runtime / ".seedvr2-source.json").read_text()) != provenance:
        raise ValueError("SeedVR2 私有 wheel 来源指纹不一致")
    run(str(python), "-I", "-m", "pip", "check")
    run(
        str(python),
        "-I",
        "-c",
        "\n".join(
            (
                "from pathlib import Path",
                "from importlib.metadata import distribution",
                "import json, sys, vllm_omni",
                "from packaging.version import Version",
                "root = Path(sys.prefix).resolve()",
                "assert Path(vllm_omni.__file__).resolve().is_relative_to(root), 'Omni 未从私有 overlay 加载'",
                "package = distribution('vllm-omni')",
                f"assert package.version == {VERSION!r}",
                "assert Version(package.version).public == '0.32.0.dev0'",
                "assert Path(package.locate_file('')).resolve().is_relative_to(root), 'Omni 元数据未来自私有 wheel'",
                "source = json.loads(package.read_text('direct_url.json'))",
                "assert source['url'].startswith('file:'), '私有 Omni 不是源码 wheel'",
                "assert not source.get('dir_info', {}).get('editable'), '禁止 editable 安装'",
                "assert (root / 'bin/vllm-omni').is_file(), '缺少私有 Omni CLI'",
                "from vllm_omni.diffusion.models.seedvr2.comfy import COMFY_VARIANT",
                "from vllm_omni.diffusion.models.seedvr2.nadit import SEEDVR2_7B_CONFIG",
                "assert COMFY_VARIANT == '7b-comfy-fp16' and SEEDVR2_7B_CONFIG['mm_layers'] == 36",
                "from vllm_omni.diffusion.models.seedvr2 import quantization, weight_manifest, calibration, joint, joint_calibration",
                "import hashlib",
                f"assets = {provenance['assets']!r}",
                "for name, expected in assets.items():",
                "    module = {'quantization.py': quantization, 'weight_manifest.py': weight_manifest, 'calibration.py': calibration, 'joint.py': joint, 'joint_calibration.py': joint_calibration}[name]",
                "    path = Path(module.__file__).resolve()",
                "    assert path.is_relative_to(root), '动态量化文件未从私有 wheel 加载'",
                "    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, '动态量化文件指纹不符'",
                "assert quantization.ADAPTER_VERSION == 'seedvr2-dynamic-v2'",
                "import base64, csv, io",
                "import vllm_omni_metal",
                "from vllm_omni_metal.quantization.linear import QuantizedLinear, build_profile, validate_profile, apply_profile",
                "from vllm_omni_metal.quantization.engine import DynamicEngine",
                "from vllm_omni_metal.quantization.capture import CaptureSession",
                "from vllm_omni_metal.quantization.adapter import DenseModelAdapter",
                "from vllm_omni_metal.quantization.runtime import load_profile, configure_model_quantization",
                "backend = distribution('vllm-omni-metal')",
                "assert Path(vllm_omni_metal.__file__).resolve().is_relative_to(root), '后端必须来自本模型私有非editable wheel'",
                "assert Path(backend.locate_file('')).resolve().is_relative_to(root)",
                "assert not json.loads(backend.read_text('direct_url.json')).get('dir_info', {}).get('editable')",
                f"expected_backend = {provenance['backend_source']!r}",
                "for relative, expected in expected_backend.items():",
                "    if relative.startswith('vllm_omni_metal/'):",
                "        path = Path(backend.locate_file(relative))",
                "        assert path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == expected, f'后端随包资源缺失或改变: {relative}'",
                "licenses = [file for file in backend.files if str(file).endswith('.dist-info/licenses/LICENSE')]",
                "assert len(licenses) == 1 and hashlib.sha256(Path(backend.locate_file(licenses[0])).read_bytes()).hexdigest() == expected_backend['LICENSE']",
                "for installed in (backend, package):",
                "    for relative, encoded, size in csv.reader(io.StringIO(installed.read_text('RECORD'))):",
                "        if not encoded:",
                "            continue",
                "        algorithm, expected = encoded.split('=', 1)",
                "        content = Path(installed.locate_file(relative)).read_bytes()",
                "        assert algorithm == 'sha256' and len(content) == int(size)",
                "        assert base64.urlsafe_b64encode(hashlib.sha256(content).digest()).rstrip(b'=').decode() == expected, f'私有wheel RECORD不符: {relative}'",
            )
        ),
    )


def prepare():
    run(
        sys.executable,
        "-I",
        str(Path(__file__).resolve().parents[1] / "check_versions.py"),
        "--omni",
    )
    base_site = base_source()
    patch = Path(__file__).resolve().parent / "patches/seedvr2-7b.patch"
    provenance = {
        "base_commit": BASE_COMMIT,
        "base_site": str(base_site),
        "base_prefix": str(Path(sys.prefix).resolve()),
        "python": sys.version,
        "patch_sha256": hashlib.sha256(patch.read_bytes()).hexdigest(),
        "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "assets": {
            target: hashlib.sha256((patch.parent.parent / source).read_bytes()).hexdigest()
            for source, target in ASSETS.items()
        },
        "version": VERSION,
        "backend_source": backend_manifest(),
    }
    fingerprint = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()[:16]
    runtime = Path.home() / ".local/share/local-model-backend/overlays/seedvr2" / fingerprint
    ready = runtime / ".ready"
    if ready.is_file() and ready.read_text() == fingerprint:
        validate_runtime(runtime, provenance)
        return runtime
    runtime.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    # 失败重试不能沿用旧 ready；只有来源和 pip check 均通过才标记完成。
    ready.unlink(missing_ok=True)
    print(f"构建 SeedVR2 私有 Omni 环境：{runtime}", file=sys.stderr)
    venv.create(runtime, with_pip=True, symlinks=True)
    python = runtime / "bin/python"
    site = (
        runtime
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    (site / "omni-metal-base.pth").write_text(str(base_site) + "\n")
    install_backend(python, provenance["backend_source"])
    with tempfile.TemporaryDirectory(prefix="seedvr2-omni-") as checkout:
        run("git", "-C", checkout, "init", "-q")
        run("git", "-C", checkout, "fetch", "-q", "--depth", "1", OMNI_URL, BASE_COMMIT)
        run("git", "-C", checkout, "checkout", "-q", "--detach", "FETCH_HEAD")
        actual = subprocess.check_output(
            ("git", "-C", checkout, "rev-parse", "HEAD"), text=True
        ).strip()
        if actual != BASE_COMMIT:
            raise ValueError("SeedVR2 Omni checkout commit 与固定基线不一致")
        run("git", "-C", checkout, "apply", "--check", str(patch))
        run("git", "-C", checkout, "apply", str(patch))
        # 模型专属源码随私有 wheel 分发，不在已安装的上游包打补丁。
        model_package = Path(checkout) / "vllm_omni/diffusion/models/seedvr2"
        for source, target in ASSETS.items():
            (model_package / target).write_bytes((patch.parent.parent / source).read_bytes())
        environment = dict(
            os.environ, VLLM_OMNI_TARGET_DEVICE="cpu", VLLM_OMNI_VERSION_OVERRIDE=VERSION
        )
        # pip 从源码构建非 editable wheel，仅写 overlay，不卸载共享基础包。
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
    (runtime / ".seedvr2-source.json").write_text(json.dumps(provenance, sort_keys=True) + "\n")
    validate_runtime(runtime, provenance)
    ready.write_text(fingerprint)
    return runtime


if __name__ == "__main__":
    try:
        print(prepare())
    except Exception as error:
        print(f"SeedVR2 私有 Omni 环境准备失败：{error}", file=sys.stderr)
        sys.exit(1)
