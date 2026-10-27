#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build a private, fingerprinted Omni wheel; never edit the shared installation."""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import venv
from importlib.metadata import distribution
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BACKEND_SOURCE = ROOT.parents[1] / "vllm-omni-metal"
OMNI_COMMIT = "9534efe802ef79ce52af31847fb144f63d259e90"


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
    if not (BACKEND_SOURCE / "vllm_omni_metal/diffusion/ops.py").is_file():
        raise ValueError("相邻后端源码缺少diffusion算子；不升级共享base/current")
    return {
        str(path.relative_to(BACKEND_SOURCE)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in files
    }


def install_backend(python, manifest):
    """旧base只提供依赖；本模型overlay拥有当前非editable后端wheel。"""
    with tempfile.TemporaryDirectory(prefix="qwen-private-backend-") as temporary:
        checkout = Path(temporary) / "source"
        for relative, expected in manifest.items():
            content = (BACKEND_SOURCE / relative).read_bytes()
            if hashlib.sha256(content).hexdigest() != expected:
                raise ValueError("私有wheel构建期间后端源码改变")
            target = checkout / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        wheels = Path(temporary) / "wheels"
        run(
            str(python),
            "-I",
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            str(checkout),
            "--wheel-dir",
            str(wheels),
        )
        (wheel,) = wheels.glob("vllm_omni_metal-*.whl")
        run(
            str(python), "-I", "-m", "pip", "install", "--no-deps", "--ignore-installed", str(wheel)
        )


def validate_backend(python, manifest):
    run(
        str(python),
        "-I",
        "-c",
        "\n".join(
            (
                "import base64,csv,hashlib,io,json,sys",
                "from pathlib import Path",
                "from importlib.metadata import distribution",
                "import vllm_omni_metal",
                "from vllm_omni_metal.diffusion import ops",
                "from vllm_omni_metal.quantization.linear import QuantizedLinear,build_profile,validate_profile,apply_profile",
                "from vllm_omni_metal.quantization.engine import DynamicEngine",
                "from vllm_omni_metal.quantization.capture import CaptureSession",
                "from vllm_omni_metal.quantization.adapter import DenseModelAdapter",
                "from vllm_omni_metal.quantization.runtime import load_profile,configure_model_quantization",
                "root=Path(sys.prefix).resolve()",
                "backend=distribution('vllm-omni-metal')",
                "assert Path(vllm_omni_metal.__file__).resolve().is_relative_to(root), '后端未从私有wheel加载'",
                "assert Path(backend.locate_file('')).resolve().is_relative_to(root)",
                "assert not json.loads(backend.read_text('direct_url.json')).get('dir_info',{}).get('editable')",
                "expected=json.loads(sys.argv[1])",
                "package_files={name for name in expected if name.startswith('vllm_omni_metal/')}",
                "installed_files={str(f) for f in backend.files or [] if str(f).startswith('vllm_omni_metal/') and '__pycache__' not in f.parts and f.suffix!='.pyc'}",
                "assert package_files==installed_files, '私有后端随包源码/native资源集合不匹配'",
                "for relative in package_files:",
                "    path=Path(backend.locate_file(relative)).resolve()",
                "    assert path.is_relative_to(root) and path.is_file()",
                "    assert hashlib.sha256(path.read_bytes()).hexdigest()==expected[relative], f'后端资源SHA不符: {relative}'",
                "licenses=[f for f in backend.files or [] if str(f).endswith('.dist-info/licenses/LICENSE')]",
                "assert len(licenses)==1 and hashlib.sha256(Path(backend.locate_file(licenses[0])).read_bytes()).hexdigest()==expected['LICENSE']",
                "for name in ('vllm-omni-metal','vllm-omni'):",
                "    package=distribution(name)",
                "    assert Path(package.locate_file('')).resolve().is_relative_to(root)",
                "    assert not json.loads(package.read_text('direct_url.json')).get('dir_info',{}).get('editable')",
                "    for relative,encoded,size in csv.reader(io.StringIO(package.read_text('RECORD'))):",
                "        if not encoded:",
                "            continue",
                "        algorithm,digest=encoded.split('=',1)",
                "        path=Path(package.locate_file(relative)).resolve()",
                "        assert path.is_relative_to(root), '私有RECORD路径越界'",
                "        content=path.read_bytes()",
                "        assert algorithm=='sha256' and len(content)==int(size)",
                "        assert base64.urlsafe_b64encode(hashlib.sha256(content).digest()).rstrip(b'=').decode()==digest, f'私有RECORD不符: {relative}'",
            )
        ),
        json.dumps(manifest, sort_keys=True),
    )


def run(*args, **kwargs):
    subprocess.run(args, check=True, stdout=sys.stderr, **kwargs)


def fingerprint(commit, base_site, manifest):
    digest = hashlib.sha256(commit.encode() + str(base_site).encode() + sys.version.encode())
    for path in (
        Path(__file__),
        ROOT / "metal_adapter.py",
        ROOT / "quantization_adapter.py",
        ROOT / "joint_adapter.py",
        ROOT / "prepare_ane.py",
        ROOT / "prepare.py",
        ROOT / "prepare_quantization.py",
        ROOT / "patches/qwen-metal.patch",
    ):
        digest.update(path.read_bytes())
    digest.update(json.dumps(manifest, sort_keys=True).encode())
    return digest.hexdigest()[:16]


def prepare(with_aot_kernels=False):
    run(sys.executable, "-I", str(ROOT.parent / "check_versions.py"), "--omni")
    source = json.loads(distribution("vllm-omni").read_text("direct_url.json"))
    if source.get("url") != "https://github.com/vllm-project/vllm-omni.git":
        raise ValueError("QwenImage 私有运行时必须派生自共享后端官方 Omni 源码")
    commit = source["vcs_info"]["commit_id"]
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("共享 Omni commit 无效")
    if commit != OMNI_COMMIT:
        raise ValueError(
            f"QwenImage 私有补丁要求官方 Omni {OMNI_COMMIT}；"
            "请显式选择匹配的基础环境，不自动更新共享部署"
        )
    base_site = Path(sysconfig.get_path("purelib")).resolve()
    manifest = backend_manifest()
    key = fingerprint(commit, base_site, manifest)
    provenance = {
        "omni_commit": commit,
        "base_site": str(base_site),
        "fingerprint": key,
        "backend_source": manifest,
        "patch_sha256": hashlib.sha256(
            (ROOT / "patches/qwen-metal.patch").read_bytes()
        ).hexdigest(),
    }
    runtime = Path.home() / ".local/share/local-model-backend/overlays/qwen-image-2.1-turbo" / key
    ready = runtime / ".ready"
    python = runtime / "bin/python"
    complete = ready.is_file() and ready.read_text() == key and python.is_file()
    ready.unlink(missing_ok=True)
    if complete and json.loads((runtime / "runtime.json").read_text()) != provenance:
        raise ValueError("Qwen ready来源manifest不匹配，未激活环境")
    if not complete:
        runtime.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        print(f"构建 QwenImage 私有 Omni 环境：{runtime}", file=sys.stderr)
        venv.create(runtime, with_pip=True, symlinks=True)
        site = (
            runtime
            / "lib"
            / f"python{sys.version_info.major}.{sys.version_info.minor}"
            / "site-packages"
        )
        (site / "omni-metal-base.pth").write_text(str(base_site) + "\n")
        install_backend(python, manifest)
        with tempfile.TemporaryDirectory(prefix="qwen-metal-wheel-") as temporary:
            checkout = Path(temporary)
            run("git", "-C", str(checkout), "init", "-q")
            run(
                "git",
                "-C",
                str(checkout),
                "fetch",
                "-q",
                "--depth",
                "1",
                "https://github.com/vllm-project/vllm-omni.git",
                commit,
            )
            run("git", "-C", str(checkout), "checkout", "-q", "--detach", "FETCH_HEAD")
            patch = ROOT / "patches/qwen-metal.patch"
            run("git", "-C", str(checkout), "apply", "--check", str(patch))
            run("git", "-C", str(checkout), "apply", str(patch))
            adapter = checkout / "vllm_omni/diffusion/models/qwen_image_21/metal_adapter.py"
            shutil.copyfile(ROOT / "metal_adapter.py", adapter)
            shutil.copyfile(
                ROOT / "quantization_adapter.py", adapter.with_name("quantization_adapter.py")
            )
            for name in ("joint_adapter.py", "prepare_ane.py", "prepare.py"):
                shutil.copyfile(ROOT / name, adapter.with_name(name))
            shutil.copyfile(
                ROOT / "prepare_quantization.py", adapter.with_name("prepare_quantization.py")
            )
            wheels = checkout / "dist"
            run(
                "uv",
                "build",
                "--quiet",
                "--wheel",
                "--python",
                str(python),
                "--out-dir",
                str(wheels),
                str(checkout),
                env=dict(
                    os.environ,
                    VLLM_OMNI_TARGET_DEVICE="cpu",
                    VLLM_OMNI_VERSION_OVERRIDE="0.32.0.dev0+cpu.qwenmetal1",
                ),
            )
            (wheel,) = wheels.glob("vllm_omni-*.whl")
            run(
                str(python),
                "-I",
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--ignore-installed",
                str(wheel),
            )
    run(str(python), "-I", "-m", "pip", "check")
    run(
        str(python),
        "-I",
        "-c",
        "import sys,hashlib,base64,json; from pathlib import Path; "
        "from importlib.metadata import distribution; import vllm_omni; "
        "from vllm_omni.diffusion.models.qwen_image_21 import metal_adapter,quantization_adapter,joint_adapter,prepare_quantization,prepare_ane,prepare; "
        "assert Path(vllm_omni.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()); "
        "dist=distribution('vllm-omni'); expected=json.loads(sys.argv[1]); "
        "modules=(metal_adapter,quantization_adapter,joint_adapter,prepare_quantization,prepare_ane,prepare); "
        "paths={m.__name__.rsplit('.',1)[-1]+'.py':Path(m.__file__).resolve() for m in modules}; "
        "assert all(p.is_relative_to(Path(sys.prefix).resolve()) for p in paths.values()); "
        "assert all(hashlib.sha256(p.read_bytes()).hexdigest()==expected[n] for n,p in paths.items()); "
        "owners={n:[f for f in dist.files or [] if Path(dist.locate_file(f)).resolve()==p] for n,p in paths.items()}; "
        "assert all(len(entries)==1 for entries in owners.values()), '私有模块不属于Omni RECORD'; "
        "assert all(entries[0].hash is not None and entries[0].hash.mode=='sha256' "
        "and entries[0].hash.value==base64.urlsafe_b64encode(hashlib.sha256(paths[n].read_bytes()).digest()).decode().rstrip('=') "
        "and entries[0].size==paths[n].stat().st_size for n,entries in owners.items()), 'Omni RECORD源码摘要不匹配'; "
        "from vllm_omni_metal.ane.projection import DenseProjection,prepare_projection; "
        "from vllm_omni_metal.ops.mps_attention import get_mps_ops,build_mps_launcher; "
        "build_mps_launcher() if get_mps_ops() is None else None",
        json.dumps(
            {
                name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                for name in (
                    "metal_adapter.py",
                    "quantization_adapter.py",
                    "prepare_quantization.py",
                    "joint_adapter.py",
                    "prepare_ane.py",
                    "prepare.py",
                )
            }
        ),
    )
    validate_backend(python, manifest)
    if with_aot_kernels:
        run(
            str(python),
            "-I",
            "-c",
            "from vllm_omni_metal.ops.aot_kernels import build_elementwise_kernels,build_diffusion_kernels; build_elementwise_kernels(); build_diffusion_kernels()",
        )
    if fingerprint(commit, base_site, backend_manifest()) != key:
        raise ValueError("构建期间源文件变化，未激活私有环境")
    (runtime / "runtime.json").write_text(json.dumps(provenance, sort_keys=True, indent=2) + "\n")
    ready.write_text(key)
    return runtime


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--with-aot-kernels",
        action="store_true",
        help="显式构建通用 elementwise 和 diffusion RoPE AOT；需要 Apple Metal compiler",
    )
    args = parser.parse_args()
    try:
        print(prepare(args.with_aot_kernels))
    except Exception as error:
        print(f"QwenImage 私有运行时准备失败：{error}", file=sys.stderr)
        sys.exit(1)
