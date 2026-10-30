#!/usr/bin/env python3
"""生成兼容模型视图与固定来源的私有像素插件 wheel；不修改共享环境或模型缓存。"""

import argparse
import base64
import configparser
import csv
from email.parser import BytesParser
import fcntl
import hashlib
import importlib
from importlib.metadata import distribution, entry_points
import io
import json
import math
import os
from pathlib import Path
import platform
import runpy
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import tomllib
import urllib.request
import zipfile

sys.dont_write_bytecode = True
PROJECT = Path(__file__).resolve().parents[2]
ADAPTERS = PROJECT / "extensions/vllm/wevisone_vl"
DIST = "wevisone-vl-metal"
PACKAGE = "wevisone_vl_metal"
VERSION = "0.1.0"
COMMIT = "9a6bcd85404505df0d05cfb5d97512ccacf35dee"
RAW_SOURCE = f"https://raw.githubusercontent.com/Tencent/WeVisOne-VL/{COMMIT}"
UPSTREAM = {
    "LICENSE": (11560, "834de5e92ef0bd5da04fd6f2f455ff35ba82104ae250a0bf13542396b64ccce8"),
    "config.py": (1345, "e3ef5729adffd8d18267a9657b0ad802c1ab0fa85cf8511a3549e3b8cbc884b3"),
    "decoder.py": (1025, "3114b5196cefc82c95fbf99b15b551758e6cdb7adcb6fe2d21b89109f14c1696"),
    "mask_decoder.py": (28791, "ce6b637cabcabf7339d92195df0df6df342bb61cdafbdcdc7d241b2a4ab7b37a"),
    "msda.py": (6161, "46ef2b5e24396a2ae052892504a247cefa059a7dbc5a5632f2b602173fc1c608"),
}
PINS = {
    "vllm": "0.32.0+cpu", "vllm-metal": "0.32.0",
    "vllm-metal-extensions": "0.1.0", "mlx": "0.32.2", "mlx-metal": "0.32.2",
    "mlx-vlm": "0.7.6", "mlx-lm": "0.32.0", "torch": "2.13.0",
    "torchvision": "0.28.0", "transformers": "5.19.0",
    "numpy": "2.3.5", "pillow": "13.0.0", "safetensors": "0.8.0",
}
OVERLAYS = Path.home() / ".local/share/local-model-backend/overlays/wevisone-vl-4b"
ENTRY_POINT = f"{PACKAGE}.endpoint:PixelEndpointPlugin"
WORKER_CLASS = f"{PACKAGE}.worker.PixelWorkerExtension"

RUNTIME = Path.home() / "Library/Application Support/wevisone-vl-4b/model"
LINKED_FILES = (
    "generation_config.json",
    "processor_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "chat_template.jinja",
    "model.safetensors",
    "mask_decoder.pt",
)


def compatible_config(config):
    """生成 mlx_vlm 0.7.x 可加载的等价配置；只改字段名/模型类型标签，不改数值语义。"""
    text = config.get("text_config") if isinstance(config, dict) else None
    if not isinstance(text, dict):
        raise ValueError("官方 config.json 缺少 text_config")
    if "rope_theta" in text or "rope_scaling" in text:
        raise ValueError("官方 config.json 已包含旧式 RoPE 字段，不再需要运行时适配")
    rope = text.get("rope_parameters")
    if (not isinstance(rope, dict)
            or set(rope) != {"rope_theta", "rope_type", "mrope_section", "mrope_interleaved"}):
        raise ValueError("官方 config.json 的 rope_parameters 结构不符")
    theta = rope["rope_theta"]
    section = rope["mrope_section"]
    head_dim = text.get("head_dim")
    if (type(theta) not in (int, float) or not math.isfinite(theta) or theta <= 0
            or rope["rope_type"] != "default" or rope["mrope_interleaved"] is not True):
        raise ValueError("官方 config.json 的 RoPE 参数不支持 mlx_vlm 适配")
    if (not isinstance(section, list) or len(section) != 3
            or any(type(value) is not int or value <= 0 for value in section)
            or type(head_dim) is not int or sum(section) != head_dim // 2):
        raise ValueError("官方 config.json 的 mrope_section 与 head_dim 不符")
    vision = config.get("vision_config")
    if not isinstance(vision, dict) or vision.get("model_type") != "qwen3_vl_vision":
        raise ValueError("官方 config.json 的 vision_config.model_type 不符")
    text["rope_theta"] = theta
    text["rope_scaling"] = {"type": "default", "mrope_section": section}
    vision["model_type"] = "qwen3_vl"


def write_config(runtime, config):
    with tempfile.NamedTemporaryFile("w", dir=runtime, prefix=".config.", suffix=".tmp",
                                     delete=False, encoding="utf-8") as file:
        temporary = Path(file.name)
        try:
            json.dump(config, file, ensure_ascii=False, indent=2)
            file.write("\n")
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, runtime / "config.json")


def prepare_runtime(source, runtime=RUNTIME):
    source = Path(source).expanduser().resolve()
    runtime = Path(runtime).expanduser()
    if runtime.is_symlink():
        raise ValueError("运行时模型目录不能是符号链接")
    runtime.mkdir(parents=True, exist_ok=True)
    runtime = runtime.resolve()
    if not runtime.is_dir():
        raise ValueError("运行时模型路径不是普通目录")
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    compatible_config(config)
    sidecar = source / "mask_decoder.pt"
    require(not sidecar.is_symlink() and sidecar.is_file()
            and sidecar.stat().st_size == 38534741
            and sha256(sidecar) == "8528460481bd09d0780c53559b08194c7abbd29dff0d51783b619158acbe7b46",
            "mask_decoder.pt 不符合固定 ModelScope 发布，拒绝建立运行时链接")
    for name in LINKED_FILES:
        target = source / name
        if target.is_symlink() or not target.is_file():
            raise ValueError(f"缓存文件缺失或路径无效：{name}")
        temporary = runtime / f".{name}.tmp"
        temporary.unlink(missing_ok=True)
        os.symlink(target, temporary)
        os.replace(temporary, runtime / name)
    write_config(runtime, config)
    return runtime


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(*args):
    # 所有构建输出留在 stderr，stdout 仅用于返回模型/overlay 路径。
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1",
                       VLLM_PLUGINS="metal,wevisone_vl_metal",
                       PYTORCH_ENABLE_MPS_FALLBACK="0")
    environment.pop("PYTHONPATH", None)
    # 不允许调用者 pip 配置把私有 addon 重定向安装进共享或其他环境。
    for name in ("PIP_TARGET", "PIP_PREFIX", "PIP_USER"):
        environment.pop(name, None)
    environment["PIP_CONFIG_FILE"] = os.devnull
    subprocess.run([str(arg) for arg in args], check=True, stdout=sys.stderr,
                   env=environment)


def adapter_sources():
    require(ADAPTERS.is_dir() and not ADAPTERS.is_symlink(), "缺少私有适配源码目录")
    sources = {}
    for path in sorted(ADAPTERS.rglob("*.py")):
        require(not path.is_symlink(), f"私有适配源码不能是符号链接：{path}")
        sources[path.relative_to(ADAPTERS).as_posix()] = sha256(path)
    require(set(sources) == {"__init__.py", "worker.py", "pixel.py", "endpoint.py",
                             "tasks.py", "render.py"},
            "私有适配源码必须完整且仅包含 __init__/worker/pixel/endpoint/tasks/render.py")
    return sources


def base_identity():
    """核对当前选定基底的固定源码、非 editable 安装和 RECORD，不执行模型。"""
    require(sys.platform == "darwin" and platform.machine() == "arm64"
            and sys.version_info[:2] == (3, 12),
            "私有插件只支持 macOS arm64 / 共享后端 Python 3.12")
    base = Path(sys.prefix).resolve()
    site = Path(sysconfig.get_path("purelib")).resolve()
    require(site.is_relative_to(base), "共享 Python 的 site-packages 越出其 prefix")
    with (PROJECT / "pyproject.toml").open("rb") as file:
        sources = tomllib.load(file)["tool"]["uv"]["sources"]
    require(sources["vllm"] == {
        "url": "https://github.com/vllm-project/vllm/archive/5df63a7203cdb9328d824a27b682e77c7b9928be.tar.gz"
    } and sources["vllm-metal"] == {
        "git": "https://github.com/vllm-project/vllm-metal.git",
        "rev": "6d6f2c37f8242f1ff1a336534adf6c2d2fe2936d"
    }, "私有插件尚未支持此 vLLM / Metal 固定源码组合")
    verifier = runpy.run_path(str(PROJECT / "common_env/prepare_optimizations.py"))
    packages = {}
    for name, expected in PINS.items():
        package = distribution(name)
        require(package.version == expected,
                f"私有插件仅支持 {name}=={expected}，实际 {package.version}")
        require(Path(package.locate_file("")).resolve() == site,
                f"{name} 不属于选定共享基底：{base}")
        direct = json.loads(package.read_text("direct_url.json") or "{}")
        require(not direct.get("dir_info", {}).get("editable"), f"禁止 editable 基底：{name}")
        record = package.read_text("RECORD")
        require(record, f"基底缺少 wheel RECORD：{name}")
        packages[name] = {
            "version": package.version, "direct_url": direct,
            "record_sha256": hashlib.sha256(record.encode()).hexdigest(),
        }
    direct = packages["vllm"]["direct_url"]
    require(direct.get("url") == sources["vllm"]["url"] and "archive_info" in direct,
            "vLLM 必须由根配置固定的官方源码归档非 editable 构建")
    direct = packages["mlx-lm"]["direct_url"]
    require(direct.get("url", "").removesuffix(".git") == "https://github.com/ml-explore/mlx-lm"
            and direct.get("vcs_info", {}).get("commit_id")
            == "9e6acca691e64d6d8bb808c328fcdea459099cca",
            "mlx-lm 必须来自已锁定的官方源码 commit")
    for name in ("vllm-metal", "vllm-metal-extensions"):
        verifier["verify_patched_package"](name)
    for name in ("vllm", "mlx-vlm", "mlx-lm"):
        # 比较实际代码/原生产物与既有 RECORD，不能重算 RECORD 掩盖修改。
        verifier["verify_records"](distribution(name))
    return {"prefix": str(base), "site": str(site), "python": sys.version,
            "packages": packages}


def plugin_inputs(identity):
    return {
        "distribution": DIST, "version": VERSION, "commit": COMMIT,
        "upstream": UPSTREAM, "adapters": adapter_sources(), "base": identity,
        "preparer_sha256": sha256(__file__),
    }


def fingerprint(inputs):
    return hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()[:24]


def build_source(build, inputs):
    """临时生成发行元数据；不复制根元数据，不导入官方完整 engine。"""
    package = build / PACKAGE
    upstream = package / "upstream"
    upstream.mkdir(parents=True)
    for name, expected in inputs["adapters"].items():
        content = (ADAPTERS / name).read_bytes()
        require(hashlib.sha256(content).hexdigest() == expected,
                f"构建期间适配源码变化：{name}")
        (package / name).write_bytes(content)
    (upstream / "__init__.py").write_text("")
    for name, (size, expected) in UPSTREAM.items():
        relative = name if name == "LICENSE" else f"wevisone_vl/{name}"
        with urllib.request.urlopen(f"{RAW_SOURCE}/{relative}", timeout=60) as response:
            require(response.status == 200, f"固定官方源码下载失败：{relative}")
            content = response.read(size + 1)
        require(len(content) == size and hashlib.sha256(content).hexdigest() == expected,
                f"固定官方源码摘要/大小不符：{relative}@{COMMIT}")
        (upstream / name).write_bytes(content)
        if name == "LICENSE":
            (build / name).write_bytes(content)
    dependencies = [f"{name}=={version}" for name, version in PINS.items()]
    (build / "pyproject.toml").write_text(
        '[build-system]\nrequires = ["setuptools>=77,<81"]\n'
        'build-backend = "setuptools.build_meta"\n\n'
        f'[project]\nname = "{DIST}"\nversion = "{VERSION}"\n'
        'requires-python = ">=3.12,<3.13"\nlicense = "Apache-2.0"\n'
        'license-files = ["LICENSE"]\n'
        f'dependencies = {json.dumps(dependencies)}\n\n'
        '[project.entry-points."vllm.endpoint_plugins"]\n'
        f'{PACKAGE} = "{ENTRY_POINT}"\n\n'
        '[tool.setuptools]\n'
        f'packages = ["{PACKAGE}", "{PACKAGE}.upstream"]\n'
        'include-package-data = false\n\n'
        f'[tool.setuptools.package-data]\n"{PACKAGE}.upstream" = ["LICENSE"]\n'
    )
    return {path.relative_to(build).as_posix(): sha256(path)
            for path in package.rglob("*") if path.is_file()}


def package_files(inputs):
    files = {f"{PACKAGE}/{name}": value for name, value in inputs["adapters"].items()}
    files.update({f"{PACKAGE}/upstream/{name}": value[1]
                  for name, value in UPSTREAM.items()})
    files[f"{PACKAGE}/upstream/__init__.py"] = hashlib.sha256(b"").hexdigest()
    return files


def launcher_source(runtime):
    return (f"#!{runtime / 'bin/python'}\nimport sys\n"
            "from vllm.entrypoints.cli.main import main\n"
            "if __name__ == '__main__':\n    sys.exit(main())\n")


def verify_wheel(wheel, expected=None):
    """同时校验完整 wheel RECORD 和源码内容，保留归档供重复验收。"""
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)), "wheel 含重复文件名")
        require(all(not Path(name).is_absolute() and ".." not in Path(name).parts
                    and not name.endswith("/") for name in names), "wheel 路径不安全")
        dist_info = f"wevisone_vl_metal-{VERSION}.dist-info/"
        record = dist_info + "RECORD"
        require(record in names, "wheel 缺少 RECORD")
        records = list(csv.reader(io.StringIO(archive.read(record).decode())))
        require(len(records) == len(names) and {row[0] for row in records} == set(names),
                "wheel RECORD 文件集合不完整")
        result = {}
        for name, encoded, size in records:
            content = archive.read(name)
            if name == record:
                require(encoded == size == "", "wheel RECORD 自身不应含摘要")
                continue
            value = hashlib.sha256(content).digest()
            require(encoded == "sha256=" + base64.urlsafe_b64encode(value).decode().rstrip("=")
                    and size == str(len(content)), f"wheel RECORD 内容不符：{name}")
            result[name] = value.hex()
        package_files = {name: value for name, value in result.items()
                         if name.startswith(PACKAGE + "/")}
        require(package_files and (expected is None or package_files == expected),
                "wheel 私有适配/官方解码器与构建输入不一致")
        require(all(name.startswith((PACKAGE + "/", dist_info)) for name in names),
                "私有 wheel 不得包含共享核心或其他发行包")
        license_path = dist_info + "licenses/LICENSE"
        require(result.get(license_path) == UPSTREAM["LICENSE"][1],
                "wheel 缺少完整官方 LICENSE")
        metadata = BytesParser().parsebytes(archive.read(dist_info + "METADATA"))
        require(metadata["Name"] == DIST and metadata["Version"] == VERSION
                and set(metadata.get_all("Requires-Dist", []))
                == {f"{name}=={version}" for name, version in PINS.items()},
                "私有 wheel 元数据或兼容版本 pins 不符")
        plugins = configparser.ConfigParser()
        plugins.read_string(archive.read(dist_info + "entry_points.txt").decode())
        require(plugins.sections() == ["vllm.endpoint_plugins"]
                and dict(plugins["vllm.endpoint_plugins"]) == {PACKAGE: ENTRY_POINT},
                "私有 wheel 必须仅提供原生像素 endpoint entry point")
        return result


def check_installed(runtime):
    """由 overlay Python 执行；只导入接口，不加载 backbone 或 decoder 权重。"""
    runtime = Path(runtime).resolve()
    require(Path(sys.prefix).resolve() == runtime, "验收必须使用 overlay 自有 Python")
    manifest = json.loads((runtime / "source-manifest.json").read_text())
    inputs = manifest["inputs"]
    identity = inputs["base"]
    site = runtime / "lib/python3.12/site-packages"
    require((site / "metal-base.pth").read_text() == identity["site"] + "\n",
            "overlay .pth 未指向选定共享基底")
    for name, expected in PINS.items():
        package = distribution(name)
        require(package.version == expected
                and str(Path(package.locate_file("")).resolve()) == identity["site"],
                f"overlay 核心依赖被复制或替换：{name}")
        require(hashlib.sha256(package.read_text("RECORD").encode()).hexdigest()
                == identity["packages"][name]["record_sha256"],
                f"共享基底 RECORD 已变化：{name}")
    wheel = runtime / "wheels" / manifest["wheel"]
    require(sha256(wheel) == manifest["wheel_sha256"], "私有 wheel 归档摘要变化")
    require(verify_wheel(wheel, package_files(inputs)) == manifest["files"],
            "私有 wheel 内容与固定适配/官方解码器清单不符")
    package = distribution(DIST)
    require(package.version == VERSION and Path(package.locate_file("")).resolve() == site,
            "私有插件未安装到 overlay")
    direct = json.loads(package.read_text("direct_url.json") or "{}")
    require(not direct.get("dir_info", {}).get("editable")
            and direct.get("url") == wheel.as_uri()
            and direct.get("archive_info", {}).get("hashes", {}).get("sha256")
            == manifest["wheel_sha256"], "私有插件必须由已验证 wheel 非 editable 安装")
    entries = {str(entry): entry for entry in package.files or ()}
    for name, expected in manifest["files"].items():
        path = Path(package.locate_file(name))
        require(not path.is_symlink() and path.resolve().is_relative_to(site)
                and path.is_file() and sha256(path) == expected,
                f"私有安装内容与 wheel 不一致：{name}")
        entry = entries.get(name)
        require(entry is not None and entry.hash is not None and entry.hash.mode == "sha256"
                and entry.hash.value == base64.urlsafe_b64encode(bytes.fromhex(expected)).decode().rstrip("=")
                and entry.size == path.stat().st_size, f"私有安装 RECORD 不符：{name}")
    actual = {path.relative_to(site).as_posix() for path in (site / PACKAGE).rglob("*")
              if path.is_file() and "__pycache__" not in path.parts}
    expected = {name for name in manifest["files"] if name.startswith(PACKAGE + "/")}
    require(actual == expected, "私有安装含未登记文件或缺少源码/许可")
    for module in ("vllm", "vllm_metal", "mlx_vlm", "torch"):
        loaded = importlib.import_module(module)
        require(Path(loaded.__file__).resolve().is_relative_to(Path(identity["site"])),
                f"{module} 未继承选定共享基底")
    matches = [entry for entry in entry_points(group="vllm.endpoint_plugins")
               if entry.name == PACKAGE]
    require(len(matches) == 1 and matches[0].value == ENTRY_POINT
            and isinstance(matches[0].load(), type), "像素 EndpointPlugin entry point 无效")
    from vllm.utils.import_utils import resolve_obj_by_qualname
    worker = resolve_obj_by_qualname(WORKER_CLASS)
    require(isinstance(worker, type) and callable(worker.wevisone_initialize)
            and callable(worker.wevisone_decode), "像素 WorkerExtension 点号路径或 RPC 接口无效")
    require("wevisone_vl.engine" not in sys.modules, "禁止导入官方完整推理 engine")
    for module in ("endpoint", "worker", "pixel", "tasks", "render", "upstream.config",
                   "upstream.decoder", "upstream.mask_decoder", "upstream.msda"):
        loaded = importlib.import_module(f"{PACKAGE}.{module}")
        require(Path(loaded.__file__).resolve().is_relative_to(site / PACKAGE),
                f"私有模块未从 overlay 加载：{module}")


def install_plugin(runtime_root=OVERLAYS, validate_only=False):
    identity = base_identity()
    inputs = plugin_inputs(identity)
    key = fingerprint(inputs)
    parent = Path(runtime_root).expanduser()
    require(not parent.is_symlink(), "overlay 根目录不能是符号链接")
    parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    parent = parent.resolve()
    require(not parent.is_relative_to(Path(identity["prefix"])), "禁止在共享基底内构建 overlay")
    runtime = parent / key
    require(not runtime.is_symlink(), "overlay 环境不能是符号链接")
    with (parent / ".prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ready = runtime / ".ready"
        complete = ready.is_file() and ready.read_text() == key
        if validate_only:
            require(complete, f"未找到当前源码/基底对应的已安装 overlay：{runtime}")
        elif not complete:
            require(not runtime.exists(), f"未完成的 overlay 路径已存在，拒绝覆盖：{runtime}")
            try:
                # 不执行 ensurepip、不安装核心；pip/setuptools 继承已校验基底。
                run(sys.executable, "-I", "-m", "venv", "--without-pip", str(runtime))
                site = runtime / "lib/python3.12/site-packages"
                (site / "metal-base.pth").write_text(identity["site"] + "\n")
                wheels = runtime / "wheels"
                wheels.mkdir()
                with tempfile.TemporaryDirectory(prefix="wevisone-metal-wheel-") as temporary:
                    build = Path(temporary)
                    expected = build_source(build, inputs)
                    run(sys.executable, "-I", "-m", "pip", "wheel", "--no-deps",
                        "--no-build-isolation", "--no-cache-dir", str(build),
                        "--wheel-dir", str(wheels))
                    candidates = list(wheels.glob("*.whl"))
                    require(len(candidates) == 1, "私有插件构建没有生成唯一 wheel")
                    wheel = candidates[0]
                    files = verify_wheel(wheel, expected)
                manifest = {"inputs": inputs, "fingerprint": key, "wheel": wheel.name,
                            "wheel_sha256": sha256(wheel), "files": files}
                (runtime / "source-manifest.json").write_text(
                    json.dumps(manifest, sort_keys=True, indent=2) + "\n")
                run(runtime / "bin/python", "-I", "-m", "pip", "install", "--no-deps",
                    "--prefix", str(runtime), "--ignore-installed", str(wheel))
                launcher = runtime / "bin/vllm"
                launcher.write_text(launcher_source(runtime))
                launcher.chmod(0o755)
                validate_plugin(runtime, inputs)
                ready.write_text(key)
            except BaseException:
                # 只删除本次新建且尚未发布的环境，不修改或清理任何共享基底。
                if runtime.is_dir() and not runtime.is_symlink():
                    shutil.rmtree(runtime)
                raise
            return runtime
        validate_plugin(runtime, inputs)
        return runtime


def validate_plugin(runtime, inputs):
    manifest = json.loads((runtime / "source-manifest.json").read_text())
    # JSON 将固定 SHA 表的 tuple 转为 list；比较规范 JSON，避免误判。
    require(json.dumps(manifest["inputs"], sort_keys=True) == json.dumps(inputs, sort_keys=True)
            and manifest["fingerprint"] == fingerprint(inputs), "私有安装指纹/基底不一致")
    require(os.access(runtime / "bin/vllm", os.X_OK), "overlay 缺少可执行 vllm launcher")
    launcher = runtime / "bin/vllm"
    require(not launcher.is_symlink() and launcher.read_text() == launcher_source(runtime),
            "overlay vllm launcher 内容不符或不是普通文件")
    run(runtime / "bin/python", "-I", "-m", "pip", "check")
    run(runtime / "bin/python", "-I", __file__, "--check-installed", runtime)
    require(plugin_inputs(base_identity()) == inputs, "准备期间基底或适配源码变化；拒绝发布")


def prepare_plugin(base_python, runtime_root=OVERLAYS, validate=False):
    """独立返回私有环境；prepare() 的模型名/模型路径两行协议保持不变。"""
    python = Path(base_python).expanduser().absolute()
    require(python.is_file() and os.access(python, os.X_OK), "选定共享 Python 不可执行")
    args = [str(python), "-I", str(Path(__file__).resolve()),
            "--runtime-root", str(Path(runtime_root).expanduser())]
    if validate:
        args.append("--validate")
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    environment.pop("PYTHONPATH", None)
    output = subprocess.check_output(args, text=True, env=environment)
    lines = output.splitlines()
    require(len(lines) == 1 and Path(lines[0]).is_absolute(), "私有准备器输出路径协议无效")
    return Path(lines[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, default=OVERLAYS,
                        help="仅将私有 overlay 放在此目录；不调用共享准备器")
    parser.add_argument("--validate", action="store_true", help="只验收匹配的已安装 overlay")
    parser.add_argument("--check-installed", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        if args.check_installed:
            check_installed(args.check_installed)
        else:
            print(install_plugin(args.runtime_root, args.validate))
    except (ValueError, KeyError, TypeError, OSError, ImportError,
            subprocess.SubprocessError, zipfile.BadZipFile) as error:
        print(f"WeVisOne 私有像素插件准备失败：{error}", file=sys.stderr)
        sys.exit(1)
