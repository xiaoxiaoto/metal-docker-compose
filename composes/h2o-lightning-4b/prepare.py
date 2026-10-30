"""离线校验手动下载的官方 H2O 发布；plist 由 start.sh 内联生成。"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

PROJECT = Path(__file__).resolve().parent
RELEASE = json.loads((PROJECT / "model-release.json").read_text())
REPOSITORY = RELEASE["repository"]
MODEL_NAME = REPOSITORY.rsplit("/", 1)[-1].lower()


def model_directory():
    hf_home = Path(os.environ.get(
        "HF_HOME", str(Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))) / "huggingface"),
    )).expanduser()
    cache = Path(os.environ.get("HF_HUB_CACHE", str(hf_home / "hub"))).expanduser()
    snapshot = cache / "models--h2oai--h2o-lightning-4b" / "snapshots" / RELEASE["revision"]
    directory = Path(os.environ.get("H2O_MODEL_DIR", str(snapshot))).expanduser().resolve()
    if "\n" in str(directory) or "\r" in str(directory):
        raise ValueError("模型目录不能包含换行")
    return directory


def hf_snapshot(directory):
    return (directory.name == RELEASE["revision"] and directory.parent.name == "snapshots"
            and directory.parent.parent.name == "models--h2oai--h2o-lightning-4b")


def validate_model(directory):
    directory = Path(directory).expanduser().resolve()
    # hf download 默认快照通过相对链接指向同仓库 blobs；不允许跨仓库或任意外链。
    blobs = None
    if hf_snapshot(directory):
        blobs = (directory.parent.parent / "blobs").resolve()
        if blobs != directory.parent.parent / "blobs":
            raise ValueError("HF blobs 目录不能指向仓库外")
    # vLLM 会扫描根目录权重；拒绝旧 label rows 和可能被当作 pickle 权重的文件。
    weights = {path.name for pattern in ("*.safetensors", "*.bin", "*.pt")
               for path in directory.glob(pattern)}
    if weights != {"model.safetensors"}:
        raise ValueError("需要官方单文件 model.safetensors；不要混入旧 label_rows.safetensors 或其他权重")
    for name, expected in RELEASE["files"].items():
        path = (directory / name).resolve()
        if not path.is_file() or not (path.is_relative_to(directory) or (blobs is not None and path.parent == blobs)):
            raise ValueError(f"发布文件缺失或指向模型目录/同仓库 HF blobs 外：{name}")
        if path.stat().st_size != expected["size"]:
            raise ValueError(f"发布文件大小不符（可能为旧版本、LFS 指针或未下载完整）：{name}")
        # 小文件使用官方 Git blob ID，LFS 文件使用官方 SHA256；流式读取，不装载权重。
        digest = hashlib.sha256() if "sha256" in expected else hashlib.sha1()
        if "git_blob" in expected:
            digest.update(f"blob {expected['size']}\0".encode())
        with path.open("rb") as file:
            for block in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != expected.get("sha256", expected.get("git_blob")):
            raise ValueError(f"发布文件摘要不符，必须使用同一官方 revision：{name}")
    config = json.loads((directory / "config.json").read_text())
    if (config.get("model_type") != "qwen3_5"
            or config.get("architectures") != ["Qwen3_5ForConditionalGeneration"]
            or config.get("head_dtype") != "float32"
            or config.get("text_config", {}).get("head_dtype") != "float32"):
        raise ValueError("需要官方 Qwen3.5 架构及 float32 logits head 配置")
    return directory


def check_model():
    directory = model_directory()
    try:
        return validate_model(directory)
    except (OSError, ValueError) as error:
        raise ValueError(
            f"{error}\n请先手动下载 {REPOSITORY}（权重 {RELEASE['weights_version']}），"
            f"revision={RELEASE['revision']} 到 {directory}；"
            "完整命令见本目录 README.md，也可用 H2O_MODEL_DIR 指定完整目录。"
            "不会下载、覆盖缓存或自动选择旧镜像。"
        ) from error


def prepare():
    directory = check_model()
    name = MODEL_NAME
    python = Path(os.environ["VLLM_BACKEND_PYTHON"])
    if not os.access(python, os.X_OK):
        raise ValueError("统一后端安装不完整，请先运行 common_env/prepare_backend.py")
    subprocess.run([str(python), str(PROJECT.parents[1] / "common_env" / "check_versions.py"), "--omni"],
                   check=True, stdout=sys.stderr)
    subprocess.run([str(python), "-c", "import torch; assert torch.backends.mps.is_available(), 'MPS 不可用'"],
                   check=True, stdout=sys.stderr)
    # 补丁只安装到模型私有 overlay，核心/API/调度仍为同一 vLLM 版本。
    runtime = subprocess.check_output(
        [str(python), "-I", str(PROJECT / "prepare_runtime.py")], text=True,
    ).strip()
    prefix = Path(runtime)
    executable = prefix / "bin/vllm"
    if not prefix.is_absolute() or not os.access(executable, os.X_OK):
        raise ValueError("私有补丁式 Metal 运行时没有可执行的 vllm")
    print(f"模型：{name}\n目录：{directory}", file=sys.stderr)

    # Docker 必须挂载同仓库根目录，保留 snapshots -> ../../blobs 相对链接。
    mount = directory.parent.parent if hf_snapshot(directory) else directory
    return name, directory, executable, prefix, mount, directory.relative_to(mount)


if __name__ == "__main__":
    try:
        if sys.argv[1:] == ["--check-model"]:
            print(check_model())
            sys.exit(0)
        if len(sys.argv) != 1:
            raise ValueError("用法：prepare.py [--check-model]（离线校验模型；默认还校验 VLLM_BACKEND_PYTHON）")
        for value in prepare():
            print(value)
    except (ValueError, KeyError, TypeError, IndexError, OSError, EOFError, subprocess.CalledProcessError) as error:
        print(f"启动准备失败：{error}", file=sys.stderr)
        sys.exit(1)
