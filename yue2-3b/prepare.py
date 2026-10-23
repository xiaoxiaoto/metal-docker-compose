#!/usr/bin/env python3
"""准备 YuE2 AR/NAR 与 FP32 VAE；不加载模型、不执行 remote code。"""

import json
import os
from pathlib import Path
import struct
import subprocess
import sys


REPOS = (("m-a-p/YuE2-3B", "yue2", "YUE2_MODEL_DIR"),
         ("m-a-p/YuE2-Vae", "yue2_vae", "YUE2_VAE_DIR"))


def validate_weights(path):
    size = path.stat().st_size
    with path.open("rb") as file:
        prefix = file.read(8)
        if len(prefix) != 8:
            raise ValueError(f"权重不完整：{path}")
        length = struct.unpack("<Q", prefix)[0]
        if not 2 <= length <= min(100_000_000, size - 8):
            raise ValueError(f"无效 safetensors（可能是 LFS 指针）：{path}")
        header = json.loads(file.read(length))
    if not isinstance(header, dict):
        raise ValueError(f"无效 safetensors header：{path}")
    offsets = [tensor["data_offsets"] for key, tensor in header.items() if key != "__metadata__"]
    if not offsets or any(not 0 <= start < end <= size - 8 - length for start, end in offsets):
        raise ValueError(f"权重截断或 data_offsets 无效：{path}")
    if max(end for _, end in offsets) != size - 8 - length:
        raise ValueError(f"权重大小与 header 不符：{path}")


def validate_model(directory, model_type):
    directory = Path(directory).expanduser().resolve()
    config = json.loads((directory / "config.json").read_text())
    if not isinstance(config, dict) or config.get("model_type") != model_type:
        raise ValueError(f"{directory} 需要 {model_type} 架构")
    if any(config.get(key) is not None for key in ("quantization", "quantization_config")):
        raise ValueError("当前部署只支持未量化权重")
    if model_type == "yue2":
        tokenizer = directory / "qwen.tiktoken"
        if not tokenizer.is_file() or not tokenizer.stat().st_size:
            raise ValueError("缺少 qwen.tiktoken；YuE2 不使用 tokenizer.json")
    else:
        if (config.get("sample_rate"), config.get("audio_channels"), config.get("latent_dim")) != (48000, 2, 64):
            raise ValueError("需要 48 kHz、双声道、64 维 YuE2 VAE")
        if config.get("release_variant") != "standard":
            raise ValueError("本部署使用标准 YuE2-Vae，不使用 legacy decoder")
    index = directory / "model.safetensors.index.json"
    names = set(json.loads(index.read_text())["weight_map"].values()) if index.exists() else {"model.safetensors"}
    if not names:
        raise ValueError("权重索引为空")
    for name in names:
        if not isinstance(name, str) or not name.endswith(".safetensors"):
            raise ValueError("无效权重文件名")
        # HF snapshots use symlinks to their blob cache; only the index's lexical path is constrained.
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("权重索引不能引用目录外文件")
        validate_weights(directory / name)
    if "\n" in str(directory) or "\r" in str(directory):
        raise ValueError("模型目录不能包含换行")
    return directory


def prepare(repo, model_type, variable):
    if os.environ.get(variable):
        return validate_model(os.environ[variable], model_type)
    source = os.environ.get("YUE2_DOWNLOAD_SOURCE", "modelscope")
    if source not in ("modelscope", "hf"):
        raise ValueError("YUE2_DOWNLOAD_SOURCE 必须是 modelscope 或 hf")
    default_cache = Path.home() / (".cache/modelscope/hub/models" if source == "modelscope" else ".cache/yue2-3b")
    cache = Path(os.environ.get("YUE2_CACHE", default_cache)).expanduser()
    # Reuse complete ModelScope caches; never unpickle or remove .mdl/.msc.
    candidates = [cache / repo]
    if "YUE2_CACHE" not in os.environ and source == "hf":
        candidates.insert(0, Path.home() / ".cache/modelscope/hub/models" / repo)
    for directory in candidates:
        try:
            return validate_model(directory, model_type)
        except (OSError, ValueError, KeyError, TypeError):
            pass
    directory = cache / repo
    if not directory.resolve().is_relative_to(cache.resolve()):
        raise ValueError("下载目标不能通过符号链接指向缓存目录外")
    print(f"从 {source} 下载 {repo} 到 {directory}（保留已有缓存）", file=sys.stderr)
    if source == "modelscope":
        subprocess.run([
            "modelscope", "download", "--model", repo,
            "--revision", "master", "--local_dir", str(directory),
        ], check=True, stdout=sys.stderr)
    else:
        from huggingface_hub import snapshot_download

        snapshot_download(repo, local_dir=directory, allow_patterns=[
            "config.json", "qwen.tiktoken", "*.safetensors", "*.safetensors.index.json",
            "*generation_config.json", "LICENSE", "THIRD_PARTY_NOTICES.md",
        ])
    return validate_model(directory, model_type)


if __name__ == "__main__":
    try:
        directories = [prepare(*item) for item in REPOS]
        print(REPOS[0][0].rsplit("/", 1)[-1].lower())
        print("\n".join(map(str, directories)))
    except Exception as error:
        print(f"模型准备失败：{error}", file=sys.stderr)
        sys.exit(1)
