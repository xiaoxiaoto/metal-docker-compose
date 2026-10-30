#!/usr/bin/env python3
"""发现、选择并下载魔搭 MOSS 模型；只解析缓存元数据，不执行它。"""

import json
import os
from pathlib import Path
import pickletools
import re
import subprocess
import sys

REPO = "openmoss/MOSS-Transcribe-Diarize"


def repository(value):
    if not isinstance(value, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", value,
    ):
        raise ValueError(f"请输入魔搭仓库名：作者/模型，例如 {REPO}")
    return value


def prompt(message):
    print(message, end="", file=sys.stderr, flush=True)
    try:
        return input().strip()
    except EOFError:
        raise EOFError("需要输入但 stdin 已关闭，请在终端运行 ./start.sh") from None


def discover_models(cache):
    models = []
    for directory in sorted(cache.glob("*/*")):
        if (directory.parent.name.startswith(".") or directory.name.startswith(".")
                or not (directory / "config.json").is_file()):
            continue
        try:
            config = json.loads((directory / "config.json").read_text())
            if not isinstance(config, dict) or config.get("model_type") != "moss_transcribe_diarize":
                continue
            validate_model(directory)
            metadata = directory / ".mdl"
            if metadata.exists():
                with metadata.open("rb") as file:
                    data = file.read(65537)
                if len(data) > 65536:
                    raise ValueError("缓存元数据过大")
                # .mdl 是 pickle；仅提取字符串，禁止反序列化执行。
                strings = [value for op, value, _ in pickletools.genops(data)
                           if op.name in ("UNICODE", "BINUNICODE", "SHORT_BINUNICODE", "BINUNICODE8")]
                repo = repository(strings[strings.index("id") + 1])
            else:
                repo = repository(f"{directory.parent.name}/{directory.name.replace('___', '.')}")
            models.append((repo, directory))
        except (ValueError, KeyError, TypeError, IndexError, OSError, EOFError) as error:
            print(f"忽略不完整或无效的缓存 {directory}：{error}", file=sys.stderr)
    return models


def select_model(cache):
    models = discover_models(cache)
    if len(models) == 1:
        repo, directory = models[0]
        print(f"自动选择唯一兼容模型：{repo}", file=sys.stderr)
        return repo, directory, False
    if models:
        print("发现多个兼容模型：", file=sys.stderr)
        for number, (repo, directory) in enumerate(models, 1):
            print(f"  {number}. {repo} — {directory}", file=sys.stderr)
        while True:
            choice = prompt(f"请选择模型编号 [1-{len(models)}]：")
            if choice.isascii() and choice.isdigit() and 1 <= int(choice) <= len(models):
                repo, directory = models[int(choice) - 1]
                return repo, directory, False
            print("编号无效，请重新输入。", file=sys.stderr)
    print("未发现完整的 MOSS 兼容模型，将从魔搭下载。", file=sys.stderr)
    while True:
        try:
            repo = repository(prompt("请输入魔搭仓库名（作者/模型）："))
            return repo, cache / repo, True
        except ValueError as error:
            print(error, file=sys.stderr)


def validate_model(directory):
    directory = Path(directory).expanduser().resolve()
    config = json.loads((directory / "config.json").read_text())
    if (config.get("model_type") != "moss_transcribe_diarize"
            or config.get("architectures") != ["MossTranscribeDiarizeForConditionalGeneration"]):
        raise ValueError("Not a MOSS-Transcribe-Diarize checkpoint")
    for part in (config, config.get("text_config", {})):
        if part.get("quantization") is not None or part.get("quantization_config") is not None:
            raise ValueError("Quantized MOSS checkpoints are not supported")
    for name in ("tokenizer.json", "tokenizer_config.json", "preprocessor_config.json",
                 "processor_config.json", "configuration_moss_transcribe_diarize.py",
                 "processing_moss_transcribe_diarize.py", "modeling_moss_transcribe_diarize.py"):
        if not (directory / name).is_file() or (directory / name).stat().st_size == 0:
            raise ValueError(f"Missing model asset: {name}")
    index = json.loads((directory / "model.safetensors.index.json").read_text())
    weights = index.get("weight_map")
    if not isinstance(weights, dict) or not weights:
        raise ValueError("Invalid safetensors weight index")
    from safetensors import safe_open
    for name in set(weights.values()):
        if not isinstance(name, str) or not name.endswith(".safetensors"):
            raise ValueError("Invalid weight filename")
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe weight filename: {name}")
        path = (directory / relative).resolve()
        # HF snapshots symlink to this repository's blobs; never allow other targets.
        blobs = directory.parent.parent / "blobs" if directory.parent.name == "snapshots" else directory
        if not path.is_file() or not (path.is_relative_to(directory) or path.is_relative_to(blobs)):
            raise ValueError(f"Missing or unsafe weight path: {name}")
        with safe_open(str(path), framework="np") as shard:
            missing = {key for key, filename in weights.items() if filename == name} - set(shard.keys())
            if missing:
                raise ValueError(f"Missing tensors in {name}: {sorted(missing)[:3]}")
    return directory


def prepare():
    explicit = os.environ.get("MOSS_MODEL_DIR")
    if explicit:
        return REPO.rsplit("/", 1)[-1].lower(), validate_model(explicit)
    cache = Path.home() / ".cache/modelscope/hub/models"
    repo, directory, download = select_model(cache)
    if download:
        if directory.exists():
            raise ValueError(f"已有缓存无效，不覆盖或删除：{directory}；请先检查该目录")
        subprocess.run([
            "modelscope", "download", "--model", repo, "--revision", "master",
            "--local_dir", str(directory),
        ], check=True, stdout=sys.stderr)
    return repo.rsplit("/", 1)[-1].lower(), validate_model(directory)


if __name__ == "__main__":
    try:
        name, directory = prepare()
        if "\n" in str(directory) or "\r" in str(directory):
            raise ValueError("Model directory cannot contain newlines")
        print(name)
        print(directory)
    except Exception as error:
        print(f"模型准备失败：{error}", file=sys.stderr)
        sys.exit(1)
