#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""发现/选择原始 Higgs TTS 3 缓存，必要时下载；不加载模型或执行缓存代码。"""

import json
import os
import pickletools
import re
import subprocess
import sys
from pathlib import Path


def repository(value):
    if not isinstance(value, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", value
    ):
        raise ValueError("请输入魔搭仓库名：作者/模型，例如 bosonai/higgs-audio-v3-tts-4b")
    return value


def validate_model(directory):
    directory = Path(directory).resolve()
    config = json.loads((directory / "config.json").read_text())
    if not isinstance(config, dict) or config.get("model_type") != "higgs_multimodal_qwen3":
        raise ValueError("模型不是当前部署兼容的 Higgs TTS 3 架构")
    text = config.get("text_config", {})
    if not isinstance(text, dict):
        raise ValueError("text_config 必须是对象")
    if any(
        value.get(key) is not None
        for value in (config, text)
        for key in ("quantization", "quantization_config")
    ):
        raise ValueError("需要原精度checkpoint；DynamicEngine仅由加载适配器消费已审批profile")
    tokenizer_config = json.loads((directory / "tokenizer_config.json").read_text())
    if not isinstance(tokenizer_config, dict):
        raise ValueError("tokenizer_config.json 必须是对象")
    tokenizer = directory / "tokenizer.json"
    if not tokenizer.is_file() or tokenizer.stat().st_size == 0:
        raise ValueError("缺少 tokenizer.json")
    index = directory / "model.safetensors.index.json"
    if index.exists():
        data = json.loads(index.read_text())
        if not isinstance(data, dict) or not isinstance(data.get("weight_map"), dict):
            raise ValueError("权重索引必须包含 weight_map 对象")
        weights = list(data["weight_map"].values())
    else:
        weights = [path.name for path in directory.glob("*.safetensors")]
    if not weights:
        raise ValueError("没有 safetensors 权重")
    for name in weights:
        if not isinstance(name, str):
            raise ValueError("权重文件名必须是字符串")
        path = (directory / name).resolve()
        if not path.is_relative_to(directory) or not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"模型权重缺失或路径无效：{name}")


def cached_repository(directory):
    metadata = directory / ".mdl"
    if not metadata.exists():
        return repository(f"{directory.parent.name}/{directory.name.replace('___', '.')}")
    with metadata.open("rb") as file:
        data = file.read(65537)
    if len(data) > 65536:
        raise ValueError("缓存元数据过大")
    # ModelScope .mdl is pickle: inspect strings only, never unpickle/execute it.
    strings = [
        value
        for op, value, _ in pickletools.genops(data)
        if op.name in ("UNICODE", "BINUNICODE", "SHORT_BINUNICODE", "BINUNICODE8")
    ]
    return repository(strings[strings.index("id") + 1])


def discover_models(cache):
    models = []
    for directory in sorted(cache.glob("*/*")):
        if directory.parent.name.startswith(".") or directory.name.startswith("."):
            continue
        # Other architectures share this cache; do not offer them as Higgs.
        try:
            config = json.loads((directory / "config.json").read_text())
            if not isinstance(config, dict) or config.get("model_type") != "higgs_multimodal_qwen3":
                continue
            validate_model(directory)
            models.append((cached_repository(directory), directory))
        except (ValueError, TypeError, IndexError, OSError, EOFError) as error:
            print(f"忽略不完整或无效的缓存 {directory}：{error}", file=sys.stderr)
    return models


def prompt(message):
    print(message, end="", file=sys.stderr, flush=True)
    try:
        return input().strip()
    except EOFError:
        raise EOFError(
            "需要输入但 stdin 已关闭，请在终端运行 ./start.sh，或设置 HIGGS_MODEL_DIR"
        ) from None


def select_model(cache):
    explicit = os.environ.get("HIGGS_MODEL_DIR")
    if explicit:
        directory = Path(explicit).expanduser().resolve()
        validate_model(directory)
        repo = cached_repository(directory) if (directory / ".mdl").is_file() else None
        return repo, directory, False
    models = discover_models(cache)
    if len(models) == 1:
        repo, directory = models[0]
        print(f"自动选择唯一兼容模型：{repo}", file=sys.stderr)
        return repo, directory, False
    if models:
        print("发现多个兼容 Higgs 模型：", file=sys.stderr)
        for number, (repo, directory) in enumerate(models, 1):
            print(f"  {number}. {repo} — {directory}", file=sys.stderr)
        while True:
            choice = prompt(f"请选择模型编号 [1-{len(models)}]：")
            if choice.isascii() and choice.isdigit() and 1 <= int(choice) <= len(models):
                repo, directory = models[int(choice) - 1]
                return repo, directory, False
            print("编号无效，请重新输入。", file=sys.stderr)
    print("未发现完整兼容的 Higgs 缓存，将从魔搭下载。", file=sys.stderr)
    while True:
        try:
            repo = repository(prompt("请输入魔搭仓库名（作者/模型）："))
            directory = cache / repo
            if not directory.resolve().is_relative_to(cache.resolve()):
                raise ValueError("下载目标不能通过符号链接指向缓存目录外")
            return repo, directory, True
        except ValueError as error:
            print(error, file=sys.stderr)


def prepare(cache):
    repo, directory, download = select_model(cache)
    if "\n" in str(directory) or "\r" in str(directory):
        raise ValueError("模型目录不能包含换行")
    print(f"模型：{repo or 'HIGGS_MODEL_DIR'}\n目录：{directory}", file=sys.stderr)
    if download:
        subprocess.run(
            [
                "modelscope",
                "download",
                "--model",
                repo,
                "--revision",
                "master",
                "--local_dir",
                str(directory),
            ],
            check=True,
            stdout=sys.stderr,
        )
    validate_model(directory)
    name = (repo or "bosonai/higgs-audio-v3-tts-4b").rsplit("/", 1)[-1].lower()
    return name, directory.resolve()


if __name__ == "__main__":
    try:
        for value in prepare(Path.home() / ".cache/modelscope/hub/models"):
            print(value)
    except (
        ValueError,
        TypeError,
        IndexError,
        OSError,
        EOFError,
        subprocess.CalledProcessError,
    ) as error:
        print(f"模型准备失败：{error}", file=sys.stderr)
        sys.exit(1)
