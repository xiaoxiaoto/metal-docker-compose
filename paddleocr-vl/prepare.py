#!/usr/bin/env python3
"""准备指定官方 PaddleOCR-VL 的 ModelScope 缓存；不加载模型或执行缓存代码。"""

import json
from pathlib import Path
import struct
import subprocess
import sys

REPOSITORY = "PaddlePaddle/PaddleOCR-VL"
MODEL_NAME = REPOSITORY.rsplit("/", 1)[-1].lower()


def local_file(directory, name):
    if not isinstance(name, str):
        raise ValueError("模型文件名必须是字符串")
    path = (directory / name).resolve()
    if not path.is_relative_to(directory) or not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"模型文件缺失或路径无效：{name}")
    return path


def validate_weights(path):
    # 检查 safetensors 数据边界，避免把半下载权重当成完整缓存；不反序列化张量。
    with path.open("rb") as file:
        size = path.stat().st_size
        prefix = file.read(8)
        if len(prefix) != 8:
            raise ValueError(f"权重头不完整：{path.name}")
        length, = struct.unpack("<Q", prefix)
        if not 2 <= length <= min(16 * 1024 * 1024, size - 8):
            raise ValueError(f"权重头长度无效：{path.name}")
        header = json.loads(file.read(length))
    if not isinstance(header, dict):
        raise ValueError("权重头必须是对象")
    tensors = [value for key, value in header.items() if key != "__metadata__"]
    if not tensors:
        raise ValueError("权重不含张量")
    end = 0
    for tensor in sorted(tensors, key=lambda value: value["data_offsets"]):
        start, stop = tensor["data_offsets"]
        if type(start) is not int or type(stop) is not int or start != end or stop < start:
            raise ValueError("权重数据偏移无效")
        end = stop
    if end != size - 8 - length:
        raise ValueError(f"权重数据不完整：{path.name}")


def validate_model(directory):
    directory = Path(directory).resolve()
    config = json.loads(local_file(directory, "config.json").read_text())
    if not isinstance(config, dict) or config.get("model_type") != "paddleocr_vl" or \
            config.get("architectures") != ["PaddleOCRVLForConditionalGeneration"]:
        raise ValueError("模型不是 PaddleOCR-VL 架构")
    if any(config.get(key) is not None for key in ("quantization", "quantization_config")):
        raise ValueError("本部署只使用官方未量化权重")
    for name in ("tokenizer_config.json", "preprocessor_config.json", "processor_config.json"):
        if not isinstance(json.loads(local_file(directory, name).read_text()), dict):
            raise ValueError(f"{name} 必须是对象")
    for name in ("tokenizer.json", "tokenizer.model", "chat_template.jinja",
                 "configuration_paddleocr_vl.py", "modeling_paddleocr_vl.py",
                 "processing_paddleocr_vl.py", "image_processing_paddleocr_vl.py"):
        local_file(directory, name)
    index = directory / "model.safetensors.index.json"
    if index.exists():
        data = json.loads(local_file(directory, index.name).read_text())
        if not isinstance(data, dict) or not isinstance(data.get("weight_map"), dict):
            raise ValueError("权重索引必须包含 weight_map 对象")
        weights = list(data["weight_map"].values())
    else:
        weights = ["model.safetensors"]
    if not weights:
        raise ValueError("没有 safetensors 权重")
    for name in set(weights):
        validate_weights(local_file(directory, name))


def prepare(cache):
    # 本部署固定官方原版，不把 1.5/1.6、量化版或其他仓库当成同一模型。
    directory = cache / REPOSITORY
    if not directory.resolve().is_relative_to(cache.resolve()):
        raise ValueError("缓存目标不能通过符号链接指向缓存目录外")
    try:
        validate_model(directory)
        print(f"使用完整官方缓存：{directory}", file=sys.stderr)
    except (ValueError, KeyError, TypeError, OSError):
        print(f"下载/补全官方模型 {REPOSITORY}（主权重约 1.92 GB）：{directory}", file=sys.stderr)
        subprocess.run([
            "modelscope", "download", "--model", REPOSITORY, "--revision", "master",
            "--local_dir", str(directory), "--exclude", "PP-DocLayoutV2/*", ".eval_results/*",
        ], check=True, stdout=sys.stderr)
        validate_model(directory)
    return MODEL_NAME, directory.resolve()


if __name__ == "__main__":
    try:
        if len(sys.argv) != 1:
            raise ValueError("用法：prepare.py")
        for value in prepare(Path.home() / ".cache/modelscope/hub/models"):
            print(value)
    except (ValueError, KeyError, TypeError, OSError, subprocess.CalledProcessError) as error:
        print(f"模型准备失败：{error}", file=sys.stderr)
        sys.exit(1)
