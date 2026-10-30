#!/usr/bin/env python3
"""校验并准备官方 Qwen-Image-2.1-Turbo 缓存，不执行缓存代码或加载张量。"""
import json
import math
from pathlib import Path
import shutil
import struct
import subprocess
import sys

REPOSITORY = "Qwen/Qwen-Image-2.1-Turbo"
MODEL_NAME = REPOSITORY.rsplit("/", 1)[-1].lower()
WEIGHT_BYTES = 32440133648


def local_file(directory, name):
    if not isinstance(name, str):
        raise ValueError("模型文件名必须是字符串")
    path = (directory / name).resolve()
    if not path.is_relative_to(directory) or not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"文件缺失或路径越界：{name}")
    return path


def read_json(directory, name):
    value = json.loads(local_file(directory, name).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{name} 必须是 JSON 对象")
    return value


def validate_weights(path):
    # 沿用 PaddleOCR 的 safetensors 边界校验，不反序列化权重。
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
    index = read_json(directory, "model_index.json")
    if index.get("_class_name") != "QwenImage21Pipeline":
        raise ValueError("需要 QwenImage21Pipeline，不是普通 Qwen-Image 或旧版 pipeline")
    components = {
        "text_encoder": ["transformers", "Qwen3VLForConditionalGeneration"],
        "processor": ["transformers", "Qwen3VLProcessor"],
        "transformer": ["diffusers", "QwenImage21Transformer2DModel"],
        "vae": ["diffusers", "AutoencoderKLQwenImage21"],
        "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
    }
    if any(index.get(name) != expected for name, expected in components.items()):
        raise ValueError("模型组件不是当前部署的 Qwen-Image 2.1 架构")
    sigmas = index.get("sample_sigmas")
    if not isinstance(sigmas, list) or len(sigmas) != 8 or any(
        type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 1
        for value in sigmas
    ) or any(a <= b for a, b in zip(sigmas, sigmas[1:])):
        raise ValueError("Turbo 必须保留官方有限、递减的 8 步 sample_sigmas")
    for component, expected in (
        ("text_encoder", "Qwen3VLForConditionalGeneration"),
        ("transformer", "QwenImage21Transformer2DModel"),
        ("vae", "AutoencoderKLQwenImage21"),
    ):
        config = read_json(directory, f"{component}/config.json")
        if component == "text_encoder":
            if config.get("architectures") != [expected] or config.get("model_type") != "qwen3_vl":
                raise ValueError("text_encoder 不是 Qwen3-VL")
        elif config.get("_class_name") != expected:
            raise ValueError(f"{component} 架构不匹配")
        for settings in (config, config.get("text_config", {}), config.get("vision_config", {})):
            if not isinstance(settings, dict) or any(
                settings.get(key) is not None for key in ("quantization", "quantization_config")
            ):
                raise ValueError("当前 Metal 部署仅支持未量化的官方组件")
        prefix = "model" if component == "text_encoder" else "diffusion_pytorch_model"
        index_name = f"{component}/{prefix}.safetensors.index.json"
        if (directory / index_name).exists():
            weight_map = read_json(directory, index_name).get("weight_map")
            if not isinstance(weight_map, dict) or not weight_map:
                raise ValueError("权重索引缺少 weight_map")
            names = set(weight_map.values())
        else:
            names = {f"{prefix}.safetensors"}
        for name in names:
            if not isinstance(name, str):
                raise ValueError("权重文件名必须是字符串")
            path = local_file(directory, f"{component}/{name}")
            if not path.is_relative_to((directory / component).resolve()):
                raise ValueError("权重索引不能越出组件目录")
            validate_weights(path)
    scheduler = read_json(directory, "scheduler/scheduler_config.json")
    if scheduler.get("_class_name") != "FlowMatchEulerDiscreteScheduler":
        raise ValueError("scheduler 架构不匹配")
    for name in ("processor/processor_config.json", "processor/tokenizer_config.json"):
        read_json(directory, name)
    for name in ("processor/tokenizer.json", "processor/chat_template.jinja"):
        local_file(directory, name)


def prepare(cache):
    directory = cache / REPOSITORY
    if not directory.resolve().is_relative_to(cache.resolve()):
        raise ValueError("缓存目录不能通过符号链接越界")
    try:
        validate_model(directory)
        print(f"使用完整官方缓存：{directory}", file=sys.stderr)
    except (ValueError, KeyError, TypeError, OSError):
        cache.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(cache).free < WEIGHT_BYTES + 5 * 1024**3:
            raise ValueError("下载/补全需预留约 30.21 GiB 权重及 5 GiB 余量；不删除现有缓存")
        print(f"下载/补全 {REPOSITORY}（约 30.21 GiB 权重）：{directory}", file=sys.stderr)
        subprocess.run([
            "modelscope", "download", "--model", REPOSITORY, "--revision", "master",
            "--local_dir", str(directory), "--exclude", "assets/*",
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
