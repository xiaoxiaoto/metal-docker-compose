#!/usr/bin/env python3
"""仅准备固定 SeedVR2 7B FP16 权重；不加载张量或执行缓存代码。"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile

REPOSITORY = "Comfy-Org/SeedVR2"
REVISION = "5360e873f15685f17acdb077b5e46d16eb6d05d7"
MANIFEST = {
    "diffusion_models/seedvr2_7b_fp16.safetensors": (16480583960, "2742ca6fee63bc5cc1773f426dd4b07b78cad27f51c9ea5cd42b035e6b592252"),
    "vae/ema_vae_fp16.safetensors": (501324814, "20678548f420d98d26f11442d3528f8b8c94e57ee046ef93dbb7633da8612ca1"),
}
WEIGHT_BYTES = sum(size for size, _ in MANIFEST.values())
MODEL_INDEX = {"_class_name": "SeedVR2Pipeline", "seedvr2_variant": "7b-comfy-fp16"}


DIT = "diffusion_models/seedvr2_7b_fp16.safetensors"
REQUIRED = {
    DIT: {
        "positive_conditioning": [58, 5120], "negative_conditioning": [64, 5120],
        "vid_in.proj.weight": [3072, 132], "txt_in.weight": [3072, 5120],
        "vid_out.proj.weight": [64, 3072],
        "emb_in.proj_out.weight": [18432, 3072],
        "blocks.35.attn.proj_qkv.txt.weight": [9216, 3072],
        "blocks.35.attn.rope.rope.freqs": [10],
    },
    "vae/ema_vae_fp16.safetensors": {
        "encoder.conv_in.weight": [128, 3, 3, 3, 3],
        "encoder.conv_out.weight": [32, 512, 3, 3, 3],
        "decoder.conv_in.weight": [512, 16, 3, 3, 3],
        "decoder.conv_out.weight": [3, 128, 3, 3, 3],
    },
}

def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON 含重复键：{key}")
        result[key] = value
    return result


def safe_path(directory, name):
    directory = Path(directory).resolve()
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("文件路径越界")
    path = directory / relative
    if not path.resolve().is_relative_to(directory):
        raise ValueError(f"符号链接路径越界：{name}")
    return path


def validate_safetensors(path, required=None):
    """有界读取 header，检查 FP16、shape、连续 offset 和精确文件边界。"""
    path = Path(path)
    size = path.stat().st_size
    with path.open("rb") as file:
        prefix = file.read(8)
        if len(prefix) != 8:
            raise ValueError("safetensors 前缀不完整")
        length, = struct.unpack("<Q", prefix)
        if not 2 <= length <= min(16 * 1024**2, size - 8):
            raise ValueError("safetensors header 长度无效")
        header = json.loads(file.read(length), object_pairs_hook=unique_object)
    if not isinstance(header, dict):
        raise ValueError("safetensors header 必须是对象")
    ranges = []
    tensors = {}
    for name, tensor in header.items():
        if name == "__metadata__":
            if not isinstance(tensor, dict) or any(not isinstance(v, str) for v in tensor.values()):
                raise ValueError("safetensors metadata 无效")
            continue
        expected_dtype = "BF16" if name in ("positive_conditioning", "negative_conditioning") else "F16"
        if not isinstance(tensor, dict) or tensor.get("dtype") != expected_dtype:
            raise ValueError(f"张量 dtype 不匹配（仅 conditioning 为 BF16）：{name}")
        shape, offsets = tensor.get("shape"), tensor.get("data_offsets")
        if not isinstance(shape, list) or any(type(n) is not int or n <= 0 for n in shape):
            raise ValueError(f"张量 shape 无效：{name}")
        if not isinstance(offsets, list) or len(offsets) != 2 or any(type(n) is not int for n in offsets):
            raise ValueError(f"张量 offset 无效：{name}")
        start, end = offsets
        if start < 0 or end < start or end - start != math.prod(shape) * 2:
            raise ValueError(f"张量大小与 shape 不符：{name}")
        ranges.append((start, end))
        tensors[name] = shape
    end = 0
    for start, stop in sorted(ranges):
        if start != end:
            raise ValueError("权重 offset 重叠或存在空洞")
        end = stop
    if end != size - 8 - length:
        raise ValueError("权重数据截断或含尾随字节")
    for name, shape in (required or {}).items():
        if tensors.get(name) != shape:
            raise ValueError(f"SeedVR2 7B 架构/conditioning 不匹配：{name}")
    return tensors


def validate_files(directory, manifest=None, required=None):
    production = manifest is None
    manifest = MANIFEST if production else manifest
    required = REQUIRED if production and required is None else (required or {})
    for name, (size, expected) in manifest.items():
        path = safe_path(directory, name)
        if not path.is_file() or path.stat().st_size != size:
            raise ValueError(f"权重缺失或大小不符：{name}")
        validate_safetensors(path, (required or {}).get(name))
        digest = hashlib.sha256()
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(8 * 1024**2), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise ValueError(f"发布 SHA256 不匹配：{name}")


def check_index(directory):
    path = safe_path(directory, "model_index.json")
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 65536:
            raise ValueError("model_index.json 路径或大小无效")
        if json.loads(path.read_text(), object_pairs_hook=unique_object) != MODEL_INDEX:
            raise ValueError("已有 model_index.json 不匹配；不覆盖用户配置")
    return path


def write_index(directory):
    path = check_index(directory)
    if path.exists():
        return
    with tempfile.NamedTemporaryFile(mode="w", dir=directory, delete=False) as file:
        temporary = Path(file.name)
        json.dump(MODEL_INDEX, file)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())
    try:
        # 原子且不覆盖并发创建的用户配置。
        try:
            os.link(temporary, path)
        except FileExistsError:
            check_index(directory)
    finally:
        temporary.unlink(missing_ok=True)


def prepare(cache, check_only=False):
    cache = Path(cache).expanduser().resolve()
    directory = safe_path(cache, REPOSITORY)
    check_index(directory)
    for name in MANIFEST:
        safe_path(directory, name)
    missing = [name for name, (size, _) in MANIFEST.items()
               if not safe_path(directory, name).is_file() or safe_path(directory, name).stat().st_size != size]
    if missing and not check_only:
        cache.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(cache).free < WEIGHT_BYTES + 5 * 1024**3:
            raise ValueError("下载需完整权重加 5 GiB 可用磁盘；不删除现有缓存")
        for name in missing:
            print(f"下载固定 revision 文件：{name}", file=sys.stderr)
            subprocess.run(["modelscope", "download", "--model", REPOSITORY,
                            "--revision", REVISION, "--local_dir", str(directory),
                            "--include", name], check=True, stdout=sys.stderr)
    validate_files(directory)
    if not check_only:
        write_index(directory)
    return "seedvr2", directory.resolve()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="仅校验缓存，不下载、不写文件")
    args = parser.parse_args()
    try:
        for value in prepare(Path.home() / ".cache/modelscope/hub/models", args.check_only):
            print(value)
    except (ValueError, TypeError, OSError, subprocess.CalledProcessError) as error:
        print(f"模型准备失败：{error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
