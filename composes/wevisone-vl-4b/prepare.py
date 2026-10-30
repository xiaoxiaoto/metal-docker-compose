#!/usr/bin/env python3
"""准备 ModelScope 固定发布的 WeVisOne-VL-4B 静态文件；不加载张量、pickle 或缓存源码。"""

import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys
import tempfile
import urllib.parse
import urllib.request

from prepare_runtime import prepare_runtime

REPOSITORY = "tencent-community/WeVisOne-VL-4B"
MODEL_NAME = REPOSITORY.rsplit("/", 1)[-1].lower()
REVISION = "855d2a73ac0b929f668e6ac4f5deccc667aba99b"
# ModelScope 该固定 revision 的 SHA256；与上游 HF tencent/WeVisOne-VL-4B 发布逐字节一致。
FILES = {
    "config.json": (1664, "69da45bf7cdb6776dc3949034ff9c4ab711abddc696adc272a8cf18bc5399697"),
    "generation_config.json": (228, "5583d8b7fa1515aaabdfbca2cf412d99a86c98d9dd57dd66cfb132504f1fa4ad"),
    "processor_config.json": (1191, "d89ef49ce9cd37fbf510158e13c1ef063d9286411c1ec9049932dbe0487143b1"),
    "tokenizer_config.json": (424, "af3b0fe802740ae544beb2b395b9b498fc942d8df2323cd0117bf1dfdec1d943"),
    "chat_template.jinja": (5292, "3636d0f0bd6bef02654cdffdc447b79cb2cef8ab02cc75267345946291a489e4"),
    "tokenizer.json": (11607393, "3bb172429b952469b51c64783f52fa01c26d1b30f1f31a4693e385a43f9cfc5e"),
    "model.safetensors": (9661250408, "002c3ac0783af7c86dc28b282c84dd4bcd0c29dcb3b5e0d584492fe3212acb4a"),
    "mask_decoder.pt": (38534741, "8528460481bd09d0780c53559b08194c7abbd29dff0d51783b619158acbe7b46"),
}

ENDPOINT = f"https://modelscope.cn/api/v1/models/{REPOSITORY}/repo"


def local_file(directory, name):
    if name not in FILES:
        raise ValueError(f"不是固定发布文件：{name}")
    path = directory / name
    if path.is_symlink() or path.resolve() != path or not path.is_file():
        raise ValueError(f"模型文件缺失或路径无效：{name}")
    return path


def validate_file(directory, name):
    path = local_file(directory, name)
    size, expected = FILES[name]
    if path.stat().st_size != size:
        raise ValueError(f"发布文件大小不符（可能未完整下载或为 LFS 指针）：{name}")
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise ValueError(f"发布文件摘要不符：{name}；需要固定 revision {REVISION}")


def validate_weights(path):
    # 只读取 safetensors JSON 头，核对 BF16 形状、连续数据边界和完整参数量。
    size = path.stat().st_size
    with path.open("rb") as file:
        prefix = file.read(8)
        if len(prefix) != 8:
            raise ValueError("权重头不完整")
        length, = struct.unpack("<Q", prefix)
        if not 2 <= length <= min(16 * 1024 * 1024, size - 8):
            raise ValueError("权重头长度无效")
        header = json.loads(file.read(length))
    if not isinstance(header, dict):
        raise ValueError("权重头必须是对象")
    tensors = [value for key, value in header.items() if key != "__metadata__"]
    if not tensors:
        raise ValueError("主权重不含张量")
    end = parameters = 0
    for tensor in sorted(tensors, key=lambda value: value["data_offsets"]):
        start, stop = tensor["data_offsets"]
        shape = tensor["shape"]
        if (tensor.get("dtype") != "BF16" or not isinstance(shape, list)
                or any(type(dimension) is not int or dimension < 0 for dimension in shape)
                or type(start) is not int or type(stop) is not int
                or start != end or stop - start != math.prod(shape) * 2):
            raise ValueError("权重不是完整连续的官方 BF16 张量")
        parameters += math.prod(shape)
        end = stop
    if end != size - 8 - length or parameters != 4830581248:
        raise ValueError("主权重数据不完整或参数量不匹配")


def validate_structure(directory):
    # 强制 safetensors 加载；不接受分片索引或混入的其他 safetensors 主权重。
    if {path.name for path in directory.glob("*.safetensors")} != {"model.safetensors"}:
        raise ValueError("本发布只能包含官方单文件 model.safetensors 主权重")
    if (directory / "model.safetensors.index.json").exists():
        raise ValueError("本发布不使用分片权重索引")
    config = json.loads(local_file(directory, "config.json").read_text())
    if (config.get("architectures") != ["Qwen3VLForConditionalGeneration"]
            or config.get("model_type") != "qwen3_vl"
            or config.get("dtype") != "bfloat16"
            or config.get("text_config", {}).get("dtype") != "bfloat16"
            or config.get("vision_config", {}).get("dtype") != "bfloat16"
            or config.get("text_config", {}).get("vocab_size") != 152680
            or config.get("eos_token_id") != 151645
            or any(config.get(key) is not None for key in ("quantization", "quantization_config"))):
        raise ValueError("需要官方未量化 Qwen3-VL / BF16 / vocab152680 配置")
    processor = json.loads(local_file(directory, "processor_config.json").read_text())
    tokenizer_config = json.loads(local_file(directory, "tokenizer_config.json").read_text())
    if (processor.get("processor_class") != "Qwen3VLProcessor"
            or processor.get("image_processor", {}).get("image_processor_type") != "Qwen2VLImageProcessor"
            or tokenizer_config.get("eos_token") != "<|im_end|>"):
        raise ValueError("官方 tokenizer / processor 配置不匹配")
    tokenizer = json.loads(local_file(directory, "tokenizer.json").read_text())
    added = {token["content"]: token["id"] for token in tokenizer["added_tokens"]}
    expected = {f"<{value}>": 151669 + value for value in range(1001)}
    expected.update({"<ref>": 152670, "</ref>": 152671, "<box>": 152672,
                     "</box>": 152673, "<point>": 152674, "</point>": 152675,
                     "<OBJ>": 152678, "<|im_end|>": 151645})
    if any(added.get(token) != identity for token, identity in expected.items()):
        raise ValueError("官方坐标、对象或终止词元缺失/ID 不符")
    validate_weights(local_file(directory, "model.safetensors"))


def validate_model(directory):
    for name in FILES:
        validate_file(directory, name)
    validate_structure(directory)


def download_file(directory, name):
    size, expected = FILES[name]
    query = urllib.parse.urlencode({"Revision": REVISION, "FilePath": name})
    target = directory / name
    temporary = None
    try:
        # 先写同目录临时文件；大小与摘要都通过后才替换目标，失败不留下半文件。
        with tempfile.NamedTemporaryFile(dir=directory, prefix=f".{name}.",
                                         suffix=".part", delete=False) as file:
            temporary = Path(file.name)
            digest = hashlib.sha256()
            with urllib.request.urlopen(f"{ENDPOINT}?{query}", timeout=60) as response:
                if response.status != 200:
                    raise ValueError(f"下载失败：HTTP {response.status}：{name}")
                length = response.headers.get("Content-Length")
                if length is not None and int(length) != size:
                    raise ValueError(f"下载大小不符：{name}：{length} != {size}")
                for block in iter(lambda: response.read(8 * 1024 * 1024), b""):
                    file.write(block)
                    digest.update(block)
        if temporary.stat().st_size != size or digest.hexdigest() != expected:
            raise ValueError(f"下载内容校验失败：{name}")
        os.replace(temporary, target)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def prepare(cache):
    cache = Path(cache).expanduser().resolve()
    directory = cache / REPOSITORY
    if directory.is_symlink():
        raise ValueError("固定官方仓库缓存路径不能通过符号链接重定向")
    directory.mkdir(parents=True, exist_ok=True)
    if directory.resolve() != directory or not directory.is_dir():
        raise ValueError("固定官方仓库缓存路径不是普通目录")
    for name in FILES:
        path = directory / name
        if path.is_symlink():
            raise ValueError(f"不覆盖符号链接：{name}")
        try:
            validate_file(directory, name)
        except (ValueError, OSError) as error:
            print(f"{error}\n从 ModelScope 下载/补全 {REPOSITORY}@{REVISION}/{name}", file=sys.stderr)
            # 来源仅此固定 ModelScope 仓库；代理和 TLS 行为沿用调用者现有环境。
            download_file(directory, name)
            validate_file(directory, name)
    validate_structure(directory)
    runtime = prepare_runtime(directory)
    print(f"使用 ModelScope 固定发布：缓存 {directory}；mlx_vlm 兼容运行时视图 {runtime}",
          file=sys.stderr)
    return MODEL_NAME, runtime


if __name__ == "__main__":
    try:
        if len(sys.argv) != 1:
            raise ValueError("用法：prepare.py")
        for value in prepare(Path.home() / ".cache/modelscope/hub/models"):
            print(value)
    except (ValueError, KeyError, TypeError, IndexError, OSError, ImportError) as error:
        print(f"模型准备失败：{error}", file=sys.stderr)
        sys.exit(1)
