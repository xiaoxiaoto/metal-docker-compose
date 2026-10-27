#!/usr/bin/env python3
"""编译已安装 Metal 源码并登记原生产物，保留私有 overlay 的 RECORD 完整性合同。"""

import base64
import csv
import hashlib
from importlib.metadata import distribution
import os
from pathlib import Path
import sys
import tempfile

sys.dont_write_bytecode = True


def file_record(path):
    digest = hashlib.sha256(path.read_bytes()).digest()
    return "sha256=" + base64.urlsafe_b64encode(digest).decode().rstrip("="), str(path.stat().st_size)


def verify_sources(package):
    """先核对原 wheel 的源码记录；不能通过重新计算 RECORD 掩盖源码篡改。"""
    if not package.files:
        raise ValueError("Metal 发行包缺少 RECORD，拒绝构建")
    for entry in package.files or ():
        if not str(entry).startswith("vllm_metal/") or str(entry).endswith(".pyc"):
            continue
        path = Path(package.locate_file(entry))
        if not entry.hash or entry.hash.mode != "sha256" or not path.is_file():
            raise ValueError(f"Metal 源码缺少有效 RECORD：{entry}")
        actual_hash, actual_size = file_record(path)
        if actual_hash != "sha256=" + entry.hash.value or actual_size != str(entry.size):
            raise ValueError(f"Metal 源码与安装 RECORD 不符：{entry}")


def register_artifacts(package, artifacts):
    root = Path(package.locate_file("")).resolve()
    records = [entry for entry in package.files or () if str(entry).endswith(".dist-info/RECORD")]
    if len(records) != 1 or not root.is_relative_to(Path(sys.prefix).resolve()):
        raise ValueError("Metal 发行记录不属于选定的隔离环境")
    record = Path(package.locate_file(records[0]))
    with record.open(newline="") as file:
        rows = {row[0]: row for row in csv.reader(file)}
    for artifact in artifacts:
        path = Path(artifact).resolve(strict=True)
        if not path.is_relative_to(root / "vllm_metal"):
            raise ValueError(f"原生产物路径越界：{path}")
        relative = path.relative_to(root).as_posix()
        rows[relative] = [relative, *file_record(path)]
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", newline="", dir=record.parent, delete=False) as file:
            temporary = Path(file.name)
            csv.writer(file).writerows(rows.values())
        os.replace(temporary, record)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    package = distribution("vllm-metal")
    verify_sources(package)
    from vllm_metal.metal import build

    artifacts = [build.build(), *build.build_metallibs(), build.build_mps()]
    artifacts += [build.mlx_version_path(), build.mps_version_path()]
    for artifact in tuple(artifacts):
        for suffix in (".sha256", ".toolchain"):
            sidecar = artifact.with_suffix(artifact.suffix + suffix)
            if sidecar.is_file():
                artifacts.append(sidecar)
    # 构建只能增加原生文件，原 wheel 的源码仍须逐字匹配。
    verify_sources(package)
    register_artifacts(package, artifacts)
    print("Metal 源码原生产物已编译并登记 RECORD：" + ", ".join(str(path) for path in artifacts))


if __name__ == "__main__":
    main()
