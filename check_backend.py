#!/usr/bin/env python3
"""验收已安装的共享后端；不安装依赖、不构建产物，也不执行模型。"""

import sys

sys.dont_write_bytecode = True

import json
import base64
import hashlib
import tomllib
from importlib import import_module
from importlib.metadata import distribution, entry_points
from importlib.resources import files
from pathlib import Path


def require(condition, message):
    """即使解释器启用优化，也必须执行部署合同检查。"""
    if not condition:
        raise RuntimeError(message)


def check_installation(plugin):
    prefix = Path(sys.prefix).resolve()
    require(
        Path(plugin.__file__).resolve().is_relative_to(prefix),
        "后端必须实际安装在当前 Python prefix 内，不能来自 editable 或其他环境",
    )
    with Path(__file__).with_name("pyproject.toml").open("rb") as file:
        sources = tomllib.load(file)["tool"]["uv"]["sources"]
    for name, source in sources.items():
        installed = distribution(name)
        require(
            Path(installed.locate_file("")).resolve().is_relative_to(prefix),
            f"{name} 发行包元数据不属于当前 Python prefix",
        )
        direct_url = json.loads(installed.read_text("direct_url.json") or "{}")
        require(
            not direct_url.get("dir_info", {}).get("editable", False),
            f"{name} 不能以 editable 方式安装",
        )
        require(
            (
                direct_url.get("url") == source["git"]
                and direct_url.get("vcs_info", {}).get("commit_id") == source["rev"]
                if "git" in source else
                direct_url.get("url") == source["url"]
                and "archive_info" in direct_url
            ),
            f"{name} 必须从 pyproject.toml 指定的源码仓库 commit 或归档构建安装",
        )

def check_native_records():
    """源码安装生成的原生库也必须受 RECORD 保护，供私有 overlay 校验复用。"""
    package = distribution("vllm-metal")
    kinds = set()
    for entry in package.files or ():
        if not str(entry).startswith("vllm_metal/") or entry.suffix not in (".so", ".metallib"):
            continue
        path = Path(package.locate_file(entry))
        require(path.is_file() and entry.hash is not None and entry.hash.mode == "sha256",
                f"Metal 原生产物缺少有效 RECORD：{entry}")
        digest = hashlib.sha256()
        with path.open("rb") as file:
            for block in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(block)
        actual = base64.urlsafe_b64encode(digest.digest()).decode().rstrip("=")
        require(actual == entry.hash.value and path.stat().st_size == entry.size,
                f"Metal 原生产物与 RECORD 不符：{entry}")
        kinds.add(entry.suffix)
    require(kinds == {".so", ".metallib"}, "Metal RECORD 缺少原生扩展或 shader libraries")



def check_cpu_contracts(torch):
    # 只在 CPU 上校验混合精度 profile，不打包权重、不执行模型。
    from vllm_omni_metal.quantization.linear import build_profile, validate_profile

    modules = {
        "w4": torch.nn.Linear(64, 8, dtype=torch.bfloat16, device="cpu").requires_grad_(False),
        "w8": torch.nn.Linear(64, 8, dtype=torch.bfloat16, device="cpu").requires_grad_(False),
        "bf16": torch.nn.Linear(7, 3, dtype=torch.bfloat16, device="cpu").requires_grad_(False),
    }
    profile = build_profile(
        modules,
        list(modules),
        scope="installation-contract-v2",
        calibration={"purpose": "installation-contract-only"},
        precisions={"w4": "w4a16", "w8": "w8a16", "bf16": "bf16"},
    )
    require(profile["schema"] == 2, "混合精度 profile schema 不符合安装合同")
    require(
        [
            (item["bits"], item["mode"], item["group_size"])
            for item in profile["modules"].values()
        ] == [(4, "affine", 64), (8, "affine", 64), (16, "identity", None)],
        "混合精度 profile 的位宽、模式或分组不符合安装合同",
    )
    require(
        validate_profile(profile, modules, scope="installation-contract-v2") == list(modules),
        "混合精度 profile 验证结果不符合安装合同",
    )

    # 保留安装器的融合 diffusion 数值合同；不触发 Metal 构建。
    from vllm_omni_metal.diffusion.ops import gated_residual_rms_adaln, qk_norm_rotary

    activation = torch.ones(1, 2, 4, device="cpu")
    normalized, residual = gated_residual_rms_adaln(
        activation, activation,
        torch.ones(4, device="cpu"), torch.ones(4, device="cpu"),
        torch.zeros(4, device="cpu"), torch.zeros(4, device="cpu"), 0.0,
    )
    torch.testing.assert_close(normalized, activation, atol=0, rtol=0)
    torch.testing.assert_close(residual, 2 * activation, atol=0, rtol=0)
    query = torch.ones(1, 1, 2, 4, device="cpu")
    key = -torch.ones(1, 1, 1, 4, device="cpu")
    rotated_q, rotated_k = qk_norm_rotary(
        query, key, None, None,
        torch.full((1, 2), 1j, dtype=torch.complex64, device="cpu"), 0.0, 0.0,
    )
    torch.testing.assert_close(
        rotated_q,
        torch.tensor([-1., 1., -1., 1.], device="cpu").expand_as(query),
        atol=0, rtol=0,
    )
    torch.testing.assert_close(
        rotated_k,
        torch.tensor([1., -1., 1., -1.], device="cpu").expand_as(key),
        atol=0, rtol=0,
    )


def main():
    import vllm_omni_metal

    check_installation(vllm_omni_metal)
    check_native_records()

    import torch
    # 使用公开加载器，按上游合同先初始化 MLX，再加载已编译扩展和 shader。
    import_module("vllm._C")
    from vllm_metal import envs
    from vllm_metal.metal import get_ops

    require(not envs.VLLM_METAL_BUILD_FROM_SOURCE, "部署验收只加载已构建产物，请取消 VLLM_METAL_BUILD_FROM_SOURCE")
    require(get_ops() is not None, "Metal paged attention 原生库尚未准备")

    for module in (
        "execution.compat", "execution.runtime", "execution.shm_connector",
        "native.attention", "native.compat", "native.runner", "native.worker",
        "diffusion.attention", "diffusion.config", "diffusion.ops", "diffusion.worker",
        "ops.mps_attention", "ops.metal_attention", "ops.aot_kernels",
        "ops.normalization", "ops.activation", "ops.rotary",
        "quantization.linear", "quantization.engine", "quantization.capture",
        "quantization.adapter", "quantization.runtime",
        "ane.cache", "ane.runtime", "ane.build", "ane.projection", "ane.allocation",
    ):
        import_module(f"vllm_omni_metal.{module}")

    # 除模块存在外，还验收安装器及共享准备器要求的公开接口。
    for module, symbols in (
        ("ane.runtime", ("DenseSwiGLU", "TiledDenseSwiGLU", "execution_scope")),
        ("quantization.linear", ("QuantizedLinear", "build_profile", "validate_profile")),
        ("quantization.capture", ("CaptureSession", "DiskCaptures")),
        ("quantization.adapter", ("DenseModelAdapter",)),
        ("quantization.runtime", ("load_profile", "configure_model_quantization")),
        ("ane.projection", ("DenseProjection", "prepare_projection")),
    ):
        imported = import_module(f"vllm_omni_metal.{module}")
        for symbol in symbols:
            getattr(imported, symbol)

    check_cpu_contracts(torch)

    from vllm_omni_metal.platform import MetalOmniPlatform

    for path in (
        MetalOmniPlatform.get_omni_ar_worker_cls(),
        MetalOmniPlatform.get_omni_generation_worker_cls(),
        MetalOmniPlatform.get_diffusion_worker_cls(),
        MetalOmniPlatform.get_diffusion_model_runner_cls(),
        MetalOmniPlatform.get_diffusion_attn_backend_cls(None, 64),
    ):
        module, _, symbol = path.rpartition(".")
        require(
            isinstance(getattr(import_module(module), symbol), type),
            f"平台返回的路径不是可导入的类：{path}",
        )

    for resource in (
        "_native/pytorch_backend/mps_ops.mm",
        "_native/metal/metal_device.h",
        "_native/metal/paged_attention_kernels.h",
        "_native/elementwise.metal",
        "_native/diffusion_rotary.metal",
        "_native/THIRD_PARTY_NOTICES.md",
        "_native/ane/ane-emitter.cc",
        "_native/ane/ane-emitter.h",
        "_native/ane/emit.cc",
        "_native/ane/bridge.swift",
        "_native/ane/LICENSE",
    ):
        require(
            files("vllm_omni_metal").joinpath(resource).is_file(),
            f"后端安装资源缺失：{resource}",
        )

    require(vllm_omni_metal.is_available(), "MPS 不可用")
    require(torch.backends.mps.is_available(), "PyTorch MPS 不可用")
    require(torch.Generator(device="mps") is not None, "无法创建 MPS Generator")
    require(
        any(
            entry.name == "metal" and entry.value == "vllm_omni_metal:register_platform"
            for entry in entry_points(group="vllm_omni.platform_plugins")
        ),
        "缺少 Omni Metal 平台 entrypoint",
    )

    # 只加载已构建的 launcher；缺失时失败，绝不调用构建入口。
    from vllm_omni_metal.ops.mps_attention import get_mps_ops

    require(get_mps_ops() is not None, "Torch-stream attention launcher 尚未构建")
    print("共享后端部署验收成功：非 editable 安装、模块与资源、平台接口、CPU 数值合同、MPS 和已构建 launcher 均通过")


if __name__ == "__main__":
    main()
