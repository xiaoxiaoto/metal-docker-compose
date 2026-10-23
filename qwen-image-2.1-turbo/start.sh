#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
prepare_only=false
if [ "$#" -eq 1 ] && [ "$1" = --prepare-only ]; then
  prepare_only=true
elif [ "$#" -ne 0 ]; then
  echo '用法：./start.sh [--prepare-only]' >&2
  exit 1
fi
cd -- "$(dirname -- "$0")"
for command in python3 modelscope launchctl curl lsof plutil; do
  command -v "$command" >/dev/null || { echo "缺少命令：$command" >&2; exit 1; }
done
if [ "$(uname -s)" != Darwin ] || [ "$(uname -m)" != arm64 ]; then
  echo "vLLM-Omni Metal 需要 Apple Silicon Mac。" >&2
  exit 1
fi
python3 -c 'import platform,sys; sys.exit(0 if sys.version_info >= (3,9) and int(platform.mac_ver()[0].split(".")[0]) >= 15 else "需要 Python >= 3.9、macOS >= 15")'
DOMAIN="gui/$(id -u)"
SERVICE="$DOMAIN/com.local.qwen-image-2.1-turbo-metal"
PLIST="$HOME/Library/Application Support/qwen-image-2.1-turbo/com.local.qwen-image-2.1-turbo-metal.plist"
if launchctl print "$SERVICE" >/dev/null 2>&1; then
  echo "Qwen-Image 任务已注册；不自动卸载现有任务，请检查日志，必要时先 ./stop.sh。" >&2
  exit 1
fi
if lsof -nP -iTCP:8099 -sTCP:LISTEN >/dev/null 2>&1; then
  echo "8099 端口已占用；不终止其他进程。" >&2
  exit 1
fi
workdir="$(mktemp -d)"
started=false
cleanup() {
  status=$?
  if [ "$status" -ne 0 ] && [ "$started" = true ]; then
    launchctl bootout "$SERVICE" || echo "请运行 ./stop.sh。" >&2
    echo "启动失败，日志：$HOME/Library/Logs/qwen-image-2.1-turbo-metal*.log" >&2
  fi
  rm -rf -- "$workdir"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
if [ -z "${VLLM_BACKEND_PYTHON:-}" ]; then
  backend="$(python3 ../prepare_backend.py)"
  export VLLM_BACKEND_PYTHON="$backend/bin/python"
fi
python="$VLLM_BACKEND_PYTHON"
cli="$(dirname -- "$python")/vllm-omni"
test -x "$python" && test -x "$cli" || { echo "统一后端安装不完整。" >&2; exit 1; }
"$python" ../check_versions.py --omni
runtime="$("$python" prepare_runtime.py)"
python="$runtime/bin/python"
cli="$runtime/bin/vllm-omni"
test -x "$cli" || { echo "QwenImage 私有运行时不完整。" >&2; exit 1; }
"$python" - <<'PY'
import subprocess
import torch
from vllm_omni.diffusion.models.qwen_image_21.pipeline_qwen_image_21 import QwenImage21Pipeline
assert torch.backends.mps.is_available(), 'MPS 不可用'
assert hasattr(QwenImage21Pipeline, 'default_num_inference_steps'), 'Omni 缺少 Turbo 模型采样计划支持'
weight_bytes = 32440133648
memory = int(subprocess.check_output(['sysctl', '-n', 'hw.memsize']))
if memory < weight_bytes + 10 * 1024**3:
    raise ValueError('常驻未量化权重约 30.21 GiB，需要至少额外 10 GiB 系统/执行余量；不自动启用 offload')
if torch.mps.recommended_max_memory() < weight_bytes + 4 * 1024**3:
    raise ValueError('Metal 推荐内存预算不足以保留权重及执行余量；不提高 MPS 水位')
PY
"$python" prepare.py > "$workdir/model"
{
  IFS= read -r model_name
  IFS= read -r model_dir
} < "$workdir/model"
"$python" - "$cli" "$model_dir" "$model_name" "$PLIST" <<'PY'
import os
from pathlib import Path
import plistlib
import sys
import tempfile

cli, directory, name, target = sys.argv[1:]
cli = str(Path(cli).resolve())
logs = Path.home() / 'Library/Logs'
logs.mkdir(parents=True, exist_ok=True)
target = Path(target)
target.parent.mkdir(parents=True, exist_ok=True)
config = {
    'Label': 'com.local.qwen-image-2.1-turbo-metal',
    # 官方单 generation pipeline；原生采样/组件由 Omni 从 model_index 解析。
    'ProgramArguments': [cli, 'serve', directory, '--omni', '--served-model-name', name,
        '--host', '127.0.0.1', '--port', '8099', '--dtype', 'bfloat16',
        '--diffusion-load-format', 'default', '--distributed-executor-backend', 'uni',
        '--diffusion-attention-backend', 'TORCH_SDPA', '--enforce-eager',
        '--vae-use-slicing', '--vae-use-tiling'],
    'WorkingDirectory': str(target.parent), 'RunAtLoad': True, 'KeepAlive': False,
    'ExitTimeOut': 30, 'AbandonProcessGroup': False,
    'EnvironmentVariables': {
        'PATH': f'{Path(cli).parent}:/usr/bin:/bin:/usr/sbin:/sbin',
        'PYTHONUNBUFFERED': '1', 'OMP_NUM_THREADS': '2',
        'VLLM_WORKER_MULTIPROC_METHOD': 'spawn',
        'VLLM_PLUGINS': 'metal,vllm_omni_register_models',
        'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
        'PYTORCH_ENABLE_MPS_FALLBACK': '0',
        'QWEN_IMAGE_METAL_GPU': os.environ.get('QWEN_IMAGE_METAL_GPU', 'auto'),
        'QWEN_IMAGE_METAL_ANE_LAYERS': os.environ.get('QWEN_IMAGE_METAL_ANE_LAYERS', '0'),
        'VLLM_OMNI_METAL_ANE': os.environ.get('VLLM_OMNI_METAL_ANE', 'off'),
        'QWEN_IMAGE_METAL_QUANTIZATION': os.environ.get(
            'QWEN_IMAGE_METAL_QUANTIZATION', os.environ.get('VLLM_OMNI_METAL_QUANTIZATION', 'off')),
        **({'QWEN_IMAGE_METAL_QUANTIZATION_PROFILE': str(Path(
            os.environ.get('QWEN_IMAGE_METAL_QUANTIZATION_PROFILE',
                           os.environ.get('VLLM_OMNI_METAL_QUANTIZATION_PROFILE'))).expanduser().resolve())}
           if os.environ.get('QWEN_IMAGE_METAL_QUANTIZATION_PROFILE',
                             os.environ.get('VLLM_OMNI_METAL_QUANTIZATION_PROFILE')) else {}),
    },
    'StandardOutPath': str(logs / 'qwen-image-2.1-turbo-metal.log'),
    'StandardErrorPath': str(logs / 'qwen-image-2.1-turbo-metal.error.log'),
}
with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as file:
    temporary = Path(file.name)
    plistlib.dump(config, file)
try:
    os.replace(temporary, target)
finally:
    temporary.unlink(missing_ok=True)
PY
plutil -lint "$PLIST" >/dev/null
if [ "$prepare_only" = true ]; then
  echo "Qwen-Image 准备完成，plist 已校验：${PLIST}；未加载模型。"
  exit 0
fi
launchctl bootstrap "$DOMAIN" "$PLIST"
started=true
echo "等待 vLLM-Omni + Omni Metal Qwen-Image 加载（最多 10 分钟）……"
deadline=$((SECONDS + 600))
until curl --connect-timeout 1 --max-time 2 -fsS http://127.0.0.1:8099/health >/dev/null 2>&1; do
  if ! state="$(launchctl print "$SERVICE" 2>/dev/null)"; then
    echo "模型任务已卸载，请检查日志。" >&2
    exit 1
  fi
  if grep -Eq 'last (exit code = -?[0-9]+|terminating signal =)' <<< "$state" && ! grep -q 'state = running' <<< "$state"; then
    echo "模型进程已退出，请检查日志。" >&2
    exit 1
  fi
  if [ "$SECONDS" -ge "$deadline" ]; then echo "模型服务未就绪。" >&2; exit 1; fi
  sleep 2
done
curl --connect-timeout 5 --max-time 10 -fsS http://127.0.0.1:8099/v1/models | "$python" -c 'import json,sys; assert any(m.get("id")==sys.argv[1] for m in json.load(sys.stdin).get("data",[])), "端口不是预期模型"' "$model_name"
echo "模型 $model_name 已启动：127.0.0.1:8099；统一接口：http://127.0.0.1/v1/images/generations（需启动/更新根目录总代理）。"
