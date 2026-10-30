#!/bin/bash
set -euo pipefail
if [ "$#" -eq 1 ] && [ "$1" = --help ]; then
  echo '用法：./start.sh [--prepare-only]；优化使用公共 METAL_* 环境变量；--help 无副作用'
  exit 0
fi
prepare_only=false
while [ "$#" -gt 0 ]; do
  case "$1" in
    --prepare-only)
      if [ "$prepare_only" = true ]; then echo '重复参数：--prepare-only' >&2; exit 1; fi
      prepare_only=true
      shift ;;
    *) echo "不支持的参数：$1" >&2; exit 1 ;;
  esac
done
cd -- "$(dirname -- "$0")"
for command in python3 modelscope launchctl curl lsof plutil; do
  command -v "$command" >/dev/null || { echo "缺少命令：$command" >&2; exit 1; }
done
if [ "$(uname -s)" != Darwin ] || [ "$(uname -m)" != arm64 ]; then
  echo '需要 Apple Silicon Mac。' >&2; exit 1
fi
python3 -c 'import platform,sys; sys.exit(0 if sys.version_info >= (3,9) and int(platform.mac_ver()[0].split(".")[0]) >= 15 else "需要 Python >= 3.9、macOS >= 15")'
DOMAIN="gui/$(id -u)"
SERVICE="$DOMAIN/com.local.seedvr2-metal"
PLIST="$HOME/Library/Application Support/seedvr2/com.local.seedvr2-metal.plist"
if launchctl print "$SERVICE" >/dev/null 2>&1; then
  echo 'SeedVR2 任务已注册；不卸载现有任务，请检查日志或先 ./stop.sh。' >&2; exit 1
fi
if lsof -nP -iTCP:8100 -sTCP:LISTEN >/dev/null 2>&1; then
  echo '8100 端口已占用；不终止其他进程。' >&2; exit 1
fi
if [ -z "${VLLM_BACKEND_PYTHON:-}" ]; then
  backend="$(python3 ../../common_env/prepare_backend.py)"
  export VLLM_BACKEND_PYTHON="$backend/bin/python"
fi
python="$VLLM_BACKEND_PYTHON"
test -x "$python" || { echo '共享后端 Python 无效。' >&2; exit 1; }
"$python" ../../common_env/check_versions.py --omni
"$python" - <<'PY'
import subprocess
import torch
assert torch.backends.mps.is_available(), 'MPS 不可用'
weight_bytes = 16480583960 + 501324814
if int(subprocess.check_output(['sysctl', '-n', 'hw.memsize'])) < weight_bytes + 10 * 1024**3:
    raise ValueError('保守预检需要权重加 10 GiB RAM；这不是实测峰值，不启用 offload')
if torch.mps.recommended_max_memory() < weight_bytes + 4 * 1024**3:
    raise ValueError('Metal 推荐预算需要权重加 4 GiB；这不是实测峰值，不改变水位')
PY
workdir="$(mktemp -d)"
started=false
cleanup() {
  status=$?
  if [ "$status" -ne 0 ] && [ "$started" = true ]; then
    launchctl bootout "$SERVICE" || echo '请运行 ./stop.sh。' >&2
    echo "启动失败，日志：$HOME/Library/Logs/seedvr2-metal*.log" >&2
  fi
  rm -rf -- "$workdir"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
"$python" prepare_runtime.py > "$workdir/runtime"
IFS= read -r runtime < "$workdir/runtime"
cli="$runtime/bin/vllm-omni"
test -x "$cli" || { echo 'SeedVR2 私有 CLI 不完整。' >&2; exit 1; }
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
import runpy
cli, directory, name, target = sys.argv[1:]
logs = Path.home() / 'Library/Logs'
logs.mkdir(parents=True, exist_ok=True)
target = Path(target)
target.parent.mkdir(parents=True, exist_ok=True)
optimization = runpy.run_path(str(Path.cwd().parents[1] / 'common_env' / 'prepare_optimizations.py'))['optimization_environment']()
unsupported = set(optimization['METAL_MODEL_OPTIMIZERS'].split(',')) - {'', 'dynamic_quantization'}
if unsupported:
    raise ValueError(f"SeedVR2 Omni 执行器未实现这些 provider: {sorted(unsupported)}")
config = {
    'Label': 'com.local.seedvr2-metal',
    'ProgramArguments': [cli, 'serve', directory, '--omni', '--served-model-name', name,
        '--model-class-name', 'SeedVR2Pipeline', '--host', '127.0.0.1', '--port', '8100',
        '--dtype', 'float16', '--enforce-eager', '--num-gpus', '1',
        '--distributed-executor-backend', 'uni', '--diffusion-attention-backend', 'TORCH_SDPA',
        '--vae-use-tiling'],
    'WorkingDirectory': str(target.parent), 'RunAtLoad': True, 'KeepAlive': False,
    'ExitTimeOut': 30, 'AbandonProcessGroup': False,
    'EnvironmentVariables': {
        'PATH': f'{Path(cli).parent}:/usr/bin:/bin:/usr/sbin:/sbin',
        'VLLM_BACKEND_PYTHON': os.environ['VLLM_BACKEND_PYTHON'],
        'PYTHONUNBUFFERED': '1', 'OMP_NUM_THREADS': '2',
        'VLLM_WORKER_MULTIPROC_METHOD': 'spawn',
        'VLLM_PLUGINS': 'metal,vllm_omni_register_models',
        'PYTORCH_ENABLE_MPS_FALLBACK': '0',
        **optimization,
    },
    'StandardOutPath': str(logs / 'seedvr2-metal.log'),
    'StandardErrorPath': str(logs / 'seedvr2-metal.error.log'),
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
  echo "SeedVR2 准备完成，plist 已校验：${PLIST}；未加载模型。"
  exit 0
fi
launchctl bootstrap "$DOMAIN" "$PLIST"
started=true
echo '等待 SeedVR2 原生 Omni + Metal pipeline（最多 1800 秒）……'
deadline=$((SECONDS + 1800))
until curl --connect-timeout 1 --max-time 2 -fsS http://127.0.0.1:8100/health >/dev/null 2>&1; do
  if ! state="$(launchctl print "$SERVICE" 2>/dev/null)"; then
    echo '模型任务已卸载，请检查日志。' >&2; exit 1
  fi
  if grep -Eq 'last (exit code = -?[0-9]+|terminating signal =)' <<< "$state" && ! grep -q 'state = running' <<< "$state"; then
    echo '模型进程已退出，请检查日志。' >&2; exit 1
  fi
  if [ "$SECONDS" -ge "$deadline" ]; then echo '模型服务未就绪。' >&2; exit 1; fi
  sleep 2
done
curl --connect-timeout 5 --max-time 10 -fsS http://127.0.0.1:8100/v1/models | "$python" -c 'import json,sys; assert any(m.get("id")==sys.argv[1] for m in json.load(sys.stdin).get("data",[])), "8100 不是预期模型"' "$model_name"
echo "seedvr2 已启动：127.0.0.1:8100；量化模式=${METAL_QUANTIZATION:-off}（实际批准/选中位宽查看加载日志）；图片/短视频仍需真实HTTP验收。"
