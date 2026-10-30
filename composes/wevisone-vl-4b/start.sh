#!/bin/bash
set -euo pipefail
prepare_only=false
if [ "$#" -eq 1 ] && [ "$1" = --prepare-only ]; then
  prepare_only=true
elif [ "$#" -ne 0 ]; then
  echo '用法：./start.sh [--prepare-only]' >&2
  exit 1
fi
cd -- "$(dirname -- "$0")"
for command in python3 launchctl curl lsof plutil; do
  command -v "$command" >/dev/null || { echo "缺少命令：$command" >&2; exit 1; }
done
if [ "$(uname -s)" != Darwin ] || [ "$(uname -m)" != arm64 ]; then
  echo "需要 Apple Silicon Mac；Linux 容器不能使用 Metal。" >&2
  exit 1
fi
python3 -c 'import platform,sys; sys.exit(0 if sys.version_info >= (3,11) and int(platform.mac_ver()[0].split(".")[0]) >= 15 else "需要 Python >= 3.11、macOS >= 15")'
DOMAIN="gui/$(id -u)"
SERVICE="$DOMAIN/com.local.wevisone-vl-4b-metal"
PLIST="$HOME/Library/Application Support/wevisone-vl-4b/com.local.wevisone-vl-4b-metal.plist"
if launchctl print "$SERVICE" >/dev/null 2>&1; then
  echo "WeVisOne-VL-4B 任务已注册；不重载或停止现有任务，请先检查日志，必要时 ./stop.sh。" >&2
  exit 1
fi
if lsof -nP -iTCP:8101 -sTCP:LISTEN >/dev/null 2>&1; then
  echo "8101 端口已占用；不停止其他进程。" >&2
  exit 1
fi
workdir="$(mktemp -d)"
started=false
cleanup() {
  status=$?
  if [ "$status" -ne 0 ] && [ "$started" = true ]; then
    launchctl bootout "$SERVICE" || echo "请运行 ./stop.sh。" >&2
    echo "启动失败，日志：$HOME/Library/Logs/wevisone-vl-4b-metal*.log" >&2
  fi
  rm -rf -- "$workdir"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
if [ -z "${VLLM_BACKEND_PYTHON:-}" ]; then
  backend="$(python3 ../../common_env/prepare_backend.py)"
  export VLLM_BACKEND_PYTHON="$backend/bin/python"
fi
python="$VLLM_BACKEND_PYTHON"
cli="$(dirname -- "$python")/vllm"
test -x "$python" && test -x "$cli" || { echo "统一后端安装不完整。" >&2; exit 1; }
"$python" ../../common_env/check_versions.py
export PYTORCH_ENABLE_MPS_FALLBACK=0
"$python" -c 'import torch, mlx.core as mx; assert torch.backends.mps.is_available(), "MPS 不可用"; assert mx.metal.is_available(), "MLX Metal 不可用"'
"$python" prepare.py > "$workdir/model"
{
  IFS= read -r model_name
  IFS= read -r model_dir
} < "$workdir/model"
"$python" prepare_runtime.py > "$workdir/plugin"
IFS= read -r overlay < "$workdir/plugin"
cli="$overlay/bin/vllm"
test -x "$cli" || { echo "WeVisOne 私有插件环境缺少 vllm launcher。" >&2; exit 1; }
"$python" - "$cli" "$model_dir" "$model_name" "$PLIST" <<'PY'
import os
from pathlib import Path
import plistlib
import runpy
import sys
import tempfile

cli, directory, name, target = sys.argv[1:]
logs = Path.home() / 'Library/Logs'
logs.mkdir(parents=True, exist_ok=True)
target = Path(target)
target.parent.mkdir(parents=True, exist_ok=True)
config = {
    'Label': 'com.local.wevisone-vl-4b-metal',
    # 一个原生 MLX backbone；像素插件在 worker 初始化阶段加载已核验的 MPS decoder。
    'ProgramArguments': [cli, 'serve', directory, '--served-model-name', name,
        '--host', '127.0.0.1', '--port', '8101', '--load-format', 'safetensors',
        '--dtype', 'bfloat16', '--max-model-len', '8192', '--max-num-seqs', '1',
        '--worker-extension-cls', 'wevisone_vl_metal.worker.PixelWorkerExtension',
        '--max-num-batched-tokens', '8192', '--gpu-memory-utilization', '0.35',
        '--enforce-eager', '--no-async-scheduling', '--no-enable-prefix-caching',
        '--mm-processor-cache-gb', '0', '--limit-mm-per-prompt', '{"image":1,"video":0,"audio":0}',
        '--mm-processor-kwargs', '{"min_pixels":16384,"max_pixels":2073600}'],
    'WorkingDirectory': str(target.parent), 'RunAtLoad': True, 'KeepAlive': False,
    'ExitTimeOut': 30, 'AbandonProcessGroup': False,
    'EnvironmentVariables': {
        'PATH': f'{Path(cli).parent}:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin',
        'PYTHONUNBUFFERED': '1', 'OMP_NUM_THREADS': '2',
        'VLLM_WORKER_MULTIPROC_METHOD': 'spawn', 'VLLM_PLUGINS': 'metal,wevisone_vl_metal',
        'PYTORCH_ENABLE_MPS_FALLBACK': '0',
        'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
        **runpy.run_path(str(Path.cwd().parents[1] / 'common_env' / 'prepare_optimizations.py'))[
            'optimization_environment'](),
    },
    'StandardOutPath': str(logs / 'wevisone-vl-4b-metal.log'),
    'StandardErrorPath': str(logs / 'wevisone-vl-4b-metal.error.log'),
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
  echo "WeVisOne-VL-4B 准备完成，plist 已校验：${PLIST}；未加载模型。"
  exit 0
fi
launchctl bootstrap "$DOMAIN" "$PLIST"
started=true
echo "等待 vLLM + vLLM Metal WeVisOne-VL-4B 加载（最多 10 分钟）……"
deadline=$((SECONDS + 600))
until curl --connect-timeout 1 --max-time 2 -fsS http://127.0.0.1:8101/health >/dev/null 2>&1; do
  if ! state="$(launchctl print "$SERVICE" 2>/dev/null)"; then
    echo "模型任务已卸载，请检查日志。" >&2
    exit 1
  fi
  if [[ "$state" =~ last\ exit\ code\ =\ -?[0-9]+ ]] && [[ "$state" != *'state = running'* ]]; then
    echo "模型进程已退出，请检查日志。" >&2
    exit 1
  fi
  if [ "$SECONDS" -ge "$deadline" ]; then echo "模型服务未就绪。" >&2; exit 1; fi
  sleep 2
done
curl --connect-timeout 5 --max-time 10 -fsS http://127.0.0.1:8101/v1/models | "$python" -c 'import json,sys; assert any(m.get("id")==sys.argv[1] for m in json.load(sys.stdin).get("data",[])), "端口不是预期模型"' "$model_name"
echo "模型 $model_name 已启动：127.0.0.1:8101；统一接口：http://127.0.0.1/v1/chat/completions、http://127.0.0.1/v1/wevisone/predict（需启动/更新根目录总代理）。"
