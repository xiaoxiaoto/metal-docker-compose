#!/bin/bash
set -euo pipefail
cd -- "$(dirname -- "$0")"

for command in docker python3 curl launchctl plutil; do
  command -v "$command" >/dev/null || { echo "缺少命令：$command" >&2; exit 1; }
done
if [ "$(uname -s)" != Darwin ] || [ "$(uname -m)" != arm64 ]; then
  echo "vLLM Metal 需要 Apple Silicon Mac。" >&2
  exit 1
fi
python3 -c 'import sys, platform; sys.exit(0 if sys.version_info >= (3, 9) and int(platform.mac_ver()[0].split(".")[0]) >= 15 else "需要 Python >= 3.9、macOS >= 15")'
# 在联网后端预检或启动任何服务前，确认用户已手动下载完整官方发布。
export MODEL_DIR
MODEL_DIR="$(python3 prepare.py --check-model)"
# Docker 不可用时不下载、不升级、不加载模型。
docker info >/dev/null

DOMAIN="gui/$(id -u)"
SERVICE="$DOMAIN/com.local.h2o-lightning-vllm"
PLIST="$HOME/Library/Application Support/h2o-lightning-shim/com.local.h2o-lightning-vllm.plist"
registered=false
if launchctl print "$SERVICE" >/dev/null 2>&1; then
  registered=true
  if ! curl --connect-timeout 1 --max-time 2 -fsS http://127.0.0.1:8000/health >/dev/null; then
    echo "H2O 任务已注册但不健康；不自动停止，请检查日志后手动 ./stop.sh。" >&2
    exit 1
  fi
else
  python3 - <<'PY'
import socket
with socket.socket() as sock:
    try:
        sock.bind(("127.0.0.1", 8000))
    except OSError:
        raise SystemExit("8000 端口已被其他服务占用；不终止该进程。")
PY
fi
workdir="$(mktemp -d)"
started=false
shim_started=false
cleanup() {
  status=$?
  if [ "$status" -ne 0 ] && [ "$shim_started" = true ]; then
    docker compose down || echo "清理本次启动的 shim 失败，请运行 ./stop.sh 重试。" >&2
  fi
  if [ "$status" -ne 0 ] && [ "$started" = true ]; then
    echo "启动失败，停止本次启动的模型。日志：$HOME/Library/Logs/h2o-lightning-vllm*.log" >&2
    launchctl bootout "$SERVICE" || echo "停止模型失败，请运行 ./stop.sh 重试。" >&2
  fi
  rm -rf -- "$workdir"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

docker compose config --quiet
if [ -z "${VLLM_BACKEND_PYTHON:-}" ]; then
  backend="$(python3 ../prepare_backend.py)"
  export VLLM_BACKEND_PYTHON="$backend/bin/python"
fi
python3 prepare.py > "$workdir/model"
{
  IFS= read -r MODEL_NAME
  IFS= read -r MODEL_DIR
  IFS= read -r EXECUTABLE
  IFS= read -r PREFIX
  IFS= read -r MODEL_MOUNT_DIR
  IFS= read -r MODEL_SUBDIR
} < "$workdir/model"
export MODEL_NAME MODEL_DIR MODEL_MOUNT_DIR MODEL_SUBDIR
python3 - "$EXECUTABLE" "$MODEL_DIR" "$MODEL_NAME" "$PREFIX" "$PLIST" "$registered" <<'PY'
import json
import os
from pathlib import Path
import plistlib
import sys
import tempfile

executable, directory, name, prefix, target, registered = sys.argv[1:]
release = json.loads(Path("model-release.json").read_text())
logs = Path.home() / 'Library/Logs'
logs.mkdir(parents=True, exist_ok=True)
target = Path(target)
target.parent.mkdir(parents=True, exist_ok=True)
config = {
    'Label': 'com.local.h2o-lightning-vllm',
    'ProgramArguments': [executable, 'serve', directory,
                         '--served-model-name', name,
                         '--host', '127.0.0.1', '--port', '8000',
                         '--dtype', 'bfloat16', '--max-model-len', '40960',
                         '--max-num-seqs', '4', '--gpu-memory-utilization', '0.55',
                         '--max-logprobs', '128',
                         '--max-num-batched-tokens', '40960',
                         '--no-enable-prefix-caching', '--no-enable-chunked-prefill',
                         '--mamba-cache-mode', 'none',
                         '--additional-config', '{"bidirectional_attention":true}',
                         '--limit-mm-per-prompt', '{"image":4,"video":0}',
                         '--mm-processor-kwargs', '{"max_pixels":1048576}'],
    'EnvironmentVariables': {
        'PATH': f'{Path(prefix)}/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin',
        'HF_HUB_OFFLINE': '1',
        'TRANSFORMERS_OFFLINE': '1',
        'VLLM_WORKER_MULTIPROC_METHOD': 'spawn',
        'H2O_MODEL_REVISION': release['revision'],
        'VLLM_METAL_BACKEND': 'mlx',
        'VLLM_METAL_BUILD_FROM_SOURCE': '0',
        'PYTORCH_ENABLE_MPS_FALLBACK': '0',
    },
    'RunAtLoad': True, 'KeepAlive': True, 'ThrottleInterval': 60,
    'StandardOutPath': str(logs / 'h2o-lightning-vllm.log'),
    'StandardErrorPath': str(logs / 'h2o-lightning-vllm.error.log'),
}
if registered == "true":
    if not target.is_file() or plistlib.loads(target.read_bytes()) != config:
        raise SystemExit("当前注册的 H2O 部署不是本次发布配置；请确认无在途请求后先 ./stop.sh 再 ./start.sh。")
    sys.exit(0)
with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as file:
    temporary = Path(file.name)
    plistlib.dump(config, file)
try:
    os.replace(temporary, target)
finally:
    temporary.unlink(missing_ok=True)
PY
plutil -lint "$PLIST" >/dev/null

if ! launchctl print "$SERVICE" >/dev/null 2>&1; then
  launchctl bootstrap "$DOMAIN" "$PLIST"
  started=true
fi

echo "等待模型加载（最多 10 分钟）……"
deadline=$((SECONDS + 600))
until curl --connect-timeout 1 --max-time 2 -fsS http://127.0.0.1:8000/health >/dev/null 2>&1; do
  if [ "$SECONDS" -ge "$deadline" ]; then
    echo "模型接口未就绪。请查看 $HOME/Library/Logs/h2o-lightning-vllm*.log" >&2
    exit 1
  fi
  sleep 2
done
curl --max-time 5 -fsS http://127.0.0.1:8000/v1/models > "$workdir/models.json"
python3 -c 'import json, sys; ids = [m["id"] for m in json.load(open(sys.argv[1]))["data"]]; sys.exit(0 if sys.argv[2] in ids else "当前运行的模型与所选模型不同，请先 ./stop.sh 再 ./start.sh")' "$workdir/models.json" "$MODEL_NAME"

# 只回收本次新建的 shim，不接管或清理启动前已有的容器。
if [ -z "$(docker compose ps --all --quiet h2o-shim)" ]; then
  shim_started=true
fi
docker compose up -d --build
echo "模型 $MODEL_NAME 和 H2O shim 已启动（shim 本机端口 8740）；统一接口：http://127.0.0.1/v1/systemone（需启动根目录总代理）。"
