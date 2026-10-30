#!/bin/bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
prepare_only=false
case "$#" in
  0) ;;
  1)
    if [ "$1" != --prepare-only ]; then echo "用法：$0 [--prepare-only]" >&2; exit 1; fi
    prepare_only=true
    ;;
  *) echo "用法：$0 [--prepare-only]" >&2; exit 1 ;;
esac
for command in python3 modelscope launchctl curl lsof plutil; do
  command -v "$command" >/dev/null || { echo "缺少命令：$command" >&2; exit 1; }
done
if [ "$(uname -s)" != Darwin ] || [ "$(uname -m)" != arm64 ]; then
  echo "需要 Apple Silicon Mac；Linux 容器不能使用 Metal。" >&2
  exit 1
fi
python3 -c 'import platform,sys; sys.exit(0 if sys.version_info >= (3,9) and int(platform.mac_ver()[0].split(".")[0]) >= 15 else "需要 Python >= 3.9、macOS >= 15")'
DOMAIN="gui/$(id -u)"
SERVICE="$DOMAIN/com.local.moss-transcribe-diarize-metal"
PLIST="$HOME/Library/Application Support/moss-transcribe-diarize/com.local.moss-transcribe-diarize-metal.plist"
if launchctl print "$SERVICE" >/dev/null 2>&1; then
  echo "MOSS 任务已注册；请先 ./stop.sh。" >&2
  exit 1
fi
if lsof -nP -iTCP:8097 -sTCP:LISTEN >/dev/null 2>&1; then
  echo "8097 端口已占用；不停止其他进程。" >&2
  exit 1
fi
workdir="$(mktemp -d)"
started=false
cleanup() {
  status=$?
  if [ "$status" -ne 0 ] && [ "$started" = true ]; then
    launchctl bootout "$SERVICE" || echo "请运行 ./stop.sh。" >&2
    echo "启动失败，日志：$HOME/Library/Logs/moss-transcribe-diarize-metal*.log" >&2
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
base="$VLLM_BACKEND_PYTHON"
test -x "$base" || { echo "统一后端安装不完整。" >&2; exit 1; }
"$base" prepare_runtime.py > "$workdir/runtime"
IFS= read -r runtime < "$workdir/runtime"
python="$runtime/bin/python"
cli="$runtime/bin/vllm"
"$python" ../../common_env/check_versions.py --omni
"$python" prepare.py > "$workdir/model"
{
  IFS= read -r model_name
  IFS= read -r model_dir
} < "$workdir/model"
"$python" - "$cli" "$model_dir" "$model_name" "$PLIST" /opt/homebrew/bin <<'PY'
import os
from pathlib import Path
import plistlib
import runpy
import sys
import tempfile

cli, directory, name, target, brew_bin = sys.argv[1:]
logs = Path.home() / 'Library/Logs'
logs.mkdir(parents=True, exist_ok=True)
target = Path(target)
target.parent.mkdir(parents=True, exist_ok=True)
config = {
    'Label': 'com.local.moss-transcribe-diarize-metal',
    'ProgramArguments': [cli, 'serve', directory,
        '--served-model-name', name,
        '--trust-remote-code', '--host', '127.0.0.1', '--port', '8097',
        '--dtype', 'bfloat16', '--max-model-len', '8192', '--max-num-seqs', '1',
        '--enforce-eager', '--no-enable-prefix-caching', '--no-enable-chunked-prefill',
        '--no-async-scheduling', '--limit-mm-per-prompt', '{"audio":1,"image":0,"video":0}'],
    # launchd may lack TCC access to Documents; all model/code paths are absolute.
    'WorkingDirectory': str(target.parent), 'RunAtLoad': True, 'KeepAlive': False,
    'ExitTimeOut': 30, 'AbandonProcessGroup': False,
    'EnvironmentVariables': {
        'PATH': f'{Path(cli).parent}:{brew_bin}:/usr/bin:/bin:/usr/sbin:/sbin',
        'PYTHONUNBUFFERED': '1', 'OMP_NUM_THREADS': '2',
        'VLLM_WORKER_MULTIPROC_METHOD': 'spawn', 'VLLM_PLUGINS': 'metal,moss_stt',
        'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
        **runpy.run_path(str(Path.cwd().parents[1] / 'common_env' / 'prepare_optimizations.py'))[
            'optimization_environment'](),
    },
    'StandardOutPath': str(logs / 'moss-transcribe-diarize-metal.log'),
    'StandardErrorPath': str(logs / 'moss-transcribe-diarize-metal.error.log'),
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
  echo "模型 $model_name 已准备：${PLIST}；未加载模型。"
  exit 0
fi
launchctl bootstrap "$DOMAIN" "$PLIST"
started=true
echo "等待 vLLM + vLLM Metal MOSS 加载……"
deadline=$((SECONDS + 600))
until curl --max-time 2 -fsS http://127.0.0.1:8097/health >/dev/null 2>&1; do
  if ! state="$(launchctl print "$SERVICE" 2>/dev/null)"; then
    echo "模型任务已卸载，请检查日志。" >&2
    exit 1
  fi
  # bootstrap may first report 'spawn scheduled'; only a completed launch has an exit code.
  if grep -Eq 'last exit code = -?[0-9]+' <<< "$state" && ! grep -q 'state = running' <<< "$state"; then
    echo "模型进程已退出，请检查日志。" >&2
    exit 1
  fi
  if [ "$SECONDS" -ge "$deadline" ]; then echo "模型服务未就绪。" >&2; exit 1; fi
  sleep 2
done
curl --max-time 5 -fsS http://127.0.0.1:8097/v1/models | "$python" -c 'import json,sys; assert any(m.get("id")==sys.argv[1] for m in json.load(sys.stdin).get("data",[])), "端口不是预期模型"' "$model_name"
echo "模型 $model_name 已启动：127.0.0.1:8097；统一接口：http://127.0.0.1/v1/audio/transcriptions（需启动根目录总代理）。"
