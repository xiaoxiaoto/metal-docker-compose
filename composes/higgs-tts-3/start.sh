#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
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
for command in python3 git modelscope launchctl curl lsof plutil; do
  command -v "$command" >/dev/null || { echo "缺少命令：$command" >&2; exit 1; }
done
if [ "$(uname -s)" != Darwin ] || [ "$(uname -m)" != arm64 ]; then
  echo "vLLM-Omni Metal 需要 Apple Silicon Mac。" >&2
  exit 1
fi
DOMAIN="gui/$(id -u)"
SERVICE="$DOMAIN/com.local.higgs-tts-3-metal"
PLIST="$HOME/Library/Application Support/higgs-tts-3/com.local.higgs-tts-3-metal.plist"
if launchctl print "$SERVICE" >/dev/null 2>&1; then
  echo "Higgs 任务已注册；先运行 ./stop.sh，再 ./start.sh，以免继续使用旧后端。" >&2
  exit 1
fi
if lsof -nP -iTCP:8095 -sTCP:LISTEN >/dev/null 2>&1; then
  echo "8095 端口已占用；不终止其他进程。" >&2
  exit 1
fi
python3 -c 'import platform,sys; sys.exit(0 if sys.version_info >= (3,9) and int(platform.mac_ver()[0].split(".")[0]) >= 15 else "需要 Python >= 3.9、macOS >= 15")'
workdir="$(mktemp -d)"
started=false
cleanup() {
  status=$?
  if [ "$status" -ne 0 ] && [ "$started" = true ]; then
    launchctl bootout "$SERVICE" || echo "请运行 ./stop.sh 停止模型。" >&2
    echo "启动失败，日志：$HOME/Library/Logs/higgs-tts-3-metal*.log" >&2
  fi
  rm -rf -- "$workdir"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

python3 prepare.py > "$workdir/model"
{
  IFS= read -r model_name
  IFS= read -r model_dir
} < "$workdir/model"

if [ -z "${VLLM_BACKEND_PYTHON:-}" ]; then
  backend="$(python3 ../../common_env/prepare_backend.py)"
  export VLLM_BACKEND_PYTHON="$backend/bin/python"
fi
base_python="$VLLM_BACKEND_PYTHON"
test -x "$base_python" || { echo "统一后端安装不完整。" >&2; exit 1; }
runtime="$("$base_python" -I prepare_runtime.py)"
python="$runtime/bin/python"
cli="$runtime/bin/vllm-omni"
test -x "$python" && test -x "$cli" || { echo "Higgs 私有后端安装不完整。" >&2; exit 1; }
"$python" -I - "$cli" "$model_dir" "$model_name" "$PLIST" "$prepare_only" <<'PY'
import os
from pathlib import Path
import plistlib
import sys
import tempfile
import shutil
import psutil

cli, directory, name, target, prepare_only = sys.argv[1:]
cli = str(Path(cli).resolve())
project = Path.cwd()
logs = Path.home() / 'Library/Logs'
logs.mkdir(parents=True, exist_ok=True)
target = Path(target)
target.parent.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(project.parent.parent / "common_env"))
from prepare_optimizations import optimization_environment
optimization = optimization_environment()
unsupported = set(filter(None, optimization['METAL_MODEL_OPTIMIZERS'].split(','))) - {'dynamic_quantization'}
if unsupported:
    raise ValueError(f'Higgs Omni 不支持这些 provider: {sorted(unsupported)}')
if optimization['METAL_QUANTIZATION'] == 'dynamic':
    print('Higgs：加载时按真实执行scope严格核验DynamicEngine批准profile。')
if prepare_only != 'true' and psutil.virtual_memory().available < 16 * 1024**3:
    raise MemoryError('启动Higgs需16GiB可用RAM，不停止其它服务')
if shutil.disk_usage(target.parent).free < 2 * 1024**3:
    raise OSError('磁盘至少保留2GiB')
config = {
    'Label': 'com.local.higgs-tts-3-metal',
    'ProgramArguments': [cli, 'serve', str(directory), '--omni',
                         '--served-model-name', name,
                         '--deploy-config', str(project / 'metal-deploy.yaml'),
                         '--host', '127.0.0.1', '--port', '8095'],
    'WorkingDirectory': str(project), 'RunAtLoad': True, 'KeepAlive': False,
    'ExitTimeOut': 30, 'AbandonProcessGroup': False,
    'EnvironmentVariables': {
        'PATH': f'{Path(cli).parent}:/usr/bin:/bin:/usr/sbin:/sbin',
        'PYTHONUNBUFFERED': '1', 'OMP_NUM_THREADS': '2',
        'VLLM_WORKER_MULTIPROC_METHOD': 'spawn',
        'VLLM_PLUGINS': 'metal,vllm_omni_register_models',
        'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
        **optimization,
        'VLLM_OMNI_METAL_ANE': 'off',
        'HIGGS_DEPLOY_CONFIG': str(project / 'metal-deploy.yaml'),
        'VLLM_ALLOW_INSECURE_SERIALIZATION': '0',
    },
    'StandardOutPath': str(logs / 'higgs-tts-3-metal.log'),
    'StandardErrorPath': str(logs / 'higgs-tts-3-metal.error.log'),
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

echo "等待统一后端的 vLLM-Omni + Omni Metal 原生双阶段加载……"
deadline=$((SECONDS + 600))
until curl --max-time 2 -fsS http://127.0.0.1:8095/health >/dev/null 2>&1; do
  if ! state="$(launchctl print "$SERVICE" 2>/dev/null)"; then
    echo "模型任务已卸载，请检查日志。" >&2
    exit 1
  fi
  # bootstrap may first report xpcproxy/spawn scheduled; only a completed launch has an exit status.
  if grep -Eq 'last (exit code = -?[0-9]+|terminating signal =)' <<< "$state" && ! grep -q 'state = running' <<< "$state"; then
    echo "模型进程已退出，请检查日志。" >&2
    exit 1
  fi
  if [ "$SECONDS" -ge "$deadline" ]; then
    echo "模型服务未就绪。" >&2
    exit 1
  fi
  sleep 2
done
# Omni /health has no JSON body; verify the served model through /v1/models.
curl --max-time 5 -fsS http://127.0.0.1:8095/v1/models | "$python" -c 'import json,sys; models=json.load(sys.stdin); assert any(m.get("id")==sys.argv[1] for m in models.get("data", [])), "8095 端口不是预期模型"' "$model_name"
echo "模型 $model_name 已启动：127.0.0.1:8095；统一接口：http://127.0.0.1/v1/audio/speech（需启动根目录总代理）。"
