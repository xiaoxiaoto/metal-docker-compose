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
for command in python3 git modelscope launchctl curl lsof plutil; do
  command -v "$command" >/dev/null || { echo "缺少命令：$command" >&2; exit 1; }
done
if [ "$(uname -s)" != Darwin ] || [ "$(uname -m)" != arm64 ]; then
  echo "vLLM-Omni Metal 需要 Apple Silicon Mac。" >&2
  exit 1
fi
export YUE2_METAL_QUANTIZATION="${YUE2_METAL_QUANTIZATION:-${VLLM_OMNI_METAL_QUANTIZATION:-off}}"
case "$YUE2_METAL_QUANTIZATION" in
  dynamic)
    export VLLM_OMNI_METAL_ANE=off
    export YUE2_METAL_QUANTIZATION_PROFILE="${YUE2_METAL_QUANTIZATION_PROFILE:-${VLLM_OMNI_METAL_QUANTIZATION_PROFILE:-$HOME/Library/Application Support/yue2-3b/dynamic-profile.json}}"
    ;;
  off) ;;
  *) echo 'YUE2_METAL_QUANTIZATION 必须是 off 或 dynamic。' >&2; exit 1 ;;
esac
case "${VLLM_OMNI_METAL_ANE:-auto}" in
  auto|off) ;;
  *) echo 'VLLM_OMNI_METAL_ANE 必须是 auto 或 off。' >&2; exit 1 ;;
esac
DOMAIN="gui/$(id -u)"
SERVICE="$DOMAIN/com.local.yue2-3b-metal"
PLIST="$HOME/Library/Application Support/yue2-3b/com.local.yue2-3b-metal.plist"
if launchctl print "$SERVICE" >/dev/null 2>&1; then
  echo "YuE2 任务已注册；先运行 ./stop.sh，再 ./start.sh，以免继续使用旧后端。" >&2
  exit 1
fi
if lsof -nP -iTCP:8096 -sTCP:LISTEN >/dev/null 2>&1; then
  echo "8096 端口已占用；不终止其他进程。" >&2
  exit 1
fi
python3 -c 'import platform,sys; sys.exit(0 if sys.version_info >= (3,9) and int(platform.mac_ver()[0].split(".")[0]) >= 15 else "需要 Python >= 3.9、macOS >= 15")'
python3 - <<'PY'
import json
import os
from pathlib import Path
import stat

if os.environ["YUE2_METAL_QUANTIZATION"] == "dynamic":
    profile = Path(os.environ["YUE2_METAL_QUANTIZATION_PROFILE"]).expanduser()
    if not profile.is_absolute() or not profile.is_file() or profile.is_symlink():
        raise SystemExit("dynamic 需要现有的绝对路径私有 profile；先运行 prepare_dynamic_quantization.py。")
    info = profile.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o600:
        raise SystemExit("dynamic profile 必须是当前用户拥有的 0600 普通文件。")
    if info.st_size > 4 * 1024**2:
        raise SystemExit("dynamic profile 超过4MiB边界。")
    data = json.loads(profile.read_text())
    profiles = data.get("profiles")
    candidates = list(profiles.values()) if isinstance(profiles, dict) else [data]
    if not candidates or any(
        not isinstance(candidate, dict)
        or candidate.get("calibration", {}).get("approved") is not True
        for candidate in candidates
    ):
        raise SystemExit("dynamic profile 必须经过完整批准。")
    selection = Path.home() / "Library/Application Support/yue2-3b/backend-wheel"
    if not os.environ.get("VLLM_OMNI_METAL_WHEEL") and not selection.is_file():
        raise SystemExit("dynamic 需要 VLLM_OMNI_METAL_WHEEL 或已保存的私有 backend-wheel 选择。")
PY
workdir="$(mktemp -d)"
started=false
cleanup() {
  status=$?
  if [ "$status" -ne 0 ] && [ "$started" = true ]; then
    launchctl bootout "$SERVICE" || echo "请运行 ./stop.sh 停止模型。" >&2
    echo "启动失败，日志：$HOME/Library/Logs/yue2-3b-metal*.log" >&2
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
# Patch a private Omni wheel; never change the common backend's model implementation.
"$python" -I prepare_runtime.py > "$workdir/runtime"
IFS= read -r runtime < "$workdir/runtime"
python="$runtime/bin/python"
cli="$runtime/bin/vllm-omni"
test -x "$python" && test -x "$cli" || { echo "YuE2 私有环境安装不完整。" >&2; exit 1; }
"$python" -I -m vllm_omni_metal.ane.build --help >/dev/null
"$python" -I ../check_versions.py --omni
# Exercise real MPS sampling/NAR/VAE before downloading large weights.
VLLM_OMNI_METAL_QUANTIZATION=off YUE2_METAL_QUANTIZATION=off VLLM_PLUGINS=metal,vllm_omni_register_models "$python" -I check_metal.py
"$python" -I prepare.py > "$workdir/model"
{
  IFS= read -r model_name
  IFS= read -r model_dir
  IFS= read -r vae_dir
} < "$workdir/model"
"$python" - "$cli" "$model_dir" "$vae_dir" "$model_name" "$PLIST" <<'PY'
import os
from pathlib import Path
import plistlib
import shutil
import sys
import tempfile

cli, directory, vae, name, target = sys.argv[1:]
project = Path.cwd()
logs = Path.home() / 'Library/Logs'
logs.mkdir(parents=True, exist_ok=True)
target = Path(target)
target.parent.mkdir(parents=True, exist_ok=True)
# launchd may not have TCC access to Documents, including the source deploy profile.
deploy = target.parent / 'metal-deploy.yaml'
shutil.copyfile(project / 'metal-deploy.yaml', deploy)
mode = os.environ['YUE2_METAL_QUANTIZATION']
profile = None
if mode == 'dynamic':
    source = Path(os.environ['YUE2_METAL_QUANTIZATION_PROFILE']).expanduser()
    profile = target.parent / 'dynamic-profile.json'
    if source != profile:
        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as file:
            temporary = Path(file.name)
            file.write(source.read_bytes())
        try:
            os.replace(temporary, profile)
        finally:
            temporary.unlink(missing_ok=True)
config = {
    'Label': 'com.local.yue2-3b-metal',
    'ProgramArguments': [cli, 'serve', str(directory), '--omni',
                         '--served-model-name', name,
                         '--deploy-config', str(deploy),
                         '--host', '127.0.0.1', '--port', '8096'],
    'WorkingDirectory': str(target.parent), 'RunAtLoad': True, 'KeepAlive': False,
    'ExitTimeOut': 30, 'AbandonProcessGroup': False,
    'EnvironmentVariables': {
        'PATH': f'{Path(cli).parent}:/usr/bin:/bin:/usr/sbin:/sbin',
        'PYTHONUNBUFFERED': '1', 'OMP_NUM_THREADS': '2',
        'VLLM_WORKER_MULTIPROC_METHOD': 'spawn',
        'VLLM_PLUGINS': 'metal,vllm_omni_register_models',
        'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
        'YUE2_VAE': vae,
        'VLLM_OMNI_METAL_ANE': os.environ.get('VLLM_OMNI_METAL_ANE', 'auto'),
        'YUE2_METAL_QUANTIZATION': mode,
    },
    'StandardOutPath': str(logs / 'yue2-3b-metal.log'),
    'StandardErrorPath': str(logs / 'yue2-3b-metal.error.log'),
}
if profile is not None:
    config['EnvironmentVariables']['YUE2_METAL_QUANTIZATION_PROFILE'] = str(profile)
with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as file:
    temporary = Path(file.name)
    plistlib.dump(config, file)
try:
    os.replace(temporary, target)
finally:
    temporary.unlink(missing_ok=True)
PY

"$python" -I -c 'import shutil; assert shutil.disk_usage(".").free >= 2 * 1024**3, "YuE2 部署需保留2GiB磁盘空间"'
plutil -lint "$PLIST" >/dev/null
if [ "$prepare_only" = true ]; then
  echo "模型 $model_name 已准备：${PLIST}；未加载模型。"
  exit 0
fi
"$python" -I -c 'import psutil; assert psutil.virtual_memory().available >= 16 * 1024**3, "YuE2 启动前需至少16GiB可用RAM；不停止其他服务"'
launchctl bootstrap "$DOMAIN" "$PLIST"
started=true

echo "等待统一后端的 vLLM-Omni + Omni Metal 原生 YuE2 单阶段加载（YuE2 私有 MPS 修复）……"
deadline=$((SECONDS + 600))
until curl --max-time 2 -fsS http://127.0.0.1:8096/health >/dev/null 2>&1; do
  if ! state="$(launchctl print "$SERVICE" 2>/dev/null)"; then
    echo "模型任务已卸载，请检查日志。" >&2
    exit 1
  fi
  # A newly bootstrapped task can be 'spawn scheduled' and have never exited.
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
curl --max-time 5 -fsS http://127.0.0.1:8096/v1/models | "$python" -I -c 'import json,sys; models=json.load(sys.stdin); assert any(m.get("id")==sys.argv[1] for m in models.get("data", [])), "8096 端口不是预期模型"' "$model_name"
echo "模型 $model_name 已启动：127.0.0.1:8096；统一接口：http://127.0.0.1/v1/audio/speech（需启动根目录总代理；请继续生成验收）。"
