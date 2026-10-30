#!/bin/bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
if [ "$#" -gt 1 ]; then
  echo "用法：./start.sh [gateway|lifecycle|h2o|higgs|yue2|moss|paddleocr|qwenimage|seedvr2|wevisone|all]" >&2
  exit 1
fi
target="${1:-gateway}"
case "$target" in
  -h|--help)
    echo "用法：./start.sh [gateway|lifecycle|h2o|higgs|yue2|moss|paddleocr|qwenimage|seedvr2|wevisone|all]（默认 gateway；模式见 runtime_mgmt/model-lifecycle.json）"
    exit 0 ;;
  gateway|lifecycle|h2o|higgs|yue2|moss|paddleocr|qwenimage|seedvr2|wevisone|all) ;;
  *) echo "未知服务：${target}；运行 ./start.sh --help 查看目标。" >&2; exit 1 ;;
esac
# H2O 由用户手动下载；缺少新版发布时不触发共享后端升级或启动其他服务。
if [ "$target" = h2o ] || [ "$target" = all ]; then
  python3 ./composes/h2o-lightning-4b/prepare.py --check-model >/dev/null
fi
commands=(docker curl launchctl python3)
for command in "${commands[@]}"; do
  command -v "$command" >/dev/null || { echo "缺少命令：$command" >&2; exit 1; }
done
docker info >/dev/null
docker compose config --quiet
python3 -I ./runtime_mgmt/lifecycle_control.py validate
# Every explicit root start rebuilds the shared backend, including gateway/lifecycle.
backend="$(python3 -I ./common_env/prepare_backend.py)"
export VLLM_BACKEND_PYTHON="$backend/bin/python"
python3 -I ./runtime_mgmt/lifecycle_control.py start
if [ "$target" = lifecycle ]; then
  python3 -I ./runtime_mgmt/lifecycle_control.py status
  exit 0
fi
if [ "$target" = gateway ] || [ "$target" = wevisone ] || [ "$target" = all ] || \
    [ -z "$(docker compose ps --status running --quiet gateway)" ] || \
    ! curl --connect-timeout 1 --max-time 2 -fsS http://127.0.0.1/health >/dev/null; then
  docker compose build
  docker compose run --rm --no-deps gateway nginx -t
  docker compose up -d
  deadline=$((SECONDS + 30))
  until curl --connect-timeout 1 --max-time 2 -fsS http://127.0.0.1/health >/dev/null; do
    if [ "$SECONDS" -ge "$deadline" ]; then
      echo "总代理未就绪，请运行 docker compose logs gateway。" >&2
      exit 1
    fi
    sleep 1
  done
fi
echo "总代理已启动：http://127.0.0.1/v1/。"

models=()
dynamic_models=()
start_model() {
  local name="$1" directory label model port
  case "$name" in
    h2o)
      directory=h2o-lightning-4b; label=com.local.h2o-lightning-vllm
      model=h2o-lightning-4b; port=8000 ;;
    higgs)
      directory=higgs-tts-3; label=com.local.higgs-tts-3-metal
      model=higgs-audio-v3-tts-4b; port=8095 ;;
    yue2)
      directory=yue2-3b; label=com.local.yue2-3b-metal
      model=yue2-3b; port=8096 ;;
    moss)
      directory=moss-transcribe-diarize; label=com.local.moss-transcribe-diarize-metal
      model=moss-transcribe-diarize; port=8097 ;;
    paddleocr)
      directory=paddleocr-vl; label=com.local.paddleocr-vl-metal
      model=paddleocr-vl; port=8098 ;;
    qwenimage)
      directory=qwen-image-2.1-turbo; label=com.local.qwen-image-2.1-turbo-metal
      model=qwen-image-2.1-turbo; port=8099 ;;
    seedvr2)
      directory=seedvr2; label=com.local.seedvr2-metal
      model=seedvr2; port=8100 ;;
    wevisone)
      directory=wevisone-vl-4b; label=com.local.wevisone-vl-4b-metal
      model=wevisone-vl-4b; port=8101 ;;
  esac
  local deployment="composes/$directory"
  if [ "$(python3 -I ./runtime_mgmt/lifecycle_control.py mode "$model")" = dynamic ]; then
    # Every explicit model start rebuilds the backend and its dependent overlays.
    # Regenerate the selected deployment before validating its cold-start contract.
    echo "$name 重新准备模型私有环境和启动配置；完成后保持冷状态。"
    "./$deployment/start.sh" --prepare-only
    python3 -I ./runtime_mgmt/lifecycle_control.py check-dynamic "$model"
    dynamic_models+=("$model")
    echo "$name 已配置 dynamic；首次推理加载，模型列表查询不唤醒。"
    return
  fi
  if [ "$name" != h2o ] && launchctl print "gui/$(id -u)/$label" >/dev/null 2>&1; then
    if ! curl --connect-timeout 1 --max-time 5 -fsS "http://127.0.0.1:$port/health" >/dev/null; then
      echo "$name 任务已注册但尚未就绪；请检查日志，必要时先运行 ./$deployment/stop.sh。" >&2
      return 1
    fi
    python3 -I ./common_env/check_versions.py --omni --plist "$HOME/Library/Application Support/$directory/$label.plist"
    echo "$name 已运行，跳过重复加载。"
  else
    echo "启动 ${name}……"
    "./$deployment/start.sh"
  fi
  if [ "$name" = h2o ]; then
    if ! curl --connect-timeout 1 --max-time 5 --retry 5 --retry-all-errors --retry-connrefused --retry-delay 1 \
        -fsS --get http://127.0.0.1:8740/v1/models --data-urlencode "model=$model" | \
        python3 -c 'import json,sys; assert any(m.get("id")==sys.argv[1] for m in json.load(sys.stdin).get("data",[])), "H2O shim 模型名称不一致"' "$model"; then
      echo "H2O vLLM 已就绪，但 shim 尚未就绪或模型名称不一致；请检查 H2O Compose。" >&2
      return 1
    fi
  fi
  models+=("$model")
  echo "$name 后端已就绪，等待最终统一入口校验。"
}

case "$target" in
  all)
    echo "按 runtime_mgmt/model-lifecycle.json 依次部署目标：resident加载，dynamic缺少部署时自动准备；任一步失败停止后续目标。"
    for service in h2o higgs yue2 moss paddleocr qwenimage seedvr2 wevisone; do
      start_model "$service"
    done ;;
  gateway) ;;
  *) start_model "$target" ;;
esac

if [ "${#models[@]}" -gt 0 ]; then
  if ! curl --connect-timeout 2 --max-time 10 -fsS http://127.0.0.1/v1/models | python3 -c '
import json, sys
result = json.load(sys.stdin)
if not isinstance(result, dict) or result.get("object") != "list" or not isinstance(result.get("data"), list):
    sys.exit("统一入口未返回 OpenAI 模型列表")
available = {model["id"] for model in result["data"] if model.get("object") == "model"}
missing = set(sys.argv[1:]) - available
if missing:
    sys.exit("指定模型未就绪：" + ", ".join(sorted(missing)))
' "${models[@]}"; then
    echo "最终统一入口校验失败；请检查指定模型及代理配置。" >&2
    exit 1
  fi
  echo "统一入口校验通过：${models[*]}。"
fi
if [ "${#dynamic_models[@]}" -gt 0 ]; then
  python3 -I ./runtime_mgmt/lifecycle_control.py status
  echo "动态目标已登记：${dynamic_models[*]}；未以模型列表冒充冷启动/推理验收。"
fi
