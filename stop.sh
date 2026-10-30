#!/bin/bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
if [ "$#" -gt 1 ]; then
  echo "用法：./stop.sh [gateway|lifecycle|h2o|higgs|yue2|moss|paddleocr|qwenimage|seedvr2|wevisone|all]" >&2
  exit 1
fi
target="${1:-gateway}"
case "$target" in
  -h|--help)
    echo "用法：./stop.sh [gateway|lifecycle|h2o|higgs|yue2|moss|paddleocr|qwenimage|seedvr2|wevisone|all]（默认 gateway；lifecycle 拒绝中断在途请求）"
    exit 0 ;;
  gateway|lifecycle|h2o|higgs|yue2|moss|paddleocr|qwenimage|seedvr2|wevisone|all) ;;
  *) echo "未知服务：${target}；运行 ./stop.sh --help 查看目标。" >&2; exit 1 ;;
esac

stop_service() {
  if [ "$1" != gateway ]; then
    command -v launchctl >/dev/null || { echo "缺少命令：launchctl" >&2; return 1; }
  fi
  case "$1" in
    gateway|lifecycle) ;;
    *)
      if python3 -I ./runtime_mgmt/lifecycle_control.py unload "$1"; then
        return 0
      else
        result=$?
        if [ "$result" -ne 3 ]; then return "$result"; fi
      fi ;;
  esac
  case "$1" in
    gateway) docker compose down ;;
    lifecycle) python3 -I ./runtime_mgmt/lifecycle_control.py stop ;;
    h2o) ./composes/h2o-lightning-4b/stop.sh ;;
    higgs) ./composes/higgs-tts-3/stop.sh ;;
    yue2) ./composes/yue2-3b/stop.sh ;;
    moss) ./composes/moss-transcribe-diarize/stop.sh ;;
    paddleocr) ./composes/paddleocr-vl/stop.sh ;;
    qwenimage) ./composes/qwen-image-2.1-turbo/stop.sh ;;
    seedvr2) ./composes/seedvr2/stop.sh ;;
    wevisone) ./composes/wevisone-vl-4b/stop.sh ;;
  esac
}

services=("$target")
if [ "$target" = all ]; then
  # Close the entry point first, then unload models in reverse startup order.
  services=(gateway lifecycle wevisone seedvr2 qwenimage paddleocr moss yue2 higgs h2o)
fi
status=0
for service in "${services[@]}"; do
  if stop_service "$service"; then
    echo "$service 已停止。"
  else
    echo "$service 停止失败；请检查日志并重试。" >&2
    if [ "$service" = lifecycle ]; then
      echo "生命周期管理器未安全退出；不继续停止模型，避免中断在途请求。" >&2
      exit 1
    fi
    status=1
  fi
done
exit "$status"
