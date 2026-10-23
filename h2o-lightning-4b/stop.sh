#!/bin/bash
set -euo pipefail
cd -- "$(dirname -- "$0")"

SERVICE="gui/$(id -u)/com.local.h2o-lightning-vllm"
status=0

# 即使 Docker 不可用，也要继续停止本机模型。
if ! docker compose down; then
  echo "关闭 Compose 失败；仍会尝试停止模型。" >&2
  status=1
fi

if launchctl print "$SERVICE" >/dev/null 2>&1; then
  if ! launchctl bootout "$SERVICE"; then
    echo "停止模型失败。" >&2
    status=1
  else
    # bootout 返回时，launchd 可能还在清理进程。
    deadline=$((SECONDS + 30))
    while launchctl print "$SERVICE" >/dev/null 2>&1; do
      if [ "$SECONDS" -ge "$deadline" ]; then
        echo "模型服务尚未卸载，请稍后重试。" >&2
        status=1
        break
      fi
      sleep 1
    done
  fi
fi

if [ "$status" -eq 0 ]; then
  echo "Compose 和模型已停止。"
fi
exit "$status"
