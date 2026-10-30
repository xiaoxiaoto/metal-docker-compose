#!/bin/bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
SERVICE="gui/$(id -u)/com.local.yue2-3b-metal"
status=0
if launchctl print "$SERVICE" >/dev/null 2>&1; then
  if ! launchctl bootout "$SERVICE"; then
    status=1
  else
    deadline=$((SECONDS + 30))
    while launchctl print "$SERVICE" >/dev/null 2>&1; do
      if [ "$SECONDS" -ge "$deadline" ]; then
        echo "模型尚未卸载，请稍后重试。" >&2
        status=1
        break
      fi
      sleep 1
    done
  fi
fi
if [ "$status" -eq 0 ]; then echo "YuE2 Metal 模型已停止（不影响总代理或其他模型）。"; fi
exit "$status"
