#!/bin/bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
if [ "$#" -ne 0 ]; then
  echo '用法：./stop.sh' >&2
  exit 1
fi
command -v launchctl >/dev/null || { echo "缺少命令：launchctl" >&2; exit 1; }
SERVICE="gui/$(id -u)/com.local.wevisone-vl-4b-metal"
if launchctl print "$SERVICE" >/dev/null 2>&1; then
  launchctl bootout "$SERVICE"
  deadline=$((SECONDS + 30))
  while launchctl print "$SERVICE" >/dev/null 2>&1; do
    if [ "$SECONDS" -ge "$deadline" ]; then
      echo "模型尚未卸载，请稍后重试。" >&2
      exit 1
    fi
    sleep 1
  done
fi
echo "WeVisOne-VL-4B 模型已停止（不影响总代理或其他模型）。"
