#!/bin/bash
set -euo pipefail
if [ "${1:-}" = --help ]; then
  echo '用法：./stop.sh（仅卸载 com.local.seedvr2-metal）；--help 无副作用'
  exit 0
fi
[ "$#" -eq 0 ] || { echo '不支持的参数' >&2; exit 1; }
cd -- "$(dirname -- "$0")"
command -v launchctl >/dev/null || { echo '缺少命令：launchctl' >&2; exit 1; }
SERVICE="gui/$(id -u)/com.local.seedvr2-metal"
if launchctl print "$SERVICE" >/dev/null 2>&1; then
  launchctl bootout "$SERVICE"
  deadline=$((SECONDS + 30))
  while launchctl print "$SERVICE" >/dev/null 2>&1; do
    if [ "$SECONDS" -ge "$deadline" ]; then
      echo '模型尚未卸载，请稍后重试。' >&2; exit 1
    fi
    sleep 1
  done
fi
echo 'SeedVR2 已停止（不影响总代理或其他模型）。'
