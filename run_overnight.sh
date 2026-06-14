#!/usr/bin/env bash
# 夜间无人值守 — 前台/tmux 运行（自动 tee 到日志）
# 用法: bash run_overnight.sh

set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs
python scripts/run_overnight.py 2>&1 | tee -a logs/overnight_stdout.log
