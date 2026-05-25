#!/usr/bin/env bash
# 启动具身导航 Web 界面
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
# AGENT_PIPELINE=debug|top|v2  (default: debug)
# WEB_MEDIA_ROOT=/autodl-tmp
exec python -m uvicorn web.app:app --host 0.0.0.0 --port "${WEB_PORT:-8765}" --reload
