#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export TZ=UTC
export PYTHONHASHSEED=0
if [ -f .venv/Scripts/python.exe ]; then
  .venv/Scripts/python.exe -m pytest "$@"
else
  .venv/bin/python -m pytest "$@"
fi
