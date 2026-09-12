#!/usr/bin/env bash
set -euo pipefail
# Template: full production calibration pipeline.
# This does NOT invent calibration data. Replace the placeholder survey JSON with real measurements.

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VIDEO="${1:-C:/Users/alexe/Downloads/ST_2026-09-04_19-30-04.mp4}"
MODEL="fisheye"  # or brown; choose explicitly; never infer only from video appearance

.venv/Scripts/python.exe calibrate.py intrinsics \
  --video calibration/prod/board_training.mp4 \
  --heldout-video calibration/prod/board_heldout.mp4 \
  --cols 8 --rows 6 --square-m 0.04 --stride 15 \
  --model "$MODEL" --output calibration/prod/intrinsic.json

# User must supply real survey measurements; placeholder survey.json below is NOT real data.
# Do NOT assume any xyz from approximate installation geometry.
.venv/Scripts/python.exe calibrate.py survey \
  --intrinsics calibration/prod/intrinsic.json \
  --survey calibration/prod/survey.json \
  --calibration-id "production-v1" --max-check-error-m 0.05 \
  --output calibration/prod/camera.json

echo "Calibration artifacts saved; verify independent check errors and survey points physically."
