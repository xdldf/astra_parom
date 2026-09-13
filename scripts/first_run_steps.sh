#!/usr/bin/env bash
# Concrete first-run sequence. Run each step one at a time in a real terminal.
# Assumes you have real measurements; does NOT invent any values.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# 1. Check the video is readable (no synthetic assumption)
echo '=== 1. Video check ==='
.venv/Scripts/python.exe -c "
import cv2; c=cv2.VideoCapture('C:/Users/alexe/Downloads/ST_2026-09-04_19-30-04.mp4')
print('Video open:', c.isOpened(), '| size:', c.get(3), 'x', c.get(4), '| fps:', c.get(5))
c.release()
"

# 2. Inspect one frame and save a reference image for manual point selection
echo '=== 2. Reference frame ==='
.venv/Scripts/python.exe annotate_road.py --video C:/Users/alexe/Downloads/ST_2026-09-04_19-30-04.mp4 --calibration calibration/prod/prototype.json --output calibration/prod/reference_inspection.jpg --frame 3000

echo 'Look at calibration/prod/reference_inspection.jpg. Identify same-side front/rear contact pixels and persistent body extreme pixels by eye or with an image viewer.'
echo 'Then fill in the JSON template: calibration/prod/example_observation_template.json (copy to your observations file).'
echo 'Every value must be from your actual observation; do not guess model/catalog dimensions.'

# 3. Interactive annotation (run this in a REAL terminal, not background, so input works)
echo '=== 3. Interactive annotation (optional alternative) ==='
echo 'If you prefer clicking: run in a real terminal (not background):'
echo '  PYTHONPATH=C:/job/astra_paroms .venv/Scripts/python.exe calibration/prod/interactive_measure.py --video C:/Users/alexe/Downloads/ST_2026-09-04_19-30-04.mp4 --calibration calibration/prod/prototype.json --track-id first-vehicle --sigma-px 2 --output calibration/prod/first_annotations.json'

# 4. After a real calibration (survey + intrinsics) exists, compute measurement
echo '=== 4. Measurement (only after real calibration exists) ==='
echo 'Replace calibration/prod/prototype.json with a real calibrated camera.json before running.'
echo 'Then run: .venv/Scripts/python.exe measure_from_annotations.py --video VIDEO --calibration calibration/prod/camera.json --observations YOUR_OBSERVATIONS.json --track-id first-vehicle --output-csv calibration/prod/result.csv'
echo 'The measurement CSV will contain length_m only if observations and calibration exist and are valid; it will be null/rejected otherwise.'
