#!/usr/bin/env python3
"""Run the capture wheel detector on all catalogue audit corrected full frames."""
import argparse
import json
from pathlib import Path
import sys
import time

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vehicle_metrology.wheels import MODEL_ID, MODEL_REVISION, MODEL_FILES, PROMPT, THRESHOLD, file_digest
from web_app.wheel_measurement import infer_wheels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda:0'], default='cuda:0')
    args = parser.parse_args()
    import torch
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    size = json.loads(args.manifest.read_text())['profile']['image_size']
    rows = [r for r in json.loads(args.comparison.read_text())['rows'] if r.get('comparison_eligible')]
    result = dict(model=MODEL_ID, revision=MODEL_REVISION, model_sha256=MODEL_FILES['model.safetensors'],
        text=PROMPT, threshold=THRESHOLD, text_threshold=THRESHOLD, device=args.device,
        comparison_sha256=file_digest(args.comparison), coordinate_manifest_sha256=file_digest(args.manifest),
        implementation_sha256=file_digest(ROOT/'web_app/wheel_measurement.py'), rows={})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    failures = []
    for row in rows:
        start = time.monotonic()
        path = args.comparison.parent/row['image']
        try:
            image = cv2.imread(str(path))
            if row.get('image_space') != 'corrected_full_resolution' or image is None or list(image.shape[1::-1]) != size:
                raise ValueError('Expected corrected full-resolution audit frame')
            details = infer_wheels(image, row['bbox'], device=args.device)
            entry = dict(status='ok', **details, image=row['image'], frame=row['frame'], vehicle_bbox=row['bbox'],
                         corrected_image_sha256=file_digest(path), elapsed_seconds=time.monotonic()-start)
        except Exception as exc:
            entry = dict(status='failed', reason=str(exc))
            failures.append(row['id'])
        result['rows'][row['id']] = entry
        temporary = args.output.with_suffix('.tmp')
        temporary.write_text(json.dumps(result, indent=2)+'\n')
        temporary.replace(args.output)
        print(row['id'], entry['status'], round(time.monotonic()-start, 2), 'seconds', flush=True)
    if failures:
        raise SystemExit('Failed detections retained; incomplete benchmark: '+', '.join(failures))


if __name__ == '__main__':
    main()
