#!/usr/bin/env python3
"""Extract pinned-model masks from the audit's existing corrected full frames."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vehicle_metrology.outline import WEIGHTS_SHA256
from web_app.outline_measurement import infer_outline


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda:0'], default='cuda:0')
    args = parser.parse_args()
    import torch
    cv2.setNumThreads(2)
    torch.set_num_threads(4)
    data = json.loads(args.comparison.read_text())
    size = json.loads(args.manifest.read_text())['profile']['image_size']
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = dict(model_sha256=WEIGHTS_SHA256, imgsz=1280, retina_masks=True, device=args.device,
        comparison_sha256=digest(args.comparison), coordinate_manifest_sha256=digest(args.manifest), masks={})
    rows = [r for r in data['rows'] if r.get('comparison_eligible')]
    failed = []
    for index, row in enumerate(rows):
        path = args.comparison.parent/row['image']
        try:
            image = cv2.imread(str(path))
            if row.get('image_space') != 'corrected_full_resolution' or image is None or list(image.shape[1::-1]) != size:
                raise ValueError('Expected corrected full-resolution audit frame')
            mask, details = infer_outline(image, row['bbox'], device=args.device)
            target = args.output_dir/(row['id']+'.npz')
            np.savez_compressed(target, mask=mask)
            results['masks'][row['id']] = dict(status='ok', sha256=digest(target),
                corrected_image_sha256=digest(path), **details)
        except Exception as exc:
            failed.append(row['id'])
            results['masks'][row['id']] = dict(status='failed', reason=str(exc))
        if (index+1) % 10 == 0:
            print(f'{index+1}/{len(rows)} frames; {len(failed)} failed', flush=True)
    (args.output_dir/'mask-manifest.json').write_text(json.dumps(results, indent=2)+'\n')
    if failed:
        raise SystemExit('Unscored mask failures retained in manifest: '+', '.join(failed))
    print(f'Saved all {len(rows)} masks. No station data or original videos changed.')


if __name__ == '__main__':
    main()
