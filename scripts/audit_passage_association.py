#!/usr/bin/env python3
"""Verify additional catalogue views using every intervening corrected video frame."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vehicle_metrology.detection import predict_vehicle_boxes
from vehicle_metrology.passage_association import associate_view
from vehicle_metrology.wheels import file_digest
from web_app.workbench import Profile, corrected

DETECTOR_SHA256 = '401cea9ab23ad19246ff7744859816bc599f350e93c9dd30367b6f0a0745d0b7'


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ('comparison', 'profile', 'extraction', 'weights', 'output-dir'):
        parser.add_argument('--'+option, type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda:0'], default='cuda:0')
    parser.add_argument('--only', nargs='+', help='Partial audit for investigation; insufficient for benchmarking')
    args = parser.parse_args()
    import torch
    from ultralytics import YOLO
    cv2.setNumThreads(2)
    torch.set_num_threads(4)
    profile = Profile.model_validate_json(args.profile.read_text())
    extraction = json.loads(args.extraction.read_text())
    if (file_digest(args.weights) != DETECTOR_SHA256
            or file_digest(args.comparison) != extraction['comparison_sha256']
            or file_digest(args.profile) != extraction['profile_sha256']):
        raise ValueError('Detector, comparison or profile provenance mismatch')
    signature = dict(extraction_sha256=file_digest(args.extraction), comparison_sha256=file_digest(args.comparison),
        profile_sha256=file_digest(args.profile), detector_sha256=DETECTOR_SHA256, device=args.device,
        imgsz=640, confidence=.3, image_space='lens_corrected_full_frame',
        association_implementation_sha256=file_digest(ROOT/'vehicle_metrology/passage_association.py'),
        detector_implementation_sha256=file_digest(ROOT/'vehicle_metrology/detection.py'),
        script_sha256=file_digest(Path(__file__)))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir/'association.json'
    result = dict(**signature, rows={})
    if path.exists():
        result = json.loads(path.read_text())
        if any(result.get(k) != v for k,v in signature.items()):
            raise ValueError('Audit provenance changed; use a new output directory')
    model = YOLO(str(args.weights))
    rows = [r for r in json.loads(args.comparison.read_text())['rows'] if r.get('comparison_eligible')]
    for row in rows:
        ident = row['id']
        if ident in result['rows'] or (args.only and ident not in args.only):
            continue
        start = time.monotonic()
        entry = extraction['rows'][ident]
        selected = entry['selected_frames']
        first, last = min(s['frame'] for s in selected), max(s['frame'] for s in selected)
        cache_path = args.output_dir/(ident+'-detections.json')
        cached = dict(signature=signature, id=ident, video=row['video'], frames={})
        if cache_path.exists():
            cached = json.loads(cache_path.read_text())
            if cached['signature'] != signature or cached['id'] != ident or cached['video'] != row['video']:
                raise ValueError('Changed detection cache provenance')
        capture = cv2.VideoCapture(row['video'])
        capture.set(cv2.CAP_PROP_POS_FRAMES, first)
        try:
            for number in range(first, last+1):
                ok, raw = capture.read()
                if not ok or raw.shape[1::-1] != profile.image_size:
                    raise ValueError(f'Cannot decode original video at {ident}/{number}')
                if str(number) not in cached['frames']:
                    image = corrected(raw, profile.lens)
                    cached['frames'][str(number)] = dict(
                        pixels_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
                        detections=predict_vehicle_boxes(model, image, .3, device=args.device, imgsz=640))
                if (number-first+1) % 25 == 0:
                    save(cache_path, cached)
            save(cache_path, cached)
        finally:
            capture.release()
        for sample in entry['samples']:
            if cached['frames'][str(sample['frame'])]['pixels_sha256'] != sample['inference_pixels_sha256']:
                raise ValueError(f'Corrected inference pixels changed: {ident}/{sample["frame"]}')
        frames = {int(k):v['detections'] for k,v in cached['frames'].items()}
        anchor = dict(frame=row['frame'], bbox=row['bbox'])
        checks = [dict(frame=s['frame'], expected_box=s['bbox'],
                       **(dict(status='anchor') if s['frame'] == row['frame'] else
                          associate_view(anchor, s, frames, profile.image_size))) for s in selected]
        result['rows'][ident] = dict(anchor=anchor, checks=checks,
            detections_file=cache_path.name, detections_sha256=file_digest(cache_path))
        save(path, result)
        print(ident, [(s['frame'], s['status'], s.get('reason')) for s in checks],
              round(time.monotonic()-start, 1), 'seconds', flush=True)
    print('Audited', len(result['rows']), 'of', len(rows), 'passages', flush=True)


if __name__ == '__main__':
    main()
