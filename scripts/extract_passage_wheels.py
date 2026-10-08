#!/usr/bin/env python3
"""Extract at most three corrected views per catalogue passage, preserving failures."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vehicle_metrology.outline import outline_features, WEIGHTS_SHA256
from vehicle_metrology.wheels import file_digest, select_wheel_pair, wheel_features, MODEL_FILES, MODEL_REVISION
from vehicle_metrology.wheel_recovery import select_passage_frames
from web_app.workbench import Profile, corrected
from web_app.outline_measurement import infer_outline
from web_app.wheel_measurement import infer_wheels


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison', type=Path, required=True)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu','cuda:0'], default='cuda:0')
    parser.add_argument('--retry-failed', action='store_true', help='Retry whole passages with failed outlines; preserve attempt history')
    args = parser.parse_args()
    import torch
    cv2.setNumThreads(2)
    torch.set_num_threads(4)
    profile = Profile.model_validate_json(args.profile.read_text())
    rows = [r for r in json.loads(args.comparison.read_text())['rows'] if r.get('comparison_eligible')]
    signature = dict(comparison_sha256=file_digest(args.comparison), profile_sha256=file_digest(args.profile),
        script_sha256=file_digest(Path(__file__)), outline_model_sha256=WEIGHTS_SHA256,
        wheel_model_sha256=MODEL_FILES['model.safetensors'], wheel_model_revision=MODEL_REVISION,
        outline_implementation_sha256=file_digest(ROOT/'web_app/outline_measurement.py'),
        wheel_implementation_sha256=file_digest(ROOT/'web_app/wheel_measurement.py'),
        selection_implementation_sha256=file_digest(ROOT/'vehicle_metrology/wheel_recovery.py'), device=args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir/'extraction.json'
    result = dict(**signature, image_space='lens_corrected_full_frame', rows={})
    if path.exists():
        result = json.loads(path.read_text())
        if any(result.get(k) != v for k,v in signature.items()):
            raise ValueError('Extraction provenance changed; use a new output directory')
    tracks = {}
    for row in rows:
        ident = row['id']
        prior = result['rows'].get(ident)
        if prior and (not args.retry_failed or all(s['status']=='ok' for s in prior['samples'])):
            continue
        if row['video'] not in tracks:
            track_path = args.comparison.parent/Path(row['video']).stem/'scout_tracks.json'
            tracks[row['video']] = {t['id']:t for t in json.loads(track_path.read_text())}
        selected = select_passage_frames(row, tracks[row['video']][ident]['observations'], profile.image_size)
        samples = []
        start = time.monotonic()
        capture = cv2.VideoCapture(row['video'])
        try:
            for frame in selected:
                sample = dict(frame=frame['frame'], expected_box=frame['bbox'], status='failed')
                try:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, frame['frame'])
                    ok, raw = capture.read()
                    if not ok or raw.shape[1::-1] != profile.image_size:
                        raise ValueError('Cannot decode original full-resolution frame')
                    image = corrected(raw, profile.lens)
                    stem = f"{ident}-{frame['frame']}"
                    image_path, mask_path = args.output_dir/(stem+'.jpg'), args.output_dir/(stem+'.npz')
                    if not cv2.imwrite(str(image_path), image, [cv2.IMWRITE_JPEG_QUALITY,95]):
                        raise ValueError('Cannot save corrected full frame')
                    sample.update(corrected_image=image_path.name, corrected_image_sha256=file_digest(image_path),
                                  inference_pixels_sha256=hashlib.sha256(image.tobytes()).hexdigest())
                    mask, details = infer_outline(image, frame['bbox'], device=args.device)
                    np.savez_compressed(mask_path, mask=mask)
                    base = outline_features(mask, profile.polygon)
                    detected = infer_wheels(image, frame['bbox'], device=args.device)
                    sample.update(status='ok', outline_features=base.tolist(), outline=details, wheels=detected,
                                  mask=mask_path.name, mask_sha256=file_digest(mask_path))
                    try:
                        pair = select_wheel_pair(detected['detections'], frame['bbox'])
                        value = wheel_features(base, pair, profile.polygon)
                        sample.update(wheel_status='ok', pair=pair, features=value.tolist())
                    except ValueError as exc:
                        sample.update(wheel_status='unavailable', wheel_reason=str(exc))
                except Exception as exc:
                    sample.update(reason=str(exc), error_type=type(exc).__name__)
                samples.append(sample)
        finally:
            capture.release()
        entry = dict(anchor_frame=row['frame'], selected_frames=selected, samples=samples)
        if prior:
            entry['previous_attempt'] = prior
        result['rows'][ident] = entry
        write_json(path, result)
        print(ident, f"{sum(s['status']=='ok' for s in samples)}/{len(samples)} frames", round(time.monotonic()-start,2), 'seconds', flush=True)
    failed = [ident for ident,r in result['rows'].items() if any(s['status']!='ok' for s in r['samples'])]
    if failed:
        raise SystemExit('Failed observations retained; investigate before interpreting the benchmark: '+', '.join(failed))


if __name__ == '__main__':
    main()
