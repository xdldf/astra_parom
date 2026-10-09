#!/usr/bin/env python3
"""Replay a frozen observation selection through the real station capture path.

Every observation remains in the report, including fragments and failures.
Catalogue references are deliberately not inputs to inference. This is an
offline integration/coverage audit, not a count of unique cars or accuracy.
"""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import sys
import time
from unittest.mock import patch

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_passage_association import DETECTOR_SHA256
from vehicle_metrology.detection import predict_vehicle_boxes
from vehicle_metrology.outline import WEIGHTS_SHA256
from vehicle_metrology.wheels import MODEL_DIRECTORY, MODEL_FILES, file_digest, verify_model
from web_app import workbench as wb, station as st
from web_app import outline_measurement as om, wheel_measurement as wm


def save(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    temporary.replace(path)


def verify_saved(entry, output):
    for name, digest in (('corrected_image', 'corrected_image_sha256'),
                         ('record_file', 'record_sha256')):
        if name in entry and file_digest(output/entry[name]) != entry[digest]:
            raise ValueError(f'Changed saved artifact: {entry[name]}')
    mask = entry.get('inference', {}).get('outline', {})
    if 'mask' in mask and file_digest(output/mask['mask']) != mask['mask_sha256']:
        raise ValueError('Changed saved outline mask')
    if entry.get('capture_status') == 'saved':
        record = json.loads((output/entry['record_file']).read_text())
        if file_digest(output/'captures'/record['full_frame_photo']) != entry['corrected_image_sha256']:
            raise ValueError('Station full-frame photo changed')


def capture_one(row, profile, output):
    ident = row['id']
    started = time.monotonic()
    entry = dict(id=ident, frame=row['frame'], video=Path(row['video']).name,
                 bbox=row['bbox'], detector_label=row.get('detector_label', 'car'),
                 prior_model_candidate=row.get('prior_model_candidate'),
                 prior_note=row.get('prior_note'), capture_status='failed')
    evidence = {}
    original_outline, original_wheels = om.infer_outline, wm.infer_wheels
    try:
        video = cv2.VideoCapture(row['video'])
        try:
            if not video.isOpened():
                raise ValueError('Video cannot be opened')
            wb.media[ident] = dict(kind='video', path=Path(row['video']),
                                  fps=video.get(cv2.CAP_PROP_FPS),
                                  frames=int(video.get(cv2.CAP_PROP_FRAME_COUNT)))
        finally:
            video.release()
        raw = wb.read_frame(ident, row['frame'])
        image = wb.corrected(raw, profile.lens)
        if image.shape[1::-1] != profile.image_size:
            raise ValueError('Corrected image size differs from profile')
        ok, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok:
            raise ValueError('Corrected frame JPEG encoding failed')
        jpeg = encoded.tobytes()
        photo = output/(ident+'-frame.jpg')
        photo.write_bytes(jpeg)
        pixel_digest = hashlib.sha256(image.tobytes()).hexdigest()
        entry.update(corrected_image=photo.name, corrected_image_sha256=file_digest(photo),
                     inference_pixels_sha256=pixel_digest, image_space='corrected_full_resolution')

        def check_pixels(frame):
            if hashlib.sha256(frame.tobytes()).hexdigest() != pixel_digest:
                raise ValueError('Inference used a different corrected frame')

        def outline(frame, box, *args, **kwargs):
            check_pixels(frame)
            try:
                mask, details = original_outline(frame, box, *args, **kwargs)
                path = output/(ident+'-mask.npz')
                np.savez_compressed(path, mask=mask)
                evidence['outline'] = dict(status='ok', mask=path.name,
                    mask_sha256=file_digest(path), details=details)
                return mask, details
            except Exception as exc:
                evidence['outline'] = dict(status='failed', reason=str(exc))
                raise

        def wheels(frame, box, *args, **kwargs):
            check_pixels(frame)
            try:
                details = original_wheels(frame, box, *args, **kwargs)
                evidence['wheels'] = dict(status='ok', details=details)
                return details
            except Exception as exc:
                evidence['wheels'] = dict(status='failed', reason=str(exc))
                raise

        with patch.object(om, 'infer_outline', outline), patch.object(wm, 'infer_wheels', wheels):
            record = st.capture(st.Capture(media_id=ident, profile=profile, frame=row['frame'],
                bbox=row['bbox'], label=entry['detector_label'], source='yolo26m', temporal=True,
                review_fallback=True, actor='Offline additional-car audit'))
        if (output/'captures'/record['full_frame_photo']).read_bytes() != jpeg:
            raise ValueError('Station photo does not match the full corrected frame')
        record_path = output/'captures'/(ident+'.json')
        save(record_path, record)
        entry.update(capture_status='saved', record_file=str(record_path.relative_to(output)),
                     record_sha256=file_digest(record_path), record_id=record['id'],
                     length_m=record['length_m'], measurement=record['source']['measurement'])
    except Exception as exc:
        entry.update(error_type=type(exc).__name__, reason=str(exc), length_m=None)
    finally:
        wb.media.pop(ident, None)
    entry.update(inference=evidence, elapsed_seconds=time.monotonic()-started)
    return entry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('selection', 'profile', 'weights', 'output-dir'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda:0'], default='cuda:0')
    parser.add_argument('--ids', nargs='+', help='Explicit subset for an integration smoke check')
    parser.add_argument('--video-dir', type=Path, help='Locate the selected video basenames in this directory')
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text())
    profile = wb.Profile.model_validate_json(args.profile.read_text())
    if (file_digest(args.profile) != selection['profile_sha256']
            or profile.detector_model != 'yolo26m' or profile.detector_imgsz != 640
            or file_digest(args.weights) != DETECTOR_SHA256):
        raise ValueError('Selection/profile/detector provenance mismatch')
    rows = selection['rows']
    ids = [r['id'] for r in rows]
    if len(set(ids)) != len(ids) or any(not re.fullmatch(r'[A-Za-z0-9_-]+', i) for i in ids):
        raise ValueError('Observation IDs must be unique safe filenames')
    if args.ids:
        if len(set(args.ids)) != len(args.ids) or set(args.ids)-set(ids):
            raise ValueError('Unknown or repeated subset ID')
        rows = [r for r in rows if r['id'] in args.ids]
    if not rows:
        raise ValueError('The observation selection is empty')
    if args.video_dir:
        rows = [dict(r, video=str((args.video_dir/Path(r['video']).name).resolve())) for r in rows]
    output = args.output_dir.resolve()
    if output.is_relative_to((ROOT/'web_app/data').resolve()):
        raise ValueError('Offline output must be outside production web_app/data')
    verify_model(ROOT/'web_app/data'/MODEL_DIRECTORY)
    if file_digest(ROOT/'web_app/data/yolo26m-seg.pt') != WEIGHTS_SHA256:
        raise ValueError('Unverified outline model')
    signature = dict(selection=file_digest(args.selection), profile=file_digest(args.profile),
        detector=DETECTOR_SHA256, outline_model=WEIGHTS_SHA256, wheel_files=MODEL_FILES,
        script=file_digest(Path(__file__)), device=args.device, selected_ids=[r['id'] for r in rows],
        videos={v:file_digest(Path(v)) for v in sorted({r['video'] for r in rows})},
        implementations={str(p.relative_to(ROOT)):file_digest(p)
            for folder in ('vehicle_metrology', 'web_app') for p in sorted((ROOT/folder).glob('*.py'))},
        packages={n:importlib.metadata.version(n) for n in ('torch', 'ultralytics', 'transformers', 'numpy')},
        opencv=cv2.__version__, python=sys.version)
    output.mkdir(parents=True, exist_ok=True)
    captures = output/'captures'
    captures.mkdir(exist_ok=True)
    result = dict(inputs_sha256=signature, accuracy_validated=False,
                  dataset_role='additional_observation_audit', rows={})
    path = output/'results.json'
    if path.exists():
        result = json.loads(path.read_text())
        if result['inputs_sha256'] != signature or set(result['rows'])-set(signature['selected_ids']):
            raise ValueError('Audit inputs changed; use a new output directory')
        for entry in result['rows'].values():
            verify_saved(entry, output)
    import torch
    from ultralytics import YOLO
    cv2.setNumThreads(2)
    torch.set_num_threads(4)
    if args.device == 'cuda:0':
        wb.require_gpu()
    model = YOLO(str(args.weights))

    def detect(frame, confidence=.3, **settings):
        if confidence != .3 or settings != dict(detector_model='yolo26m', imgsz=640):
            raise ValueError('Unexpected temporal detector settings')
        return predict_vehicle_boxes(model, frame, confidence, device=args.device, imgsz=640)

    with patch.object(st, 'DATA', captures), patch.object(st, 'DB', captures/'station.sqlite3'), \
         patch.object(wb, 'require_gpu', lambda: args.device), patch.object(wb, 'detect_vehicles', detect):
        for row in rows:
            if row['id'] not in result['rows']:
                entry = capture_one(row, profile, output)
                result['rows'][row['id']] = entry
                save(path, result)
                print(row['id'], entry['capture_status'], entry.get('length_m'), flush=True)
    result['summary'] = dict(observations=len(rows), completed=len(result['rows']),
        statuses=dict(Counter(r.get('measurement', {}).get('status', r['capture_status'])
                              for r in result['rows'].values())),
        limitations=['Observations may duplicate the same passage.',
                     'No independent physical ground truth; not an accuracy or detector-recall score.'])
    save(path, result)
    print(json.dumps(result['summary'], ensure_ascii=False))


if __name__ == '__main__':
    main()
