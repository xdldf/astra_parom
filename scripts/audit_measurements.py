#!/usr/bin/env python3
"""Reproducible sampled-video audit of the station's actual measurement path.

CPU inference is an explicit offline option, not a live-station fallback.
Without independent ground truth this reports coverage, not physical accuracy.
"""
import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from vehicle_metrology.bbox_scale import measure_box
from vehicle_metrology.detection import predict_vehicle_boxes, box_iou
from vehicle_metrology.evaluation import evaluate
from vehicle_metrology.video import sha256_file, write_json
from web_app.workbench import Profile, corrected, profile_scale


def audit(videos, profile, detector, output, *, device, step_seconds=10.,
          start_seconds=0., end_seconds=None, imgsz=640, compare_imgsz=None, confidence=.35):
    if not math.isfinite(step_seconds) or step_seconds <= 0:
        raise ValueError('step_seconds must be finite and positive')
    if not math.isfinite(start_seconds) or start_seconds < 0:
        raise ValueError('start_seconds must be finite and nonnegative')
    if end_seconds is not None and (not math.isfinite(end_seconds) or end_seconds <= start_seconds):
        raise ValueError('end_seconds must be finite and greater than start_seconds')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    scale = profile_scale(profile)
    manifest, frames, samples, differences = [], [], [], []
    hashes = set()
    for source in videos:
        source = Path(source)
        if not source.is_file():
            raise ValueError(f'Local video does not exist: {source}')
        digest = sha256_file(source)
        if digest in hashes:
            raise ValueError('The same video was supplied more than once')
        hashes.add(digest)
        capture = cv2.VideoCapture(str(source))
        try:
            fps = capture.get(cv2.CAP_PROP_FPS)
            count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if not math.isfinite(fps) or fps <= 0 or count <= 0:
                raise ValueError(f'Unusable video metadata: {source}')
            stop = count if end_seconds is None else min(count, math.ceil(end_seconds*fps))
            first = round(start_seconds*fps)
            if first >= stop:
                raise ValueError(f'No frames in requested interval: {source}')
            indices = list(range(first, stop, max(1, round(step_seconds*fps))))
            manifest.append(dict(path=str(source.resolve()), sha256=digest, bytes=source.stat().st_size,
                                 fps=fps, frame_count=count, sampled_frames=indices))
            for index in indices:
                capture.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, raw = capture.read()
                if not ok:
                    raise ValueError(f'Cannot decode {source.name}, frame {index}')
                if raw.shape[1::-1] != profile.image_size:
                    raise ValueError('Calibration and video resolutions differ')
                frame = corrected(raw, profile.lens)
                detections = predict_vehicle_boxes(detector, frame, confidence, device=device, imgsz=imgsz)
                comparison = (predict_vehicle_boxes(detector, frame, confidence, device=device, imgsz=compare_imgsz)
                              if compare_imgsz else [])
                used = set()
                for number, detection in enumerate(detections):
                    measured = measure_box(detection['bbox'], profile.polygon, scale, profile.image_size,
                                           profile.measurement_line_x, profile.line_tolerance_px)
                    # Scale diagnostics are stored once in summary.json, not in every sample.
                    measured.pop('calibration_diagnostics', None)
                    detection.update(measured)
                    ident = f'{digest[:16]}:{index}:{number}'
                    detection['sample_id'] = ident
                    samples.append(dict(track_id=ident, video=source.name, frame=index, seconds=index/fps,
                                        label=detection['label'], length_m=detection['length_m'],
                                        status=detection['status'], reasons=[] if detection['length_m'] is not None else [detection['status']]))
                    matches = [(box_iou(detection['bbox'], other['bbox']), j, other)
                               for j, other in enumerate(comparison) if j not in used]
                    if matches:
                        overlap, j, other = max(matches, key=lambda row: row[0])
                        if overlap >= .5 and detection['depth'] is not None:
                            used.add(j)
                            differences.append(dict(sample_id=ident, iou=overlap,
                                                    width_delta_px=other['bbox'][2]-detection['bbox'][2],
                                                    bottom_delta_px=sum(other['bbox'][1::2])-sum(detection['bbox'][1::2])))
                frames.append(dict(video_sha256=digest, video=source.name, frame=index,
                                   seconds=index/fps, detections=detections))
                if any(d['length_m'] is not None for d in detections):
                    overlay = frame.copy()
                    for d in detections:
                        if d['length_m'] is None:
                            continue
                        x, y, w, h = map(round, d['bbox'])
                        cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, 255, 255), 2)
                        cv2.putText(overlay, f"{d['sample_id'].split(':')[-1]}: approx {d['length_m']:.3f} m",
                                    (x, max(20, y-10)), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 255, 255), 2)
                    cv2.imwrite(str(output/f'{digest[:16]}_{index}.jpg'), overlay)
            print(f'{source.name}: {len(indices)} sampled frames', flush=True)
        finally:
            capture.release()
    summary = dict(target_tolerance_m=profile.accuracy_tolerance_m, accuracy_status='unvalidated_no_independent_ground_truth',
                   scope='Sampled detection observations, not a vehicle census or independent passage count.',
                   sampled_frames=len(frames), detection_samples=len(samples),
                   measured_samples=sum(s['length_m'] is not None for s in samples),
                   statuses=dict(Counter(s['status'] for s in samples)), scale=scale,
                   device=str(device), imgsz=imgsz, compare_imgsz=compare_imgsz,
                   confidence=confidence, videos=manifest,
                   resolution_comparison=dict(matched_samples=len(differences),
                       median_abs_width_delta_px=float(np.median([abs(d['width_delta_px']) for d in differences])) if differences else None,
                       max_abs_width_delta_px=max((abs(d['width_delta_px']) for d in differences), default=None)))
    write_json(output/'summary.json', summary)
    write_json(output/'detections.json', frames)
    write_json(output/'resolution_comparison.json', differences)
    with (output/'measurements.csv').open('w', newline='', encoding='utf-8') as stream:
        fields=['track_id', 'video', 'frame', 'seconds', 'label', 'length_m', 'status']
        writer=csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader(); writer.writerows(samples)
    # Do not prefill truth with predictions or guessed catalogue dimensions.
    template=output/'ground_truth_template.csv'
    if not template.exists():
        with template.open('w', newline='', encoding='utf-8') as stream:
            writer=csv.writer(stream); writer.writerow(['track_id', 'vehicle_id', 'length_m'])
            writer.writerows((s['track_id'], '', '') for s in samples if s['length_m'] is not None)
    return summary, samples


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--videos', nargs='+', required=True, type=Path)
    parser.add_argument('--profile', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--model', type=Path, default=ROOT/'models/workbench/yolo26n.pt')
    parser.add_argument('--device', required=True, help='cpu for offline analysis, or CUDA device such as 0')
    parser.add_argument('--step-seconds', type=float, default=10.)
    parser.add_argument('--start-seconds', type=float, default=0.)
    parser.add_argument('--end-seconds', type=float)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--compare-imgsz', type=int)
    parser.add_argument('--ground-truth', type=Path, help='Independently measured CSV, associated to output sample IDs')
    parser.add_argument('--tolerance-m', type=float, help='Override the profile accuracy target (metres)')
    args=parser.parse_args()
    if not args.model.is_file():
        parser.error('A local model file is required; this audit never downloads weights')
    # Refuse any destination that could replace supplied evidence.
    sources={p.resolve() for p in [*args.videos, args.profile, args.model] + ([args.ground_truth] if args.ground_truth else [])}
    if any(p.parent == args.output_dir.resolve() for p in sources):
        parser.error('Keep audit output in a separate directory from input evidence')
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    data=json.loads(args.profile.read_text(encoding='utf-8'))
    if args.tolerance_m is not None:
        data['accuracy_tolerance_m']=args.tolerance_m
    profile=Profile.model_validate(data)
    summary, samples=audit(args.videos, profile, YOLO(str(args.model)), args.output_dir,
                          device=args.device, step_seconds=args.step_seconds, start_seconds=args.start_seconds,
                          end_seconds=args.end_seconds, imgsz=args.imgsz, compare_imgsz=args.compare_imgsz)
    summary['model_sha256']=sha256_file(args.model)
    summary['profile_sha256']=sha256_file(args.profile)
    if args.ground_truth:
        report=evaluate(samples, args.ground_truth, tolerance_m=profile.accuracy_tolerance_m)
        report['warnings'].append('Audit IDs are frame observations. Associate physical vehicles and adjudicate missed passages separately.')
        write_json(args.output_dir/'evaluation.json', report)
        summary['accuracy_status']='evaluated_supplied_ground_truth_sample_only' if report['metrics']['count'] else 'unvalidated_no_matched_ground_truth'
    write_json(args.output_dir/'summary.json', summary)
    print(json.dumps({k:summary[k] for k in ['sampled_frames', 'detection_samples', 'measured_samples', 'accuracy_status']}))


if __name__ == '__main__':
    main()
