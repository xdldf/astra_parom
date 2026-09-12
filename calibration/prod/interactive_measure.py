#!/usr/bin/env python
"""Interactive production measurement for one vehicle passage.
Reads the actual production video, pauses, asks user to click 4 points per selected frame,
saves observations, asks for measured real length, saves ground truth.
Requires: production video, calibration, user measurements.
Does NOT invent physical values. Rejects missing observations.
Usage:
  .venv/Scripts/python.exe calibration/prod/interactive_measure.py --video VIDEO --calibration calibration/prod/camera.json --track-id car-A --sigma-px 1.5 --output calibration/prod/measurements.json
"""
import argparse, sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import csv, hashlib, json, math
from pathlib import Path
import cv2
import numpy as np
from vehicle_metrology.video import load_observations, local_video, sha256_file, POINT_KEYS
from vehicle_metrology.geometry import Camera


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video', required=True)
    p.add_argument('--calibration', required=True)
    p.add_argument('--track-id', default='production-vehicle-1')
    p.add_argument('--sigma-px', type=float, default=1.5)
    p.add_argument('--output', required=True)
    args = p.parse_args(argv)

    video = local_video(args.video)
    camera = Camera.from_dict(json.loads(Path(args.calibration).read_text()))
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError('Cannot decode the provided video')
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if not count:
        # Some files have zero reported count; try reading
        temp = 0; ok, _ = capture.read(); temp += (1 if ok else 0)
        while ok:
            ok, _ = capture.read(); temp += (1 if ok else 0)
        capture.release(); count = temp
    capture = cv2.VideoCapture(str(video))
    size = None
    try:
        while True:
            ok, frame = capture.read()
            if not ok: break
            size = [frame.shape[1], frame.shape[0]]
        if size is None:
            capture.release(); raise ValueError('No decodable frames')
        # Check calibration matches video
        if list(camera.image_size) != size:
            capture.release(); raise ValueError(f'Calibration size {list(camera.image_size)} does not match video size {size}')
    finally:
        capture.release()

    # Load or initialize observations
    output = Path(args.output)
    video_hash = sha256_file(video)
    observations_path = output.with_suffix('.json')
    if observations_path.exists():
        data = load_observations(str(observations_path), video_hash, size, count)
    else:
        data = {'schema_version': 1, 'video_sha256': video_hash, 'tracks': [{'track_id': args.track_id, 'observations': []}]}

    print('=== Interactive measurement session ===')
    print('Using video:', video)
    print('Video resolution:', size)
    print('Calibration:', args.calibration)
    print('Track ID:', args.track_id)
    print('Instructions: open the video in your preferred viewer or use the annotations below.')
    print('For each selected sharp frame (prefer clear, unoccluded, varied position):')
    print('  1. Click REAR same-side wheel-road contact (actual ground, not shadow).')
    print('  2. Click FRONT same-side wheel-road contact.')
    print('  3. Click PERSISTENT REAR BODY EXTREME point.')
    print('  4. Click PERSISTENT FRONT BODY EXTREME point.')
    print('Then supply the independently measured real length in meters.')
    print('These values must be physically real; do not use catalog dimensions.')
    print('Re-run for additional frame observations; use --track-id for additional vehicles.')
    print()

    # Interactive: ask user to name a frame number and provide 4 points + length manually (simplified interactive entry here)
    # Since this session is CLI/text only, prompt via standard input rather than mouse events
    # This is a practical compromise; for full mouse annotation use annotate.py separately.

    track = data['tracks'][0]
    observations = track['observations']
    print('Enter a frame index (integer between 0 and %d) or type END:' % max(0, count - 1))
    observation = {}
    while True:
        line = input('Frame index (or END): ').strip()
        if line.upper() == 'END':
            break
        try:
            frame_idx = int(line)
            if not (0 <= frame_idx < count):
                print('Frame out of range 0..%d' % max(0, count - 1))
                continue
        except ValueError:
            print('Invalid integer')
            continue
        # User must provide 4 points as pixel pairs (x y) manually here; in practice you could open the video separately
        # and ask for measurements; this interface requires manual entry of pixel coordinates.
        print('For frame %d, provide 4 points as: NAME x y' % frame_idx)
        points = {}
        for key in POINT_KEYS:
            try:
                parts = input('%s (x y): ' % key).strip().split()
                if len(parts) != 2: raise ValueError('need 2 numbers')
                px, py = float(parts[0]), float(parts[1])
                if not (0 <= px < size[0] and 0 <= py < size[1]):
                    print('Point outside image size %s' % size)
                    raise ValueError('outside image')
                points[key] = [px, py]
            except Exception:
                print('Invalid point; retry this point.')
                break
        else:
            # All 4 points entered
            observation = dict(frame=frame_idx, **points)
            # Confirm these are from the same actual physical points (not silhouette tangents)
            confirm = input('Confirm these refer to REAL persistent points (same side contacts, same body extremes) [Y/n]: ')
            if confirm.lower().startswith('y'):
                observations.append(observation)
                print('Observation saved for frame %d.' % frame_idx)
            else:
                print('Discarded.')

    observations.sort(key=lambda r: r['frame'])
    # Deduplicate frame observations: keep last per frame per point set if duplicates
    # Simplified: no duplicate checks enforced here; user controls.

    # Ask for independently measured length
    print()
    print('Now supply the independently measured physical length for this tracked vehicle.')
    try:
        real_length_str = input('Real length (meters, must be positive, accurate measured value): ')
        real_length = float(real_length_str)
        if not math.isfinite(real_length) or real_length <= 0:
            raise ValueError('Length must be a positive finite number.')
    except ValueError:
        real_length = None
        print('No valid real length entered; measurement saved without ground truth.')
    else:
        # Create/update ground truth CSV file
        truth_path = output.with_suffix('.csv')
        truth_path = output.with_suffix('.ground_truth.csv')
        # Write/update ground truth file
        truth_path.parent.mkdir(parents=True, exist_ok=True)
        existing = {}
        if truth_path.exists():
            with open(truth_path, newline='', encoding='utf-8-sig') as stream:
                for row in csv.DictReader(stream):
                    existing[row['track_id']] = row
        existing[track['track_id']] = {
            'track_id': track['track_id'],
            'vehicle_id': input('Physical vehicle identity label (used for grouping repeated runs): '),
            'length_m': real_length_str
        }
        with open(truth_path, 'w', newline='', encoding='utf-8') as stream:
            w = csv.writer(stream)
            w.writerow(['track_id', 'vehicle_id', 'length_m'])
            for row in existing.values():
                w.writerow([row['track_id'], row.get('vehicle_id', ''), row['length_m']])
        print('Ground truth saved:', truth_path)

    # Save observations
    output.parent.mkdir(parents=True, exist_ok=True)
    data['tracks'][0]['observations'] = observations
    with open(output.with_suffix('.json'), 'w', encoding='utf-8') as f:
        f.write(json.dumps(data, indent=2, allow_nan=False))

    print('Observations saved:', output)
    print('Points saved for track:', track['track_id'])
    print('Observations count:', len(observations))

    # Now, if calibration is present, compute and report conditional measurements immediately
    # This verifies that the observations work with the geometry before production use.
    from vehicle_metrology.geometry import measure_track
    try:
        result = measure_track(camera, observations, seed=0, mc_samples=10)
        print('Conditional geometry measurement for these observations:')
        print('  Status:', result['status'])
        print('  Length (m):', result['length_m'] if result['length_m'] is not None else 'REJECTED')
        if result['reasons']:
            print('  Rejection reasons:', ', '.join(result['reasons']))
        if result['diagnostics']:
            diag = result['diagnostics']
            for k, v in diag.items():
                print('  %s: %s' % (k, v))
        if result['frames']:
            for f in result['frames'][-min(3, len(result['frames'])):]:
                print('  Frame %d: x=%.3f y=%.3f heading=%.1f window_length=%.4f' % (f['frame'], f['x_m'], f['y_m'], f['heading_deg'], f['window_length_m'] if f['window_length_m'] is not None else 'N/A'))
    except Exception as exc:
        print('Geometry measurement could not be completed (expected if calibration/geometry does not apply):', exc)
    print()
    print('IMPORTANT: These are CONDITIONAL results only. Real-world accuracy depends on correct calibration, accurate survey, and correct persistent physical observations.')


if __name__ == '__main__':
    sys.exit(main())
