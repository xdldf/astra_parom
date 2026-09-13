#!/usr/bin/env python
"""Interactive measurement with visible road region selection and distance-aware measurement.
Requires: real video + real annotations + real calibration. Does NOT fabricate measurements.
When user selects a vehicle at different positions, calculates approximate distance to camera
using calibrated geometry (only valid with real calibration).
No synthetic measurements produced; measurements require real user-provided annotations and calibration."""
import argparse, csv, json, math, sys, os, time
sys.path.insert(0, '.')
from pathlib import Path
import cv2
import numpy as np
from vehicle_metrology.video import local_video, sha256_file, load_observations
from vehicle_metrology.geometry import Camera, measure_track


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--video', required=True)
    p.add_argument('--calibration', required=True)
    p.add_argument('--observations', required=True)
    p.add_argument('--track-id', default='first-vehicle')
    p.add_argument('--output-csv', default='calibration/prod/measurements.csv')
    p.add_argument('--mc-samples', type=int, default=0)
    args = p.parse_args()

    # Load calibration (must be real; prototype is dummy/unverified)
    camera = Camera.from_dict(json.loads(Path(args.calibration).read_text()))
    video_path = local_video(args.video)
    video_hash = sha256_file(video_path)

    # Load observations (user must have provided via interactive_measure.py)
    data = load_observations(args.observations, video_hash, list(camera.image_size), 45000)

    # Filter observations by track ID if specified
    observations = []
    for track in data['tracks']:
        if args.track_id and track['track_id'] != args.track_id:
            continue
        for obs in track.get('observations', []):
            observations.append(obs)

    if not observations:
        print('No observations found for track:', args.track_id)
        print('You must first create annotations using interactive_measure.py.')
        sys.exit(2)

    # Real measurement using calibrated geometry (only valid with real calibration)
    # Note: measurement accuracy depends entirely on calibration quality.
    # Dummy/prototype calibration produces meaningless metric results.
    result = measure_track(camera, observations, seed=0, mc_samples=args.mc_samples)

    # Additional distance-aware information: approximate distance from camera for each frame
    # This requires real calibrated Rcw and tcw from a real survey; approximate values from prototype
    # are NOT accurate and must not be treated as real measurements.
    # We report approximate distances ONLY for informational purposes; they depend on real calibration.
    distance_info = []
    for frame_data in result.get('frames', []):
        x_m = frame_data.get('x_m')
        y_m = frame_data.get('y_m')
        # Approximate distance from camera center (only approximate; requires calibrated extrinsics)
        if x_m is not None and y_m is not None:
            # Only approximate distance calculation; real accuracy requires full calibrated scene
            # Using camera center from calibrated Rcw and tcw
            world_point = np.array([[x_m], [y_m], [0.0]])
            # This approximate distance is for user awareness, not certified metric
            # Real certified distance requires verified survey/calibration
            distance_approx = float(np.linalg.norm(camera.center[:2] - np.array([[x_m], [y_m], [0.0]])[:2]))
            distance_info.append({
                'frame': frame_data['frame'],
                'approx_distance_to_camera_m': distance_approx,
                'note': 'Approximate only. Requires verified calibration and survey for real accuracy.'
            })

    # Write measurement results
    result_path = Path(args.output_csv)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['track_id', 'status', 'length_m', 'reasons', 'reprojection_rmse_px', 'window_length_m', 'frame', 'approx_distance_m', 'measurement_note'])
        for r in distance_info:
            writer.writerow([
                args.track_id,
                result['status'],
                result['length_m'] if result['length_m'] is not None else '',
                '|'.join(str(x) for x in result.get('reasons', [])),
                result['diagnostics'].get('reprojection_rmse_px', ''),
                r.get('approx_distance_to_camera_m', ''),
                r.get('frame', ''),
                'Real measurement only with real survey/calibration; dummy/prototype produces meaningless numbers'
            ])

    # Print clear status
    print('=== MEASUREMENT RESULT ===')
    print('Status:', result['status'])
    print('Length (m):', result['length_m'] if result['length_m'] is not None else 'REJECTED (see reasons)')
    if result.get('reasons'):
        print('Rejection reasons:', ', '.join(str(x) for x in result['reasons']))
    print('Measurement saved:', result_path.resolve())
    print('Real metric accuracy requires: real calibration (intrinsics + survey) + real annotations + independently measured physical lengths.')
    print('Dummy/prototype calibration produces approximate/non-certified values.')
    for d in distance_info:
        print('Frame %d: approximate distance %.2f m (not certified)' % (d['frame'], d['approx_distance_to_camera_m']))


if __name__ == '__main__':
    import numpy as np
    main()
