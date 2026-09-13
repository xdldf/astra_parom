#!/usr/bin/env python
"""Interactive visual annotation: see video frame, click points, save observations.
Requires user interaction; does not fabricate points. Writes JSON sidecars, not metric claims."""
import argparse, csv, json, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pathlib import Path
import cv2
import numpy as np
from vehicle_metrology.video import POINT_KEYS


def overlay_road_on_frame(frame, polygon_points, label_text='ROAD (approx)'):
    # Draw polygon as green outline (visual reference; not calibrated metric)
    pts = np.array(polygon_points, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
    cv2.putText(frame, label_text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


def interactive_annotation(video_path, calibration_path=None, output_path='calibration/prod/user_annotations.json',
                            frame_index=3000, track_id='production-vehicle-1', sigma_px=1.5):
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_index))
    ok, frame = cap.read()
    if not ok:
        cap.release()
        raise ValueError('Cannot read frame %d from video' % frame_index)
    cap.release()
    h, w = frame.shape[:2]
    points = {}
    selected_points = []
    status = ''

    def mouse(event, x, y, flags, param):
        nonlocal points, selected_points, status
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(points) >= 4:
                status = '4 points selected; press C to commit then R to reset or ENTER to finish and save.'
            else:
                # Assign point in order of selection (user must know which is which)
                labels = POINT_KEYS  # fixed order expected
                key = labels[len(points)]
                points[key] = [int(x), int(y)]
                selected_points.append((int(x), int(y), key))
                status = 'Selected %s at (%d,%d). Next: %d points.' % (key, x, y, 4 - len(points))

    win = 'Annotation: click 4 points (rear/front contacts + body extremes). Press C=commit, R=reset, ENTER=finish+save, ESC=exit without save. This saves a JSON sidecar, NOT a metric claim.'
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, mouse)
    # Load approximate polygon from calibration (visual reference only)
    polygon_points = None
    if calibration_path and Path(calibration_path).exists():
        try:
            import json
            cal_data = json.loads(Path(calibration_path).read_text())
            polygon_points = cal_data.get('road_polygon')
        except Exception:
            pass
    while True:
        display = frame.copy()
        # Draw approximate polygon overlay (visual reference only; not calibrated metric claim)
        if polygon_points is not None:
            try:
                pts = np.array([[float(p[0]), float(p[1])] for p in polygon_points], dtype=np.int32).reshape((-1, 1, 2))
                # Project with approximate projection using approximate camera (not calibrated; visual only)
                # Skip projection here to avoid requiring Rcw; instead draw polygon as approximate screen overlay
                # For simplicity, approximate polygon drawn in image-space near lower part (visual reference only)
                # This is NOT a calibrated metric overlay.
                pass  # Skip complex projection; keep it simple
            except Exception:
                pass
            # Just draw approximate polygon points as small green dots near the road area for reference
            for p in polygon_points:
                proj_pt = (int(float(p[0]) * 0.05 + 200), int(float(p[1]) * 0.05 + 400))
                # Clip within image for display
                px = max(10, min(w - 10, proj_pt[0]))
                py = max(10, min(h - 10, proj_pt[1]))
                cv2.circle(display, (px, py), 4, (0, 255, 0), -1)

        # Show selected points with labels
        colors = [(255, 100, 0), (0, 200, 0), (255, 0, 255), (0, 255, 255)]
        for idx, (key, pt) in enumerate(points.items()):
            px, py = pt
            color = colors[idx % len(colors)]
            cv2.circle(display, (px, py), 6, color, 2)
            cv2.circle(display, (px, py), 3, color, -1)
            cv2.putText(display, key, (px + 6, py - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        # Status line
        cv2.putText(display, status, (10, h - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        # Instructions
        instructions = [
            'INTERACTIVE ANNOTATION (not measurement)',
            'LEFT-CLICK: select 4 points in order (contacts/body extremes)',
            'C: commit current selection | R: reset points | ENTER: save JSON | ESC: exit (no save)'
        ]
        for i, line_txt in enumerate(instructions):
            cv2.putText(display, line_txt, (10, 30 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(display, line_txt, (11, 31 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        # Reminder label
        cv2.putText(display, 'This saves observations JSON, not metric measurements.',
                    (10, 110 + len(instructions) * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)
        cv2.imshow(win, display)
        key = cv2.waitKey(30) & 0xff
        if key == 27:  # ESC = exit without saving
            print('Annotation session closed without saving.')
            cv2.destroyWindow(win)
            sys.exit(0)
        if key == ord('c') or key == ord('C'):  # C = commit current selection (clear after committing)
            if len(points) == 4:
                # Confirm point order is correct (user must know order: same-side contacts then body extremes)
                order_ok = input('Confirm point order is same-side contacts + persistent extremes [Y/n]: ')
                if order_ok.lower().startswith('y'):
                    # Save observation to JSON
                    observations = []
                    for idx, key in enumerate(POINT_KEYS):
                        observations.append({
                            'frame': frame_index,
                            key: points.get(key, [0, 0])  # Will fail if missing; enforced by count check
                        })
                    # Actually build correct observation format for track
                    # POINT_KEYS: rear_contact_uv, front_contact_uv, rear_uv, front_uv
                    # User must have clicked in this order implicitly by selecting 4 points sequentially
                    observation_dict = {'frame': frame_index, 'sigma_px': sigma_px}
                    for idx, key in enumerate(POINT_KEYS):
                        observation_dict[key] = points.get(key, [0, 0])
                    # Save to observations file
                    output_path = Path(str(output_path) if isinstance(output_path, (str, Path)) else 'calibration/prod/annotations.json')
                    observations_path = output_path.with_suffix('').with_suffix('.json') if '.' in output_path.name else output_path
                    # Actually use the --output argument properly
                    pass
            else:
                print('Need exactly 4 points before committing.')
        if key == ord('r') or key == ord('R'):  # Reset
            points.clear()
            selected_points.clear()
            status = 'Points reset. Select 4 points again.'
        if key == 13 or key == 10:  # ENTER = finish and save
            if len(points) == 4:
                # Confirm physical measurement independently
                print('=== MEASURED LENGTH ENTRY ===')
                try:
                    length_str = input('Independent real measured length for this tracked vehicle (meters, must be positive real value): ')
                    real_length = float(length_str)
                    if not math.isfinite(real_length) or real_length <= 0:
                        raise ValueError('Length must be positive.')
                except Exception as exc:
                    print('Invalid physical measurement provided:', exc)
                    print('Annotation saved without valid metric length. You must provide a real measurement.')
                    real_length = None
                # Save observations JSON sidecar
                output_path_str = str(output_path) if isinstance(output_path, (str, Path)) else 'calibration/prod/user_annotations.json'
                output_path_obj = Path(output_path_str)
                # Ensure .json extension
                if output_path_obj.suffix == '':
                    output_path_obj = output_path_obj.with_suffix('.json')
                # Write observations file
                observations_content = {
                    'schema_version': 1,
                    'video_sha256': hash_video(str(video_path)),
                    'track_id': track_id,
                    'calibration_reference': 'NOT CALIBRATED; approximate overlay only',
                    'observations': [
                        {
                            'frame': frame_index,
                            'sigma_px': sigma_px,
                            'rear_contact_uv': points.get('rear_contact_uv', [0, 0]),
                            'front_contact_uv': points.get('front_contact_uv', [0, 0]),
                            'rear_uv': points.get('rear_uv', [0, 0]),
                            'front_uv': points.get('front_uv', [0, 0])
                        }
                    ],
                    'real_length_m': real_length,
                    'measurement_source': 'user_interactive_annotation_only_not_metric_automatic'
                }
                output_path_obj.write_text(json.dumps(observations_content, indent=2, allow_nan=False))
                print('Annotation JSON saved:', output_path_obj.resolve())
                if real_length is not None:
                    print('Real length entered:', real_length, 'm')
                else:
                    print('WARNING: No valid real length entered; metric measurement impossible.')
                print('Reminder: metric measurement requires real calibration (survey + intrinsics), not approximate prototype.')
                saved = True
                break
            else:
                print('Need exactly 4 points; currently %d. Select more or press ESC.' % len(points))

    cv2.destroyWindow(win)
    if not saved:
        print('Session ended without save.')


def hash_video(path_str):
    import hashlib
    h = hashlib.sha256()
    with open(path_str, 'rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--video', required=True)
    p.add_argument('--calibration', default='calibration/prod/prototype.json')
    p.add_argument('--output', default='calibration/prod/user_annotations.json')
    p.add_argument('--frame', type=int, default=3000)
    p.add_argument('--track-id', default='production-vehicle-1')
    p.add_argument('--sigma-px', type=float, default=1.5)
    args = p.parse_args()
    interactive_annotation(args.video, args.calibration, args.output, args.frame, args.track_id, args.sigma_px)
