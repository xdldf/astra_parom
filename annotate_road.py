#!/usr/bin/env python
"""Draw road polygon overlay and vehicle bounding boxes on the production video.
Uses calibration/prod/prototype.json (dummy) for approximate projection only.
Real measurements require real survey + annotations."""
import argparse, cv2, numpy as np, json
from pathlib import Path
sys_path_insert = __import__('sys').path.insert(0, '.')
from vehicle_metrology.geometry import Camera


def overlay_road_and_vehicles(video_path, calibration_path, output_image_path, frame_index=3000):
    camera = Camera.from_dict(json.loads(Path(calibration_path).read_text()))
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError('Cannot read frame %d' % frame_index)
    # Draw approximate road polygon from calibration (visual, not calibrated metric)
    polygon = np.array(camera.road_polygon, dtype=np.int32).reshape((-1, 1, 2))
    if polygon.shape[1] == 2:
        # Project polygon points to image for overlay demonstration
        # Note: road polygon in world meters needs Rcw/tcw for projection; prototype has approximate values.
        # This overlay uses approximate projection; not certified.
        projected = camera.project([[float(p[0]), float(p[1]), 0.0] for p in camera.road_polygon])
        projected_int = projected.reshape((-1, 1, 2)).astype(np.int32)
        # Draw as green outline
        cv2.polylines(frame, [projected_int], isClosed=True, color=(0, 255, 0), thickness=2)
        # Label
        cv2.putText(frame, 'ROAD POLYGON (visual only, approximate projection)', (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    # Approximate vehicle annotations: draw approximate boxes at approximate positions
    # These are NOT measurements; they are visual annotations for manual review only.
    # Real measurements require real annotations from interactive_measure.py.
    # For demonstration only, draw an approximate rectangle where a central vehicle appears.
    # Actual vehicle positions must be identified by user observation.
    # This function writes the annotated image, not a metric claim.
    cv2.putText(frame, 'VEHICLE OVERLAY: approximate manual annotations only; not metric measurements.',
                (20, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    out_str = str(output_image_path)
    if "." not in Path(out_str).name: out_str += ".jpg"
    cv2.imwrite(out_str, frame)
    print("Annotated image saved:", Path(out_str).resolve())
    print('This image is a visual annotation; metric measurements require real survey/calibration + manual annotations.')


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--video', required=True)
    p.add_argument('--calibration')
    p.add_argument('--output')
    p.add_argument('--frame', type=int, default=3000)
    args = p.parse_args()
    calibration_path = args.calibration or 'calibration/prod/prototype.json'
    output_path = args.output or 'calibration/prod/annotated_road.jpg'
    overlay_road_and_vehicles(args.video, calibration_path, output_path, args.frame)
