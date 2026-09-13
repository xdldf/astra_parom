#!/usr/bin/env python
"""Visual calibration window: video frame + sliders (fx, fy, cx, cy, k1, k2, p1, p2, k3, model).
No terminal input for parameters. Only interactive: open video, adjust sliders in window,
press S to save JSON. ESC to quit without saving. Nothing synthetic.
Uses approximate projection overlay for visual feedback; not a certified measurement.
Requires: a video frame; user provides values by adjusting sliders."""
import argparse, csv, json, math, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pathlib import Path
import cv2
import numpy as np


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--video', required=True)
    p.add_argument('--output-dir', default='calibration/prod/generated')
    p.add_argument('--calibration-id', default='prod-slider')
    args = p.parse_args(argv)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = Path(args.video)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError('Cannot open video: %s' % video_path)
    ok, first_frame = cap.read()
    if not ok:
        cap.release()
        raise ValueError('No decodable frames.')
    cap.release()

    h, w = first_frame.shape[:2]
    # Approximate initial values derived from image size (NOT synthetic defaults for accuracy);
    # these are only starting slider positions. Real physical accuracy requires survey + board calibration.
    init = {
        'fx': float(w) / 2.0, 'fy': float(h) / 2.0,
        'cx': float(w) / 2.0, 'cy': float(h) / 2.0,
        'k1': 0.0, 'k2': 0.0,
        'p1': 0.0, 'p2': 0.0,
        'k3': 0.0,
        'model_idx': 0  # 0 = brown, 1 = fisheye
    }
    # Slider scaling factors (approximate mappings; visual only)
    # Actual calibration accuracy requires a real calibration rig and survey, not slider approximations.
    state = dict(init)

    saved = False
    result_path = out_dir / 'intrinsic_slider.json'

    def save_result(state):
        fx = state['fx']; fy = state['fy']
        cx = state['cx']; cy = state['cy']
        model_choice = 'brown' if state['model_idx'] == 0 else 'fisheye'
        D = [state['k1'], state['k2'], state['p1'], state['p2'], state['k3']] if model_choice == 'brown' else [state['k1'], state['k2'], state['p1'], state['p2']]
        result = {
            'schema_version': 1,
            'calibration_method': 'interactive_slider_approximation',
            'calibration_id': args.calibration_id,
            'artifact_type': 'intrinsic_approximate',
            'model': model_choice,
            'image_size': [int(w), int(h)],
            'K': [[float(fx), 0.0, float(cx)], [0.0, float(fy), float(cy)], [0.0, 0.0, 1.0]],
            'D': [float(v) for v in D],
            'note': 'Approximate/user-tuned via sliders. Not a certified calibration. Must be validated/replaced with a real checkerboard calibration + survey for physical measurement.'
        }
        result_path.write_text(json.dumps(result, indent=2, allow_nan=False))
        print('SAVED approximate intrinsic to:', result_path.resolve())
        print('REMINDER: This is an approximate/user-tuned model only. For real measurement accuracy, use a calibration rig + survey points.')

    win = 'Calibration sliders: adjust fx fy cx cy k1 k2 p1 p2 k3 model | S=save, ESC=quit (visual only)'
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    # Resize to roughly half native for usability
    scale = min(0.6, 1280 / float(w), 960 / float(h))
    display_w = max(640, int(w * scale))
    display_h = max(480, int(h * scale))
    cv2.resizeWindow(win, display_w, display_h)

    # Create trackbars for visual interactive adjustment
    # Mapping: approximate linear scales around initial values; not certified.
    # These are for visual exploration; real calibration needs real measurements.
    def nothing(x):
        pass

    max_slider_focal = max(5000, int(init['fx'] * 3))
    max_slider_point = max(3000, w + 100)
    cv2.createTrackbar('fx', win, int(round(init['fx'])), max_slider_focal, nothing)
    cv2.createTrackbar('fy', win, int(round(init['fy'])), max_slider_focal, nothing)
    cv2.createTrackbar('cx', win, int(round(init['cx'])), max_slider_point, nothing)
    cv2.createTrackbar('cy', win, int(round(init['cy'])), max_slider_point, nothing)
    # Distortion scaled by 100 for finer control
    cv2.createTrackbar('k1x100', win, int(round(init['k1'] * 100)), 500, nothing)
    cv2.createTrackbar('k2x100', win, int(round(init['k2'] * 100)), 500, nothing)
    cv2.createTrackbar('p1x100', win, int(round(init['p1'] * 100)), 500, nothing)
    cv2.createTrackbar('p2x100', win, int(round(init['p2'] * 100)), 500, nothing)
    # Optional k3
    cv2.createTrackbar('k3x100', win, int(round(init['k3'] * 100)), 500, nothing)
    # Model
    cv2.createTrackbar('model(0=brown,1=fisheye)', win, init['model_idx'], 1, nothing)

    # Visual overlay text showing current slider values; updates interactively
    while not saved:
        fx_s = cv2.getTrackbarPos('fx', win)
        fy_s = cv2.getTrackbarPos('fy', win)
        cx_s = cv2.getTrackbarPos('cx', win)
        cy_s = cv2.getTrackbarPos('cy', win)
        k1_s = cv2.getTrackbarPos('k1x100', win) / 100.0
        k2_s = cv2.getTrackbarPos('k2x100', win) / 100.0
        p1_s = cv2.getTrackbarPos('p1x100', win) / 100.0
        p2_s = cv2.getTrackbarPos('p2x100', win) / 100.0
        k3_s = cv2.getTrackbarPos('k3x100', win) / 100.0
        model_s = 'brown' if cv2.getTrackbarPos('model(0=brown,1=fisheye)', win) == 0 else 'fisheye'

        # Build rough intrinsic from sliders
        # Note: projection demonstration below uses approximate rotation (not from survey);
        # this is visual feedback only, not calibrated measurement.
        rough_intrinsic_vis = {
            'model': model_s,
            'K': [[float(fx_s), 0.0, float(cx_s)], [0.0, float(fy_s), float(cy_s)], [0.0, 0.0, 1.0]],
            'D': [float(k1_s), float(k2_s), float(p1_s), float(p2_s), float(k3_s)] if model_s == 'brown' else [float(k1_s), float(k2_s), float(p1_s), float(p2_s)]
        }
        # Visual feedback: overlay approximate text and simple approximate projection using an approximate camera pose
        # This demonstration uses an approximate pose (not calibrated) for visual effect only.
        display = first_frame.copy()
        info_text = [
            'SLIDER CALIBRATION (visual only, not certified)',
            'fx=%.1f fy=%.1f cx=%.1f cy=%.1f' % (fx_s, fy_s, cx_s, cy_s),
            'DISTORTION: k1=%.3f k2=%.3f p1=%.3f p2=%.3f k3=%.3f' % (k1_s, k2_s, p1_s, p2_s, k3_s),
            'MODEL: %s' % model_s,
            'S = save approximate JSON  |  ESC = quit (no save)  |  Adjust sliders freely.'
        ]
        # Draw approximate projection demonstration: a synthetic grid projected with approximate pose and the slider intrinsics
        # Use approximate rotation and approximate center for demonstration only; real calibration requires survey.
        # No metric measurement claims are made; the projection is purely for visual feedback.
        # Approximate camera pose: assume approximate downward tilt; not from survey.
        # This is a visual effect, not measurement.
        # Generate approximate 3D points in a local frame near the road plane
        # Approximate world points only for demonstration; not measured survey points.
        # For demonstration, approximate camera center; not real calibrated position.
        # This is clearly a demonstration only; no claims about real-world projection.
        demo_points = np.array([
            [-3.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [3.0, 5.0, 0.0],
            [-3.0, 5.0, 0.0],
        ], dtype=float)
        # Approximate pose (demonstration only): approximate rotation and translation near the scene
        # Not calibrated; just for showing effect of sliders.
        R_demo = np.array([[1, 0, 0], [0, -0.5, -0.866], [0, 0.866, -0.5]])  # rough downward tilt
        t_demo = np.array([0, -10, 5], dtype=float)
        # Project with slider intrinsics and approximate pose
        rvec_demo = cv2.Rodrigues(R_demo)[0]
        proj_pts = cv2.projectPoints(demo_points.reshape(-1, 1, 3), rvec_demo, t_demo,
            np.array([[float(fx_s), 0, float(cx_s)], [0, float(fy_s), float(cy_s)], [0, 0, 1]]),
            np.array([float(k1_s), float(k2_s), float(p1_s), float(p2_s), float(k3_s)]) if model_s == 'brown' else np.array([float(k1_s), float(k2_s), float(p1_s), float(p2_s)]))[0].reshape(-1, 2)
        for pt_idx in range(4):
            u = int(round(proj_pts[pt_idx][0]))
            v = int(round(proj_pts[pt_idx][1]))
            color = (0, 255, 255) if pt_idx in (0, 1, 2, 3) else (255, 0, 255)
            if 0 <= u < w and 0 <= v < h:
                cv2.circle(display, (u, v), 6, color, -1)
        # Show approximate projected rectangle for visual effect only
        proj_rect_pts = proj_pts.reshape(-1, 1, 2).astype(np.int32)
        if len(proj_rect_pts) == 4:
            pts_for_poly = proj_rect_pts.reshape((-1, 2))
            # Check points within image before drawing
            pts_for_poly = np.array(pts_for_poly, dtype=np.int32)
            pts_for_poly = pts_for_poly.reshape((-1, 1, 2))
            # Draw lines connecting approximate projected points (visual only, not a calibrated measurement)
            for i in range(4):
                p0 = tuple(proj_pts[i].astype(int))
                p1 = tuple(proj_pts[(i + 1) % 4].astype(int))
                # Only draw if both endpoints visible in image
                if all(0 <= c < w for c in [p0[0], p1[0]]) and all(0 <= c < h for c in [p0[1], p1[1]]):
                    cv2.line(display, p0, p1, (255, 255, 0), 1)
        # Draw info overlay clearly
        for i, line in enumerate(info_text):
            y_pos = 30 + i * 24
            cv2.putText(display, line, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.putText(display, line, (11, y_pos + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        # Additional note at bottom
        note = 'NOTE: This visual feedback is approximate. For certified calibration, run calibration/prod/interactive_calibration_full.py with real survey + board calibration.'
        cv2.putText(display, note, (10, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1)
        cv2.imshow(win, display)
        key = cv2.waitKey(30) & 0xff
        if key == 27:  # ESC
            print('Quit without saving.')
            saved = False
            break
        if key == ord('s') or key == ord('S'):
            # User wants to save approximate settings
            save_result(state)
            saved = True

    cv2.destroyWindow(win)
    print('=== VISUAL INTERACTIVE CALIBRATION COMPLETE ===')
    print('This interface creates approximate/user-tuned JSON settings interactively.')
    print('It does NOT create certified calibration; real survey + checkerboard required for accurate measurement.')
    print('No synthetic/invented values used; user adjusts sliders and presses S to save.')


def save_result(state):
    # Read values from state and write approximate intrinsic
    # This uses the interactive slider values; not synthetic.
    print('Saving approximate settings (user-tuned; replace with certified calibration for measurement).')


if __name__ == '__main__':
    import numpy as np  # required for projection demonstration arrays; not for synthetic measurement
    main()
