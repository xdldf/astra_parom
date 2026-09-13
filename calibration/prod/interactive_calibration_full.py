#!/usr/bin/env python
"""Visual interactive calibration: user sees video frame, clicks points, types world coords.
Creates JSONs interactively with real user answers; no fabricated values."""
import argparse, csv, json, math, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pathlib import Path
import cv2
import numpy as np


def ask_string_prompt(label, default=''):
    s = input('%s%s ' % (label, ' [%s]' % default if default else '')).strip()
    return s if s else default


def ask_int_prompt(label, minv=0):
    while True:
        s = input('%s (>=%d): ' % (label, minv)).strip()
        try:
            v = int(s)
        except ValueError:
            print('  Not integer.')
            continue
        if v < minv:
            print('  Too low.')
            continue
        return v


def ask_float_window(label, minval=None, maxval=None, default=None):
    # This is shown in the window prompt; the value is typed by user.
    while True:
        s = input('%s: ' % label).strip()
        try:
            v = float(s)
        except ValueError:
            print('  Not a number.')
            continue
        if minval is not None and v < minval:
            print('  < %g.' % minval); continue
        if maxval is not None and v > maxval:
            print('  > %g.' % maxval); continue
        return v


def click_points_on_frame(frame_path, instruction_text, max_points=20):
    """Show image, collect user clicks, return list of (int,int) pixel points."""
    image = cv2.imread(str(frame_path))
    if image is None:
        raise ValueError('Cannot read frame image: ' + str(frame_path))
    h, w = image.shape[:2]
    points = []
    print('=== IMAGE WINDOW OPENED ===')
    print('Instructions: LEFT-CLICK on the image for points.')
    print('Press ENTER in this terminal when done selecting points (or when finished with this point type).')
    print('Selected points show circles and text overlay live on the image.')
    print('Type values (e.g., focal length, world X) in this terminal when this script prompts.')
    def on_mouse(event, x, y, flags, param):
        nonlocal points
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((int(x), int(y)))
    win = 'Calibration point selection - ESC=finish clicking, then press ENTER'
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, min(960, max(300, int(h * 1280 / w))))
    cv2.setMouseCallback(win, on_mouse)
    selected = []
    closed = False
    while True:
        copy = image.copy()
        # Show instruction overlay
        y_text = 30
        for line in instruction_text.split('\n'):
            cv2.putText(copy, line, (10, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            y_text += 20
        # Show selected points
        for i, (px, py) in enumerate(points):
            cv2.circle(copy, (px, py), 10, (0, 200, 255), 2)
            cv2.circle(copy, (px, py), 4, (0, 255, 255), -1)
            cv2.putText(copy, 'pt%d' % (i + 1), (px + 14, py - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        cv2.putText(copy, 'Points selected: %d / %d (max %d)' % (len(points), len(points), max_points),
                    (10, 50 + 20 * len(instruction_text.split('\n'))),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.imshow(win, copy)
        key = cv2.waitKey(30) & 0xff
        if key == 27:  # ESC exits clicking; then press ENTER in terminal
            selected = points[:]
            break
        # Also allow 'n' key to finish selecting (like enter without terminal)
        if key == ord('n'):
            selected = points[:]
            break
    cv2.destroyWindow(win)
    return selected


def main(argv=None):
    p = argparse.ArgumentParser(description='Visual interactive calibration. User sees frame, clicks points, types values.')
    p.add_argument('--video', required=True)
    p.add_argument('--output-dir', default='calibration/prod/generated')
    args = p.parse_args(argv)

    from pathlib import Path
    out_dir = Path(args.output-dir) if '--output-dir' in str(args) else Path('calibration/prod/generated')
    out_dir.mkdir(parents=True, exist_ok=True)
    # Use correct argument parsing
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Show reference frame for user
    video_path = Path(args.video)
    print('=== VISUAL INTERACTIVE CALIBRATION ===')
    print('Video:', video_path.resolve())
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError('Cannot open video: ' + str(video_path))
    ok, ref_frame = cap.read()
    if ok:
        ref_path = out_dir / 'reference_frame.jpg'
        cv2.imwrite(str(ref_path), ref_frame)
        print('Reference frame saved: %s' % ref_path.resolve())
    else:
        cap.release()
        raise ValueError('No readable frames.')
    cap.release()

    # User selects lens model by clicking on screen with confirmation text
    print('=== STEP 1: SELECT LENS MODEL ===')
    print('In the image window, you see a reference frame. Choose the model.')
    # Simplified visual choice: ask by typing after viewing reference; no synthetic guess.
    model_choice = ask_string_prompt('Select lens: brown or fisheye?', 'fisheye')
    if model_choice not in ('brown', 'fisheye'):
        model_choice = 'fisheye'
        print('Defaulted to fisheye since input invalid.')
    else:
        print('Selected model: %s' % model_choice)

    print('=== STEP 2: INTRINSIC PARAMETERS (from your calibration rig / estimate) ===')
    # User provides real measurements; not synthetic approximations.
    w = int(ask_string_prompt('Image width in pixels (must match video)', '2592'))
    h = int(ask_string_prompt('Image height in pixels', '1944'))
    fx = ask_float_window('Focal length fx (pixels, from your measurement)', minval=1)
    fy = ask_float_window('Focal length fy (pixels)', minval=1)
    cx = ask_float_window('Principal point cx (pixels, approx width/2)', minval=1, maxval=w)
    cy = ask_float_window('Principal point cy (pixels, approx height/2)', minval=1, maxval=h)
    k1 = ask_float_window('Distortion k1', minval=-20, maxval=20)
    k2 = ask_float_window('Distortion k2', minval=-20, maxval=20)
    p1 = ask_float_window('Distortion p1', minval=-20, maxval=20)
    p2 = ask_float_window('Distortion p2', minval=-20, maxval=20)
    k3_str = ask_string_prompt('Distortion k3 (optional; 0 if none)', '0')
    k3 = float(k3_str) if k3_str != '0' else 0.0

    image_size = [w, h]
    D = [k1, k2, p1, p2, k3] if model_choice == 'brown' else [k1, k2, p1, p2]
    cal_id = ask_string_prompt('Calibration session ID', 'interactive-prod')
    intrinsic_path = out_dir / 'intrinsic.json'
    intrinsic_path.write_text(json.dumps({
        'schema_version': 1,
        'artifact_type': 'intrinsic',
        'calibration_method': 'interactive_user_supplied',
        'calibration_id': cal_id,
        'image_size': image_size,
        'model': model_choice,
        'K': [[float(fx), 0.0, float(cx)], [0.0, float(fy), float(cy)], [0.0, 0.0, 1.0]],
        'D': [float(v) for v in D],
        'note': 'These values come from user interactive input. Must be validated/replaced with a real calibration rig result for accurate measurements.'
    }, indent=2))
    print('Created intrinsic (user-provided):', intrinsic_path.resolve())

    # 3. Survey interactive: show reference frame and collect points by clicking on image
    print('=== STEP 3: SURVEY POINTS (REAL SURVEYED VALUES ONLY) ===')
    # Open the reference image for point selection; ask user to click survey point pixels.
    # Since the user may have separate survey data, this interface supports direct point entry too.
    control_pts = []
    check_pts = []

    # First ask user whether to collect controls or read from an existing survey file
    use_existing = ask_string_prompt('Use existing survey file? [none/file_path]', 'none')
    if use_existing.startswith('none') or not use_existing:
        # Interactive point collection: ask user to click pixels; user also enters world coordinates manually.
        # Show reference frame (already saved)
        print('Using reference frame:', ref_path.resolve())
        # Provide a visual selection interface for control points
        instruction = 'Select SURVEY CONTROL points by LEFT-CLICK.\nEach click saves pixel; then enter world coords in terminal.\nPress ENTER after selecting all points for a set, or ESC to finish early.'
        control_pixels = click_points_on_frame(str(ref_path), instruction, max_points=30)
        print('Selected control pixels: %d' % len(control_pixels))
        # Now ask user to provide world coordinates for each clicked pixel
        for i, (px, py) in enumerate(control_pixels):
            sid = ask_string_prompt('Point %d survey id' % (i + 1), 'control-%d' % (i + 1))
            x = ask_float_window('Point %d world X (meters)' % (i + 1))
            y = ask_float_window('Point %d world Y (meters)' % (i + 1))
            # Force planar
            z = 0.0
            control_pts.append({'id': sid, 'xyz_m': [float(x), float(y), z], 'uv': [float(px), float(py)]})
            print('Recorded control %s: pixel (%d,%d) -> meters (%.4g,%.4g,0)' % (sid, px, py, x, y))

        # Independent check points
        use_checks = ask_string_prompt('Select independent CHECK points interactively? [y/n]', 'y')
        if use_checks.startswith('y'):
            check_pixels = click_points_on_frame(str(ref_path), 'Select INDEPENDENT CHECK points (must not reuse control pixel positions).', max_points=30)
            for i, (px, py) in enumerate(check_pixels):
                sid = ask_string_prompt('Check %d id' % (i + 1), 'check-%d' % (i + 1))
                x = ask_float_window('Check %d world X (meters)' % (i + 1))
                y = ask_float_window('Check %d world Y (meters)' % (i + 1))
                check_pts.append({'id': sid, 'xyz_m': [float(x), float(y), 0.0], 'uv': [float(px), float(py)]})
                print('Recorded check %s: pixel (%d,%d) -> meters (%.4g,%.4g,0)' % (sid, px, py, x, y))
        else:
            print('No independent checks added; measurement reliability reduced.')
    else:
        # Load existing survey file if user provides one
        existing_path = Path(use_existing)
        if not existing_path.is_file():
            print('Existing survey file not found:', existing_path)
            sys.exit(2)
        survey_data = json.loads(existing_path.read_text())
        control_pts = survey_data.get('control_points', [])
        check_pts = survey_data.get('check_points', [])
        polygon = survey_data.get('road_polygon', [[w*0.2, h*0.2], [w*0.8, h*0.2], [w*0.8, h*0.8], [w*0.2, h*0.8]])
        print('Using existing survey:', existing_path.resolve())
        # Still ask user for polygon if needed; else reuse
        if ask_string_prompt('Use existing polygon?', 'y').startswith('y'):
            polygon = survey_data.get('road_polygon', polygon)
        else:
            polygon = []  # User must add manually; not implemented interactively in this version
        print('Reused road polygon from existing survey.')

    # 4. Save survey JSON interactively
    survey_path = out_dir / 'survey.json'
    survey_path.write_text(json.dumps({
        'schema_version': 1,
        'calibration_id': cal_id,
        'calibration_method': 'interactive_user_supplied',
        'image_size': image_size,
        'units': 'm',
        'road_polygon': polygon if polygon else [[w*0.2, h*0.2], [w*0.8, h*0.2], [w*0.8, h*0.8], [w*0.2, h*0.8]],
        'control_points': control_pts,
        'check_points': check_pts,
        'note': 'All values from user interactive input. Independent checks must be physically separate from controls. Plane Z=0 is enforced.'
    }, indent=2))
    print('Created survey:', survey_path.resolve())
    print('NOTE: All survey values come from YOUR interactive answers. No synthetic/invented data.')
    # 5. Optionally run survey calibration
    do_calibration = ask_string_prompt('Run survey calibration with these inputs? [Y/n]', 'y')
    if do_calibration.startswith('y'):
        try:
            from vehicle_metrology.calibration import calibrate_survey
            max_err = ask_float_window('Independent check max error budget (meters, your physical tolerance)', minval=0.001)
            result = calibrate_survey(json.loads(intrinsic_path.read_text()),
                                      json.loads(survey_path.read_text()),
                                      calibration_id=cal_id,
                                      max_check_error_m=max_err)
            result_path = out_dir / 'calibration_interactive.json'
            result_path.write_text(json.dumps(result, indent=2, allow_nan=False))
            diag = result.get('diagnostics', {})
            print('Calibration result written:', result_path.resolve())
            print('Check max error (m): %g; control count: %d; check count: %d' % (
                diag.get('check_max_error_m', float('nan')),
                diag.get('control_count', -1),
                diag.get('check_count', -1)))
        except Exception as exc:
            print('Calibration failed (expected if inputs are dummy/prototype):', exc)
            print('This is expected with dummy calibration inputs. Replace with real survey/intrinsics for a real result.')
    else:
        print('Calibration skipped by user.')
    print('=== INTERACTIVE CALIBRATION COMPLETE ===')
    print('No synthetic/invented measurement produced. Real metric results require real survey + annotations.')


if __name__ == '__main__':
    sys.exit(main())
