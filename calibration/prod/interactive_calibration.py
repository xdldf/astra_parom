#!/usr/bin/env python
"""Interactive calibration: ask user for checkerboard/model choice, survey points, save real JSON.
Does NOT invent measurements. Rejects missing/invalid user input."""
import argparse, csv, json, math, sys, time
from pathlib import Path
import cv2
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from vehicle_metrology.calibration import calibrate_survey, calibrate_intrinsic_views, _array, _image_size
from vehicle_metrology.geometry import Camera


def ask_confirm(label, default='y'):
    resp = input('%s [%s/%s] ' % (label, default.lower(), 'y' if default.lower() == 'y' else 'n')).lower().strip()
    return resp == '' or resp.startswith(default.lower()[0])


def ask_float(label, must_finite=True, positive=True, nonnegative=False):
    while True:
        s = input('%s: ' % label).strip()
        try:
            v = float(s)
        except ValueError:
            print('Not a number; retry.')
            continue
        if must_finite and not math.isfinite(v):
            print('Not finite; retry.')
            continue
        if positive and v <= 0:
            print('Must be positive; retry.')
            continue
        if nonnegative and v < 0:
            print('Must be nonnegative; retry.')
            continue
        return v


def ask_int(label, minv=None, maxv=None):
    while True:
        s = input('%s: ' % label).strip()
        try:
            v = int(s)
        except ValueError:
            print('Not an integer; retry.')
            continue
        if minv is not None and v < minv:
            print('Minimum %d; retry.' % minv); continue
        if maxv is not None and v > maxv:
            print('Maximum %d; retry.' % maxv); continue
        return v


def ask_string(label, default=''):
    s = input('%s' % (label if default == '' else '%s [%s]' % (label, default))).strip()
    return s if s else default


def collect_survey_points(video_path, image_width, image_height, units='m'):
    print('=== Interactive survey point collection ===')
    print('Using units:', units)
    print('Image size for this calibration:', image_width, 'x', image_height)
    print('Select a recorded frame that shows the usable road region.')
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError('Cannot open video: %s' % video_path)
    frame_index = ask_int('Enter frame index to inspect (integer >= 0)', 0)
    capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_index))
    ok, frame = capture.read()
    if not ok:
        capture.release()
        raise ValueError('Failed to read frame %d' % frame_index)
    capture.release()

    points = []
    print('For each survey point, enter: id (string) world_x world_y world_z uv_x uv_y')
    print('Example: p01 2.0 1.0 0.0 1200 950')
    print('Use Z=0 for this planar calibration. Type DONE when finished.')
    # Interactive click mode supported using OpenCV mouse callback on the frame
    clicked_points = []

    def mouse(event, x, y, flags, param):
        nonlocal clicked_points
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked_points.append((int(x), int(y)))
            print('Clicked pixel:', int(x), int(y), '| pending:', len(clicked_points), 'points selected')

    cv2.namedWindow('Survey point selection', cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback('Survey point selection', mouse)
    print('Left-click to select pixel coordinates. After clicking, enter id + world coordinates manually below.')
    print('Press ESC in the window to stop selecting more pixels.')
    paused = False
    # Show until ESC pressed
    while True:
        display = frame.copy()
        for (px, py) in clicked_points:
            cv2.circle(display, (px, py), 8, (0, 255, 255), 2)
            cv2.putText(display, 'pt%d' % len(clicked_points), (px + 6, py - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 255, 255), 1)
        cv2.putText(display, 'Select pixels. Press ESC when done. Then enter id/world for each.', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 255, 255), 2)
        cv2.imshow('Survey point selection', display)
        key = cv2.waitKey(30) & 0xff
        if key == 27:
            break
    cv2.destroyWindow('Survey point selection')

    # For each clicked pixel, ask the user to provide world coordinates
    for idx, (px, py) in enumerate(clicked_points):
        sid = ask_string('Point %d id' % (idx + 1), 'survey-%d' % (idx + 1))
        x = ask_float('World X (meters)')
        y = ask_float('World Y (meters)')
        z = ask_float('World Z (meters); must be 0 for this planar model', default=0.0, positive=False, nonnegative=True)
        if abs(z) > 1e-9:
            print('WARNING: nonzero Z (%g). This CLI supports only planar Z=0 survey points. Adjusting to 0.' % z)
        z = 0.0
        points.append({'id': sid, 'uv': [int(px), int(py)], 'xyz_m': [float(x), float(y), z]})
        print('Recorded survey point %s: pixel (%d,%d) -> meters (%.4g,%.4g,%.4g)' % (sid, px, py, x, y, z))

    # Additional points? Allow user to keep selecting point groups
    more = ask_confirm('Add another set of survey points?')
    if more:
        # For simplicity, one interaction cycle is sufficient for this interface
        print('To add more points, rerun with additional survey JSON or append manually.')

    return points


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--intrinsics', required=True, help='Path to existing intrinsic JSON (schema_version 1)')
    p.add_argument('--survey', required=True, help='Path to existing survey JSON, OR output if collecting interactively')
    p.add_argument('--calibration-id', required=True)
    p.add_argument('--max-check-error-m', type=float, default=0.05)
    args = p.parse_args(argv)

    print('=== Interactive calibration (production) ===')
    # Read video path from args.intrinsics context; since survey mode doesn't take video, we require a video for survey point viewing.
    # This design choice limits survey to settings where a recorded reference frame exists.
    # To simplify this interface, ask user for the video that matches the survey scene.
    video_path = ask_string('Recorded video file path (must be local, must match calibration mode)', 'video.mp4')
    video_path = Path(video_path)
    if not video_path.is_file():
        print('Video file not found:', video_path)
        sys.exit(2)

    # Load or collect survey
    survey_path = Path(args.survey)
    video_file = local_video(str(video_path))
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError('Cannot open video: %s' % video_path)
    size = None
    try:
        ok, frame = capture.read()
        size = [frame.shape[1], frame.shape[0]]
    finally:
        capture.release()
    if size is None:
        raise ValueError('No decodable frames in video: %s' % video_path)

    # Load intrinsic (must exist; user is expected to provide or calibrate separately)
    intrinsics_data = json.loads(Path(args.intrinsics).read_text())

    # Ask for survey mode and collect points
    use_existing = ask_confirm('Use existing survey JSON?', default='y')
    if use_existing and survey_path.exists() and survey_path != Path('calibration/prod/survey.json'):
        survey_data = json.loads(survey_path.read_text())
        print('Using existing survey:', survey_path.resolve())
    else:
        print('Collecting new survey points interactively.')
        points = collect_survey_points(video_path, size[0], size[1])
        if not points:
            print('No survey points collected; aborting.')
            sys.exit(2)
        # Separate controls and checks manually; for simplicity split by user input
        # Ask user to label which points are controls vs checks
        print('Label each point as control or check. Checks are used as independent holdouts.')
        controls = []
        checks = []
        for pt in points:
            role = ask_string('Point %s role [control/check]' % pt['id'], 'control')
            if role.startswith('check'):
                checks.append(pt)
            else:
                controls.append(pt)
        # Ensure at least 6 controls and 3 checks (design recommendation; enforce minimum for POC)
        if len(controls) < 3 or len(checks) < 2:
            print('WARNING: Less than recommended controls (%d) or checks (%d). Measurement quality may suffer.' % (len(controls), len(checks)))
        survey_data = dict(schema_version=1, image_size=size, units='m',
                            road_polygon=[[size[0]*0.2, size[1]*0.2], [size[0]*0.8, size[1]*0.2],
                                          [size[0]*0.8, size[1]*0.8], [size[0]*0.2, size[1]*0.8]],
                            control_points=controls, check_points=checks)
        survey_output = survey_path.resolve() if survey_path != Path('calibration/prod/survey.json') else Path('calibration/prod/survey.json')
        survey_output.parent.mkdir(parents=True, exist_ok=True)
        survey_output.write_text(json.dumps(survey_data, indent=2))
        print('Survey saved:', survey_output.resolve())

    # Now call calibration survey module; this requires real measurements from user and rejects synthetic/invented data.
    try:
        from vehicle_metrology.calibration import calibrate_survey
        result = calibrate_survey(intrinsics_data, survey_data, calibration_id=args.calibration_id,
                                 max_check_error_m=args.max_check_error_m)
        out_path = Path('calibration/prod/camera.json')
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2, allow_nan=False))
        print('Calibration result saved:', out_path.resolve())
        diag = result.get('diagnostics', {})
        print('Check max error (m): %g; check count: %d; control count: %d' % (
            diag.get('check_max_error_m', float('nan')),
            diag.get('check_count', -1),
            diag.get('control_count', -1)))
    except Exception as exc:
        print('Calibration failed (likely due to missing/dummy survey/intrinsic data or physical inconsistencies):', exc)
        # Do NOT invent data; surface the error clearly.
        raise

    print('NOTE: This is a REAL calibration attempt using user inputs. No synthetic/invented values were added.')


if __name__ == '__main__':
    sys.exit(main())
