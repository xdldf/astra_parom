"""Recompute short passage evidence from original media, never browser estimates."""
import math

import cv2

from vehicle_metrology.temporal import measure_passage, apply_passage
from web_app import workbench as wb

WINDOW_SECONDS = .45
MAX_FRAMES = 31


def capture_candidate(detection):
    return detection['length_m'] is not None or (detection.get('at_measurement_line') and
                                                 detection['status'] in {'outside_calibration','calibration_review'})


def refine(profile, anchor_box, frames):
    """Frames yield distinct (index, original pixels) from one bounded passage."""
    observations = []
    for index, raw in frames:
        if len(observations) >= MAX_FRAMES:
            break
        if raw is None or tuple(raw.shape[1::-1]) != profile.image_size:
            continue
        frame = wb.corrected(raw, profile.lens)
        detections = wb.detect_vehicles(frame, .3, detector_model=profile.detector_model,
                                         imgsz=profile.detector_imgsz)
        observations.append(dict(frame=index, detections=detections))
    return measure_passage(observations, anchor_box, profile.polygon, wb.profile_scale(profile),
                           profile.image_size, line_x=profile.measurement_line_x,
                           line_tolerance_px=profile.line_tolerance_px,
                           tolerance_m=profile.accuracy_tolerance_m)


def video_frames(item, anchor):
    fps = item.get('fps', 0)
    if not math.isfinite(fps) or fps <= 0:
        return
    radius = max(1, round(fps*WINDOW_SECONDS))
    first, last = max(0, anchor-radius), min(item['frames']-1, anchor+radius)
    indices = sorted(set([anchor, *range(first, last+1, max(1, math.ceil((last-first+1)/(MAX_FRAMES-1))))]))
    selected = set(indices)
    capture = cv2.VideoCapture(str(item['path']))
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, first)
        for index in range(first, last+1):
            ok, raw = capture.read()
            if not ok:
                break
            if index in selected:
                yield index, raw
    finally:
        capture.release()


def refine_video(payload, result):
    measured = result['detections'][0]
    item = wb.media[payload.media_id]
    if measured['length_m'] is None or item['kind'] != 'video':
        return
    passage = refine(payload.profile, payload.bbox, video_frames(item, payload.frame))
    passage.update(detector_model=payload.profile.detector_model, imgsz=payload.profile.detector_imgsz,
                   anchor_frame=payload.frame, window_seconds=WINDOW_SECONDS)
    result['detections'][0] = apply_passage(measured, passage)
