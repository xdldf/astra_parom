"""Recompute short passage evidence from original media, never browser estimates."""
import math

import cv2

from vehicle_metrology.detection import box_iou
from vehicle_metrology.temporal import measure_passage, apply_passage
from web_app import workbench as wb

WINDOW_SECONDS = .45
MAX_FRAMES = 31


def crossing_fraction(profile, before_box, after_box):
    line = profile.measurement_line_x
    if line is None:
        raise ValueError('A measurement line is required')
    width, height = profile.image_size
    for box in (before_box, after_box):
        x, y, w, h = box
        if (not all(math.isfinite(v) for v in box) or min(x,y)<0 or min(w,h)<=2
                or x+w>width or y+h>height):
            raise ValueError('Invalid crossing box')
    before = before_box[0]+before_box[2]/2-line
    after = after_box[0]+after_box[2]/2-line
    if before*after >= 0:
        raise ValueError('Observed boxes must straddle the measurement line')
    return abs(before)/(abs(before)+abs(after))


def crossing_match(detections, expected_box):
    """Recover the same vehicle, never a neighbour merely close to the line."""
    matches = sorted(((box_iou(expected_box,d['bbox']),d) for d in detections),
                     key=lambda pair:pair[0],reverse=True)
    if not matches or matches[0][0]<.5 or (len(matches)>1 and matches[1][0]>.4):
        return None
    candidate = matches[0][1]
    return candidate if candidate.get('at_measurement_line') and capture_candidate(candidate) else None


def find_video_crossing(profile, item, before_frame, before_box, after_frame, after_box):
    """Search actual neighbours of a skipped crossing; keep the original line gate."""
    fps = item.get('fps',0)
    if (item.get('kind')!='video' or not math.isfinite(fps) or fps<=0
            or not 0<=before_frame<after_frame<item['frames']
            or (after_frame-before_frame)/fps>2):
        raise ValueError('Crossing requires two video observations at most two seconds apart')
    fraction = crossing_fraction(profile,before_box,after_box)
    predicted = round(before_frame+fraction*(after_frame-before_frame))
    radius = min((MAX_FRAMES-1)//2,max(2,round(fps*WINDOW_SECONDS)))
    first,last = max(before_frame,predicted-radius),min(after_frame,predicted+radius)
    capture = cv2.VideoCapture(str(item['path']))
    examined = 0
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES,first)
        for index in range(first,last+1):
            ok,raw = capture.read()
            if not ok:
                break
            result = wb.render_raw(raw,wb.FrameRequest(profile=profile,frame=index,detect=True),include_image=False)
            examined += 1
            t = (index-before_frame)/(after_frame-before_frame)
            expected = [a+t*(b-a) for a,b in zip(before_box,after_box)]
            candidate = crossing_match(result['detections'],expected)
            if candidate is not None:
                return dict(frame=index,detection=candidate,examined_frames=examined,predicted_frame=predicted)
    finally:
        capture.release()
    return dict(frame=None,examined_frames=examined,predicted_frame=predicted,
                reason='Не найден кадр этого автомобиля у линии. Проверьте проезд вручную.')


def capture_candidate(detection):
    return detection['length_m'] is not None or (detection.get('at_measurement_line') and
                                                 detection['status'] in {'outside_calibration','calibration_review'})


def refinement_candidate(measured):
    # A jittery first box can fall outside depth support while its actual
    # neighbours at the line are calibrated. Let those frames prove the fit.
    return measured['length_m'] is not None or (measured.get('at_measurement_line') and
                                               measured['status']=='outside_calibration')


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
    options=dict(line_x=profile.measurement_line_x,line_tolerance_px=profile.line_tolerance_px,
                 tolerance_m=profile.accuracy_tolerance_m)
    args=(observations,anchor_box,profile.polygon,wb.profile_scale(profile),profile.image_size)
    strict=measure_passage(*args,**options)
    if strict['length_m'] is not None or profile.measurement_mode=='strict':return strict
    estimated=measure_passage(*args,**options,estimate=True)
    estimated['strict_review_reasons']=strict['reasons']
    return estimated


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
    if not refinement_candidate(measured) or item['kind'] != 'video':
        return
    passage = refine(payload.profile, payload.bbox, video_frames(item, payload.frame))
    passage.update(detector_model=payload.profile.detector_model, imgsz=payload.profile.detector_imgsz,
                   anchor_frame=payload.frame, window_seconds=WINDOW_SECONDS)
    result['detections'][0] = apply_passage(measured, passage,allow_estimate=payload.profile.measurement_mode=='estimate')
