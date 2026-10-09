"""Optional capture-time silhouette estimation shared by video and IP cameras."""
from functools import lru_cache
import hashlib
from pathlib import Path
import threading

import cv2
import numpy as np

from vehicle_metrology.detection import box_iou
from vehicle_metrology.outline import WEIGHTS_NAME, WEIGHTS_SHA256, outline_features, predict_outline

ROOT = Path(__file__).resolve().parents[1]
lock = threading.Lock()
UNAVAILABLE_WARNING = 'Контурная оценка недоступна. Длина не назначена; исходная оценка рамки и полный кадр сохранены для проверки.'


@lru_cache(maxsize=1)
def load_model():
    from ultralytics import YOLO
    path = ROOT/'web_app/data'/WEIGHTS_NAME
    if not path.is_file():
        raise ValueError('Run scripts/prepare_outline_model.py to install the outline model')
    if hashlib.sha256(path.read_bytes()).hexdigest() != WEIGHTS_SHA256:
        raise ValueError('Outline model checksum mismatch')
    return YOLO(str(path))


def infer_outline(image, expected_box, *, device=None):
    from web_app.workbench import require_gpu
    # Device override is only exposed to offline Python runners, never HTTP.
    device = require_gpu() if device is None else device
    with lock:
        result = load_model().predict(image, device=device, imgsz=1280, classes=[2, 5, 7],
                                      conf=.25, retina_masks=True, verbose=False)[0]
    matches = []
    for i, (x0, y0, x1, y1) in enumerate(result.boxes.xyxy.cpu().numpy()):
        box = [float(x0), float(y0), float(x1-x0), float(y1-y0)]
        matches.append((box_iou(expected_box, box), i, box))
    matches.sort(reverse=True)
    if not matches or matches[0][0] < .5 or result.masks is None:
        raise ValueError('No matching vehicle silhouette')
    # Class-aware NMS can retain the same car as both car and truck. Only
    # collapse near-identical boxes AND masks; overlapping cars remain ambiguous.
    iou, i, box = matches[0]
    mask = result.masks.data[i].cpu().numpy().astype(np.uint8)
    competing = []
    duplicates = 0
    for candidate_iou, candidate_index, candidate_box in matches[1:]:
        if candidate_iou <= .4:
            continue
        candidate_mask = result.masks.data[candidate_index].cpu().numpy().astype(bool)
        same_box = box_iou(box, candidate_box) >= .9
        intersection = np.count_nonzero((mask != 0) & candidate_mask)
        union = np.count_nonzero((mask != 0) | candidate_mask)
        if same_box and union and intersection/union >= .9:
            duplicates += 1
        else:
            competing.append(candidate_index)
    if competing:
        raise ValueError('Ambiguous vehicle silhouette')
    if mask.shape != image.shape[:2]:
        raise ValueError('Silhouette resolution differs from corrected frame')
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    contour = max(contours, key=cv2.contourArea)
    return mask, dict(bbox=box, iou=iou, confidence=float(result.boxes.conf[i]),
                      detector_class=int(result.boxes.cls[i]),
                      contour=contour[:, 0].tolist(), image_space='lens_corrected_full_frame',
                      duplicate_proposals_ignored=duplicates)


def apply_outline(profile, measured, image, *, label):
    calibration = profile.outline_calibration
    if calibration is None or profile.measurement_mode != 'estimate':
        return measured
    # Preserve capture/visibility rules. An outline must never make a clipped
    # vehicle complete or turn a missing-line trigger into a successful one.
    if (measured.get('depth') is None or measured['status'] in {'clipped', 'outside_road', 'waiting_for_line'}
            or {'clipped', 'missed_measurement_line'} & set(measured.get('quality_reasons', []))):
        return measured
    result = dict(measured, warnings=list(measured.get('warnings', [])),
                  quality_reasons=list(measured.get('quality_reasons', [])))
    evidence = dict(calibration_id=calibration.calibration_id, method='catalogue_outline_regression',
                    reference_source='catalogue', accuracy_validated=False, status='unavailable',
                    model=WEIGHTS_NAME, model_sha256=WEIGHTS_SHA256, imgsz=1280,
                    estimate_source='single_corrected_capture_frame',
                    baseline_length_m=measured.get('length_m'), baseline_status=measured['status'])
    result['outline'] = evidence
    if label not in {'car', 'manual car', 'selected car'}:
        evidence.update(status='not_applicable', reason='Catalogue outline model covers passenger vehicles only')
        return result
    try:
        if 'ambiguous_vehicle_association' in measured.get('temporal', {}).get('reasons', []):
            raise ValueError('Ambiguous vehicle identity in neighbouring frames')
        if image is None or image.shape[1::-1] != profile.image_size:
            raise ValueError('Missing corrected full-resolution capture frame')
        mask, details = infer_outline(image, measured['bbox'])
        features = outline_features(mask, profile.polygon)
        candidate = predict_outline(features, calibration.model_dump())
        evidence.update(details, features=features.tolist(), candidate_length_m=candidate)
        lower, upper = np.asarray(calibration.feature_bounds).T
        # This is a coarse applicability guard, not an accuracy interval.
        margin = .1*(upper-lower)
        if np.any(features < lower-margin) or np.any(features > upper+margin):
            evidence['failure_code'] = 'outside_feature_range'
            raise ValueError('Silhouette is outside the catalogue training feature range')
        evidence.update(status='estimated', length_m=candidate)
        result.update(length_m=candidate, status='outline_estimate', approximate=True,
                      position_model='catalogue_trained_visible_outline', cm_per_px=100*candidate/measured['bbox'][2],
                      coefficient=None, accuracy_validated=False)
        result['quality_reasons'].append('catalogue_outline_estimate')
        result['warnings'].append('Длина оценена по контуру кузова и каталожным эталонам. Точность 10 см для каждого автомобиля пока не достигнута. Проверьте результат.')
    except Exception as exc:
        # An unavailable GPU/model or failed mask must not discard a car/photo.
        evidence['reason'] = str(exc)
        result['quality_reasons'].append('outline_unavailable')
        result.update(length_m=None, approximate=False, status='capture_review', cm_per_px=None, coefficient=None)
        result['warnings'].append(UNAVAILABLE_WARNING)
    return result
