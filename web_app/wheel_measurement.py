"""Optional wheel refinement of an available outline capture estimate."""
from functools import lru_cache
from pathlib import Path
import threading

import cv2
import numpy as np

from vehicle_metrology.wheels import (
    MODEL_DIRECTORY, MODEL_ID, MODEL_REVISION, MODEL_FILES, PROMPT, THRESHOLD,
    verify_model, wheel_roi, select_wheel_pair, wheel_features, predict_wheels,
)

ROOT = Path(__file__).resolve().parents[1]
lock = threading.Lock()


@lru_cache(maxsize=1)
def load_model():
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    directory = ROOT/'web_app/data'/MODEL_DIRECTORY
    verify_model(directory)
    processor = AutoProcessor.from_pretrained(directory, local_files_only=True, trust_remote_code=False)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        directory, local_files_only=True, trust_remote_code=False, use_safetensors=True).eval()
    return processor, model


def infer_wheels(image, expected_box, *, device=None):
    import torch
    from PIL import Image
    from web_app.workbench import require_gpu
    # Only offline Python runners may override the production CUDA requirement.
    device = require_gpu() if device is None else device
    left, top, right, bottom = wheel_roi(expected_box, image.shape[1::-1])
    roi = Image.fromarray(cv2.cvtColor(image[top:bottom, left:right], cv2.COLOR_BGR2RGB))
    with lock, torch.inference_mode():
        processor, model = load_model()
        model.to(device)
        inputs = processor(images=roi, text=PROMPT, return_tensors='pt').to(device)
        outputs = model(**inputs)
        detection = processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=THRESHOLD, text_threshold=THRESHOLD,
            target_sizes=[roi.size[::-1]])[0]
        boxes, scores = detection['boxes'].cpu().tolist(), detection['scores'].cpu().tolist()
    detections = [dict(xyxy=[l+left, t+top, r+left, b+top], score=score)
                  for (l, t, r, b), score in zip(boxes, scores)]
    return dict(roi=[left, top, right, bottom], detections=detections,
                image_space='lens_corrected_full_frame')


def apply_wheels(profile, measured, image, *, label):
    calibration = profile.wheel_calibration
    if calibration is None:
        return measured
    result = dict(measured, warnings=list(measured.get('warnings', [])),
                  quality_reasons=list(measured.get('quality_reasons', [])))
    evidence = dict(calibration_id=calibration.calibration_id, method='catalogue_outline_wheel_regression',
        reference_source='catalogue', accuracy_validated=False, status='fallback',
        model=MODEL_ID, revision=MODEL_REVISION, model_sha256=MODEL_FILES['model.safetensors'],
        text=PROMPT, threshold=THRESHOLD, text_threshold=THRESHOLD,
        estimate_source='single_corrected_capture_frame',
        baseline_length_m=measured.get('length_m'), baseline_status=measured['status'])
    result['wheels'] = evidence
    outline = measured.get('outline', {})
    if (label not in {'car', 'manual car', 'selected car'} or outline.get('status') != 'estimated'
            or measured.get('depth') is None or measured['status'] != 'outline_estimate'
            or 'clipped' in measured.get('quality_reasons', [])
            or 'ambiguous_vehicle_association' in measured.get('temporal', {}).get('reasons', [])):
        evidence.update(status='not_applicable', reason='Requires an accepted passenger-car outline estimate')
        return result
    try:
        if image is None or image.shape[1::-1] != profile.image_size:
            raise ValueError('Missing corrected full-resolution capture frame')
        details = infer_wheels(image, measured['bbox'])
        evidence.update(details)
        pair = select_wheel_pair(details['detections'], measured['bbox'])
        features = wheel_features(outline['features'], pair, profile.polygon)
        candidate = predict_wheels(features, calibration.model_dump())
        evidence.update(pair=pair, features=features.tolist(), candidate_length_m=candidate)
        lower, upper = np.asarray(calibration.feature_bounds).T
        margin = .1*(upper-lower)
        if np.any(features < lower-margin) or np.any(features > upper+margin):
            raise ValueError('Wheel features are outside the catalogue training range')
        evidence.update(status='estimated', length_m=candidate)
        result.update(length_m=candidate, status='wheel_estimate', approximate=True,
            position_model='catalogue_trained_outline_and_visible_wheels',
            cm_per_px=100*candidate/measured['bbox'][2], coefficient=None, accuracy_validated=False)
        result['quality_reasons'] = [r for r in result['quality_reasons'] if r != 'catalogue_outline_estimate']
        result['quality_reasons'].append('catalogue_wheel_estimate')
        result['warnings'].append('Оценка контура уточнена по видимым колёсам. Каталожное сравнение не гарантирует точность 10 см.')
    except Exception as exc:
        # Missing wheels must never be represented by invented zero features.
        # Keep the existing outline estimate and save why refinement failed.
        evidence['reason'] = str(exc)
        result['quality_reasons'].append('wheel_refinement_unavailable')
        result['warnings'].append('Уточнение по колёсам недоступно; сохранена оценка контура кузова.')
    return result
