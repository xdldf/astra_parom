"""Visible-wheel image features; no claim to reconstruct hidden tyre contacts."""
import hashlib

import numpy as np

from .bbox_scale import road_cross_section
from .detection import box_iou
from .outline import FEATURE_NAMES as OUTLINE_FEATURE_NAMES

MODEL_ID = 'IDEA-Research/grounding-dino-tiny'
MODEL_REVISION = 'a2bb814dd30d776dcf7e30523b00659f4f141c71'
MODEL_DIRECTORY = 'grounding-dino-tiny'
MODEL_FILES = {
    'added_tokens.json': '909e96cb32d92ce728a01bc99850cbba26196d74115c17ebeb019275412588f2',
    'config.json': 'eec82c5ab66e16df12a9a212e68ac011779927c2536cf9078658e35d85f0c67a',
    'model.safetensors': '1a2412ef99bd74bcd3c2a246fa1e48581f8889a1300c9051974741314fc042f3',
    'preprocessor_config.json': '8454179ba95e2ad22947835aad7b45862a601fc0055ab88bf1ee70892d3aea60',
    'special_tokens_map.json': 'b6d346be366a7d1d48332dbc9fdf3bf8960b5d879522b7799ddba59e76237ee3',
    'tokenizer.json': 'd241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66',
    'tokenizer_config.json': 'd40ab645b68211910b9170d22433d43186a6ec8ee6fd10ba170524b25bf4fb56',
    'vocab.txt': '07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3',
}
PROMPT = 'car wheel. car tire.'
THRESHOLD = .25
FEATURE_NAMES = (*OUTLINE_FEATURE_NAMES, 'wheel_bottom_proxy_depth', 'wheel_visible_aspect')


def file_digest(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def verify_model(directory):
    for name, expected in MODEL_FILES.items():
        path = directory/name
        if not path.is_file() or file_digest(path) != expected:
            raise ValueError(f'Missing or unverified wheel model file: {name}; run scripts/prepare_wheel_model.py')


def wheel_roi(vehicle, image_size):
    x, y, w, h = vehicle
    if not np.isfinite(vehicle).all() or min(w, h) <= 0:
        raise ValueError('Invalid vehicle box for wheel detection')
    width, height = image_size
    roi = (max(0, int(x-.08*w)), max(0, int(y-.1*h)),
           min(width, int(x+1.08*w)), min(height, int(y+1.25*h)))
    if roi[2] <= roi[0] or roi[3] <= roi[1]:
        raise ValueError('Empty wheel detection region')
    return roi


def select_wheel_pair(detections, vehicle):
    """Require two unambiguous side wheels, not two best out of many candidates."""
    x, y, w, h = vehicle
    if not np.isfinite(vehicle).all() or min(w, h) <= 0:
        raise ValueError('Invalid vehicle box for wheel association')
    candidates = []
    for detection in sorted(detections, key=lambda item: -item['score']):
        l, t, r, b = detection['xyxy']
        score = detection['score']
        if not np.isfinite([l, t, r, b, score]).all() or not THRESHOLD <= score <= 1:
            continue
        cx, cy = (l+r)/2, (t+b)/2
        if not (x < cx < x+w and y+.45*h < cy < y+1.25*h
                and .04*w < r-l < .3*w and 5 < b-t < .6*w):
            continue
        box = [l, t, r-l, b-t]
        if any(box_iou(box, item['bbox']) > .4 for item in candidates):
            continue
        candidates.append(dict(bbox=box, center=[cx, cy], score=score))
    if len(candidates) != 2:
        raise ValueError('An unambiguous pair of visible wheels was not found')
    a, b = sorted(candidates, key=lambda item: item['center'][0])
    if (not .3*w < b['center'][0]-a['center'][0] < .85*w
            or max(a['bbox'][2], b['bbox'][2])/min(a['bbox'][2], b['bbox'][2]) > 1.6):
        raise ValueError('Visible wheel pair has unsupported separation or size ratio')
    return [a, b]


def wheel_features(outline, pair, polygon):
    if pair is None or len(pair) != 2:
        raise ValueError('Wheel features require an observed pair; missing wheels cannot be zero-filled')
    base = np.asarray(outline, float)
    boxes = np.asarray([p['bbox'] for p in pair], float)
    if (base.shape != (7,) or boxes.shape != (2, 4) or not np.isfinite(base).all()
            or not np.isfinite(boxes).all() or np.any(boxes[:, 2:] <= 0)):
        raise ValueError('Invalid outline or wheel features')
    center_x = np.mean(boxes[:, 0]+boxes[:, 2]/2)
    # Bottoms may stop at the barrier. Top + horizontal diameter is only an
    # image proxy used by the regression, never a measured ground contact.
    proxy_y = np.mean(boxes[:, 1]+boxes[:, 2])
    section = road_cross_section(polygon, center_x, proxy_y)
    if section is None:
        raise ValueError('Visible wheels have no road cross-section')
    return np.r_[base, section['depth'], np.mean(boxes[:, 3]/boxes[:, 2])]


def predict_wheels(features, calibration):
    x, mean, scale, coefficient = [np.asarray(value, float) for value in
        (features, calibration['mean'], calibration['scale'], calibration['coefficient'])]
    if any(a.shape != (9,) or not np.isfinite(a).all() for a in (x, mean, scale, coefficient)) or np.any(scale <= 0):
        raise ValueError('Invalid wheel regression')
    prediction = float(((x-mean)/scale) @ coefficient + calibration['intercept'])
    if not np.isfinite(prediction) or not 0 < prediction <= 100:
        raise ValueError('Wheel length outside the supported numeric range')
    return prediction
