"""Corrected-image silhouette features for catalogue-trained length estimates."""
import hashlib
import json

import numpy as np

from .bbox_scale import road_cross_section

WEIGHTS_NAME = 'yolo26m-seg.pt'
WEIGHTS_SHA256 = '16b636f04e8fb6a325b3370f22dc5e5535ff473e384f4d041fd28d788f6ee9f5'
WEIGHTS_URL = 'https://huggingface.co/Ultralytics/YOLO26/resolve/main/yolo26m-seg.pt'
FEATURE_NAMES = ('outline_width', 'road_depth', 'outline_height', 'horizontal_position',
                 'upper_body_width', 'middle_body_width', 'lower_body_width')


def geometry_signature(profile):
    geometry = {key: profile[key] for key in ('image_size', 'lens', 'polygon')}
    return hashlib.sha256(json.dumps(geometry, sort_keys=True).encode()).hexdigest()


def outline_features(mask, polygon):
    """Visible silhouette only; hidden tyres are not reconstructed or imputed."""
    mask = np.asarray(mask)
    if mask.ndim != 2 or mask.size == 0 or not np.isfinite(mask).all():
        raise ValueError('A finite full-frame two-dimensional mask is required')
    ys, xs = np.nonzero(mask)
    if len(xs) < 100:
        raise ValueError('Insufficient vehicle silhouette')
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    width, height = x1-x0, y1-y0
    if min(width, height) < 10 or min(x0, y0) <= 1 or x1 >= mask.shape[1]-2 or y1 >= mask.shape[0]-2:
        raise ValueError('Clipped or degenerate silhouette')
    section = road_cross_section(polygon, (x0+x1)/2, y1)
    if section is None:
        raise ValueError('Silhouette has no road cross-section')
    span = section['near'][1]-section['far'][1]
    features = [width/span, section['depth'], height/span, (x0+x1)/2/mask.shape[1]]
    for low, high in ((.4, .65), (.65, .85), (.85, .99)):
        selected = xs[(ys >= y0+height*low) & (ys <= y0+height*high)]
        if len(selected) < 10:
            raise ValueError('Incomplete body silhouette')
        features.append(float(np.diff(np.percentile(selected, [1, 99]))[0]/span))
    return np.asarray(features, float)


def predict_outline(features, calibration):
    x = np.asarray(features, float)
    mean = np.asarray(calibration['mean'], float)
    scale = np.asarray(calibration['scale'], float)
    coefficient = np.asarray(calibration['coefficient'], float)
    if any(a.shape != (7,) or not np.isfinite(a).all() for a in (x, mean, scale, coefficient)) or np.any(scale <= 0):
        raise ValueError('Invalid outline regression')
    prediction = float(((x-mean)/scale) @ coefficient + calibration['intercept'])
    if not np.isfinite(prediction) or not 0 < prediction <= 100:
        raise ValueError('Outline length outside the supported numeric range')
    return prediction
