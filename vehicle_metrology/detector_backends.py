"""Optional detector adapters for reproducible local comparisons.

RF-DETR uses sparse, one-based COCO IDs and RGB arrays; Ultralytics uses
contiguous, zero-based IDs and BGR arrays. Keep those contracts explicit.
"""
import math
from pathlib import Path

import cv2

from .detection import predict_vehicle_boxes, suppress_duplicate_boxes


RF_VEHICLE_CLASSES = {3: 'car', 4: 'motorcycle', 6: 'bus', 8: 'truck'}
RF_NATIVE_RESOLUTIONS = {'rfdetr-medium': 576, 'rfdetr-large': 704}


def predict_rfdetr_boxes(model, frame, confidence=.3):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = model.predict(rgb, threshold=confidence, include_source_image=False)
    detections = []
    for xyxy, score, class_id in zip(result.xyxy, result.confidence, result.class_id):
        label = RF_VEHICLE_CLASSES.get(int(class_id))
        coords = [float(v) for v in xyxy]
        score = float(score)
        if label is None or not all(math.isfinite(v) for v in [*coords, score]) or score < confidence:
            continue
        x1, y1, x2, y2 = coords
        # Preserve coordinates, including clipped edges, for downstream review.
        if x2 <= x1 or y2 <= y1:
            continue
        detections.append(dict(bbox=[x1,y1,x2-x1,y2-y1], confidence=score, label=label))
    return suppress_duplicate_boxes(detections)


class DetectorBackend:
    def __init__(self, family, weights, *, device='cpu', resolution=None):
        weights = Path(weights)
        if not weights.is_file():
            raise ValueError('Supply an existing local checkpoint; download official weights separately')
        self.family, self.device = family, device
        if family in RF_NATIVE_RESOLUTIONS:
            from rfdetr import RFDETRMedium, RFDETRLarge
            self.resolution = resolution or RF_NATIVE_RESOLUTIONS[family]
            if self.resolution != RF_NATIVE_RESOLUTIONS[family]:
                raise ValueError('This adapter uses the RF-DETR variant\'s native resolution')
            factory = RFDETRMedium if family == 'rfdetr-medium' else RFDETRLarge
            self.model = factory(pretrain_weights=str(weights.resolve()), device=device)
        elif family in {'yolo','rtdetr'}:
            from ultralytics import YOLO, RTDETR
            self.resolution = resolution or 640
            if self.resolution not in {640,1280}:
                raise ValueError('Use 640 or 1280 for the Ultralytics comparison')
            self.model = (RTDETR if family == 'rtdetr' else YOLO)(str(weights))
        else:
            raise ValueError('Unknown detector family')

    def predict(self, frame, confidence=.3):
        if self.family in RF_NATIVE_RESOLUTIONS:
            return predict_rfdetr_boxes(self.model, frame, confidence)
        return predict_vehicle_boxes(self.model, frame, confidence,
                                     device=self.device, imgsz=self.resolution)
