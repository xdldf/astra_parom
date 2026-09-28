"""Shared detector output handling for the station and offline audits."""


def box_iou(a, b):
    x, y, w, h = a
    u, v, bw, bh = b
    intersection = max(0, min(x+w, u+bw)-max(x, u)) * max(0, min(y+h, v+bh)-max(y, v))
    union = w*h + bw*bh - intersection
    return intersection/union if union > 0 else 0.


def suppress_duplicate_boxes(detections, threshold=.8):
    """Remove near-identical boxes across classes; keep partially overlapping cars.

    YOLO end-to-end heads may bypass NMS even with agnostic_nms=True.
    This conservative final check also covers those exported/model versions.
    """
    kept = []
    for detection in sorted(detections, key=lambda d: d['confidence'], reverse=True):
        if not any(box_iou(detection['bbox'], other['bbox']) > threshold for other in kept):
            kept.append(detection)
    return kept


def predict_vehicle_boxes(model, frame, confidence=.35, *, device=0, imgsz=640):
    result = model.predict(frame, conf=confidence, classes=[2, 3, 5, 7],
                           agnostic_nms=True, imgsz=imgsz, device=device, verbose=False)[0]
    detections = []
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        detections.append(dict(bbox=[x1, y1, x2-x1, y2-y1],
                               confidence=float(box.conf[0]), label=result.names[int(box.cls[0])]))
    return suppress_duplicate_boxes(detections)
