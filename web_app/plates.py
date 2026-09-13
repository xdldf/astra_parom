"""Local CUDA plate detection/OCR. No footage leaves this machine."""
import re
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import numpy as np
from vehicle_metrology.plate_text import normalize_plate


def front_vehicle(image):
    from web_app.workbench import detect_vehicles
    vehicles=detect_vehicles(image)
    if not vehicles:
        return None
    # For the fixed front camera, the nearest road contact is lowest in the image.
    return max(vehicles,key=lambda v:(v['bbox'][1]+v['bbox'][3],v['bbox'][2]*v['bbox'][3]))['bbox']


def foreground_plate(detections, target):
    tx,ty,tw,th=target
    eligible=[d for d in detections if tx <= (d.bounding_box.x1+d.bounding_box.x2)/2 <= tx+tw
              and ty+th*.45 <= (d.bounding_box.y1+d.bounding_box.y2)/2 <= ty+th]
    return sorted(eligible,key=lambda d:d.bounding_box.y2,reverse=True)[:1]


def same_vehicle(a,b):
    x=max(a[0],b[0]); y=max(a[1],b[1])
    overlap=max(0,min(a[0]+a[2],b[0]+b[2])-x)*max(0,min(a[1]+a[3],b[1]+b[3])-y)
    return overlap/max(1,a[2]*a[3]+b[2]*b[3]-overlap)>.25

LOCK = threading.Lock()
MODEL = None
ERROR = None
DETECTOR = 'yolo-v9-t-640-license-plate-end2end'
OCR = 'cct-s-v1-global-model'
POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix='plate-evidence')
JOBS = set()
JOB_LOCK = threading.Lock()


def enqueue(ident):
    with JOB_LOCK:
        if ident in JOBS:
            return
        JOBS.add(ident)
    POOL.submit(process_record, ident)


def resume_pending():
    from web_app import station as st
    with st.connect() as db:
        pending=[json.loads(row['data']) for row in db.execute('SELECT data FROM vehicles')]
    for record in pending:
        if record.get('plate_ocr',{}).get('state')=='queued' and record['status'] not in {'Оплачен','Подтвержден'}:
            enqueue(record['id'])


def process_record(ident):
    from web_app import station as st, workbench as wb
    import cv2
    try:
        with st.connect() as db:
            record = st.find(db, ident)
        front = (record.get('source') or {}).get('front_camera')
        samples = []
        stored=(record.get('source') or {}).get('front_samples')
        if stored:
            anchor_image=cv2.imread(str(st.DATA/record['front_photo']))
            anchor=front_vehicle(anchor_image)
            for sample in stored:
                image=cv2.imread(str(st.DATA/sample['photo']))
                if image is not None:
                    samples.append((sample['frame'],image,recognize(image,reference_box=anchor) if anchor else []))
        elif front and front.get('media_id'):
            anchor=front_vehicle(wb.read_frame(front['media_id'], front['frame']))
            item = wb.media[front['media_id']]
            # A short window provides repeat readings and clearer evidence during motion.
            indices = sorted({max(0, min(item['frames']-1, front['frame']+round(t*item['fps'])))
                              for t in (-.6, -.3, 0, .3, .6)})
            for index in indices:
                image = wb.read_frame(front['media_id'], index)
                samples.append((index, image, recognize(image,reference_box=anchor) if anchor else []))
        elif record.get('front_photo'):
            image = cv2.imread(str(st.DATA/record['front_photo']))
            if image is None:
                raise ValueError('Cannot read front photo')
            samples.append((None, image, recognize(image)))
        else:
            raise ValueError('No front camera photo for recognition')
        groups = {}
        for index, image, candidates in samples:
            for candidate in candidates:
                group = groups.setdefault(candidate['text'], {'frames': set(), 'best': candidate, 'image': image, 'frame': index})
                group['frames'].add(index)
                if candidate['confidence'] > group['best']['confidence']:
                    group.update(best=candidate, image=image, frame=index)
        evidence = []
        for group in sorted(groups.values(), key=lambda g:(len(g['frames']),g['best']['confidence']), reverse=True)[:12]:
            candidate = dict(group['best'], votes=len(group['frames']), frame=group['frame'])
            image = group['image']
            x1,y1,x2,y2 = candidate['bbox']
            margin = max(6, round((x2-x1)*.12))
            crop = image[max(0,y1-margin):min(image.shape[0],y2+margin),max(0,x1-margin):min(image.shape[1],x2+margin)]
            if not crop.size:
                continue
            candidate['photo'] = uuid.uuid4().hex+'-plate.jpg'
            if not cv2.imwrite(str(st.DATA/candidate['photo']), crop):
                raise ValueError('Cannot save plate evidence')
            evidence.append(candidate)
        result = dict(state='review' if evidence else 'not_found', candidates=evidence,
                      detector=DETECTOR, ocr=OCR, device='CUDA:0',
                      association='operator_review', completed_at=st.now())
        save_result(ident, result)
    except Exception as exc:
        save_result(ident, {'state':'error', 'error':str(exc), 'candidates':[]})
    finally:
        with JOB_LOCK:
            JOBS.discard(ident)


def save_result(ident, result):
    from web_app import station as st
    with st.connect() as db:
        db.execute('BEGIN IMMEDIATE')
        old = st.find(db, ident)
        # Evidence must never overwrite an operator's number or a closed transaction.
        if old['status'] in {'Оплачен', 'Подтвержден'}:
            return
        record = dict(old, plate_ocr=result, version=old['version']+1, updated_at=st.now())
        db.execute('UPDATE vehicles SET version=?,data=? WHERE id=?',
                   (record['version'], json.dumps(record,ensure_ascii=False), ident))
        st.event(db, record, 'OCR', 'plate_recognition', 'Результат OCR; сопоставление автомобиля проверяет оператор', old)


def engine():
    global MODEL
    if MODEL is None:
        import torch  # Load the same CUDA/cuDNN DLLs as vehicle inference first.
        import onnxruntime as ort
        from fast_alpr import ALPR
        from fast_plate_ocr.inference import hub as ocr_hub
        from open_image_models.detection.core import hub as detection_hub
        if not torch.cuda.is_available():
            raise RuntimeError('Plate recognition requires NVIDIA CUDA')
        ort.preload_dlls()
        cache = Path(__file__).parent/'data'/'plate-models'
        ocr_hub.MODEL_CACHE_DIR = cache/'ocr'
        detection_hub.MODEL_CACHE_DIR = cache/'detector'
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.log_severity_level = 3
        providers = [('CUDAExecutionProvider', {'device_id': 0})]
        model = ALPR(detector_model=DETECTOR, ocr_model=OCR,
                     detector_providers=providers, ocr_providers=providers,
                     ocr_device='cuda', detector_sess_options=options, ocr_sess_options=options)
        # Reject a runtime which silently fell back to CPU after a CUDA load failure.
        def sessions(obj, depth=0):
            if isinstance(obj, ort.InferenceSession):
                return [obj]
            if depth > 4 or not hasattr(obj, '__dict__'):
                return []
            return [s for value in vars(obj).values() for s in sessions(value, depth+1)]
        found = sessions(model)
        if len(found) < 2 or any('CUDAExecutionProvider' not in s.get_providers() for s in found):
            raise RuntimeError('Plate detector/OCR CUDA provider failed to initialize')
        MODEL = model
    return MODEL


def recognize(image, reference_box=None):
    """Return coordinates in the original frame, with raw model evidence."""
    import cv2
    target=front_vehicle(image)
    if target is None or (reference_box is not None and not same_vehicle(target,reference_box)):
        return []
    with LOCK:
        model = engine()
        from types import SimpleNamespace
        # Detect plates first; read text only for the foreground vehicle.
        detections=model.detector.predict(image)
        # One foreground number only. Never fall back to a following vehicle.
        detections=foreground_plate(detections,target)
        results=[SimpleNamespace(detection=d,ocr=model.ocr.predict(image[max(0,d.bounding_box.y1):d.bounding_box.y2,max(0,d.bounding_box.x1):d.bounding_box.x2])) for d in detections]
        alternatives = []
        for result in results:
            b = result.detection.bounding_box
            reads = [result.ocr] if result.ocr else []
            crop = image[max(0,b.y1):b.y2,max(0,b.x1):b.x2]
            if crop.size and b.width/max(1,b.height) < 2.5:
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                _, mask = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                contours,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    contour = max(contours,key=cv2.contourArea)
                    points = cv2.approxPolyDP(contour,.03*cv2.arcLength(contour,True),True).reshape(-1,2)
                    if len(points)==4 and cv2.contourArea(contour)>.4*b.width*b.height:
                        sums=points.sum(1); diffs=np.diff(points,axis=1).ravel()
                        quad=np.float32([points[sums.argmin()],points[diffs.argmin()],points[sums.argmax()],points[diffs.argmax()]])
                        if len(np.unique(quad,axis=0))==4:
                            crop=cv2.warpPerspective(crop,cv2.getPerspectiveTransform(quad,np.float32([[0,0],[199,0],[199,149],[0,149]])),(200,150))
                # Reflow two-line plates before OCR, retaining the original crop as evidence.
                split=crop.shape[0]//2
                if split:
                    strip=cv2.hconcat([cv2.resize(crop[:split],(200,75)),cv2.resize(crop[split:],(200,75))])
                    read=model.ocr.predict(strip)
                    if read: reads.append(read)
            alternatives.append(max(reads,key=ocr_confidence) if reads else None)
    candidates = []
    for result, reading in zip(results, alternatives):
        if reading is None:
            continue
        text = normalize_plate(re.sub(r'[^A-ZА-Я0-9]', '', reading.text.upper()))
        if re.search(r'[A-Z]',text):
            continue
        if len(text) < 4:
            continue
        b = result.detection.bounding_box
        if b.x1 <= 1 or b.y1 <= 1 or b.x2 >= image.shape[1]-1 or b.y2 >= image.shape[0]-1:
            continue
        candidates.append(dict(text=text, raw_text=reading.text,
            confidence=ocr_confidence(reading),
            detection_confidence=float(result.detection.confidence),
            bbox=[int(b.x1), int(b.y1), int(b.x2), int(b.y2)]))
    return candidates


def ocr_confidence(reading):
    probabilities=reading.confidence
    # Padding tokens are highly confident and must not inflate the number score.
    if isinstance(probabilities,list):
        probabilities=probabilities[:len(reading.text)]
    return float(np.mean(probabilities)) if np.size(probabilities) else 0.0
