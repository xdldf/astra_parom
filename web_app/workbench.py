"""Visual bbox calibration API, separate from the surveyed 3D workflow."""
import base64
import threading
import uuid
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, ConfigDict, model_validator, model_serializer
from vehicle_metrology.bbox_scale import fit_scale, measure_box
from vehicle_metrology.detection import predict_vehicle_boxes

router = APIRouter(prefix='/api/workbench')
DATA = Path(__file__).parent / 'data' / 'workbench'
DATA.mkdir(parents=True, exist_ok=True)
media = {}
model = None
larger_models = {}
model_lock = threading.Lock()


def require_gpu():
    import torch
    if not torch.cuda.is_available():
        raise HTTPException(503, 'GPU detection is required. CUDA is unavailable. '
                            'Install requirements-gpu.txt and check the NVIDIA driver. '
                            'CPU fallback is disabled.')
    return 0


class Lens(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    k1: float = Field(0, ge=-.8, le=.8)
    k2: float = Field(0, ge=-.5, le=.5)
    focal: float = Field(1, ge=.4, le=2)
    zoom: float = Field(1, ge=.5, le=2)
    cx: float = Field(.5, ge=.2, le=.8)
    cy: float = Field(.5, ge=.2, le=.8)
    tilt_deg: float = Field(0, ge=-30, le=30)


class Reference(BaseModel):
    vehicle_id: str | None = Field(None, max_length=64)
    verified_by: str | None = Field(None, max_length=100)
    verified_at: str | None = Field(None, max_length=64)
    model_config = ConfigDict(allow_inf_nan=False)
    bbox: tuple[float, float, float, float]
    length_m: float = Field(gt=0, le=40)
    frame: int = Field(0, ge=0)

    @model_serializer(mode='wrap')
    def serialize_reference(self, handler):
        data=handler(self)
        for key in ('vehicle_id','verified_by','verified_at'):
            if data[key] is None:
                del data[key]
        return data


class MetricRuler(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    points: list[tuple[float, float]] = Field(min_length=3, max_length=100)
    step_m: float = Field(1, gt=0, le=20)


class SurveyCalibration(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    projection: tuple[tuple[float,float,float],tuple[float,float,float],tuple[float,float,float]]
    support_world: list[tuple[float,float]] = Field(min_length=4,max_length=100)
    lens: Lens
    ruler_check_error_m: float = Field(ge=0)
    ground_check_error_m: float = Field(ge=0)
    source: Literal['visible_barrier_marks_and_posts'] = 'visible_barrier_marks_and_posts'

    @model_validator(mode='after')
    def usable(self):
        H=np.asarray(self.projection)
        if abs(np.linalg.det(H))<1e-9 or np.linalg.cond(H)>1e9:
            raise ValueError('Singular survey projection')
        support=np.asarray(self.support_world,np.float32)
        if not cv2.isContourConvex(support) or abs(cv2.contourArea(support))<1:
            raise ValueError('Survey support must be a convex metric polygon')
        z=np.c_[support,np.ones(len(support))]@H[2]
        if np.min(z)*np.max(z)<=0:
            raise ValueError('Survey support crosses the projection horizon')
        return self


class Profile(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    version: Literal[1] = 1
    image_size: tuple[int, int]
    lens: Lens = Field(default_factory=Lens)
    polygon: list[tuple[float, float]] = Field(default_factory=list, max_length=100)
    references: list[Reference] = Field(default_factory=list, max_length=100)
    metric_rulers: list[MetricRuler] = Field(default_factory=list, max_length=20)
    survey_calibration: SurveyCalibration | None = None
    measurement_line_x: float | None = Field(None, ge=0)
    line_tolerance_px: float = Field(10, ge=1, le=100)
    accuracy_tolerance_m: float = Field(.1, gt=0, le=1)
    measurement_mode: Literal['estimate','strict'] = 'estimate'
    detector_model: Literal['yolo26n','yolo26m','yolo26l','rtdetr-l','rtdetr-x'] = 'yolo26n'
    detector_imgsz: Literal[640,1280] = 640

    @model_validator(mode='after')
    def geometry(self):
        w, h = self.image_size
        if not 0 < w <= 20000 or not 0 < h <= 20000:
            raise ValueError('Invalid image dimensions')
        if self.measurement_line_x is not None and self.measurement_line_x >= w:
            raise ValueError('Measurement line must be inside the image')
        if self.polygon:
            if len(self.polygon) < 4:
                raise ValueError('Draw at least four road points')
            if any(not (0 <= x < w and 0 <= y < h) for x,y in self.polygon):
                raise ValueError('Road points must be inside the image')
            # Reject crossings, repeated vertices and degenerate edges.
            p = np.asarray(self.polygon)
            if len(set(self.polygon)) != len(p) or abs(cv2.contourArea(p.astype(np.float32))) < 10:
                raise ValueError('Road polygon is degenerate')
            def cross(a,b,c):
                u,v = b-a,c-a
                return float(u[0]*v[1]-u[1]*v[0])
            for i in range(len(p)):
                for j in range(i+1,len(p)):
                    if j == i+1 or (i == 0 and j == len(p)-1):
                        continue
                    a,b,c,d = p[i],p[(i+1)%len(p)],p[j],p[(j+1)%len(p)]
                    if cross(a,b,c)*cross(a,b,d) <= 0 and cross(c,d,a)*cross(c,d,b) <= 0:
                        if np.all(np.maximum(np.minimum(a,b),np.minimum(c,d)) <= np.minimum(np.maximum(a,b),np.maximum(c,d))):
                            raise ValueError('Road edges must not cross or touch')
        for r in self.references:
            x,y,bw,bh = r.bbox
            if x <= 1 or y <= 1 or bw <= 2 or bh <= 2 or x+bw >= w-1 or y+bh >= h-1:
                raise ValueError('Reference cars must be fully inside the image')
        for ruler in self.metric_rulers:
            if any(not (0 <= x < w and 0 <= y < h) for x,y in ruler.points):
                raise ValueError('Ruler marks must be inside the original image')
        if (self.references or self.metric_rulers) and not self.polygon:
            raise ValueError('Draw the road before adding reference cars')
        if self.survey_calibration:
            if self.survey_calibration.lens != self.lens:
                raise ValueError('Survey projection belongs to different lens settings; recalibrate it')
            if self.references or self.metric_rulers or not self.polygon:
                raise ValueError('Survey calibration needs its road polygon and cannot mix with empirical references')
        profile_scale(self)
        return self


class FrameRequest(BaseModel):
    profile: Profile
    frame: int = Field(0, ge=0)
    detect: bool = False
    confidence: float = Field(.3, ge=.05, le=.95)
    boxes: list[tuple[float,float,float,float]] = Field(default_factory=list, max_length=200)


def detector_settings(profile):
    return dict(model=profile.detector_model,imgsz=profile.detector_imgsz)


def profile_scale(profile):
    if profile.survey_calibration:
        s=profile.survey_calibration
        error=max(s.ruler_check_error_m,s.ground_check_error_m)
        return dict(status='survey_calibrated' if error<=profile.accuracy_tolerance_m else 'survey_review',
            projection=s.projection,support_world=s.support_world,
            diagnostics=dict(survey=True,target_tolerance_m=profile.accuracy_tolerance_m,
                ruler_check_error_m=s.ruler_check_error_m,ground_check_error_m=s.ground_check_error_m,
                accuracy_validated=False,source=s.source,
                note='Raised barrier marks; ground inferred from visible posts. Vehicle accuracy is unvalidated.'))
    from web_app.calibration_references import eligible
    return cached_scale(tuple(profile.polygon),
                        tuple((r.bbox, r.length_m, bool(r.vehicle_id), r.frame) for r in eligible(profile)),
                        tuple((tuple(r.points), r.step_m) for r in profile.metric_rulers),
                        profile.accuracy_tolerance_m)


@lru_cache(maxsize=64)
def cached_scale(polygon, references, rulers, tolerance_m=.1):
    return fit_scale(polygon, [dict(bbox=bbox, length_m=length, vehicle_id=verified, frame=frame) for bbox,length,verified,frame in references],
                     [dict(points=points, step_m=step) for points,step in rulers], tolerance_m=tolerance_m)


@lru_cache(maxsize=6)
def correction_maps(w, h, values):
    lens = Lens(**dict(values))
    f = max(w,h)*lens.focal
    k = np.array([[f,0,w*lens.cx],[0,f,h*lens.cy],[0,0,1]], dtype=float)
    new = k.copy()
    new[0,0] *= lens.zoom
    new[1,1] *= lens.zoom
    maps = cv2.initUndistortRectifyMap(k, np.array([lens.k1,lens.k2,0,0,0.]), None, new, (w,h), cv2.CV_16SC2)
    return maps


def corrected(image, lens):
    h,w = image.shape[:2]
    if not lens.k1 and not lens.k2 and lens.zoom == 1:
        frame = image
    else:
        maps = correction_maps(w,h,tuple(sorted(lens.model_dump().items())))
        frame = cv2.remap(image, *maps, cv2.INTER_LINEAR)
    if lens.tilt_deg:
        # Positive slider values rotate right (clockwise) in image coordinates.
        rotation = cv2.getRotationMatrix2D(((w-1)/2, (h-1)/2), -lens.tilt_deg, 1)
        frame = cv2.warpAffine(frame, rotation, (w,h), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return frame


def read_frame(key, index):
    if key not in media:
        # Uploaded files survive a code/server restart; recover only UUID-named media.
        if len(key) != 32 or any(c not in '0123456789abcdef' for c in key):
            raise HTTPException(404, 'Media not found')
        paths = [p for p in DATA.glob(key+'.*') if p.suffix.lower() in
                 {'.png','.jpg','.jpeg','.bmp','.mp4','.mov','.avi','.mkv'}]
        if len(paths) != 1:
            raise HTTPException(404, 'Upload the image or video again; this session has ended.')
        path = paths[0]
        if path.suffix.lower() in {'.png','.jpg','.jpeg','.bmp'}:
            media[key] = dict(path=path,kind='image',frames=1,fps=0)
        else:
            cap = cv2.VideoCapture(str(path))
            try:
                media[key] = dict(path=path,kind='video',frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                                  fps=cap.get(cv2.CAP_PROP_FPS))
            finally:
                cap.release()
    item = media[key]
    if index >= item['frames']:
        raise HTTPException(400, 'Frame is outside the recording')
    if item['kind'] == 'image':
        frame = cv2.imread(str(item['path']))
    else:
        cap = cv2.VideoCapture(str(item['path']))
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES,index)
            ok, frame = cap.read()
            if not ok:
                frame = None
        finally:
            cap.release()
    if frame is None:
        raise HTTPException(400, 'Cannot decode this frame')
    return frame


@router.post('/media')
def upload(file: UploadFile = File(...)):
    suffix = Path(file.filename or '').suffix.lower()
    if suffix not in {'.jpg','.jpeg','.png','.bmp','.mp4','.avi','.mov','.mkv'}:
        raise HTTPException(400, 'Choose an image or recorded video')
    key = uuid.uuid4().hex
    path = DATA / (key+suffix)
    with path.open('wb') as stream:
        while chunk := file.file.read(1024*1024):
            stream.write(chunk)
    is_image = suffix in {'.jpg','.jpeg','.png','.bmp'}
    frame = cv2.imread(str(path)) if is_image else None
    count,fps = 1,0
    if not is_image:
        cap = cv2.VideoCapture(str(path))
        try:
            _,frame = cap.read()
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
        finally:
            cap.release()
    if frame is None or count < 1:
        path.unlink(missing_ok=True)
        raise HTTPException(400, 'This file cannot be decoded')
    h,w = frame.shape[:2]
    media[key] = dict(path=path, kind='image' if is_image else 'video', frames=count, fps=fps)
    return dict(id=key, image_size=[w,h], frames=count, fps=fps, name=Path(file.filename).name)


@router.post('/frame/{key}')
def render(key: str, request: FrameRequest):
    raw = read_frame(key,request.frame)
    return render_raw(raw, request)


@router.post('/camera-frame')
def camera_frame():
    from web_app.ip_cameras import side_snapshot
    jpeg=side_snapshot()
    raw=cv2.imdecode(np.frombuffer(jpeg,np.uint8),cv2.IMREAD_COLOR)
    if raw is None:
        raise HTTPException(409,'Не удалось прочитать кадр боковой камеры.')
    h,w=raw.shape[:2]
    key=uuid.uuid4().hex
    path=DATA/(key+'.jpg')
    path.write_bytes(jpeg)
    media[key]=dict(path=path,kind='image',frames=1,fps=0)
    return dict(id=key,image_size=[w,h],frames=1,fps=0,source='ip_camera',
                captured_at=datetime.now(timezone.utc).isoformat(),name='Боковая камера · кадр для калибровки')


class DetectorChoice(BaseModel):
    detector_model: Literal['yolo26n','yolo26m','yolo26l','rtdetr-l','rtdetr-x']
    detector_imgsz: Literal[640,1280] = 640


@router.post('/detector/check')
def prepare_detector(choice: DetectorChoice):
    """Load and exercise the requested weights before interrupting a running source."""
    detect_vehicles(np.zeros((640,640,3),dtype=np.uint8),
                    detector_model=choice.detector_model,imgsz=choice.detector_imgsz)
    return {'model':choice.detector_model,'imgsz':choice.detector_imgsz,'ready':True}


def detect_vehicles(frame, confidence=.35, *, detector_model='yolo26n', imgsz=640):
    global model
    device = require_gpu()
    try:
        from ultralytics import YOLO, RTDETR
        with model_lock:
            if detector_model not in {'yolo26n','yolo26m','yolo26l','rtdetr-l','rtdetr-x'} or imgsz not in {640,1280}:
                raise ValueError('Unsupported detector configuration')
            if detector_model == 'yolo26n':
                if model is None:
                    bundled = Path(__file__).resolve().parents[1]/'models/workbench/yolo26n.pt'
                    model = YOLO(str(bundled if bundled.exists() else DATA/'yolo26n.pt'))
                selected_model = model
            else:
                if detector_model not in larger_models:
                    # Ultralytics resolves only these allowlisted official model names.
                    factory = RTDETR if detector_model.startswith('rtdetr-') else YOLO
                    larger_models[detector_model] = factory(str(DATA/(detector_model+'.pt')))
                selected_model = larger_models[detector_model]
            detections = predict_vehicle_boxes(selected_model, frame, confidence, device=device, imgsz=imgsz)
    except Exception as exc:
        raise HTTPException(503, f'{detector_model} GPU inference failed: {exc}. CPU fallback is disabled.') from exc
    return detections


def render_raw(raw, request, include_image=True, include_frame=False):
    global model
    if tuple(raw.shape[1::-1]) != request.profile.image_size:
        raise HTTPException(400, 'Calibration resolution does not match this media')
    frame = corrected(raw, request.profile.lens)
    detections = []
    if request.detect:
        detections = detect_vehicles(frame, request.confidence,
                                     detector_model=request.profile.detector_model,
                                     imgsz=request.profile.detector_imgsz)
        for detection in detections:
            detection.update(source=request.profile.detector_model, imgsz=request.profile.detector_imgsz)
    else:
        for b in request.boxes:
            x,y,w,h = b
            if not all(np.isfinite(b)) or x < 0 or y < 0 or w <= 2 or h <= 2 or x+w > raw.shape[1] or y+h > raw.shape[0]:
                raise HTTPException(400, 'Invalid bbox coordinates')
            detections.append(dict(bbox=list(b), label='selected car', confidence=None))
    refs = [r.model_dump() for r in request.profile.references]
    scale = profile_scale(request.profile)
    for d in detections:
        d.update(measure_box(d['bbox'],request.profile.polygon,scale,request.profile.image_size,
                            request.profile.measurement_line_x,request.profile.line_tolerance_px,
                            estimate=request.profile.measurement_mode=='estimate'))
    encoded_image = None
    if include_image:
        _,encoded = cv2.imencode('.jpg',frame,[cv2.IMWRITE_JPEG_QUALITY,92])
        encoded_image = base64.b64encode(encoded).decode()
    result = dict(image=encoded_image, detections=detections, scale=scale,
                references=[measure_box(r['bbox'],request.profile.polygon,scale,request.profile.image_size) for r in refs])
    if include_frame:
        result['frame_image'] = frame
    return result


@router.post('/validate-profile')
def validate_profile(profile: Profile):
    return profile


@router.get('/operator-calibration')
def operator_calibration():
    path = DATA.parent/'operator-calibration.json'
    if not path.exists():
        raise HTTPException(404, 'Import a calibration JSON')
    return Profile.model_validate_json(path.read_text(encoding='utf-8'))


@router.post('/approved-references')
def approved_references(profile: Profile):
    from web_app.calibration_references import merge
    return merge(profile)


@router.post('/operator-calibration')
def save_operator_calibration(profile: Profile):
    profile = approved_references(profile)
    path = DATA.parent/'operator-calibration.json'
    temporary = path.with_suffix('.'+uuid.uuid4().hex+'.tmp')
    temporary.write_text(profile.model_dump_json(), encoding='utf-8')
    temporary.replace(path)
    return profile


@router.get('/video/{key}')
def video_stream(key: str):
    # Resolve through the same UUID validation/recovery as frame requests.
    if key not in media:
        read_frame(key, 0)
    item = media[key]
    if item['kind'] != 'video':
        raise HTTPException(400, 'This media is not a video')
    return FileResponse(item['path'], media_type='video/mp4')
