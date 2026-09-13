"""Paced file camera: sequential decoder, latest-frame inference, bounded buffers."""
import threading
import time
import uuid
import json
import os
from pathlib import Path
from functools import lru_cache
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import cv2
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from web_app import workbench as wb

router = APIRouter(prefix='/api/stream')
sessions = {}
registry_lock = threading.Lock()


@lru_cache(maxsize=256)
def plate_label(text):
    font=ImageFont.truetype(str(Path(os.environ.get('WINDIR','C:/Windows'))/'Fonts'/'arial.ttf'),20)
    label=Image.new('RGB',(max(80,len(text)*16),28),(20,32,40))
    ImageDraw.Draw(label).text((3,1),text,font=font,fill=(255,220,40))
    return np.asarray(label)[:,:,::-1].copy()


class Start(BaseModel):
    media_id: str
    profile: wb.Profile
    frame: int = Field(0, ge=0)
    front_media_id: str | None = None
    front_offset_seconds: float = Field(3, ge=-3600, le=3600, allow_inf_nan=False)


def paired_frame(side_frame, side_fps, front_fps, offset):
    return round((side_frame/side_fps + offset)*front_fps)


class Camera:
    def __init__(self, request):
        wb.read_frame(request.media_id, request.frame)  # UUID and frame validation
        item = wb.media[request.media_id]
        if item['kind'] != 'video':
            raise HTTPException(400, 'Choose a video')
        wb.require_gpu()
        self.request = request
        self.path, self.fps = str(item['path']), item['fps']
        self.front = None
        if request.front_media_id:
            wb.read_frame(request.front_media_id, 0)
            self.front = wb.media[request.front_media_id]
            if self.front['kind'] != 'video':
                raise HTTPException(400, 'Front camera must be a video')
            first = paired_frame(request.frame,self.fps,self.front['fps'],request.front_offset_seconds)
            if not 0 <= first < self.front['frames']:
                raise HTTPException(422, 'Selected time is outside the overlapping recordings')
        self.stop = threading.Event()
        self.condition = threading.Condition()
        self.latest = None
        self.front_latest = None
        self.plate_result = None
        self.plate_error = None
        self.jpeg = None
        self.front_jpeg = None
        self.front_frame = None
        self.sequence = 0
        self.result = None
        self.error = None
        self.ended = False
        self.touched = time.monotonic()
        self.display_frame = request.frame
        self.started = time.monotonic()
        self.display_count = 0
        self.inference_count = 0
        self.id = uuid.uuid4().hex

    def start(self):
        for fn in (self.produce, self.infer, self.recognize_plates):
            threading.Thread(target=fn, daemon=True).start()

    def produce(self):
        cap = cv2.VideoCapture(self.path)
        front_cap = cv2.VideoCapture(str(self.front['path'])) if self.front else None
        front_next = None
        front_raw = None
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, self.request.frame)
            index = self.request.frame
            clock = time.monotonic()
            while not self.stop.is_set():
                if time.monotonic()-self.touched > 30:
                    break
                target = self.request.frame + int((time.monotonic()-clock)*self.fps)
                # Decode sequentially even when skipping display frames. Never seek
                # from a keyframe on each inference request.
                while index < target:
                    if not cap.grab():
                        return
                    index += 1
                ok, raw = cap.read()
                if not ok:
                    break
                if tuple(raw.shape[1::-1]) != self.request.profile.image_size:
                    raise ValueError('Calibration resolution does not match video')
                front_encoded = None
                if front_cap:
                    front_index=paired_frame(index,self.fps,self.front['fps'],self.request.front_offset_seconds)
                    if front_index>=self.front['frames']:
                        break
                    if front_next is None:
                        front_cap.set(cv2.CAP_PROP_POS_FRAMES,front_index)
                        front_next=front_index
                    while front_next<front_index:
                        if not front_cap.grab():
                            raise ValueError('Front camera decode ended')
                        front_next+=1
                    if front_next==front_index:
                        ok,front_raw=front_cap.read()
                        if not ok:
                            raise ValueError('Cannot decode front camera')
                        front_next+=1
                    fh,fw=front_raw.shape[:2]
                    front_preview=cv2.resize(front_raw,(960,round(fh*960/fw)))
                    self.front_latest=(front_index,front_raw)
                    plates=self.plate_result
                    if plates and abs(front_index-plates['frame'])/self.front['fps']<.6:
                        for plate in plates['candidates']:
                            x1,y1,x2,y2=[round(v*960/fw) for v in plate['bbox']]
                            cv2.rectangle(front_preview,(x1,y1),(x2,y2),(0,210,255),2)
                            label=plate_label(plate['text'])
                            ly=max(0,y1-label.shape[0]);lx=max(0,min(x1,front_preview.shape[1]-label.shape[1]))
                            front_preview[ly:ly+label.shape[0],lx:lx+label.shape[1]]=label
                    cv2.putText(front_preview,f'FRONT {front_index/self.front["fps"]:.2f}s | SIDE {index/self.fps:.2f}s',
                                (12,28),cv2.FONT_HERSHEY_SIMPLEX,.6,(60,255,180),2)
                    _,front_encoded=cv2.imencode('.jpg',front_preview,[cv2.IMWRITE_JPEG_QUALITY,80])
                with self.condition:
                    self.latest = (index, raw)
                    self.condition.notify_all()
                h,w = raw.shape[:2]
                width = min(1280,w)
                preview = cv2.resize(raw, (width,round(h*width/w)))
                preview = wb.corrected(preview,self.request.profile.lens)
                scale = width/w
                result = self.result
                if result and abs(index-result['frame'])/self.fps < .5:
                    for d in result['detections']:
                        x,y,bw,bh = [round(v*scale) for v in d['bbox']]
                        cv2.rectangle(preview,(x,y),(x+bw,y+bh),(80,240,130),2)
                        label = f"{d['length_m']:.2f} m" if d['length_m'] is not None else ''
                        cv2.putText(preview,label,(x,max(20,y-6)),cv2.FONT_HERSHEY_SIMPLEX,.6,(80,240,130),2)
                line = self.request.profile.measurement_line_x
                if line is not None:
                    x=round(line*scale)
                    cv2.line(preview,(x,0),(x,preview.shape[0]),(220,130,220),1)
                ok, encoded = cv2.imencode('.jpg',preview,[cv2.IMWRITE_JPEG_QUALITY,80])
                if ok:
                    with self.condition:
                        self.jpeg=encoded.tobytes()
                        if front_encoded is not None:
                            self.front_jpeg=front_encoded.tobytes()
                            self.front_frame=front_index
                        self.sequence+=1
                        self.display_frame=index
                        self.display_count+=1
                        self.condition.notify_all()
                index+=1
                self.stop.wait(max(0,(index-self.request.frame)/self.fps-(time.monotonic()-clock)))
        except Exception as exc:
            self.error=str(exc)
        finally:
            cap.release()
            if front_cap:
                front_cap.release()
            self.ended=True
            self.stop.set()
            with self.condition:
                self.condition.notify_all()
            with registry_lock:
                sessions.pop(self.id,None)

    def infer(self):
        last=-1
        while not self.stop.is_set():
            with self.condition:
                self.condition.wait_for(lambda:self.stop.is_set() or (self.latest and self.latest[0]!=last),timeout=1)
                if self.stop.is_set():
                    break
                if not self.latest:
                    continue
                index,raw=self.latest
            try:
                result=wb.render_raw(raw,wb.FrameRequest(profile=self.request.profile,frame=index,detect=True),include_image=False)
                self.result={**result,'frame':index}
                self.inference_count+=1
                last=index
                self.error=None
            except Exception as exc:
                self.error=str(exc)
                self.stop.wait(1)

    def recognize_plates(self):
        if not self.front:
            return
        from web_app.plates import recognize
        last=-1
        while not self.stop.is_set():
            sample=self.front_latest
            if sample and sample[0]!=last:
                last,raw=sample
                try:
                    self.plate_result={'frame':last,'candidates':recognize(raw)}
                    self.plate_error=None
                except Exception as exc:
                    self.plate_error=str(exc)
                    self.stop.wait(3)
            self.stop.wait(.2)

    def frames(self, front=False):
        last=-1
        while not self.stop.is_set():
            self.touched=time.monotonic()
            with self.condition:
                self.condition.wait_for(lambda:self.stop.is_set() or self.sequence!=last,timeout=1)
                jpeg,last=(self.front_jpeg if front else self.jpeg),self.sequence
            if jpeg:
                yield b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: '+str(len(jpeg)).encode()+b'\r\n\r\n'+jpeg+b'\r\n'


def get(key):
    camera=sessions.get(key)
    if camera is None:
        raise HTTPException(404,'Stream ended; press Start to replay')
    camera.touched=time.monotonic()
    return camera


@router.post('/start')
def start(request: Start):
    camera=Camera(request)
    with registry_lock:
        if len(sessions)>=4:
            raise HTTPException(409,'Stop another video stream first')
        sessions[camera.id]=camera
    camera.start()
    return {'id':camera.id}


@router.get('/configuration')
def configuration():
    path=wb.DATA.parent/'camera-pair.json'
    if not path.exists():
        raise HTTPException(404,'Select camera recordings')
    data=json.loads(path.read_text(encoding='utf-8'))
    data['profile']=wb.operator_calibration()
    return data


@router.get('/{key}/video')
def video(key: str):
    return StreamingResponse(get(key).frames(),media_type='multipart/x-mixed-replace; boundary=frame',headers={'Cache-Control':'no-store'})


@router.get('/{key}/front')
def front_video(key: str):
    camera=get(key)
    if not camera.front:
        raise HTTPException(404,'No front camera')
    return StreamingResponse(camera.frames(front=True),media_type='multipart/x-mixed-replace; boundary=frame',headers={'Cache-Control':'no-store'})


@router.get('/{key}/state')
def state(key: str):
    c=get(key)
    elapsed=max(.001,time.monotonic()-c.started)
    return {'result':c.result,'frame':c.display_frame,'error':c.error,'ended':c.ended,
            'plates':c.plate_result,'plate_error':c.plate_error,
            'front_frame':c.front_frame,
            'front_seconds':None if c.front_frame is None else c.front_frame/c.front['fps'],
            'sync_error_ms':None if c.front_frame is None else 1000*(c.front_frame/c.front['fps']-c.display_frame/c.fps-c.request.front_offset_seconds),
            'display_fps':c.display_count/elapsed,'inference_fps':c.inference_count/elapsed}


@router.delete('/{key}')
def close(key: str):
    c=sessions.get(key)
    if c:
        c.stop.set()
    return {'stopped':True}
