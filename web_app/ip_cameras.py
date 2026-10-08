"""Two persistent network receivers; bounded receive-time pairing and server capture."""
import json
import asyncio
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from urllib.parse import urlsplit

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel, Field, field_validator
from web_app import workbench as wb
from web_app.front_history import FrontHistory

router=APIRouter(prefix='/api/ip')
CONFIG=wb.DATA.parent/'ip-cameras.json'
guard=threading.RLock()
active=None


class Settings(BaseModel):
    side_url: str=Field('',max_length=2048)
    front_url: str=Field('',max_length=2048)
    offset_seconds: float=Field(0,ge=-2,le=2,allow_inf_nan=False)
    tolerance_ms: int=Field(200,ge=40,le=1000)
    auto_measure: bool=True
    autostart: bool=False
    profile: wb.Profile | None=None

    @field_validator('side_url','front_url')
    @classmethod
    def network_url(cls,value):
        if value:
            parsed=urlsplit(value)
            if parsed.scheme not in {'rtsp','rtsps','http','https'} or not parsed.hostname or any(c in value for c in '\r\n'):
                raise ValueError('Укажите адрес видеопотока rtsp://, http:// или https://')
        return value


def settings():
    return Settings.model_validate_json(CONFIG.read_text(encoding='utf-8')) if CONFIG.exists() else Settings()


def public_config(cfg):
    # Credentials and full URLs never return to the browser or station records.
    data=cfg.model_dump(exclude={'side_url','front_url','profile'})
    data['calibrated']=cfg.profile is not None
    data['detector']=configured_detector(cfg)
    for name in ('side','front'):
        url=getattr(cfg,name+'_url')
        data[name+'_configured']=bool(url)
        data[name+'_host']=urlsplit(url).hostname if url else ''
    return data


def configured_detector(cfg):
    try:
        profile=cfg.profile or wb.operator_calibration()
        return wb.detector_settings(profile)
    except HTTPException:
        return None


@dataclass(frozen=True)
class Packet:
    seq: int
    stamp: float
    jpeg: bytes

    def image(self):
        return cv2.imdecode(np.frombuffer(self.jpeg,np.uint8),cv2.IMREAD_COLOR)


class PassageFrames:
    """Pin a bounded original-frame window while inference uses the rolling buffer."""
    def __init__(self, anchor, epoch):
        from web_app.temporal_capture import EXTENDED_WINDOW_SECONDS
        self.anchor=anchor
        self.epoch=epoch
        self.end=anchor.stamp+EXTENDED_WINDOW_SECONDS
        self.lock=threading.Lock()
        self.packets={anchor.seq:anchor}
        # Only the inference/capture worker accesses the decoded observations.
        self.observations=[]
        self.passage=None

    def extend(self, packets):
        from web_app.temporal_capture import WINDOW_SECONDS, EXTENDED_WINDOW_SECONDS, MAX_FRAMES, MAX_EXTENDED_FRAMES
        with self.lock:
            bins={}
            for packet in [*self.packets.values(),*packets]:
                offset=packet.stamp-self.anchor.stamp
                if packet.seq==self.anchor.seq or abs(offset)>EXTENDED_WINDOW_SECONDS:
                    continue
                outer=abs(offset)>WINDOW_SECONDS
                count=(MAX_EXTENDED_FRAMES-MAX_FRAMES)//2 if outer else (MAX_FRAMES-1)//2
                start=WINDOW_SECONDS if outer else 0.
                span=EXTENDED_WINDOW_SECONDS-WINDOW_SECONDS if outer else WINDOW_SECONDS
                slot=min(count-1,int((abs(offset)-start)/span*count))
                target=start+(slot+.5)*span/count
                key=(outer,offset<0,slot)
                distance=abs(abs(offset)-target)
                if key not in bins or distance<bins[key][0]:
                    bins[key]=(distance,packet)
            # Reserve sparse neighbours separately so dense near-anchor frames
            # cannot evict all evidence of a later slow crossing.
            ordered=sorted((value[1] for value in bins.values()),
                           key=lambda p:(abs(p.stamp-self.anchor.stamp)>WINDOW_SECONDS,
                                         abs(p.stamp-self.anchor.stamp)))
            selected=[self.anchor];size=len(self.anchor.jpeg)
            for packet in ordered:
                if size+len(packet.jpeg)>48*1024*1024:continue
                selected.append(packet);size+=len(packet.jpeg)
            self.packets={p.seq:p for p in selected}

    def snapshot(self, *, extended=False):
        from web_app.temporal_capture import WINDOW_SECONDS
        with self.lock:
            return sorted((p for p in self.packets.values()
                           if extended or abs(p.stamp-self.anchor.stamp)<=WINDOW_SECONDS),key=lambda p:p.seq)

    def measure(self, profile, box, *, extended=False, packets=None):
        from web_app.temporal_capture import refine, WINDOW_SECONDS, EXTENDED_WINDOW_SECONDS, MAX_FRAMES, MAX_EXTENDED_FRAMES
        packets=self.snapshot(extended=extended) if packets is None else packets
        initial_reasons=list(self.passage['reasons']) if self.passage else []
        passage=refine(profile,box,((p.seq,p.image()) for p in packets),observations=self.observations,
                       max_frames=MAX_EXTENDED_FRAMES if extended else MAX_FRAMES)
        passage.update(detector_model=profile.detector_model,imgsz=profile.detector_imgsz,
                       anchor_frame=self.anchor.seq,
                       window_seconds=EXTENDED_WINDOW_SECONDS if extended else WINDOW_SECONDS)
        passage['diagnostics'].update(buffered_frames=len(packets),
            frame_span_seconds=packets[-1].stamp-packets[0].stamp if packets else 0,
            extended_time_window=extended,initial_review_reasons=initial_reasons if extended else list(passage['reasons']))
        self.passage=passage
        return passage


class Receiver:
    def __init__(self,url,stop,opener=None,save_fps=15):
        self.url=url
        self.stop=stop
        self.opener=opener or self.open
        self.save_fps=save_fps
        self.lock=threading.Lock()
        self.packets=deque(maxlen=100)
        self.size=0
        self.status='Подключение…'
        self.seq=0
        self.last_seen=0
        self.epoch=0
        self.windows=[]

    @staticmethod
    def open(url):
        return cv2.VideoCapture(url,cv2.CAP_FFMPEG,[cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,5000,cv2.CAP_PROP_READ_TIMEOUT_MSEC,3000])

    def snapshot(self):
        with self.lock:
            return list(self.packets)

    def retain_passage(self, anchor, epoch, earlier=()):
        with self.lock:
            if epoch!=self.epoch:
                raise HTTPException(409,'Камера переподключилась во время измерения')
            window=PassageFrames(anchor,epoch)
            window.extend([*earlier,*self.packets,anchor])
            latest=self.packets[-1].stamp if self.packets else anchor.stamp
            self.windows=[w for w in self.windows if w.epoch==epoch and w.end>latest]
            if latest<window.end:
                if len(self.windows)>=4:
                    raise HTTPException(409,'Слишком много одновременных проездов для проверки кадров')
                self.windows.append(window)
            return window

    def append(self, packet):
        with self.lock:
            while self.packets and (len(self.packets)>=100 or self.size+len(packet.jpeg)>48*1024*1024):
                self.size-=len(self.packets.popleft().jpeg)
            self.packets.append(packet)
            self.size+=len(packet.jpeg)
            for window in self.windows:
                if window.epoch==self.epoch:window.extend([packet])
            self.windows=[w for w in self.windows if w.epoch==self.epoch and w.end>packet.stamp]

    def release_passage(self, window):
        with self.lock:
            self.windows=[w for w in self.windows if w is not window]

    def run(self):
        while not self.stop.is_set():
            cap=None
            try:
                self.status='Подключение…'
                cap=self.opener(self.url)
                if not cap.isOpened():
                    raise RuntimeError()
                self.epoch+=1
                last_saved=0
                while not self.stop.is_set():
                    ok,raw=cap.read()
                    stamp=time.monotonic()
                    if not ok or raw is None:
                        raise RuntimeError()
                    self.last_seen=stamp
                    self.status='Камера работает'
                    # Side evidence needs adjacent frames; both buffers stay byte bounded.
                    if stamp-last_saved<1/self.save_fps:
                        continue
                    last_saved=stamp
                    ok,jpeg=cv2.imencode('.jpg',raw,[cv2.IMWRITE_JPEG_QUALITY,90])
                    if not ok:
                        continue
                    self.seq+=1
                    packet=Packet(self.seq,stamp,jpeg.tobytes())
                    self.append(packet)
            except Exception:
                self.status='Нет связи · переподключение'
            finally:
                if cap is not None:
                    cap.release()
                with self.lock:
                    self.packets.clear()
                    self.size=0
                    self.windows.clear()
            self.stop.wait(2)
        self.status='Остановлена'


def pair_packets(side,front,offset,tolerance,now):
    if not side or not front or now-side[-1].stamp>1 or now-front[-1].stamp>1:
        return None
    target=min(side[-1].stamp,front[-1].stamp-offset)
    a=min(side,key=lambda p:abs(p.stamp-target))
    b=min(front,key=lambda p:abs(p.stamp-(a.stamp+offset)))
    delta=b.stamp-a.stamp-offset
    return (a,b,delta) if abs(delta)<=tolerance else None


def iou(a,b):
    area=max(0,min(a[0]+a[2],b[0]+b[2])-max(a[0],b[0]))*max(0,min(a[1]+a[3],b[1]+b[3])-max(a[1],b[1]))
    return area/max(1,a[2]*a[3]+b[2]*b[3]-area)


class Station:
    def __init__(self,cfg,profile):
        self.cfg=cfg
        self.profile=profile
        self.id=uuid.uuid4().hex
        self.stop=threading.Event()
        self.side=Receiver(cfg.side_url,self.stop,save_fps=30)
        self.front=Receiver(cfg.front_url,self.stop)
        self.condition=threading.Condition()
        self.latest=None
        self.result=None
        self.plates=None
        self.error=None
        self.plate_error=None
        self.jpeg=None
        self.front_jpeg=None
        self.sequence=0
        self.tracks=[]
        self.threads=[]
        self.capture_lock=threading.Lock()
        self.saved={}
        self.last_capture=None
        self.front_history=FrontHistory()

    def start(self):
        for fn in (self.side.run,self.front.run,self.produce,self.infer,self.read_plates):
            t=threading.Thread(target=fn,daemon=True)
            self.threads.append(t)
            t.start()

    def close(self):
        self.stop.set()
        with self.condition:self.condition.notify_all()
        for t in self.threads:
            t.join(timeout=1.5)

    def paired(self):
        return pair_packets(self.side.snapshot(),self.front.snapshot(),self.cfg.offset_seconds,self.cfg.tolerance_ms/1000,time.monotonic())

    @staticmethod
    def fresh(receiver):
        packets=receiver.snapshot()
        return packets[-1] if packets and time.monotonic()-packets[-1].stamp<1 else None

    def produce(self):
        last_side=last_front=-1
        while not self.stop.wait(.025):
            pair=self.paired()
            a=self.fresh(self.side);b=self.fresh(self.front)
            if pair:a,b,_=pair
            if not a:self.jpeg=None;self.result=None
            if not b:self.front_jpeg=None;self.plates=None
            changed=False
            if a and a.seq!=last_side:
                last_side=a.seq
                try:
                    raw=a.image();h,w=raw.shape[:2]
                    preview=wb.corrected(cv2.resize(raw,(960,round(h*960/w))),self.profile.lens)
                    result=self.result
                    if result and abs(result['stamp']-a.stamp)<.6:
                        for d in result['detections']:
                            x,y,bw,bh=[round(v*960/w) for v in d['bbox']]
                            cv2.rectangle(preview,(x,y),(x+bw,y+bh),(70,240,140),2)
                            if d['length_m'] is not None:cv2.putText(preview,f"{d['length_m']:.2f} m",(x,max(20,y-4)),0,.6,(70,240,140),2)
                    if self.profile.measurement_line_x is not None:
                        x=round(self.profile.measurement_line_x*960/w)
                        cv2.line(preview,(x,0),(x,preview.shape[0]),(220,120,220),1)
                    self.jpeg=cv2.imencode('.jpg',preview)[1].tobytes();changed=True
                except Exception:self.error='Не удалось подготовить боковое изображение'
            if b and b.seq!=last_front:
                last_front=b.seq
                try:
                    front=b.image();fh,fw=front.shape[:2]
                    front=cv2.resize(front,(960,round(fh*960/fw)))
                    from web_app.video_stream import plate_label
                    if self.plates and abs(self.plates['stamp']-b.stamp)<.6:
                        for p in self.plates['candidates']:
                            x,y,x2,y2=[round(v*960/fw) for v in p['bbox']]
                            cv2.rectangle(front,(x,y),(x2,y2),(0,220,255),2)
                            label=plate_label(p['text']);lx=max(0,min(x,960-label.shape[1]));ly=max(0,min(y-28,front.shape[0]-28))
                            front[ly:ly+28,lx:lx+label.shape[1]]=label
                    self.front_jpeg=cv2.imencode('.jpg',front)[1].tobytes();changed=True
                except Exception:self.plate_error='Не удалось подготовить фронтальное изображение'
            if changed:
                with self.condition:
                    self.sequence+=1;self.condition.notify_all()

    def infer(self):
        from web_app.temporal_capture import capture_candidate, review_candidate, crossing_match, WINDOW_SECONDS, MAX_FRAMES
        from web_app.passage_tracking import match_track
        last=-1
        epoch=None
        while not self.stop.wait(.015):
            a=self.fresh(self.side)
            if not a or a.seq<=last:
                try:
                    self.flush_reviews(time.monotonic())
                except Exception:
                    self.error='Не удалось сохранить кадр автомобиля для проверки.'
                continue
            pair=self.paired() or (a,None,None)
            a,b,delta=pair
            if a.seq<=last:continue
            last=a.seq
            current_epoch=self.side.epoch
            if epoch!=current_epoch:
                self.tracks=[];epoch=current_epoch
            try:
                # Keep pre-line evidence before a slow detector can evict it.
                earlier=self.side.snapshot()
                self.flush_pending(a.stamp)
                result=wb.render_raw(a.image(),wb.FrameRequest(profile=self.profile,frame=a.seq,detect=True),include_image=False)
                if self.stop.is_set() or current_epoch!=self.side.epoch:continue
                result.update(frame=a.seq,stamp=a.stamp)
                self.result=result
                self.error=None
                used=set();self.tracks=[t for t in self.tracks if a.stamp-t['stamp']<3 or t.get('pending') or t.get('review')]
                for index,d in enumerate(result['detections']):
                    track=match_track(self.tracks,result['detections'],index,used,a.stamp)
                    if track is None:
                        track=dict(id=uuid.uuid4().hex,sent=False,box=d['bbox'],stamp=a.stamp)
                        self.tracks.append(track)
                    previous=track['box'];previous_time=track['stamp']
                    track.update(box=d['bbox'],stamp=a.stamp);used.add(track['id'])
                    if not self.cfg.auto_measure or track['sent'] or track.get('pending'):continue
                    if review_candidate(d):
                        distance=abs(d.get('line_offset_px') or 0)+(self.profile.image_size[0] if d['status']=='clipped' else 0)
                        if not track.get('review') or distance<track['review'][3]:
                            track['review']=(pair,d,current_epoch,distance)
                    candidate=d;capture_pair=pair
                    line=self.profile.measurement_line_x
                    if not capture_candidate(d) and line is not None:
                        before=previous[0]+previous[2]/2-line;after=d['bbox'][0]+d['bbox'][2]/2-line
                        if before*after<0 and a.stamp-previous_time<2:
                            when=previous_time+(a.stamp-previous_time)*abs(before)/(abs(before)+abs(after))
                            packets=[p for p in earlier if previous_time<=p.stamp<=a.stamp
                                     and abs(p.stamp-when)<=WINDOW_SECONDS]
                            for packet in sorted(packets,key=lambda p:abs(p.stamp-when))[:MAX_FRAMES]:
                                pp=pair_packets([packet],self.front.snapshot(),self.cfg.offset_seconds,self.cfg.tolerance_ms/1000,time.monotonic())
                                if not pp:pp=(packet,None,None)
                                rr=wb.render_raw(packet.image(),wb.FrameRequest(profile=self.profile,frame=packet.seq,detect=True),include_image=False)
                                t=(packet.stamp-previous_time)/(a.stamp-previous_time)
                                expected=[v+t*(w-v) for v,w in zip(previous,d['bbox'])]
                                match=crossing_match(rr['detections'],expected)
                                if match is not None:
                                    candidate=match;capture_pair=pp;break
                    if capture_candidate(candidate):
                        track.pop('review',None)
                        track['passage_frames']=self.side.retain_passage(capture_pair[0],current_epoch,earlier)
                        track['pending']=(capture_pair,candidate,current_epoch)
                self.flush_reviews(a.stamp)
            except Exception:
                self.error='Измерение не выполнено. Проверьте камеры и настройку.'

    def flush_reviews(self, stamp):
        """Give centred detection priority, then persist the closest real frame."""
        for track in self.tracks:
            review=track.get('review')
            if not review or track['sent'] or track.get('pending'):
                continue
            pair,candidate,epoch,_=review
            if epoch!=self.side.epoch or not self.cfg.auto_measure:
                track.pop('review',None)
                continue
            if stamp-track['stamp']<.8 and stamp-pair[0].stamp<2:
                continue
            self.capture(pair,candidate,track['id'],epoch=epoch,review_fallback=True)
            track.update(sent=True)
            track.pop('review',None)

    def flush_pending(self, stamp):
        from web_app.temporal_capture import WINDOW_SECONDS, EXTENDED_WINDOW_SECONDS, needs_more_evidence
        for track in self.tracks:
            pending=track.get('pending')
            if not pending or stamp-pending[0][0].stamp < WINDOW_SECONDS:
                continue
            pair, candidate, epoch=pending
            window=track.get('passage_frames')
            max_age=3+EXTENDED_WINDOW_SECONDS if window else 3
            if epoch != self.side.epoch or not self.cfg.auto_measure:
                track.pop('pending',None)
                track.pop('passage_frames',None)
                self.side.release_passage(window)
                continue
            if stamp-pair[0].stamp>max_age:
                # Slow inference must not erase the immutable observation.
                self.capture(pair,candidate,track['id'],epoch=epoch,review_fallback=True)
                track.update(sent=True,pending=None)
                track.pop('passage_frames',None)
                self.side.release_passage(window)
                continue
            if window:
                passage=window.passage or window.measure(self.profile,candidate['bbox'])
                if needs_more_evidence(passage) and not passage['diagnostics']['extended_time_window']:
                    if stamp<window.end:
                        continue  # Receiver keeps the originals; resume live tracking meanwhile.
                    window.measure(self.profile,candidate['bbox'],extended=True)
            self.capture(pair,candidate,track['id'],temporal=True,epoch=epoch,
                         passage_frames=window)
            track.update(sent=True,pending=None)
            track.pop('passage_frames',None)
            self.side.release_passage(window)

    def capture(self,pair,d,track_id,*,temporal=False,epoch=None,passage_frames=None,review_fallback=False):
        from web_app import station as st
        from web_app.temporal_capture import refinement_candidate, EXTENDED_WINDOW_SECONDS, needs_more_evidence
        a,b,delta=pair
        max_age=3+EXTENDED_WINDOW_SECONDS if passage_frames else 3
        if self.stop.is_set() or (not review_fallback and (self.fresh(self.side) is None or time.monotonic()-a.stamp>max_age)):
            raise HTTPException(409,'Нет свежего изображения боковой камеры')
        with self.capture_lock:
            if track_id in self.saved:return self.saved[track_id]
            expected_epoch=self.side.epoch if epoch is None else epoch
            if expected_epoch!=self.side.epoch or (passage_frames and passage_frames.epoch!=expected_epoch):
                raise HTTPException(409,'Камера переподключилась во время измерения')
            # Freeze first: rendering the anchor must not erase its neighbours.
            if temporal:
                window=passage_frames or self.side.retain_passage(a,expected_epoch)
                packets=window.snapshot()
            result=wb.render_raw(a.image(),wb.FrameRequest(profile=self.profile,frame=a.seq,boxes=[d['bbox']]),
                                 include_image=False,include_frame=True)
            if temporal and refinement_candidate(result['detections'][0]):
                from vehicle_metrology.temporal import apply_passage
                if expected_epoch != self.side.epoch:
                    raise HTTPException(409,'Камера переподключилась во время измерения')
                passage=window.passage or window.measure(self.profile,d['bbox'],packets=packets)
                if (needs_more_evidence(passage) and not passage['diagnostics']['extended_time_window']
                        and expected_epoch==self.side.epoch and not self.stop.is_set()):
                    passage=window.measure(self.profile,d['bbox'],extended=True)
                if expected_epoch != self.side.epoch or self.stop.is_set():
                    raise HTTPException(409,'Измерение прервано переподключением или остановкой камеры')
                passage['diagnostics']['pinned_passage']=passage_frames is not None
                result['detections'][0]=apply_passage(result['detections'][0],passage,
                    allow_estimate=self.profile.measurement_mode=='estimate')
            if expected_epoch!=self.side.epoch or self.stop.is_set():
                raise HTTPException(409,'Измерение прервано переподключением или остановкой камеры')
            if review_fallback:
                from web_app.temporal_capture import prepare_review_measurement
                prepare_review_measurement(self.profile,result)
            payload=st.Capture(media_id=self.id,profile=self.profile,frame=a.seq,bbox=d['bbox'],label=d['label'],source=self.profile.detector_model,actor='Камеры',temporal=temporal,review_fallback=review_fallback)
            paired=None;samples=[]
            if b is not None:
                paired=dict(kind='ip',frame=b.seq,side_seconds=a.stamp,front_seconds=b.stamp,
                            offset_seconds=self.cfg.offset_seconds,sync_error_ms=delta*1000,
                            clock='server_receive_monotonic',association='operator_review')
            front_image=b.image() if b else None
            evidence=[]
            if b is not None and self.front_history.frames:
                from web_app.plates import front_vehicle
                try:
                    evidence=self.front_history.select(b,front_vehicle(front_image),self.front.epoch)
                except Exception:
                    pass  # A plate-model failure must not discard a measured vehicle.
            if b is not None and not evidence:
                samples=sorted(self.front.snapshot(),key=lambda p:abs(p.stamp-b.stamp))[:5]
                samples=[(p.seq,p.image()) for p in samples if abs(p.stamp-b.stamp)<.5]
            record=st.persist_capture(payload,result,front_image,paired,samples,
                                      camera_note=None if b else 'Нет согласованного фронтального снимка: номер нужно проверить вручную',
                                      front_evidence=evidence)
            self.saved[track_id]=record
            if len(self.saved)>1000:self.saved.pop(next(iter(self.saved)))
            self.last_capture=record['id']
            if temporal:self.side.release_passage(window)
            return record

    def read_plates(self):
        from web_app.plates import recognize, front_vehicle
        last=-1
        while not self.stop.wait(.25):
            b=self.fresh(self.front)
            if not b or b.seq==last:continue
            last=b.seq
            try:
                epoch=self.front.epoch
                image=b.image();box=front_vehicle(image)
                candidates=recognize(image,target_box=box) if box is not None else []
                if epoch!=self.front.epoch:continue
                self.front_history.observe(b,box,candidates,epoch)
                self.plates=dict(stamp=b.stamp,frame=b.seq,candidates=candidates)
                self.plate_error=None
            except Exception:self.plate_error='Номер временно недоступен'

    async def frames(self,front):
        seq=-1
        while not self.stop.is_set():
            if self.sequence!=seq:
                seq=self.sequence;data=self.front_jpeg if front else self.jpeg
                if data:yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'+data+b'\r\n'
            # Share already encoded frames without occupying a worker thread per viewer.
            await asyncio.sleep(.05)


@router.get('/configuration')
def get_config():return public_config(settings())


@router.post('/configuration')
def configure(cfg: Settings):
    with guard:
        if active:raise HTTPException(409,'Сначала остановите камеры')
        old=settings()
        cfg.side_url=cfg.side_url or old.side_url;cfg.front_url=cfg.front_url or old.front_url
        cfg.profile=cfg.profile or old.profile
        if not cfg.side_url or not cfg.front_url:raise HTTPException(422,'Укажите адреса обеих камер')
        temporary=CONFIG.with_suffix('.tmp')
        temporary.write_text(cfg.model_dump_json(),encoding='utf-8');temporary.replace(CONFIG)
    return public_config(cfg)


@router.post('/calibration')
def calibration(profile: wb.Profile):
    global active
    with guard:
        profile=wb.approved_references(profile)
        cfg=settings();previous=active
        if previous:
            if len(profile.polygon)<4 or not (profile.references or profile.metric_rulers or profile.survey_calibration):
                raise HTTPException(422,'Нужна дорога и эталоны или мерные линии')
            if wb.detector_settings(previous.profile)!=wb.detector_settings(profile):
                wb.prepare_detector(wb.DetectorChoice(**profile.model_dump()))
        cfg.profile=profile
        temporary=CONFIG.with_suffix('.tmp')
        # Disk and model failures must leave the previous cameras running.
        temporary.write_text(cfg.model_dump_json(),encoding='utf-8')
        replacement=Station(cfg,profile) if previous else None
        if previous:previous.close()
        try:
            if replacement:replacement.start()
            temporary.replace(CONFIG)
        except Exception as exc:
            if replacement:replacement.close()
            temporary.unlink(missing_ok=True)
            active=None
            if previous:
                restored=Station(previous.cfg,previous.profile)
                try:
                    restored.start();active=restored
                except Exception:
                    restored.close()
                    raise HTTPException(503,'Настройка не применена; камеры остановлены. Подключите их повторно.') from exc
            raise HTTPException(503,'Настройка не применена; предыдущая настройка сохранена.') from exc
        active=replacement
        return {'saved':True,'image_size':profile.image_size,'profile':profile,
                'running':active is not None,'restarted':previous is not None,
                'id':active.id if active else None,'detector':wb.detector_settings(profile)}


@router.post('/detector')
def change_detector(choice: wb.DetectorChoice):
    # Read the latest source profile under the same lock as apply. Only these
    # two fields change; a stale browser cannot overwrite the road calibration.
    with guard:
        profile=current_calibration().model_copy(update=choice.model_dump())
        return calibration(profile)


@router.get('/calibration')
def current_calibration():
    with guard:
        if active:
            return active.profile
        cfg=settings()
        return cfg.profile or wb.operator_calibration()


@router.post('/start')
def start():
    global active
    with guard:
        if active:return {'running':True,'id':active.id}
        cfg=settings()
        if not cfg.side_url or not cfg.front_url:raise HTTPException(422,'Сначала настройте адреса камер')
        profile=cfg.profile or wb.operator_calibration()
        if not isinstance(profile,wb.Profile):profile=wb.Profile.model_validate(profile)
        if not (profile.references or profile.metric_rulers or profile.survey_calibration) or len(profile.polygon)<4:raise HTTPException(422,'Сначала загрузите настройку измерения')
        wb.require_gpu()
        active=Station(cfg,profile);active.start()
        return {'running':True,'id':active.id}


@router.post('/stop')
def stop():
    global active
    with guard:
        if active:active.close()
        active=None
    return {'running':False}


@router.post('/automatic')
def automatic(enabled: bool):
    with guard:
        if active:active.cfg.auto_measure=enabled
        cfg=settings();cfg.auto_measure=enabled
        temporary=CONFIG.with_suffix('.tmp')
        temporary.write_text(cfg.model_dump_json(),encoding='utf-8');temporary.replace(CONFIG)
    return {'enabled':enabled}


@router.get('/state')
def state(compact: bool=False):
    c=active
    if not c:return {'running':False,'detector':configured_detector(settings())}
    pair=c.paired()
    side_ready=c.fresh(c.side) is not None;front_ready=c.fresh(c.front) is not None
    return dict(running=True,id=c.id,ready=pair is not None,side_ready=side_ready,front_ready=front_ready,side=c.side.status,front=c.front.status,
                error=c.error,plate_error=c.plate_error,plates=c.plates if front_ready else None,
                result=c.result if side_ready and not compact else None,sync_error_ms=pair[2]*1000 if pair else None,
                auto_measure=c.cfg.auto_measure,last_capture=c.last_capture,
                detector=wb.detector_settings(c.profile))


@router.get('/{camera}/video')
def video(camera: str):
    if camera not in {'side','front'} or not active:raise HTTPException(404,'Камеры не запущены')
    return StreamingResponse(active.frames(camera=='front'),media_type='multipart/x-mixed-replace; boundary=frame',headers={'Cache-Control':'no-store'})


def side_snapshot():
    # Use the receiver's uncorrected full-resolution image, never the annotated
    # browser preview. Calibration applies the lens correction exactly once.
    with guard:
        camera=active
        if camera:
            packet=camera.fresh(camera.side)
            if packet is None:
                raise HTTPException(409,'Нет свежего кадра боковой камеры. Проверьте подключение и повторите.')
            return packet.jpeg
        url=settings().side_url
    if not url:raise HTTPException(409,'Сначала сохраните адрес боковой камеры в окне оператора.')
    cap=None
    try:
        cap=Receiver.open(url);ok,raw=cap.read()
        if not ok or raw is None:raise ValueError()
        ok,encoded=cv2.imencode('.jpg',raw,[cv2.IMWRITE_JPEG_QUALITY,95])
        if not ok:raise ValueError()
        return encoded.tobytes()
    except Exception:raise HTTPException(409,'Не удалось получить снимок камеры. Проверьте адрес и подключение.')
    finally:
        if cap is not None:cap.release()


@router.get('/side/snapshot')
def snapshot():
    image=side_snapshot()
    return Response(image,media_type='image/jpeg',headers={'Content-Disposition':'attachment; filename="camera-calibration.jpg"','Cache-Control':'no-store'})


def startup():
    if settings().autostart:
        try:start()
        except Exception:pass  # UI remains reachable to repair configuration.
