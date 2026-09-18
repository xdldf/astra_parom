"""Persistent local operator station. Prices and transitions are server-owned."""
import base64
import csv
import hashlib
import io
import json
import re
import uuid
from functools import lru_cache
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Query, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

from vehicle_metrology.plate_text import normalize_plate
from vehicle_metrology.tariffs import CATEGORIES, catalog, quote
from web_app import workbench
from web_app.station_store import connect_database, summary

router=APIRouter(prefix='/api/station')
DATA=Path(__file__).parent/'data'/'station'
DB=DATA/'station.sqlite3'
STATUSES=['Новый','Требует проверки','Подтвержден','Оплачен','Отклонён']


def now():
    return datetime.now(timezone(timedelta(hours=9))).isoformat(timespec='seconds')


def connect():
    return connect_database(DB)


def find(db,ident):
    row=db.execute('SELECT data FROM vehicles WHERE id=?',(ident,)).fetchone()
    if not row:
        raise HTTPException(404,'Запись не найдена')
    return json.loads(row['data'])


def event(db,record,actor,action,reason,before=None):
    db.execute('INSERT INTO audit(vehicle_id,timestamp,actor,action,reason,before_json,after_json) VALUES(?,?,?,?,?,?,?)',
               (record['id'],now(),actor,action,reason,json.dumps(before,ensure_ascii=False),json.dumps(record,ensure_ascii=False)))


class Fields(BaseModel):
    model_config=ConfigDict(allow_inf_nan=False)
    plate: str=Field('',max_length=24)
    category: str='car'
    length_m: float | None=Field(None,gt=0,le=100)
    load_capacity_t: float | None=Field(None,gt=0,le=1000)
    manual_rub: int | None=Field(None,ge=0,le=10000000,strict=True)
    actor: str=Field('Оператор',min_length=1,max_length=100)
    reason: str=Field('',max_length=1000)
    station_id: str=Field('default',pattern=r'^[a-zA-Z0-9_-]{1,64}$')

    @model_validator(mode='after')
    def validate_category(self):
        self.plate=normalize_plate(self.plate)
        if re.search('[A-Z]',self.plate):
            raise ValueError('Используйте русские буквы номера')
        if self.category not in CATEGORIES:
            raise ValueError('Неизвестная категория ТС')
        if not self.actor.strip():
            raise ValueError('Укажите оператора')
        return self


class Edit(Fields):
    version: int=Field(ge=1)
    action: Literal['edit','confirm','reject','pay']='edit'


class Capture(BaseModel):
    media_id: str
    profile: workbench.Profile
    frame: int=Field(0,ge=0)
    bbox: tuple[float,float,float,float]
    label: Literal['car','truck','bus','motorcycle','manual car','selected car']='car'
    source: Literal['manual','yolo26x','yolo26n']='manual'
    actor: str=Field('Оператор',min_length=1,max_length=100)
    front_media_id: str | None = None
    front_session_id: str | None = None
    front_offset_seconds: float=Field(3,ge=-3600,le=3600,allow_inf_nan=False)


def new_record(fields,source=None):
    stamp=now()
    tariff=quote(fields.category,fields.length_m,fields.manual_rub,fields.load_capacity_t)
    return dict(id=uuid.uuid4().hex,version=1,created_at=stamp,updated_at=stamp,
                plate=normalize_plate(fields.plate),category=fields.category,length_m=fields.length_m,
                load_capacity_t=fields.load_capacity_t,
                measured_length_m=source.get('measured_length_m') if source else None,
                tariff=tariff,manual_rub=fields.manual_rub,status='Требует проверки',
                actor=fields.actor,source=source,side_photo=None,front_photo=None)


@router.get('/catalog')
def tariffs():
    return dict(categories=CATEGORIES,tariffs=catalog(),statuses=STATUSES,
                policy='Приложение №1; раздел 6 — приказ №37/ПЯ от 08.09.2026, действует с 08.09.2026. Диапазоны без округления и заполнения промежутков. Грузоподъёмность берётся из документов, не из изображения. Категорию подтверждает оператор.')


@router.get('/health')
def health():
    import torch
    available=torch.cuda.is_available()
    return dict(online=True,gpu_available=available,gpu=torch.cuda.get_device_name(0) if available else None,
                detector='YOLO26n',front_camera=(workbench.DATA.parent/'camera-pair.json').exists(),camera_source='recorded_video_pair',plate_recognition=True)


@router.post('/quote')
def calculate(fields: Fields):
    return quote(fields.category,fields.length_m,fields.manual_rub,fields.load_capacity_t)


@router.post('/vehicles')
def create(fields: Fields):
    if fields.manual_rub is not None and not fields.reason.strip():
        raise HTTPException(422,'Укажите причину ручного тарифа')
    record=new_record(fields)
    with connect() as db:
        db.execute('INSERT INTO vehicles VALUES(?,?,?,?)',(record['id'],None,1,json.dumps(record,ensure_ascii=False)))
        event(db,record,fields.actor,'create',fields.reason)
    return record


@router.post('/capture')
def capture(payload: Capture):
    # Recalculate from the original uploaded frame and submitted calibration.
    result=workbench.render_raw(workbench.read_frame(payload.media_id,payload.frame),
            workbench.FrameRequest(profile=payload.profile,frame=payload.frame,boxes=[payload.bbox]),
            include_image=False,include_frame=True)
    front_image=None
    paired=None
    evidence=[]
    if payload.front_media_id:
        from web_app.video_stream import paired_frame
        workbench.read_frame(payload.front_media_id,0)
        side=workbench.media[payload.media_id]
        front=workbench.media[payload.front_media_id]
        if side['kind']!='video' or front['kind']!='video':
            raise HTTPException(422,'Paired capture requires two videos')
        fi=paired_frame(payload.frame,side['fps'],front['fps'],payload.front_offset_seconds)
        if fi<0:
            raise HTTPException(422,'Front frame precedes recording')
        front_image=workbench.read_frame(payload.front_media_id,fi)
        paired=dict(media_id=payload.front_media_id,frame=fi,offset_seconds=payload.front_offset_seconds,
                    side_seconds=payload.frame/side['fps'],front_seconds=fi/front['fps'],
                    association='synchronized_frame_operator_review')
        if payload.front_session_id:
            from web_app.video_stream import sessions
            from web_app.ip_cameras import Packet
            from web_app.plates import front_vehicle
            camera=sessions.get(payload.front_session_id)
            if (camera and camera.front_history.frames and camera.request.media_id==payload.media_id
                    and camera.request.front_media_id==payload.front_media_id):
                try:
                    evidence=camera.front_history.select(Packet(fi,fi/front['fps'],b''),front_vehicle(front_image),0)
                except Exception:
                    pass
    return persist_capture(payload,result,front_image,paired,front_evidence=evidence)


def persist_capture(payload,result,front_image=None,paired=None,front_samples=None,camera_note=None,front_evidence=None):
    measured=result['detections'][0]
    if measured['status'] in {'outside_road','clipped','waiting_for_line','outside_calibration'}:
        raise HTTPException(422,'Автомобиль должен быть целиком в кадре, на дороге, в области калибровки и у линии измерения (если она включена).')
    category={'car':'car','bus':'bus','truck':'truck','motorcycle':'motorcycle'}.get(payload.label,'car')
    identity=[payload.media_id,payload.frame,[round(v,1) for v in payload.bbox]]
    if paired:
        identity.append(paired)
    source_key=hashlib.sha256(json.dumps(identity).encode()).hexdigest()
    source=dict(media_id=payload.media_id,frame=payload.frame,bbox=list(payload.bbox),label=payload.label,
                detector=payload.source,measurement=measured,measured_length_m=measured['length_m'],
                calibration=payload.profile.model_dump())
    if paired:
        source['front_camera']=paired
    if camera_note:
        source['camera_note']=camera_note
    record=new_record(Fields(category=category,length_m=measured['length_m'],actor=payload.actor),source)
    with connect() as db:
        existing=db.execute('SELECT data FROM vehicles WHERE source_key=?',(source_key,)).fetchone()
        if existing:return json.loads(existing['data'])
    image=result.get('frame_image')
    if image is None:
        image=cv2.imdecode(np.frombuffer(base64.b64decode(result['image']),np.uint8),cv2.IMREAD_COLOR)
    x,y,bw,bh=map(int,payload.bbox)
    margin=max(20,int(bw*.08))
    crop=image[max(0,y-margin):min(image.shape[0],y+bh+margin),max(0,x-margin):min(image.shape[1],x+bw+margin)]
    # Encode outside the SQLite write transaction so operators can continue saving.
    photos={record['id']+'-side.jpg':crop}
    record['side_photo']=record['id']+'-side.jpg'
    if front_evidence and front_image is not None and paired:
        # Keep the simultaneous image as evidence; the readable cab can pass
        # earlier than the centre of a long truck reaches the measurement line.
        selected=front_evidence[0]['packet']
        earlier=selected.image()
        if earlier is not None:
            filename=uuid.uuid4().hex+'-front.jpg'
            photos[filename]=front_image
            source['synchronized_front_photo']=filename
            source['front_evidence']=dict(frame=selected.seq,seconds=selected.stamp,
                offset_seconds=selected.stamp-paired['front_seconds'],association='tracked_passage_operator_review')
            front_image=earlier
            source['camera_note']='Фото номера выбрано из того же непрерывного проезда фронтальной камеры. Сверьте кабину и номер с боковым снимком.'
            # Reuse the live OCR results; do not run a second GPU search per car.
            front_samples=None
            source['front_samples']=[]
            for item in front_evidence:
                packet=item['packet'];filename=uuid.uuid4().hex+'-front.jpg'
                pixels=packet.image()
                if pixels is None:continue
                photos[filename]=pixels
                source['front_samples'].append(dict(frame=packet.seq,photo=filename,
                    target_box=item['box'],candidates=item['candidates']))
    if front_image is not None:
        record['front_photo']=record['id']+'-front.jpg'
        photos[record['front_photo']]=front_image
        record['plate_ocr']={'state':'queued','candidates':[]}
    if front_samples:
        source['front_samples']=[]
        for frame_index,sample in front_samples:
            filename=uuid.uuid4().hex+'-front.jpg'
            photos[filename]=sample
            source['front_samples'].append(dict(frame=frame_index,photo=filename))
    encoded={}
    for filename,pixels in photos.items():
        ok,jpeg=cv2.imencode('.jpg',pixels,[cv2.IMWRITE_JPEG_QUALITY,92])
        if not ok:raise HTTPException(500,'Не удалось подготовить снимок')
        encoded[filename]=jpeg.tobytes()
    with connect() as db:
        db.execute('BEGIN IMMEDIATE')
        existing=db.execute('SELECT data FROM vehicles WHERE source_key=?',(source_key,)).fetchone()
        if existing:
            return json.loads(existing['data'])
        for filename,jpeg in encoded.items():
            (DATA/filename).write_bytes(jpeg)
        db.execute('INSERT INTO vehicles VALUES(?,?,?,?)',(record['id'],source_key,1,json.dumps(record,ensure_ascii=False)))
        event(db,record,payload.actor,'capture','Импорт из измерения; требуется проверка оператором')
    if front_image is not None:
        from web_app.plates import enqueue
        enqueue(record['id'])
    return record


@router.post('/vehicles/{ident}/recognize-plate')
def recognize_plate(ident: str):
    from web_app.plates import enqueue
    with connect() as db:
        db.execute('BEGIN IMMEDIATE')
        old=find(db,ident)
        if old['status'] in {'Оплачен','Подтвержден'}:
            raise HTTPException(409,'Сначала откройте запись для редактирования')
        if not old.get('front_photo'):
            raise HTTPException(422,'Нет фото фронтальной камеры')
        if old.get('plate_ocr',{}).get('state')=='queued':
            enqueue(ident)
            return old
        record=dict(old,plate_ocr={'state':'queued','candidates':[],'reread':True},version=old['version']+1,updated_at=now())
        db.execute('UPDATE vehicles SET version=?,data=? WHERE id=?',(record['version'],json.dumps(record,ensure_ascii=False),ident))
    enqueue(ident)
    return record


def filters(plate='',status='',category='',length_min=None,length_max=None,date_from=None,date_to=None):
    if status and status not in STATUSES:
        raise HTTPException(422,'Неизвестный статус')
    if category and category not in CATEGORIES:
        raise HTTPException(422,'Неизвестная категория')
    if date_from and date_to and date_from>date_to:
        raise HTTPException(422,'Начальная дата позже конечной')
    if length_min is not None and length_max is not None and length_min>length_max:
        raise HTTPException(422,'Минимальная длина больше максимальной')
    clauses,values=[],[]
    for column,value in (('status',status),('category',category)):
        if value:
            clauses.append(column+'=?');values.append(value)
    if plate:
        clauses.append('instr(plate,?)>0');values.append(normalize_plate(plate))
    for column,operator,value in (('length_m','>=',length_min),('length_m','<=',length_max),
                                   ('created_at','>=',date_from.isoformat() if date_from else None),
                                   ('created_at','<=',date_to.isoformat()+'T99' if date_to else None)):
        if value is not None:
            clauses.append(column+operator+'?');values.append(value)
    return ' AND '.join(clauses) or '1', values


def conditional(request, response, token):
    etag='"'+hashlib.sha256(token.encode()).hexdigest()+'"'
    response.headers.update({'ETag':etag,'Cache-Control':'private, no-cache'})
    if request.headers.get('if-none-match')==etag:
        return Response(status_code=304,headers=dict(response.headers))


@router.get('/vehicles')
def vehicles(request: Request,response: Response,plate: str='',status: str='',category: str='',
             length_min: float|None=Query(None,allow_inf_nan=False),
             length_max: float|None=Query(None,allow_inf_nan=False),date_from: date|None=None,date_to: date|None=None,
             limit: int=Query(250,ge=1,le=250),offset: int=Query(0,ge=0),snapshot: int|None=Query(None,ge=0)):
    where,args=filters(plate,status,category,length_min,length_max,date_from,date_to)
    with connect() as db:
        db.execute('BEGIN')
        revision=db.execute("SELECT value FROM settings WHERE key='vehicle_revision'").fetchone()[0]
        unchanged=conditional(request,response,str(DB)+revision+str(request.url))
        if unchanged is not None:return unchanged
        # A frozen upper rowid keeps older pages stable when cameras add new cars.
        if snapshot is None:
            snapshot=db.execute('SELECT COALESCE(MAX(seq),0) FROM vehicle_list').fetchone()[0]
        where+=' AND seq<=?';args.append(snapshot)
        totals=dict(db.execute(f"""SELECT COUNT(*) AS count,
            COALESCE(SUM(CASE WHEN status IN ('Подтвержден','Оплачен') THEN amount_rub ELSE 0 END),0) AS amount_rub,
            COALESCE(SUM(CASE WHEN status='Оплачен' THEN amount_rub ELSE 0 END),0) AS paid_rub,
            COALESCE(SUM(status IN ('Требует проверки','Отклонён')),0) AS issues
            FROM vehicle_list WHERE {where}""",args).fetchone())
        rows=[summary(r) for r in db.execute(f"""SELECT page.*, json_extract(v.data, '$.side_photo') AS side_photo,
            json_extract(v.data, '$.front_photo') AS front_photo,
            json_extract(v.data, '$.source.front_evidence.offset_seconds') AS front_photo_offset_seconds
            FROM (SELECT * FROM vehicle_list WHERE {where} ORDER BY seq DESC LIMIT ? OFFSET ?) AS page
            JOIN vehicles v ON v.id=page.id ORDER BY page.seq DESC""",args+[limit,offset])]
    return dict(rows=rows,**totals,limit=limit,offset=offset,snapshot=snapshot,has_more=offset+len(rows)<totals['count'])


@router.get('/vehicles/{ident}')
def vehicle(ident: str,request: Request,response: Response):
    with connect() as db:
        db.execute('BEGIN')
        row=db.execute('SELECT version FROM vehicles WHERE id=?',(ident,)).fetchone()
        if row is None:raise HTTPException(404,'Запись не найдена')
        unchanged=conditional(request,response,str(DB)+ident+str(row['version']))
        if unchanged is not None:return unchanged
        record=find(db,ident)
        if record['category']=='truck' and record.get('length_m',0) and record['length_m']>11.9 and record['tariff']['amount_rub'] is None:
            record['tariff']=quote('truck',record['length_m'])
        record['history']=[dict(row) for row in db.execute('SELECT timestamp,actor,action,reason FROM audit WHERE vehicle_id=? ORDER BY id DESC',(ident,))]
        return record


@router.post('/vehicles/{ident}')
def update(ident: str,fields: Edit):
    with connect() as db:
        db.execute('BEGIN IMMEDIATE')
        old=find(db,ident)
        if old['version']!=fields.version:
            raise HTTPException(409,'Запись изменена в другом окне. Обновите список и выберите автомобиль снова.')
        if old['status']=='Оплачен':
            raise HTTPException(409,'Оплаченная запись закрыта для изменений')
        if fields.action=='pay' and old['status']!='Подтвержден':
            raise HTTPException(409,'Сначала подтвердите автомобиль')
        if fields.action=='pay':
            if any(old.get(k)!=getattr(fields,k) for k in ('category','length_m','manual_rub','load_capacity_t')) or old['plate']!=normalize_plate(fields.plate):
                raise HTTPException(422,'Перед оплатой подтвердите изменения данных')
            record=dict(old,status='Оплачен')
        elif fields.action=='reject':
            if not fields.reason.strip():
                raise HTTPException(422,'Укажите причину отклонения')
            record=dict(old,status='Отклонён')
        else:
            changed=any(old.get(k)!=getattr(fields,k) for k in ('category','length_m','manual_rub','load_capacity_t')) or old['plate']!=normalize_plate(fields.plate)
            if (changed or fields.manual_rub is not None) and not fields.reason.strip():
                raise HTTPException(422,'Укажите причину изменения')
            tariff=quote(fields.category,fields.length_m,fields.manual_rub,fields.load_capacity_t)
            if fields.action=='confirm' and tariff['amount_rub'] is None:
                raise HTTPException(422,'Тариф не определён. Уточните данные или задайте ручной тариф с причиной.')
            record=dict(old,plate=normalize_plate(fields.plate),category=fields.category,length_m=fields.length_m,
                        manual_rub=fields.manual_rub,load_capacity_t=fields.load_capacity_t,tariff=tariff,
                        status='Подтвержден' if fields.action=='confirm' else 'Требует проверки')
        record.update(version=old['version']+1,updated_at=now(),actor=fields.actor)
        db.execute('UPDATE vehicles SET version=?,data=? WHERE id=?',(record['version'],json.dumps(record,ensure_ascii=False),ident))
        if fields.action=='confirm':
            db.execute("INSERT OR REPLACE INTO settings VALUES(?,?)",(client_key(fields.station_id),ident))
        elif fields.action in {'edit','reject'}:
            db.execute("DELETE FROM settings WHERE (key='client_vehicle' OR key LIKE 'client_vehicle:%') AND value=?",(ident,))
        event(db,record,fields.actor,fields.action,fields.reason,old)
    return record


def client_key(station_id):
    return 'client_vehicle' if station_id=='default' else 'client_vehicle:'+station_id


@router.get('/client')
def client_screen(request: Request,response: Response,station_id: str=Query('default',pattern=r'^[a-zA-Z0-9_-]{1,64}$')):
    with connect() as db:
        db.execute('BEGIN')
        row=db.execute("SELECT value FROM settings WHERE key=?",(client_key(station_id),)).fetchone()
        version=db.execute('SELECT version FROM vehicles WHERE id=?',(row['value'],)).fetchone() if row else None
        unchanged=conditional(request,response,str(DB)+station_id+(row['value']+str(version[0]) if row and version else 'empty'))
        if unchanged is not None:return unchanged
        record=find(db,row['value']) if row else None
        if record and record['status'] not in {'Подтвержден','Оплачен'}: record=None
        if record:
            record={k:record.get(k) for k in ('id','version','status','plate','category','length_m','load_capacity_t',
                                             'tariff','front_photo','side_photo')}
        return dict(vehicle=record)


@router.get('/photos/{filename}')
def photo(filename: str, thumbnail: bool=False):
    if not re.fullmatch(r'[0-9a-f]{32}-(?:side|front|plate)\.jpg',filename):
        raise HTTPException(404,'Фото не найдено')
    path=DATA/filename
    if not path.is_file(): raise HTTPException(404,'Фото не найдено')
    if thumbnail:
        stat=path.stat()
        data=photo_thumbnail(str(path),stat.st_mtime_ns,stat.st_size)
        return Response(data,media_type='image/jpeg',headers={'Cache-Control':'private, max-age=86400'})
    return FileResponse(path)


@lru_cache(maxsize=512)
def photo_thumbnail(path: str, modified: int, size: int):
    image=cv2.imread(path)
    if image is None:raise HTTPException(404,'Фото не найдено')
    height,width=image.shape[:2]
    ratio=min(1,160/width,100/height)
    if ratio<1:
        image=cv2.resize(image,(max(1,round(width*ratio)),max(1,round(height*ratio))),interpolation=cv2.INTER_AREA)
    ok,encoded=cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,75])
    if not ok:raise HTTPException(500,'Не удалось подготовить фото')
    return encoded.tobytes()


@router.post('/vehicles/{ident}/front-photo')
def front_photo(ident: str,version: int=Form(...),actor: str=Form('Оператор'),file: UploadFile=File(...)):
    content=file.file.read(15*1024*1024+1)
    if len(content)>15*1024*1024: raise HTTPException(413,'Фото должно быть меньше 15 МБ')
    image=cv2.imdecode(np.frombuffer(content,np.uint8),cv2.IMREAD_COLOR)
    if image is None: raise HTTPException(422,'Не удалось открыть изображение')
    with connect() as db:
        db.execute('BEGIN IMMEDIATE')
        old=find(db,ident)
        if old['version']!=version: raise HTTPException(409,'Обновите запись перед загрузкой фото')
        if old['status']=='Оплачен': raise HTTPException(409,'Оплаченная запись закрыта для изменений')
        record=dict(old,front_photo=uuid.uuid4().hex+'-front.jpg',version=old['version']+1,updated_at=now())
        if old.get('source'):
            record['source']={k:v for k,v in old['source'].items() if k not in
                {'front_camera','front_evidence','front_samples','synchronized_front_photo','camera_note'}}
        record.pop('plate_ocr',None)
        cv2.imwrite(str(DATA/record['front_photo']),image)
        db.execute('UPDATE vehicles SET version=?,data=? WHERE id=?',(record['version'],json.dumps(record,ensure_ascii=False),ident))
        event(db,record,actor,'front_photo','Добавлено фото спереди',old)
    return record


@router.get('/reports.csv')
def report_csv(status: str='',category: str='',date_from: date|None=None,date_to: date|None=None):
    where,args=filters(status=status,category=category,date_from=date_from,date_to=date_to)
    def safe(s):
        text=str(s if s is not None else '')
        return "'"+text if text.lstrip().startswith(('=','+','-','@')) or text.startswith(('\t','\r','\n')) else text
    def stream():
        out=io.StringIO()
        writer=csv.writer(out,delimiter=';',lineterminator='\r\n')
        writer.writerow(['Дата','Время','Гос. номер','Длина, м','Категория','Тариф, руб.','Статус','Оператор','Пункт тарифа','Грузоподъёмность, т'])
        yield '\ufeff'+out.getvalue()
        # Page limits apply only to the UI. Export all matching rows in bounded batches.
        with connect() as db:
            cursor=db.execute(f'SELECT * FROM vehicle_list WHERE {where} ORDER BY seq DESC',args)
            while batch:=cursor.fetchmany(250):
                out.seek(0);out.truncate(0)
                for r in batch:
                    writer.writerow([safe(v) for v in [r['created_at'][:10],r['created_at'][11:19],r['plate'],
                        '' if r['length_m'] is None else str(r['length_m']).replace('.',','),CATEGORIES[r['category']],
                        r['amount_rub'],r['status'],r['actor'],r['code'],
                        '' if r['load_capacity_t'] is None else str(r['load_capacity_t']).replace('.',',')]])
                yield out.getvalue()
    return StreamingResponse(stream(),media_type='text/csv; charset=utf-8',
                    headers={'Content-Disposition':'attachment; filename="ferry_report.csv"'})
