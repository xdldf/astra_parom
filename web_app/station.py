"""Persistent local operator station. Prices and transitions are server-owned."""
import base64
import csv
import hashlib
import io
import json
import re
import sqlite3
import uuid
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, ConfigDict, Field, model_validator

from vehicle_metrology.plate_text import normalize_plate
from vehicle_metrology.tariffs import CATEGORIES, catalog, quote
from web_app import workbench

router=APIRouter(prefix='/api/station')
DATA=Path(__file__).parent/'data'/'station'
DB=DATA/'station.sqlite3'
STATUSES=['Новый','Требует проверки','Подтвержден','Оплачен','Отклонён']


def now():
    return datetime.now(timezone(timedelta(hours=9))).isoformat(timespec='seconds')


def connect():
    DATA.mkdir(parents=True,exist_ok=True)
    db=sqlite3.connect(DB,timeout=15)
    db.row_factory=sqlite3.Row
    db.executescript('''
        CREATE TABLE IF NOT EXISTS vehicles(id TEXT PRIMARY KEY, source_key TEXT UNIQUE, version INTEGER NOT NULL, data TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS audit(id INTEGER PRIMARY KEY, vehicle_id TEXT, timestamp TEXT, actor TEXT, action TEXT, reason TEXT, before_json TEXT, after_json TEXT);
        CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY, value TEXT);
    ''')
    return db


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
    manual_rub: int | None=Field(None,ge=0,le=10000000,strict=True)
    actor: str=Field('Оператор',min_length=1,max_length=100)
    reason: str=Field('',max_length=1000)

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
    front_offset_seconds: float=Field(3,ge=-3600,le=3600,allow_inf_nan=False)


def new_record(fields,source=None):
    stamp=now()
    tariff=quote(fields.category,fields.length_m,fields.manual_rub)
    return dict(id=uuid.uuid4().hex,version=1,created_at=stamp,updated_at=stamp,
                plate=normalize_plate(fields.plate),category=fields.category,length_m=fields.length_m,
                measured_length_m=source.get('measured_length_m') if source else None,
                tariff=tariff,manual_rub=fields.manual_rub,status='Требует проверки',
                actor=fields.actor,source=source,side_photo=None,front_photo=None)


@router.get('/catalog')
def tariffs():
    return dict(categories=CATEGORIES,tariffs=catalog(),statuses=STATUSES,
                policy='Точные диапазоны Приложения №1, без округления и заполнения промежутков. Категория подтверждается оператором.')


@router.get('/health')
def health():
    import torch
    available=torch.cuda.is_available()
    return dict(online=True,gpu_available=available,gpu=torch.cuda.get_device_name(0) if available else None,
                detector='YOLO26n',front_camera=(workbench.DATA.parent/'camera-pair.json').exists(),camera_source='recorded_video_pair',plate_recognition=True)


@router.post('/quote')
def calculate(fields: Fields):
    return quote(fields.category,fields.length_m,fields.manual_rub)


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
    result=workbench.render(payload.media_id,workbench.FrameRequest(profile=payload.profile,
            frame=payload.frame,boxes=[payload.bbox]))
    measured=result['detections'][0]
    front_image=None
    paired=None
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
    return persist_capture(payload,result,front_image,paired)


def persist_capture(payload,result,front_image=None,paired=None,front_samples=None,camera_note=None):
    measured=result['detections'][0]
    if measured['status'] in {'outside_road','clipped','waiting_for_line'}:
        raise HTTPException(422,'Автомобиль должен быть целиком в кадре, на дороге и у линии измерения (если она включена).')
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
    if front_samples:
        source['front_samples']=[]
        for frame_index,sample in front_samples:
            filename=uuid.uuid4().hex+'-front.jpg'
            DATA.mkdir(parents=True,exist_ok=True)
            if not cv2.imwrite(str(DATA/filename),sample):
                raise HTTPException(500,'Не удалось сохранить снимок номера')
            source['front_samples'].append(dict(frame=frame_index,photo=filename))
    record=new_record(Fields(category=category,length_m=measured['length_m'],actor=payload.actor),source)
    with connect() as db:
        db.execute('BEGIN IMMEDIATE')
        existing=db.execute('SELECT data FROM vehicles WHERE source_key=?',(source_key,)).fetchone()
        if existing:
            return json.loads(existing['data'])
        record['side_photo']=record['id']+'-side.jpg'
        image=cv2.imdecode(np.frombuffer(base64.b64decode(result['image']),np.uint8),cv2.IMREAD_COLOR)
        x,y,bw,bh=map(int,payload.bbox)
        margin=max(20,int(bw*.08))
        crop=image[max(0,y-margin):min(image.shape[0],y+bh+margin),max(0,x-margin):min(image.shape[1],x+bw+margin)]
        cv2.imwrite(str(DATA/record['side_photo']),crop)
        if front_image is not None:
            record['front_photo']=record['id']+'-front.jpg'
            cv2.imwrite(str(DATA/record['front_photo']),front_image)
            record['plate_ocr']={'state':'queued','candidates':[]}
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
        record=dict(old,plate_ocr={'state':'queued','candidates':[]},version=old['version']+1,updated_at=now())
        db.execute('UPDATE vehicles SET version=?,data=? WHERE id=?',(record['version'],json.dumps(record,ensure_ascii=False),ident))
    enqueue(ident)
    return record


def filtered(db,plate='',status='',category='',length_min=None,length_max=None,date_from=None,date_to=None):
    if status and status not in STATUSES:
        raise HTTPException(422,'Неизвестный статус')
    if category and category not in CATEGORIES:
        raise HTTPException(422,'Неизвестная категория')
    if date_from and date_to and date_from>date_to:
        raise HTTPException(422,'Начальная дата позже конечной')
    if length_min is not None and length_max is not None and length_min>length_max:
        raise HTTPException(422,'Минимальная длина больше максимальной')
    records=[]
    for row in db.execute('SELECT data FROM vehicles ORDER BY rowid DESC'):
        r=json.loads(row['data'])
        if normalize_plate(plate) not in normalize_plate(r['plate']): continue
        if status and r['status']!=status: continue
        if category and r['category']!=category: continue
        if length_min is not None and (r['length_m'] is None or r['length_m']<length_min): continue
        if length_max is not None and (r['length_m'] is None or r['length_m']>length_max): continue
        d=date.fromisoformat(r['created_at'][:10])
        if date_from and d<date_from: continue
        if date_to and d>date_to: continue
        records.append(r)
    return records


@router.get('/vehicles')
def vehicles(plate: str='',status: str='',category: str='',length_min: float|None=None,
             length_max: float|None=None,date_from: date|None=None,date_to: date|None=None):
    with connect() as db:
        rows=filtered(db,plate,status,category,length_min,length_max,date_from,date_to)
    return dict(rows=rows,count=len(rows),amount_rub=sum(r['tariff']['amount_rub'] or 0 for r in rows if r['status'] in {'Подтвержден','Оплачен'}),
                paid_rub=sum(r['tariff']['amount_rub'] or 0 for r in rows if r['status']=='Оплачен'),
                issues=sum(r['status'] in {'Требует проверки','Отклонён'} for r in rows))


@router.get('/vehicles/{ident}')
def vehicle(ident: str):
    with connect() as db:
        record=find(db,ident)
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
            if any(old[k]!=getattr(fields,k) for k in ('category','length_m','manual_rub')) or old['plate']!=normalize_plate(fields.plate):
                raise HTTPException(422,'Перед оплатой подтвердите изменения данных')
            record=dict(old,status='Оплачен')
        elif fields.action=='reject':
            if not fields.reason.strip():
                raise HTTPException(422,'Укажите причину отклонения')
            record=dict(old,status='Отклонён')
        else:
            changed=any(old[k]!=getattr(fields,k) for k in ('category','length_m','manual_rub')) or old['plate']!=normalize_plate(fields.plate)
            if (changed or fields.manual_rub is not None) and not fields.reason.strip():
                raise HTTPException(422,'Укажите причину изменения')
            tariff=quote(fields.category,fields.length_m,fields.manual_rub)
            if fields.action=='confirm' and tariff['amount_rub'] is None:
                raise HTTPException(422,'Тариф не определён. Уточните данные или задайте ручной тариф с причиной.')
            record=dict(old,plate=normalize_plate(fields.plate),category=fields.category,length_m=fields.length_m,
                        manual_rub=fields.manual_rub,tariff=tariff,
                        status='Подтвержден' if fields.action=='confirm' else 'Требует проверки')
        record.update(version=old['version']+1,updated_at=now(),actor=fields.actor)
        db.execute('UPDATE vehicles SET version=?,data=? WHERE id=?',(record['version'],json.dumps(record,ensure_ascii=False),ident))
        if fields.action=='confirm':
            db.execute("INSERT OR REPLACE INTO settings VALUES('client_vehicle',?)",(ident,))
        elif fields.action in {'edit','reject'}:
            db.execute("DELETE FROM settings WHERE key='client_vehicle' AND value=?",(ident,))
        event(db,record,fields.actor,fields.action,fields.reason,old)
    return record


@router.get('/client')
def client_screen():
    with connect() as db:
        row=db.execute("SELECT value FROM settings WHERE key='client_vehicle'").fetchone()
        record=find(db,row['value']) if row else None
        if record and record['status'] not in {'Подтвержден','Оплачен'}: record=None
        return dict(vehicle=record)


@router.get('/photos/{filename}')
def photo(filename: str):
    if not re.fullmatch(r'[0-9a-f]{32}-(?:side|front|plate)\.jpg',filename):
        raise HTTPException(404,'Фото не найдено')
    path=DATA/filename
    if not path.is_file(): raise HTTPException(404,'Фото не найдено')
    return FileResponse(path)


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
        record=dict(old,front_photo=ident+'-front.jpg',version=old['version']+1,updated_at=now())
        cv2.imwrite(str(DATA/record['front_photo']),image)
        db.execute('UPDATE vehicles SET version=?,data=? WHERE id=?',(record['version'],json.dumps(record,ensure_ascii=False),ident))
        event(db,record,actor,'front_photo','Добавлено фото спереди',old)
    return record


@router.get('/reports.csv')
def report_csv(status: str='',category: str='',date_from: date|None=None,date_to: date|None=None):
    with connect() as db:
        rows=filtered(db,status=status,category=category,date_from=date_from,date_to=date_to)
    out=io.StringIO()
    writer=csv.writer(out,delimiter=';',lineterminator='\r\n')
    writer.writerow(['Дата','Время','Гос. номер','Длина, м','Категория','Тариф, руб.','Статус','Оператор','Пункт тарифа'])
    def safe(s):
        text=str(s if s is not None else '')
        return "'"+text if text.lstrip().startswith(('=','+','-','@')) or text.startswith(('\t','\r','\n')) else text
    for r in rows:
        writer.writerow([safe(v) for v in [r['created_at'][:10],r['created_at'][11:19],r['plate'],
            '' if r['length_m'] is None else str(r['length_m']).replace('.',','),CATEGORIES[r['category']],
            r['tariff']['amount_rub'],r['status'],r['actor'],r['tariff']['code']]])
    return Response('\ufeff'+out.getvalue(),media_type='text/csv; charset=utf-8',
                    headers={'Content-Disposition':'attachment; filename="ferry_report.csv"'})
