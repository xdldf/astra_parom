"""Persistent full-frame labeling queue and calibration JSON export."""
import json
import csv
import io
import threading
import uuid
from typing import Literal

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict, Field, model_validator

from web_app import station as st, workbench as wb
from web_app.calibration_references import geometry_key, PREFIX, invalidate

router = APIRouter(prefix='/api/station/evaluation')
export_lock = threading.Lock()


class Review(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    version: int = Field(ge=1)
    actor: str = Field(min_length=1,max_length=100)
    action: Literal['label','skip']
    actual_length_m: float | None = Field(None,gt=0,le=100)
    physical_vehicle_id: str | None = Field(None,max_length=100)
    dataset_role: Literal['calibration','validation'] = 'calibration'
    reference_source: Literal['physical_measurement','catalogue','unknown'] = 'unknown'
    reference_uncertainty_m: float | None = Field(None,gt=0,le=1)

    @model_validator(mode='after')
    def validate_review(self):
        if not self.actor.strip() or (self.action == 'label' and self.actual_length_m is None):
            raise ValueError('Введите оператора и фактическую длину автомобиля')
        self.physical_vehicle_id = (self.physical_vehicle_id or '').strip() or None
        if self.action == 'label' and self.dataset_role == 'validation' and (
                not self.physical_vehicle_id or self.reference_source != 'physical_measurement'
                or self.reference_uncertainty_m is None):
            raise ValueError('Для независимой проверки нужны ID автомобиля, физический замер и погрешность замера')
        return self


@router.get('')
def queue(state: Literal['pending','skipped','labeled']='pending', offset: int=Query(0,ge=0),
          limit: int=Query(25,ge=1,le=100)):
    where = "json_extract(data,'$.full_frame_photo') IS NOT NULL AND COALESCE(json_extract(data,'$.evaluation.state'),'pending')=?"
    with st.connect() as db:
        count=db.execute('SELECT COUNT(*) FROM vehicles WHERE '+where,(state,)).fetchone()[0]
        rows=db.execute('SELECT data FROM vehicles WHERE '+where+' ORDER BY rowid LIMIT ? OFFSET ?',
                        (state,limit,offset)).fetchall()
    records=[]
    for row in rows:
        r=json.loads(row['data'])
        records.append({k:r.get(k) for k in ('id','version','created_at','plate','measured_length_m',
                       'full_frame_photo','evaluation')})
        records[-1]['bbox']=r['source']['bbox']
        records[-1]['image_size']=r['source']['calibration']['image_size']
        records[-1]['measurement']=r['source']['measurement']
    return dict(rows=records,count=count,offset=offset,has_more=offset+len(records)<count)


@router.get('/validation.csv')
def validation_csv():
    """Physical reference values only; predictions never fill reference columns."""
    with st.connect() as db:
        rows=db.execute("SELECT data FROM vehicles WHERE json_extract(data,'$.evaluation.state')='labeled' AND json_extract(data,'$.evaluation.dataset_role')='validation' ORDER BY rowid").fetchall()
    output=io.StringIO(newline='')
    fields=['track_id','vehicle_id','length_m','uncertainty_m','reference_source','dataset_role']
    writer=csv.DictWriter(output,fieldnames=fields);writer.writeheader()
    for row in rows:
        record=json.loads(row['data']);label=record['evaluation']
        writer.writerow(dict(track_id=record['id'],vehicle_id=label['physical_vehicle_id'],
            length_m=label['actual_length_m'],uncertainty_m=label['reference_uncertainty_m'],
            reference_source=label['reference_source'],dataset_role='validation'))
    return Response(output.getvalue(),media_type='text/csv',headers={
        'Content-Disposition':'attachment; filename="validation-lengths.csv"'})


def samples_for(profile):
    key=geometry_key(profile)
    with st.connect() as db:
        rows=db.execute("SELECT data FROM vehicles WHERE json_extract(data,'$.evaluation.state')='labeled' AND json_extract(data,'$.evaluation.geometry')=? ORDER BY rowid",(key,)).fetchall()
    result=[]
    for row in rows:
        r=json.loads(row['data']);label=r['evaluation'];source=r['source']
        result.append(wb.EvaluationSample(vehicle_id=r['id'],bbox=source['bbox'],frame=source['frame'],
            measured_length_m=r.get('measured_length_m'),actual_length_m=label['actual_length_m'],
            full_frame_photo=r['full_frame_photo'],
            verified_by=label['actor'],verified_at=label['updated_at'],
            physical_vehicle_id=label.get('physical_vehicle_id'),
            dataset_role=label.get('dataset_role','calibration'),
            reference_source=label.get('reference_source','unknown'),
            reference_uncertainty_m=label.get('reference_uncertainty_m')))
    return result


@router.post('/{ident}')
def review(ident: str, payload: Review):
    with st.connect() as db:
        db.execute('BEGIN IMMEDIATE')
        old=st.find(db,ident)
        if old['version'] != payload.version:
            raise HTTPException(409,'Запись изменена. Обновите кадр перед сохранением.')
        if not old.get('full_frame_photo') or not old.get('source',{}).get('calibration'):
            raise HTTPException(422,'У этой записи нет полного кадра. Выберите новый проезд.')
        profile=wb.Profile.model_validate(old['source']['calibration'])
        key=geometry_key(profile)
        if payload.action == 'label' and payload.physical_vehicle_id:
            others=db.execute("SELECT data FROM vehicles WHERE id<>? AND json_extract(data,'$.evaluation.physical_vehicle_id')=?",
                              (ident,payload.physical_vehicle_id)).fetchall()
            other_labels=[json.loads(row['data']).get('evaluation',{}) for row in others]
            if any(label.get('dataset_assigned',label.get('state')=='labeled') and
                   label.get('dataset_role','calibration') != payload.dataset_role for label in other_labels):
                raise HTTPException(409,'Этот автомобиль уже относится к другой выборке. Все его проезды должны быть в одной выборке.')
        # Once exposed to fitting, a label cannot become independent evidence merely
        # by changing a select box. Keep the assignment sticky through skip/reopen.
        previous=old.get('evaluation',{})
        assigned=previous.get('dataset_assigned',previous.get('state')=='labeled')
        if not assigned and db.execute("SELECT 1 FROM audit WHERE vehicle_id=? AND action='evaluation_label' LIMIT 1",(ident,)).fetchone():
            assigned=True
        if assigned and payload.action == 'label' and previous.get('dataset_role','calibration') != payload.dataset_role:
            raise HTTPException(409,'Назначение выборки уже закреплено. Для независимой проверки используйте другой автомобиль.')
        if assigned and previous.get('physical_vehicle_id') and payload.action=='label' and previous['physical_vehicle_id'] != payload.physical_vehicle_id:
            raise HTTPException(409,'ID уже проверенного автомобиля нельзя менять: это нарушит разделение выборок.')
        if payload.action=='label' and payload.dataset_role=='validation' and db.execute(
                "SELECT 1 FROM audit WHERE vehicle_id=? AND action='calibration_reference' LIMIT 1",(ident,)).fetchone():
            raise HTTPException(409,'Автомобиль уже участвовал в проверке эталонов калибровки. Выберите новый независимый автомобиль.')
        record=dict(old)
        label=dict(state='labeled' if payload.action=='label' else 'skipped',
                   actual_length_m=payload.actual_length_m if payload.action=='label' else None,
                   actor=payload.actor.strip(),updated_at=st.now(),geometry=key,reference_added=False,
                   physical_vehicle_id=payload.physical_vehicle_id,
                   dataset_role=payload.dataset_role,reference_source=payload.reference_source,
                   reference_uncertainty_m=payload.reference_uncertainty_m,
                   dataset_assigned=assigned or payload.action=='label')
        if previous and payload.action == 'skip':
            for field in ('physical_vehicle_id','dataset_role','reference_source','reference_uncertainty_m'):
                label[field]=previous.get(field,label[field])
        record['evaluation']=label
        # This explicit length review is independent of cashier confirmation.
        # Keep unsuitable outlines as evaluation data without using them as scale.
        if (payload.action=='label' and payload.dataset_role=='calibration'
                and not profile.survey_calibration and old['status']!='Отклонён'):
            try:
                ref=wb.Reference(bbox=old['source']['bbox'],length_m=payload.actual_length_m,
                    frame=old['source']['frame'],vehicle_id=ident,verified_by=label['actor'],verified_at=label['updated_at'])
                wb.Profile.model_validate({**profile.model_dump(),'metric_rulers':[],
                    'references':[dict(bbox=ref.bbox,length_m=ref.length_m,frame=ref.frame)]})
            except ValueError:
                pass
            else:
                item=dict(geometry=key,reference=ref.model_dump())
                record['calibration_reference']=item
                db.execute('INSERT OR REPLACE INTO settings VALUES(?,?)',(PREFIX+ident,json.dumps(item)))
                label['reference_added']=True
        if not label['reference_added']:
            record.pop('calibration_reference',None)
            db.execute('DELETE FROM settings WHERE key=?',(PREFIX+ident,))
        record.update(version=old['version']+1,updated_at=st.now())
        db.execute('UPDATE vehicles SET version=?,data=? WHERE id=?',(record['version'],json.dumps(record,ensure_ascii=False),ident))
        st.event(db,record,payload.actor,'evaluation_'+payload.action,
                 'Проверена фактическая длина' if payload.action=='label' else 'Кадр пропущен',old)
    invalidate()
    # SQLite is the review source of truth; refresh an atomic JSON snapshot too.
    # Serialize exports so an older request cannot overwrite a newer label.
    with export_lock:
        from web_app.calibration_references import merge
        data=merge(profile).model_dump_json(indent=2)
        directory=st.DATA/'calibrations'
        directory.mkdir(exist_ok=True)
        path=directory/(key+'.json')
        temporary=directory/(key+'.'+uuid.uuid4().hex+'.tmp')
        try:
            temporary.write_text(data,encoding='utf-8')
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    return record


@router.get('/{ident}/calibration.json')
def export(ident: str):
    from web_app.calibration_references import merge
    with st.connect() as db:
        record=st.find(db,ident)
    source=record.get('source') or {}
    if not source.get('calibration'):
        raise HTTPException(422,'Нет исходной калибровки')
    profile=merge(wb.Profile.model_validate(source['calibration']))
    return Response(profile.model_dump_json(indent=2),media_type='application/json',
        headers={'Content-Disposition':'attachment; filename="calibration-reviewed.json"'})
