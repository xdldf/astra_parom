import csv
import io
import json

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from vehicle_metrology.tariffs import quote,catalog
from web_app.main import app
from web_app import station,workbench


@pytest.mark.parametrize('category,length,price,code',[
    ('car',4,1390,'1.1'),('car',4.01,1650,'1.2'),('car',4.6,1650,'1.2'),
    ('car',4.61,1900,'1.3'),('car',5,1900,'1.3'),('car',5.001,2200,'1.4'),
    ('car_trailer',None,1200,'2'),('boat_trailer',None,2000,'3'),
    ('bus',5,2800,'4.1'),('bus',5.01,3600,'4.2'),('bus',10,3600,'4.2'),('bus',10.02,5000,'4.3'),
    ('motorcycle',None,380,'5'),('truck',7.09,2700,'6.1'),('truck',7.1,5350,'6.2'),
    ('truck',8.09,5350,'6.2'),('truck',8.1,6450,'6.3'),('truck',10.09,6450,'6.3'),
    ('truck',10.1,8500,'6.4'),('truck',11.9,8500,'6.4'),('tractor',7.1,5650,'7.1'),
    ('tractor',11.9,5650,'7.1'),('road_train',12,9050,'7.2'),('road_train',12.1,10800,'7.3'),
    ('road_train',15,10800,'7.3'),('road_train',15.1,14000,'7.4'),('oversize',None,22600,'7.5'),
    ('lowbed',12,10750,'8.1'),('lowbed',12.1,11850,'8.2'),('lowbed',14,11850,'8.2'),
    ('lowbed',14.1,14300,'8.3'),('lowbed',16,14300,'8.3')])
def test_all_tariff_rows(category,length,price,code):
    result=quote(category,length)
    assert result['amount_rub']==price
    assert result['code']==code


@pytest.mark.parametrize('category,length',[
    ('car',4.005),('car',4.605),('bus',5.005),('bus',10.01),('truck',7.095),
    ('truck',11.91),('tractor',7),('tractor',12),('road_train',12.05),
    ('road_train',15.05),('lowbed',14.05),('lowbed',16.01),('car',None)])
def test_gaps_are_not_invented_or_rounded(category,length):
    result=quote(category,length)
    assert result['amount_rub'] is None
    assert result['warnings']


def test_catalog_and_boundary_warning():
    assert len(catalog())==22
    assert quote('car',4.92)['boundary_m']==5
    assert quote('car',4.3)['boundary_m'] is None
    assert quote('car',4.605,1650)['mode']=='manual'


@pytest.fixture
def client(tmp_path,monkeypatch):
    monkeypatch.setattr(station,'DATA',tmp_path/'station')
    monkeypatch.setattr(station,'DB',tmp_path/'station'/'test.sqlite3')
    return TestClient(app)


def edit_payload(record,action='confirm',**changes):
    return dict(plate=record['plate'],category=record['category'],length_m=record['length_m'],
                manual_rub=record['manual_rub'],version=record['version'],action=action,actor='Тест',**changes)


def test_confirm_client_pay_persistence_and_audit(client):
    assert client.get('/api/station/vehicles').json()['rows']==[]
    record=client.post('/api/station/vehicles',json={'plate':'а123вс14','category':'car','length_m':4.92}).json()
    assert record['tariff']['amount_rub']==1900
    assert client.get('/api/station/client').json()['vehicle'] is None
    url='/api/station/vehicles/'+record['id']
    confirmed=client.post(url,json=edit_payload(record)).json()
    assert confirmed['status']=='Подтвержден'
    assert client.get('/api/station/client').json()['vehicle']['id']==record['id']
    assert client.post(url,json=edit_payload(record)).status_code==409
    paid=client.post(url,json=edit_payload(confirmed,'pay')).json()
    assert paid['status']=='Оплачен'
    assert client.post(url,json=edit_payload(paid,'edit')).status_code==409
    summary=client.get('/api/station/vehicles').json()
    assert summary['amount_rub']==summary['paid_rub']==1900
    # New request/connection reads the same database and full audit trail.
    detail=client.get(url).json()
    assert [h['action'] for h in detail['history']]==['pay','confirm','create']


def test_manual_override_reason_and_rejection_clear_client(client):
    record=client.post('/api/station/vehicles',json={'length_m':4.605}).json()
    url='/api/station/vehicles/'+record['id']
    assert client.post(url,json=edit_payload(record)).status_code==422
    payload=edit_payload(record);payload['manual_rub']=1650
    assert client.post(url,json=payload).status_code==422
    payload['reason']='Уточнение границы по Приложению №1'
    confirmed=client.post(url,json=payload).json()
    assert confirmed['tariff']['amount_rub']==1650
    rejected=client.post(url,json=edit_payload(confirmed,'reject',reason='Повторное обнаружение')).json()
    assert rejected['status']=='Отклонён'
    assert client.get('/api/station/client').json()['vehicle'] is None
    assert client.get('/api/station/vehicles').json()['amount_rub']==0


def test_report_filters_and_csv_formula_escape(client):
    client.post('/api/station/vehicles',json={'plate':'=1+1','category':'motorcycle'})
    client.post('/api/station/vehicles',json={'plate':'Б222','category':'bus','length_m':8})
    result=client.get('/api/station/vehicles',params={'category':'motorcycle'}).json()
    assert result['count']==1
    assert client.get('/api/station/vehicles',params={'length_min':7}).json()['count']==1
    assert client.get('/api/station/vehicles',params={'date_from':'2999-01-01'}).json()['count']==0
    assert client.get('/api/station/vehicles',params={'date_from':'2026-12-01','date_to':'2026-01-01'}).status_code==422
    report=client.get('/api/station/reports.csv',params={'category':'motorcycle'})
    assert report.content.startswith(b'\xef\xbb\xbf')
    rows=list(csv.reader(io.StringIO(report.content.decode('utf-8-sig')),delimiter=';'))
    assert len(rows)==2
    assert rows[1][2]=="'=1+1"
    assert rows[1][5]=='380'


def test_capture_recomputes_measurement_stores_photo_and_deduplicates(client,monkeypatch):
    monkeypatch.setattr(workbench,'read_frame',lambda *_:np.zeros((500,600,3),np.uint8))
    polygon=[[20,200],[580,200],[580,450],[20,450]]
    references=[{'bbox':[100,150,100,75],'length_m':5},{'bbox':[100,300,200,125],'length_m':5}]
    payload={'media_id':'test','profile':{'image_size':[600,500],'polygon':polygon,'references':references,'measurement_line_x':150},
             'frame':0,'bbox':[100,150,100,75],'label':'car','source':'yolo26x'}
    first=client.post('/api/station/capture',json=payload)
    assert first.status_code==200,first.text
    record=first.json()
    assert record['measured_length_m']==pytest.approx(5)
    assert record['tariff']['amount_rub']==1900
    assert client.get('/api/station/photos/'+record['side_photo']).status_code==200
    assert client.post('/api/station/capture',json=payload).json()['id']==record['id']
    assert client.get('/api/station/vehicles').json()['count']==1
    payload['profile']['measurement_line_x']=300
    assert client.post('/api/station/capture',json=payload).status_code==422


def test_front_photo_upload(client):
    record=client.post('/api/station/vehicles',json={}).json()
    _,image=cv2.imencode('.png',np.zeros((20,30,3),np.uint8))
    response=client.post('/api/station/vehicles/'+record['id']+'/front-photo',
                         data={'version':1,'actor':'Тест'},files={'file':('photo.png',image.tobytes(),'image/png')})
    assert response.status_code==200
    assert response.json()['version']==2
    assert client.get('/api/station/photos/'+response.json()['front_photo']).status_code==200


def test_synchronized_capture_saves_front_and_side(client,monkeypatch):
    from web_app import plates
    queued=[]
    monkeypatch.setattr(plates,'enqueue',queued.append)
    from web_app.video_stream import paired_frame
    assert paired_frame(250,25,20,3)==260
    for index in [0,1,3023,44900]:
        mapped=paired_frame(index,25,20.016546271258903,3)
        assert abs(mapped/20.016546271258903-index/25-3)<=.5/20.016546271258903
    calls=[]
    def read(key,index):
        calls.append((key,index))
        return np.full((500,600,3),120 if key=='front' else 0,np.uint8)
    monkeypatch.setattr(workbench,'read_frame',read)
    monkeypatch.setitem(workbench.media,'side',{'kind':'video','fps':25,'frames':1000})
    monkeypatch.setitem(workbench.media,'front',{'kind':'video','fps':20,'frames':1000})
    payload={'media_id':'side','front_media_id':'front','front_offset_seconds':3,
      'profile':{'image_size':[600,500],'polygon':[[20,200],[580,200],[580,450],[20,450]],
        'references':[{'bbox':[100,150,100,75],'length_m':5}]},
      'frame':250,'bbox':[100,150,100,75],'label':'car','source':'yolo26n'}
    response=client.post('/api/station/capture',json=payload)
    assert response.status_code==200,response.text
    record=response.json()
    assert queued==[record['id']]
    assert record['plate_ocr']['state']=='queued'
    assert ('front',260) in calls
    assert record['source']['front_camera']['front_seconds']==13
    assert record['source']['front_camera']['side_seconds']==10
    front=client.get('/api/station/photos/'+record['front_photo'])
    assert front.status_code==200
    decoded=cv2.imdecode(np.frombuffer(front.content,np.uint8),cv2.IMREAD_COLOR)
    assert decoded.shape==(500,600,3)
    assert decoded.mean()==pytest.approx(120,abs=1)
    assert client.get('/api/station/photos/'+record['side_photo']).status_code==200
    assert client.post('/api/station/capture',json=payload).json()['id']==record['id']
    payload['front_offset_seconds']=-20
    assert client.post('/api/station/capture',json=payload).status_code==422
