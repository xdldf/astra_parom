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
    assert len(catalog())==26
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


@pytest.mark.parametrize('capacity,price,code',[
    (15.001,9200,'6.6'),(24,9200,'6.6'),(24.001,10950,'6.7'),
    (30,10950,'6.7'),(30.001,14250,'6.8'),(60,14250,'6.8')])
def test_september_capacity_tariffs(capacity,price,code):
    result=quote('truck_capacity',4,load_capacity_t=capacity)
    assert (result['amount_rub'],result['code'])==(price,code)
    assert result['boundary_m'] is None


@pytest.mark.parametrize('capacity',[None,15,24.0005,30.0005])
def test_capacity_gaps_and_missing_capacity_are_not_inferred(capacity):
    assert quote('truck_capacity',20,load_capacity_t=capacity)['amount_rub'] is None


def test_truck_trailer_and_capacity_persistence(client):
    assert quote('truck_trailer',11.9)['amount_rub'] is None
    assert quote('truck_trailer',11.901)['amount_rub']==9000
    r=client.post('/api/station/vehicles',json={'category':'truck_capacity','length_m':8,'load_capacity_t':25}).json()
    assert r['tariff']['amount_rub']==10950
    payload=edit_payload(r);payload['load_capacity_t']=25
    confirmed=client.post('/api/station/vehicles/'+r['id'],json=payload).json()
    assert confirmed['status']=='Подтвержден'
    payload=edit_payload(confirmed,'pay');payload['load_capacity_t']=26
    assert client.post('/api/station/vehicles/'+r['id'],json=payload).status_code==422
    payload['load_capacity_t']=25
    assert client.post('/api/station/vehicles/'+r['id'],json=payload).json()['status']=='Оплачен'


def seed_records(count):
    records=[]
    with station.connect() as db:
        for i in range(count):
            r=station.new_record(station.Fields(category='motorcycle',plate=str(i)))
            r['status']='Оплачен'
            r['source']={'calibration':{'large_fixture':'x'*20000}}
            db.execute('INSERT INTO vehicles VALUES(?,?,?,?)',(r['id'],None,1,json.dumps(r)))
            records.append(r)
    return records


def test_bounded_projection_paging_full_totals_and_export(client):
    records=seed_records(301)
    response=client.get('/api/station/vehicles');first=response.json()
    assert len(first['rows'])==250 and first['count']==301 and first['has_more']
    assert len(response.content)<150000
    assert all('source' not in r for r in first['rows'])
    assert first['amount_rub']==first['paid_rub']==301*380
    assert first['rows'][0]['id']==records[-1]['id']
    headers={'If-None-Match':response.headers['etag']}
    assert client.get('/api/station/vehicles',headers=headers).status_code==304
    new=client.post('/api/station/vehicles',json={'category':'motorcycle'}).json()
    assert client.get('/api/station/vehicles',headers=headers).status_code==200
    second=client.get('/api/station/vehicles',params={'offset':250,'snapshot':first['snapshot']}).json()
    assert len(second['rows'])==51 and not second['has_more']
    assert not ({r['id'] for r in first['rows']} & {r['id'] for r in second['rows']})
    assert new['id'] not in {r['id'] for r in second['rows']}
    report=client.get('/api/station/reports.csv')
    assert len(list(csv.reader(io.StringIO(report.content.decode('utf-8-sig')),delimiter=';')))==303
    assert client.get('/api/station/vehicles?limit=251').status_code==422
    assert client.get('/api/station/vehicles?offset=-1').status_code==422
    assert client.get('/api/station/vehicles?plate=300').json()['count']==1


def test_parallel_operator_updates_and_payments(client):
    from concurrent.futures import ThreadPoolExecutor
    import threading
    r=client.post('/api/station/vehicles',json={'category':'motorcycle'}).json()
    url='/api/station/vehicles/'+r['id']
    for action in ('confirm','pay'):
        barrier=threading.Barrier(2)
        def update(actor):
            payload=edit_payload(r,action);payload['actor']=actor
            barrier.wait(timeout=5)
            return TestClient(app).post(url,json=payload)
        with ThreadPoolExecutor(max_workers=2) as pool:
            results=list(pool.map(update,['Первый','Второй']))
        assert sorted(x.status_code for x in results)==[200,409]
        r=next(x.json() for x in results if x.status_code==200)
    assert [h['action'] for h in client.get(url).json()['history']]==['pay','confirm','create']


def test_parallel_independent_records_and_customer_screens(client):
    from concurrent.futures import ThreadPoolExecutor
    records=[client.post('/api/station/vehicles',json={'category':'motorcycle'}).json() for _ in range(12)]
    def confirm(index):
        payload=edit_payload(records[index]);payload['station_id']='desk-'+str(index)
        return TestClient(app).post('/api/station/vehicles/'+records[index]['id'],json=payload)
    with ThreadPoolExecutor(max_workers=6) as pool:
        results=list(pool.map(confirm,range(len(records))))
    assert all(r.status_code==200 for r in results)
    for i,record in enumerate(records):
        assert client.get('/api/station/client?station_id=desk-'+str(i)).json()['vehicle']['id']==record['id']
    assert client.get('/api/station/client').json()['vehicle'] is None
    r=results[0].json();payload=edit_payload(r,'edit');payload['station_id']='desk-other'
    assert client.post('/api/station/vehicles/'+r['id'],json=payload).status_code==200
    assert client.get('/api/station/client?station_id=desk-0').json()['vehicle'] is None


def test_legacy_database_migrates_without_repricing(client):
    import sqlite3
    station.DATA.mkdir(parents=True)
    r=station.new_record(station.Fields(category='truck',length_m=9))
    r['tariff']['amount_rub']=1234
    r.pop('load_capacity_t')
    with sqlite3.connect(station.DB) as db:
        db.execute('CREATE TABLE vehicles(id TEXT PRIMARY KEY,source_key TEXT UNIQUE,version INTEGER NOT NULL,data TEXT NOT NULL)')
        db.execute('INSERT INTO vehicles VALUES(?,?,?,?)',(r['id'],None,1,json.dumps(r)))
    response=client.get('/api/station/vehicles').json()
    assert response['rows'][0]['tariff']['amount_rub']==1234
    assert client.get('/api/station/vehicles/'+r['id']).json()['tariff']['amount_rub']==1234
    with station.connect() as db:
        assert db.execute('PRAGMA journal_mode').fetchone()[0]=='wal'
    with pytest.raises(sqlite3.ProgrammingError):db.execute('SELECT 1')


def test_ocr_invalidates_list_and_detail_etags(client):
    from web_app import plates
    r=client.post('/api/station/vehicles',json={}).json();url='/api/station/vehicles/'+r['id']
    listing=client.get('/api/station/vehicles');detail=client.get(url)
    plates.save_result(r['id'],{'state':'review','candidates':[{'text':'А123ВС14'}]})
    latest=client.get('/api/station/vehicles',headers={'If-None-Match':listing.headers['etag']})
    assert latest.status_code==200
    assert latest.json()['rows'][0]['plate_ocr']['candidates'][0]['text']=='А123ВС14'
    assert client.get(url,headers={'If-None-Match':detail.headers['etag']}).status_code==200



def test_simultaneous_capture_is_idempotent_and_has_no_orphan_photos(client,monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    monkeypatch.setattr(workbench,'read_frame',lambda *_:np.zeros((500,600,3),np.uint8))
    payload={'media_id':'parallel-test','profile':{'image_size':[600,500],
             'polygon':[[20,200],[580,200],[580,450],[20,450]],
             'references':[{'bbox':[100,150,100,75],'length_m':5}]},
             'frame':0,'bbox':[100,150,100,75],'label':'car'}
    with ThreadPoolExecutor(max_workers=4) as pool:
        results=list(pool.map(lambda _:TestClient(app).post('/api/station/capture',json=payload),range(4)))
    assert all(r.status_code==200 for r in results)
    assert len({r.json()['id'] for r in results})==1
    assert client.get('/api/station/vehicles').json()['count']==1
    assert len(list(station.DATA.glob('*.jpg')))==1


def test_customer_etag_is_scoped_to_desk_and_changes_with_publication(client):
    first=client.get('/api/station/client?station_id=one')
    r=client.post('/api/station/vehicles',json={'category':'motorcycle'}).json()
    payload=edit_payload(r);payload['station_id']='two'
    assert client.post('/api/station/vehicles/'+r['id'],json=payload).status_code==200
    assert client.get('/api/station/client?station_id=one',headers={'If-None-Match':first.headers['etag']}).status_code==304
    displayed=client.get('/api/station/client?station_id=two')
    assert displayed.json()['vehicle']['id']==r['id']
    assert 'source' not in displayed.json()['vehicle']
    assert displayed.headers['etag']!=first.headers['etag']


def test_list_photo_references_and_small_thumbnail(client):
    record=client.post('/api/station/vehicles',json={'category':'car','length_m':4}).json()
    filename=record['id']+'-side.jpg'
    pixels=np.full((600,1200,3),170,dtype=np.uint8)
    cv2.imwrite(str(station.DATA/filename),pixels)
    with station.connect() as db:
        record['side_photo']=filename
        db.execute('UPDATE vehicles SET data=? WHERE id=?',(json.dumps(record),record['id']))
    row=client.get('/api/station/vehicles').json()['rows'][0]
    assert row['side_photo']==filename
    assert row['front_photo'] is None
    assert 'source' not in row
    original=client.get('/api/station/photos/'+filename)
    thumb=client.get('/api/station/photos/'+filename+'?thumbnail=true&v=1')
    assert thumb.status_code==200
    decoded=cv2.imdecode(np.frombuffer(thumb.content,np.uint8),cv2.IMREAD_COLOR)
    assert decoded.shape[:2]==(80,160)
    assert len(thumb.content)<len(original.content)
    assert 'max-age' in thumb.headers['cache-control']
    assert client.get('/api/station/photos/invalid.jpg?thumbnail=true').status_code==404


def test_long_truck_requires_category_choice_not_automatic_reclassification(client):
    q=quote('truck',17.16)
    assert q['amount_rub'] is None
    assert 'road_train' in q['category_options']
    assert not any('Нужен ручной тариф' in w for w in q['warnings'])
    record=client.post('/api/station/vehicles',json={'category':'truck','length_m':17.16}).json()
    assert record['category']=='truck' and record['tariff']['amount_rub'] is None
    url='/api/station/vehicles/'+record['id']
    payload=edit_payload(record)
    assert client.post(url,json=payload).status_code==422
    payload['category']='road_train'
    payload['reason']='Уточнение типа состава'
    response=client.post(url,json=payload)
    assert response.status_code==200
    assert response.json()['tariff']['amount_rub']==14000
    assert response.json()['tariff']['code']=='7.4'


def test_video_capture_uses_only_matching_session_front_history(client,monkeypatch):
    from types import SimpleNamespace
    from web_app import video_stream,plates
    from web_app.ip_cameras import Packet
    from web_app.front_history import FrontHistory
    monkeypatch.setattr(plates,'enqueue',lambda _:None)
    monkeypatch.setattr(plates,'front_vehicle',lambda image:[0,0,600,500])
    monkeypatch.setattr(workbench,'read_frame',lambda key,index:np.full((500,600,3),40,np.uint8))
    for name in ('front','side'):
        monkeypatch.setitem(workbench.media,name,{'kind':'video','fps':10,'frames':300})
    history=FrontHistory()
    jpeg=cv2.imencode('.jpg',np.full((500,600,3),200,np.uint8))[1].tobytes()
    candidate=dict(text='А123ВС14',confidence=.95,bbox=[100,300,200,350])
    history.observe(Packet(20,2,jpeg),[0,0,600,500],[candidate],0)
    for second in range(3,11):history.observe(Packet(second*10,second,b''),[0,0,600,500],[],0)
    monkeypatch.setitem(video_stream.sessions,'video-session',SimpleNamespace(
        front_history=history,request=SimpleNamespace(media_id='side',front_media_id='front')))
    payload=dict(media_id='side',front_media_id='front',front_session_id='video-session',front_offset_seconds=0,
        profile={'image_size':[600,500],'polygon':[[20,200],[580,200],[580,450],[20,450]],
                 'references':[{'bbox':[100,150,100,75],'length_m':5}]},
        frame=100,bbox=[100,150,100,75],label='truck')
    result=client.post('/api/station/capture',json=payload)
    assert result.status_code==200,result.text
    assert result.json()['source']['front_evidence']['offset_seconds']==-8
    assert cv2.imread(str(station.DATA/result.json()['front_photo'])).mean()==pytest.approx(200,abs=1)
    video_stream.sessions['video-session'].request.media_id='other-recording'
    payload['frame']=101
    result=client.post('/api/station/capture',json=payload)
    assert result.status_code==200,result.text
    assert 'front_evidence' not in result.json()['source']
    assert cv2.imread(str(station.DATA/result.json()['front_photo'])).mean()==pytest.approx(40,abs=1)
