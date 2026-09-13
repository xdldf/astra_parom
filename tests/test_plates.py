import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from web_app import plates, station
from web_app.main import app


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(station,'DATA',tmp_path)
    monkeypatch.setattr(station,'DB',tmp_path/'station.sqlite3')
    monkeypatch.setattr(plates,'enqueue',lambda _:None)
    return TestClient(app)


def with_photo(client, plate=''):
    r=client.post('/api/station/vehicles',json={'plate':plate,'length_m':4.5}).json()
    _,jpg=cv2.imencode('.jpg',np.full((80,160,3),128,np.uint8))
    return client.post('/api/station/vehicles/'+r['id']+'/front-photo',
        data={'version':r['version'],'actor':'test'},files={'file':('front.jpg',jpg.tobytes(),'image/jpeg')}).json()


def test_recognition_evidence_keeps_operator_number(client,monkeypatch):
    r=with_photo(client,'А777АА14')
    monkeypatch.setattr(plates,'recognize',lambda image:[dict(text='A123BC14',raw_text='A123BC14',confidence=.95,detection_confidence=.9,bbox=[20,20,100,55])])
    url='/api/station/vehicles/'+r['id']
    queued=client.post(url+'/recognize-plate',json={}).json()
    assert queued['plate_ocr']['state']=='queued'
    plates.process_record(r['id'])
    result=client.get(url).json()
    assert result['plate']=='А777АА14'
    assert result['version']==queued['version']+1
    candidate=result['plate_ocr']['candidates'][0]
    assert candidate['text']=='A123BC14' and candidate['votes']==1
    assert client.get('/api/station/photos/'+candidate['photo']).status_code==200
    assert result['history'][0]['action']=='plate_recognition'


def test_ambiguous_does_not_assign_plate(client,monkeypatch):
    r=with_photo(client)
    monkeypatch.setattr(plates,'recognize',lambda image:[dict(text=t,raw_text=t,confidence=.99,detection_confidence=.9,bbox=[20,20,100,55]) for t in ['A123BC14','B456EE14']])
    plates.process_record(r['id'])
    result=client.get('/api/station/vehicles/'+r['id']).json()
    assert result['plate']==''
    assert len(result['plate_ocr']['candidates'])==2
    assert result['plate_ocr']['association']=='operator_review'


def test_error_no_plate_and_closed_records(client,monkeypatch):
    r=with_photo(client)
    url='/api/station/vehicles/'+r['id']
    monkeypatch.setattr(plates,'recognize',lambda image:[])
    plates.process_record(r['id'])
    assert client.get(url).json()['plate_ocr']['state']=='not_found'
    def fail(image): raise RuntimeError('CUDA unavailable')
    monkeypatch.setattr(plates,'recognize',fail)
    plates.process_record(r['id'])
    result=client.get(url).json()
    assert result['plate_ocr']['state']=='error'
    confirmed=client.post(url,json={'version':result['version'],'action':'confirm','plate':'','category':'car','length_m':4.5}).json()
    assert confirmed['status']=='Подтвержден'
    assert client.post(url+'/recognize-plate',json={}).status_code==409
    plates.save_result(r['id'],{'state':'review'})
    assert client.get(url).json()['version']==confirmed['version']


def test_confidence_excludes_padding():
    assert plates.ocr_confidence(SimpleNamespace(text='AB',confidence=[.5,.7,1,1,1]))==pytest.approx(.6)


def test_no_source_rejected(client):
    r=client.post('/api/station/vehicles',json={}).json()
    assert client.post('/api/station/vehicles/'+r['id']+'/recognize-plate',json={}).status_code==422


def test_cyrillic_input_and_search(client):
    r=client.post('/api/station/vehicles',json={'plate':' y123xC14 '}).json()
    assert r['plate']=='У123ХС14'
    assert client.get('/api/station/vehicles',params={'plate':'Y123XC'}).json()['count']==1


def test_foreground_plate_no_rear_fallback():
    def detection(x1,y1,x2,y2):
        return SimpleNamespace(bounding_box=SimpleNamespace(x1=x1,y1=y1,x2=x2,y2=y2))
    rear=detection(110,110,160,130)
    near=detection(260,700,360,750)
    assert plates.foreground_plate([rear,near],[100,200,500,600])==[near]
    assert plates.foreground_plate([rear],[100,200,500,600])==[]
    assert not plates.same_vehicle([100,200,500,600],[10,10,50,50])
