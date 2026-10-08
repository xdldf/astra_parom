import json

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from web_app.main import app
from web_app import station as st, workbench as wb


@pytest.fixture
def setup(tmp_path,monkeypatch):
    monkeypatch.setattr(st,'DATA',tmp_path)
    monkeypatch.setattr(st,'DB',tmp_path/'station.sqlite3')
    raw=np.zeros((500,600,3),np.uint8)
    raw[:80,:80]=[10,180,250]  # Evidence outside the vehicle crop must survive.
    monkeypatch.setattr(wb,'read_frame',lambda *args:raw.copy())
    profile=wb.Profile(image_size=(600,500),polygon=[(20,200),(580,200),(580,450),(20,450)],
        references=[dict(bbox=[100,150,100,75],length_m=5)],measurement_line_x=300)
    payload=dict(media_id='evaluation-test',profile=profile.model_dump(),frame=7,
                 bbox=[120,150,100,75],review_fallback=True)
    client=TestClient(app)
    response=client.post('/api/station/capture',json=payload)
    assert response.status_code==200,response.text
    return client,profile,response.json(),tmp_path,payload


def review(client,record,action='label',length=4.725,**overrides):
    payload=dict(version=record['version'],actor='Reviewer',action=action,actual_length_m=length)
    payload.update(overrides)
    return client.post('/api/station/evaluation/'+record['id'],json=payload)


def test_off_line_capture_preserves_entire_images_and_original_estimate(setup):
    client,profile,record,path,payload=setup
    assert record['length_m']==pytest.approx(5)
    assert record['source']['measurement']['approximate']
    assert 'missed_measurement_line' in record['source']['measurement']['quality_reasons']
    response=client.get('/api/station/photos/'+record['full_frame_photo'])
    assert response.status_code==200
    image=cv2.imdecode(np.frombuffer(response.content,np.uint8),cv2.IMREAD_COLOR)
    assert image.shape==(500,600,3)
    assert image[20,20,1]==pytest.approx(180,abs=3)
    assert 'raw_frame_photo' not in record
    assert not list(path.glob('*-raw.jpg'))
    assert client.post('/api/station/capture',json=payload).json()['id']==record['id']
    assert len(list(path.glob('*.jpg')))==2
    payload['profile']['measurement_mode']='strict';payload['frame']=8
    strict=client.post('/api/station/capture',json=payload).json()
    assert strict['length_m'] is None and strict['full_frame_photo']
    assert strict['source']['measurement']['status']=='capture_review'


def test_label_exports_real_length_without_rewriting_measurement_or_cashier_state(setup):
    client,profile,record,path,_=setup
    assert client.get('/api/station/evaluation').json()['count']==1
    response=review(client,record)
    assert response.status_code==200,response.text
    saved=response.json()
    assert saved['status']==record['status'] and saved['length_m']==record['length_m']
    assert saved['evaluation']['reference_added']
    assert client.get('/api/station/evaluation').json()['count']==0
    assert client.get('/api/station/evaluation?state=labeled').json()['count']==1
    data=client.get('/api/station/evaluation/'+record['id']+'/calibration.json').json()
    assert data['references'][-1]['length_m']==4.725
    sample=data['evaluation_samples'][0]
    assert sample['measured_length_m']==5 and sample['actual_length_m']==4.725
    assert sample['full_frame_photo']==record['full_frame_photo']
    assert sample['bbox']==record['source']['bbox']
    assert sample['verified_by']=='Reviewer'
    assert json.loads(next((path/'calibrations').glob('*.json')).read_text())==data
    # The usual calibration download/save flow includes the review too.
    result=client.post('/api/workbench/approved-references',json=profile.model_dump()).json()
    assert result==data
    assert review(client,record,length=7).status_code==409
    assert client.get('/api/station/evaluation/'+record['id']+'/calibration.json').json()==data


def test_skip_is_persistent_reversible_and_revokes_stale_json_reference(setup):
    client,profile,record,path,_=setup
    labeled=review(client,record).json()
    old=client.get('/api/station/evaluation/'+record['id']+'/calibration.json').json()
    skipped=review(client,labeled,'skip',None).json()
    assert skipped['evaluation']['state']=='skipped'
    assert client.get('/api/station/evaluation?state=skipped').json()['rows'][0]['id']==record['id']
    data=client.post('/api/workbench/approved-references',json=old).json()
    assert len(data['references'])==1 and not data['evaluation_samples']
    restored=review(client,skipped,length=4.8).json()
    assert restored['evaluation']['state']=='labeled'
    data=client.get('/api/station/evaluation/'+record['id']+'/calibration.json').json()
    assert data['evaluation_samples'][0]['actual_length_m']==4.8
    assert len(data['references'])==2


@pytest.mark.parametrize('length',[None,0,-1,101,'not-a-length'])
def test_invalid_real_lengths_do_not_advance_queue(setup,length):
    client,_,record,_,_=setup
    assert review(client,record,length=length).status_code==422
    assert client.get('/api/station/evaluation').json()['count']==1


def test_labels_stay_with_compatible_camera_geometry(setup):
    client,profile,record,_,_=setup
    review(client,record)
    changed=profile.model_dump();changed['lens']['k1']=.1
    data=client.post('/api/workbench/approved-references',json=changed).json()
    assert not data['evaluation_samples'] and len(data['references'])==1
    assert client.get('/evaluation').status_code==200


def test_clipped_detection_is_saved_for_evaluation_but_never_used_as_scale(setup):
    client,_,_,_,payload=setup
    payload.update(frame=10,bbox=[0,150,100,75])
    response=client.post('/api/station/capture',json=payload)
    assert response.status_code==200,response.text
    record=response.json()
    assert record['measured_length_m'] is None
    assert record['source']['measurement']['quality_reasons']==['clipped']
    saved=review(client,record).json()
    assert not saved['evaluation']['reference_added']
    data=client.get('/api/station/evaluation/'+record['id']+'/calibration.json').json()
    assert data['evaluation_samples'][0]['actual_length_m']==4.725
    assert len(data['references'])==1


VALIDATION=dict(dataset_role='validation',physical_vehicle_id='field-car-01',
                reference_source='physical_measurement',reference_uncertainty_m=.01)


def test_validation_sample_exported_but_excluded_from_every_calibration_path(setup):
    client,profile,record,path,_=setup
    saved=review(client,record,**VALIDATION).json()
    assert saved['evaluation']['dataset_assigned']
    assert not saved['evaluation']['reference_added']
    exported=client.get('/api/station/evaluation/'+record['id']+'/calibration.json').json()
    assert len(exported['references'])==1
    sample=exported['evaluation_samples'][0]
    for key,value in VALIDATION.items(): assert sample[key]==value
    import csv,io
    rows=list(csv.DictReader(io.StringIO(client.get('/api/station/evaluation/validation.csv').text)))
    assert len(rows)==1 and rows[0]['track_id']==record['id']
    assert float(rows[0]['length_m'])==4.725 and float(rows[0]['uncertainty_m'])==.01
    assert rows[0]['vehicle_id']=='field-car-01'
    response=client.post('/api/station/vehicles/'+record['id']+'/calibration-reference',json=dict(
        version=saved['version'],actor='Reviewer',enabled=True,actual_length_m=4.725,verified=True))
    assert response.status_code==409
    skipped=review(client,saved,'skip',None).json()
    assert len(client.get('/api/station/evaluation/validation.csv').text.splitlines())==1
    assert skipped['evaluation']['dataset_role']=='validation'
    assert review(client,skipped).status_code==409  # Cannot turn a held-out sample into fitting data.
    restored=review(client,skipped,**VALIDATION).json()
    assert not restored['evaluation']['reference_added']
    assert len(client.post('/api/workbench/approved-references',json=exported).json()['references'])==1


@pytest.mark.parametrize('change',[{'physical_vehicle_id':None},{'physical_vehicle_id':' '},
    {'reference_source':'catalogue'},{'reference_uncertainty_m':None},{'reference_uncertainty_m':0}])
def test_validation_requires_independent_reference_metadata(setup,change):
    client,_,record,_,_=setup
    assert review(client,record,**{**VALIDATION,**change}).status_code==422
    assert client.get('/api/station/evaluation').json()['count']==1


def test_physical_vehicle_cannot_cross_dataset_split_or_be_renamed(setup):
    client,_,record,_,payload=setup
    saved=review(client,record,**VALIDATION).json()
    payload['frame']=30
    second=client.post('/api/station/capture',json=payload).json()
    assert review(client,second,physical_vehicle_id='field-car-01').status_code==409
    assert review(client,second,**VALIDATION).status_code==200
    assert review(client,saved,**{**VALIDATION,'physical_vehicle_id':'different'}).status_code==409


def test_first_skip_does_not_lock_role_but_fitting_cannot_be_reclassified_as_validation(setup):
    client,_,record,_,payload=setup
    skipped=review(client,record,'skip',None).json()
    assert not skipped['evaluation']['dataset_assigned']
    assert review(client,skipped,**VALIDATION).status_code==200
    payload['frame']=31
    other=client.post('/api/station/capture',json=payload).json()
    fitted=review(client,other).json()
    skipped=review(client,fitted,'skip',None).json()
    assert review(client,skipped,**{**VALIDATION,'physical_vehicle_id':'new-id'}).status_code==409


def test_long_combination_truth_is_retained_without_entering_legacy_box_scale(setup):
    client,_,record,_,_=setup
    saved=review(client,record,length=45).json()
    assert saved['evaluation']['actual_length_m']==45
    assert not saved['evaluation']['reference_added']
    data=client.get('/api/station/evaluation/'+record['id']+'/calibration.json').json()
    assert data['evaluation_samples'][0]['actual_length_m']==45
