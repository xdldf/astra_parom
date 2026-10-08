from copy import deepcopy

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from vehicle_metrology.outline import geometry_signature, outline_features, predict_outline
from web_app import workbench as wb, station as st, outline_measurement as om
from web_app.main import app


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(st, 'DATA', tmp_path)
    monkeypatch.setattr(st, 'DB', tmp_path/'station.sqlite3')
    profile = wb.Profile(image_size=(600,500), polygon=[(20,200),(580,200),(580,450),(20,450)],
        references=[dict(bbox=[250,150,100,75], length_m=5)], measurement_line_x=300)
    mask = np.zeros((500,600), np.uint8)
    mask[150:225,250:350] = 1
    features = outline_features(mask, profile.polygon)
    calibration = dict(calibration_id='test-outline', geometry_signature=geometry_signature(profile.model_dump(mode='json')),
        mean=features.tolist(), scale=[1]*7, coefficient=[1,0,0,0,0,0,0], intercept=4.2,
        feature_bounds=[[float(x)-1,float(x)+1] for x in features], training_count=20,training_family_count=4,
        reference_manifest_sha256='0'*64)
    profile = wb.Profile.model_validate({**profile.model_dump(),'outline_calibration':calibration})
    image = np.zeros((500,600,3),np.uint8)
    image[:80,:80] = [10,180,250]
    monkeypatch.setattr(wb,'read_frame',lambda *args:image.copy())
    monkeypatch.setattr(om,'infer_outline',lambda *args:(mask,dict(contour=[[250,150],[349,150],[349,224]],iou=.98)))
    payload=dict(media_id='outline-video', profile=profile.model_dump(), frame=5,
                 bbox=[250,150,100,75],label='car')
    return tmp_path, profile, mask, payload


def test_capture_runs_outline_and_keeps_clean_corrected_full_frame_and_baseline(setup):
    path,profile,mask,payload=setup
    response=TestClient(app).post('/api/station/capture',json=payload)
    assert response.status_code==200,response.text
    record=response.json();measured=record['source']['measurement']
    assert record['length_m']==record['measured_length_m']==pytest.approx(4.2)
    assert measured['outline']['baseline_length_m']==pytest.approx(5)
    assert measured['approximate'] and not measured['outline']['accuracy_validated']
    assert measured['outline']['reference_source']=='catalogue'
    saved=cv2.imread(str(path/record['full_frame_photo']))
    assert saved.shape==(500,600,3)
    assert saved[20,20,1]==pytest.approx(180,abs=3)
    assert not saved[150,250].any()  # Outline is metadata, never burnt into evidence.
    assert len(list(path.glob('*-frame.jpg')))==1 and not list(path.glob('*raw*'))


def test_failed_outline_keeps_vehicle_photo_without_assigning_fallback_length(setup,monkeypatch):
    path,_,_,payload=setup
    def broken(*args):raise RuntimeError('model unavailable')
    monkeypatch.setattr(om,'infer_outline',broken)
    record=TestClient(app).post('/api/station/capture',json=payload).json()
    assert record['length_m'] is None and record['status']=='Требует проверки'
    assert (path/record['full_frame_photo']).is_file()
    assert record['source']['measurement']['outline']['baseline_length_m']==pytest.approx(5)
    assert record['source']['measurement']['outline']['reason']=='model unavailable'


def test_outline_profile_rejects_stale_geometry_and_strict_mode(setup):
    _,profile,_,_=setup
    data=profile.model_dump()
    for patch in [dict(lens={**data['lens'],'k1':.01}),dict(measurement_mode='strict'),dict(image_size=(601,500))]:
        with pytest.raises(ValueError):wb.Profile.model_validate({**data,**patch})
    corrupted=deepcopy(data);corrupted['outline_calibration']['scale']=[0]*7
    with pytest.raises(ValueError):wb.Profile.model_validate(corrupted)


def test_non_car_does_not_use_passenger_regression(setup,monkeypatch):
    _,_,_,payload=setup
    monkeypatch.setattr(om,'infer_outline',lambda *args:pytest.fail('Unexpected passenger inference'))
    record=TestClient(app).post('/api/station/capture',json={**payload,'label':'truck'}).json()
    assert record['length_m']==pytest.approx(5)
    assert record['source']['measurement']['outline']['status']=='not_applicable'


def test_unsupported_silhouette_requires_review_and_does_not_erase_candidate(setup):
    _,profile,mask,payload=setup
    data=profile.model_dump();data['outline_calibration']['feature_bounds']=[[10,11]]*7
    record=TestClient(app).post('/api/station/capture',json={**payload,'profile':data}).json()
    assert record['length_m'] is None
    evidence=record['source']['measurement']['outline']
    assert evidence['candidate_length_m']==pytest.approx(4.2)
    assert 'outside' in evidence['reason']


def test_clipped_masks_and_invalid_outputs_are_not_lengths(setup):
    _,profile,mask,_=setup
    clipped=mask.copy();clipped[150:200,:300]=1
    with pytest.raises(ValueError,match='Clipped'):outline_features(clipped,profile.polygon)
    data=profile.outline_calibration.model_dump();data['intercept']=1000
    with pytest.raises(ValueError,match='numeric range'):
        predict_outline(outline_features(mask,profile.polygon),data)


def test_ip_camera_uses_the_same_capture_outline_path(setup,monkeypatch):
    import time
    from web_app import ip_cameras as ip
    path,profile,_,_=setup
    camera=ip.Station(ip.Settings(),profile)
    jpeg=cv2.imencode('.jpg',np.zeros((500,600,3),np.uint8))[1].tobytes()
    stamp=time.monotonic()-.3
    packets=[ip.Packet(i,stamp+.04*i,jpeg) for i in range(7)]
    camera.side.packets.extend(packets)
    monkeypatch.setattr(wb,'detect_vehicles',lambda *args,**kwargs:[dict(bbox=[250,150,100,75],label='car')])
    record=camera.capture((packets[3],None,None),dict(bbox=[250,150,100,75],label='car'),
                          'car',temporal=True,epoch=camera.side.epoch)
    assert record['length_m']==pytest.approx(4.2)
    assert record['source']['measurement']['temporal']['length_m']==pytest.approx(5)
    assert record['source']['measurement']['outline']['baseline_length_m']==pytest.approx(5)
    assert (path/record['full_frame_photo']).is_file()


def test_an_outline_cannot_override_ambiguous_temporal_identity(setup,monkeypatch):
    _,profile,_,_=setup
    monkeypatch.setattr(om,'infer_outline',lambda *args:pytest.fail('Ambiguous vehicle must stay in review'))
    measured=dict(status='temporal_estimate',length_m=4.5,depth=.5,
                  temporal=dict(reasons=['ambiguous_vehicle_association']))
    reviewed=om.apply_outline(profile,measured,None,label='car')
    assert reviewed['length_m'] is None
    assert reviewed['outline']['baseline_length_m']==4.5
    assert 'Ambiguous' in reviewed['outline']['reason']
