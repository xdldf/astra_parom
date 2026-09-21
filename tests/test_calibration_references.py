import json
import pytest
from fastapi.testclient import TestClient
from web_app.main import app
from web_app import station, workbench, ip_cameras


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(station, 'DATA', tmp_path/'station')
    monkeypatch.setattr(station, 'DB', tmp_path/'station'/'test.sqlite3')
    monkeypatch.setattr(workbench, 'DATA', tmp_path/'media')
    monkeypatch.setattr(ip_cameras, 'CONFIG', tmp_path/'ip.json')
    monkeypatch.setattr(ip_cameras, 'active', None)
    (tmp_path/'media').mkdir()
    profile=workbench.Profile(image_size=(600,500),polygon=[(20,200),(580,200),(580,450),(20,450)],
                              references=[workbench.Reference(bbox=(100,220,100,100),length_m=4)])
    record=station.new_record(station.Fields(length_m=4),source=dict(
        bbox=[100,220,100,100],frame=8,calibration=profile.model_dump(),measured_length_m=4))
    record['created_at']='2020-01-01T12:00:00+09:00'
    record['side_photo']='saved-side.jpg'
    with station.connect() as db:
        db.execute('INSERT INTO vehicles VALUES(?,?,?,?)',(record['id'],'test',1,json.dumps(record)))
    return TestClient(app),profile,record


def approve(client, record, action='confirm', **overrides):
    body=dict(version=record['version'],plate='',category='car',length_m=4,
              action=action,actor='Human',reason='Проверка')
    body.update(overrides)
    r=client.post('/api/station/vehicles/'+record['id'],json=body)
    assert r.status_code==200,r.text
    return r.json()


def verify(client, record, **overrides):
    body=dict(version=record['version'],actor='Verifier',enabled=True,actual_length_m=4.5,verified=True)
    body.update(overrides)
    return client.post('/api/station/vehicles/'+record['id']+'/calibration-reference',json=body)


def merged(client, profile):
    r=client.post('/api/workbench/approved-references',json=profile if isinstance(profile,dict) else profile.model_dump())
    assert r.status_code==200,r.text
    return r.json()


@pytest.mark.parametrize('status',['Новый','Требует проверки','Отклонён','Подтвержден','Оплачен'])
def test_status_alone_is_not_human_approval(setup,status):
    client,profile,record=setup
    record['status']=status
    with station.connect() as db:
        db.execute('UPDATE vehicles SET data=? WHERE id=?',(json.dumps(record),record['id']))
    assert verify(client,record).status_code==409
    assert len(merged(client,profile)['references'])==1


def test_old_approved_verified_car_in_json_once_with_provenance(setup):
    client,profile,record=setup
    record=approve(client,record)
    assert len(merged(client,profile)['references'])==1  # Approval alone does not add an estimate.
    assert verify(client,record,verified=False).status_code==422
    assert verify(client,record,actual_length_m=None).status_code==422
    response=verify(client,record)
    assert response.status_code==200,response.text
    result=merged(client,profile)
    assert len(result['references'])==2
    ref=result['references'][-1]
    assert ref['length_m']==4.5 and ref['verified_by']=='Verifier' and ref['vehicle_id']==record['id']
    assert merged(client,result)==result
    assert client.post('/api/workbench/operator-calibration',json=profile.model_dump()).json()==result
    assert client.post('/api/ip/calibration',json=profile.model_dump()).status_code==200
    assert ip_cameras.settings().profile.references[-1].vehicle_id==record['id']
    assert verify(client,record).status_code==409  # Concurrent/stale verification cannot overwrite.


@pytest.mark.parametrize('action',['edit','reject'])
def test_unapproved_record_is_removed_even_from_old_json(setup,action):
    client,profile,record=setup
    record=approve(client,record)
    record=verify(client,record).json()
    old_json=merged(client,profile)
    approve(client,record,action)
    assert len(merged(client,old_json)['references'])==1
    record=client.get('/api/station/vehicles/'+record['id']).json()
    assert not record.get('calibration_reference')
    record=approve(client,record)
    assert len(merged(client,old_json)['references'])==1  # Requires new explicit verification.


def test_paid_reference_can_be_withdrawn_without_changing_payment(setup):
    client,profile,record=setup
    record=approve(client,approve(client,record),'pay')
    record=verify(client,record).json()
    old=merged(client,profile)
    result=verify(client,record,enabled=False,verified=False,actual_length_m=None)
    assert result.status_code==200,result.text
    assert result.json()['status']=='Оплачен'
    assert len(merged(client,old)['references'])==1


@pytest.mark.parametrize('change',[{'image_size':[601,500]},{'lens':{'k1':.1}},
                                  {'polygon':[[21,200],[580,200],[580,450],[20,450]]}])
def test_changed_coordinate_system_excludes_reference(setup,change):
    client,profile,record=setup
    verify(client,approve(client,record))
    data=profile.model_dump();data.update(change)
    assert len(merged(client,data)['references'])==1


def test_missing_source_and_clipped_bbox_rejected(setup):
    client,profile,record=setup
    record=approve(client,record)
    for source in [None,dict(calibration=profile.model_dump(),bbox=[0,220,100,100])]:
        record['source']=source
        with station.connect() as db:
            db.execute('UPDATE vehicles SET data=? WHERE id=?',(json.dumps(record),record['id']))
        assert verify(client,record).status_code==422


def test_running_profile_stops_using_reference_after_revocation(setup):
    from vehicle_metrology.bbox_scale import measure_box
    client,profile,record=setup
    record=verify(client,approve(client,record)).json()
    saved=workbench.Profile.model_validate(merged(client,profile))
    scale=workbench.profile_scale(saved)
    assert scale['status']=='verified_local'
    assert measure_box((100,220,100,100),saved.polygon,scale,saved.image_size)['length_m']==pytest.approx(4.5)
    assert measure_box((100,320,100,100),saved.polygon,scale,saved.image_size)['length_m'] is None
    verify(client,record,enabled=False)
    # Same in-memory object, without reloading or restarting the camera.
    assert workbench.profile_scale(saved)['status']=='single_reference'
    assert workbench.profile_scale(saved)['intercept']==pytest.approx(25)


def test_rulers_keep_priority_over_verified_cars(setup):
    client,profile,record=setup
    verify(client,approve(client,record))
    data=profile.model_dump()
    data['metric_rulers']=[dict(points=[[100,320],[200,320],[300,320]],step_m=1)]
    saved=workbench.Profile.model_validate(merged(client,data))
    assert len(saved.references)==2
    assert workbench.profile_scale(saved)['status']=='ruler_calibrated'
