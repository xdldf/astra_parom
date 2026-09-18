import io
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from web_app.main import app
from web_app import workbench
from vehicle_metrology.bbox_scale import road_depth, fit_scale, measure_box

POLY = [[20,200],[580,200],[580,450],[20,450]]
REFS = [dict(bbox=[100,150,100,75],length_m=5,frame=0),
        dict(bbox=[100,300,200,125],length_m=5,frame=0)]


def test_perspective_and_known_lengths():
    scale = fit_scale(POLY,REFS)
    far = measure_box(REFS[0]['bbox'],POLY,scale,(600,500))
    near = measure_box(REFS[1]['bbox'],POLY,scale,(600,500))
    assert far['length_m'] == pytest.approx(5)
    assert near['length_m'] == pytest.approx(5)
    assert far['cm_per_px'] == pytest.approx(near['cm_per_px']*2)
    assert far['coefficient'] > near['coefficient']
    assert measure_box([100,250,150,75],POLY,scale,(600,500))['length_m'] == pytest.approx(5)


def test_road_depth_uses_local_edges_and_rejects_outside():
    p = [[20,100],[580,200],[580,450],[20,350]]
    assert road_depth(p,300,275) == pytest.approx(.5)
    assert road_depth(p,10,275) is None
    assert road_depth(p,300,120) is None


def test_unsupported_and_invalid_calibration():
    assert measure_box([100,100,50,40],POLY,None,(600,500))['length_m'] is None
    assert fit_scale(POLY,REFS[:1])['status'] == 'single_reference'
    with pytest.raises(ValueError,match='too close'):
        fit_scale(POLY,[REFS[0],REFS[0]])
    with pytest.raises(ValueError,match='invalid perspective'):
        fit_scale(POLY,[dict(REFS[0],length_m=1),REFS[1]])
    assert measure_box([0,300,100,100],POLY,fit_scale(POLY,REFS),(600,500))['status']=='clipped'


def test_api_upload_preview_profile_and_detector(tmp_path,monkeypatch):
    monkeypatch.setattr(workbench,'DATA',tmp_path)
    client=TestClient(app)
    assert client.get('/').status_code==200
    assert client.get('/static/app.js').status_code==200
    image=np.zeros((500,600,3),np.uint8)
    cv2.line(image,(20,20),(580,480),(255,255,255),3)
    _,buf=cv2.imencode('.png',image)
    response=client.post('/api/workbench/media',files={'file':('../../frame.png',io.BytesIO(buf.tobytes()),'image/png')})
    assert response.status_code==200
    item=response.json()
    profile=dict(version=1,image_size=[600,500],lens={},polygon=POLY,references=REFS)
    assert client.post('/api/workbench/validate-profile',json=profile).status_code==200
    payload=dict(profile=profile,frame=0,boxes=[REFS[0]['bbox']])
    result=client.post('/api/workbench/frame/'+item['id'],json=payload)
    assert result.status_code==200,result.text
    assert result.json()['detections'][0]['length_m']==pytest.approx(5)
    assert result.json()['image']
    profile['image_size']=[601,500]
    assert client.post('/api/workbench/frame/'+item['id'],json=payload).status_code==400
    profile['image_size']=[600,500]
    profile['polygon']=[[20,200],[580,450],[580,200],[20,450]]
    assert client.post('/api/workbench/validate-profile',json=profile).status_code==422


def test_lens_identity_and_correction():
    image=np.random.default_rng(0).integers(0,255,(100,150,3),dtype=np.uint8)
    assert np.array_equal(workbench.corrected(image,workbench.Lens()),image)
    assert not np.array_equal(workbench.corrected(image,workbench.Lens(k1=-.2)),image)


def test_inference_uses_corrected_frame_and_preserves_geometry(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock
    monkeypatch.setattr(workbench,'DATA',tmp_path)
    image=np.random.default_rng(1).integers(0,255,(500,600,3),dtype=np.uint8)
    path=tmp_path/'input.png'
    cv2.imwrite(str(path),image)
    monkeypatch.setitem(workbench.media,'test',dict(path=path,frames=1,kind='image'))
    box=SimpleNamespace(xyxy=np.array([[100,150,200,225]]),conf=np.array([.95]),cls=np.array([2]))
    detector=Mock()
    detector.predict.return_value=[SimpleNamespace(boxes=[box],names={2:'car'})]
    monkeypatch.setattr(workbench,'model',detector)
    monkeypatch.setattr(workbench,'require_gpu',lambda: 0)
    profile=dict(image_size=[600,500],lens={'k1':-.2,'tilt_deg':7.5},polygon=POLY,references=REFS)
    response=TestClient(app).post('/api/workbench/frame/test',json=dict(profile=profile,detect=True))
    assert response.status_code==200,response.text
    actual=detector.predict.call_args.args[0]
    assert np.array_equal(actual,workbench.corrected(image,workbench.Lens(k1=-.2,tilt_deg=7.5)))
    assert detector.predict.call_args.kwargs['classes']==[2,3,5,7]
    assert detector.predict.call_args.kwargs['device']==0
    d=response.json()['detections'][0]
    assert d['label']=='car'
    assert d['bbox']==[100,150,100,75]
    assert d['length_m']==pytest.approx(5)


def test_measurement_line_uses_center_not_bbox_edge():
    scale = fit_scale(POLY, REFS)
    bbox = [100, 150, 100, 75]
    # The bbox spans the line at 110, but its center is 150.
    off = measure_box(bbox, POLY, scale, (600,500), 110, 5)
    assert off['status'] == 'waiting_for_line'
    assert off['length_m'] is None
    assert off['line_offset_px'] == 40
    assert not off['at_measurement_line']
    for line in (145,150,155):
        on = measure_box(bbox, POLY, scale, (600,500), line, 5)
        assert on['at_measurement_line']
        assert on['length_m'] == pytest.approx(5)
        assert on['bottom'] == [150,225]
        assert on['center'] == [150,187.5]
    assert measure_box(bbox, POLY, scale, (600,500),156,5)['length_m'] is None


def test_line_profile_round_trip_and_validation():
    client = TestClient(app)
    data = dict(image_size=[600,500],polygon=POLY,references=REFS,
                measurement_line_x=300,line_tolerance_px=8)
    result = client.post('/api/workbench/validate-profile',json=data)
    assert result.status_code == 200
    assert result.json()['polygon'] == POLY
    assert result.json()['references'] == REFS
    assert result.json()['measurement_line_x'] == 300
    data['measurement_line_x']=600
    assert client.post('/api/workbench/validate-profile',json=data).status_code == 422


def test_uploaded_image_survives_server_restart(tmp_path,monkeypatch):
    monkeypatch.setattr(workbench,'DATA',tmp_path)
    monkeypatch.setattr(workbench,'media',{})
    key='a'*32
    image=np.zeros((100,120,3),np.uint8)
    cv2.imwrite(str(tmp_path/(key+'.png')),image)
    assert np.array_equal(workbench.read_frame(key,0),image)
    assert workbench.media[key]['frames']==1


def test_tilt_rotates_both_directions_without_changing_resolution():
    image=np.zeros((101,151,3),np.uint8)
    image[48:53,113:118]=255
    left=workbench.corrected(image,workbench.Lens(tilt_deg=-10))
    right=workbench.corrected(image,workbench.Lens(tilt_deg=10))
    assert left.shape == right.shape == image.shape
    assert np.where(left[:,:,0]>100)[0].mean() < 50
    assert np.where(right[:,:,0]>100)[0].mean() > 50
    assert np.array_equal(image,workbench.corrected(image,workbench.Lens(tilt_deg=0)))


def test_tilt_profile_round_trip_and_older_profiles():
    client=TestClient(app)
    data=dict(image_size=[600,500],lens={'tilt_deg':-4.2})
    response=client.post('/api/workbench/validate-profile',json=data)
    assert response.status_code == 200
    assert response.json()['lens']['tilt_deg'] == -4.2
    data['lens']={}
    assert client.post('/api/workbench/validate-profile',json=data).json()['lens']['tilt_deg']==0
    data['lens']={'tilt_deg':31}
    assert client.post('/api/workbench/validate-profile',json=data).status_code==422


def test_detection_refuses_cpu_fallback(monkeypatch):
    import torch
    from unittest.mock import Mock
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    monkeypatch.setattr(workbench,'read_frame',lambda *_:np.zeros((500,600,3),np.uint8))
    detector=Mock()
    monkeypatch.setattr(workbench,'model',detector)
    response=TestClient(app).post('/api/workbench/frame/test',json={
        'profile':{'image_size':[600,500]},'detect':True})
    assert response.status_code==503
    assert 'CPU fallback is disabled' in response.json()['detail']
    detector.predict.assert_not_called()


RULER_POLY=[[10,100],[590,100],[590,490],[10,490]]

def ruler(xs,y,step=1):
    return {'points':[[x,y] for x in xs],'step_m':step}


def test_rulers_integrate_center_compression_across_the_whole_car():
    # True metric marks 0,1,2,3,4: pixels/metre is 100,50,50,100.
    scale=fit_scale(RULER_POLY,[],[ruler([100,200,250,300,400],300)])
    for box,expected in [([100,200,300,100],4),([200,200,100,100],2),([100,200,100,100],1),([150,200,200,100],3)]:
        measured=measure_box(box,RULER_POLY,scale,(600,500))
        assert measured['length_m']==pytest.approx(expected)
        assert measured['status']=='ruler_calibrated'
    # Width at only the center would incorrectly give six metres, not four.
    assert measure_box([100,200,300,100],RULER_POLY,scale,(600,500))['length_m']!=6


def test_rulers_interpolate_perspective_and_refuse_extrapolation():
    scale=fit_scale(RULER_POLY,[],[ruler([50,150,250,350,450,550],200),
                                ruler([50,250,450],400)])
    assert measure_box([150,150,200,150],RULER_POLY,scale,(600,500))['length_m']==pytest.approx(200/150)
    assert measure_box([350,100,200,100],RULER_POLY,scale,(600,500))['length_m']==pytest.approx(2)
    for box in ([150,50,200,100],[150,350,200,100],[20,150,200,150],[350,150,200,150]):
        r=measure_box(box,RULER_POLY,scale,(600,500))
        assert r['length_m'] is None and r['status']=='outside_calibration'


def test_ruler_validation_and_profile_roundtrip():
    from pydantic import ValidationError
    data={'image_size':[600,500],'polygon':RULER_POLY,'metric_rulers':[ruler([100,200,250,400],300)]}
    profile=workbench.Profile.model_validate(data)
    assert workbench.Profile.model_validate_json(profile.model_dump_json()).metric_rulers==profile.metric_rulers
    assert not profile.references
    reverse=dict(data,metric_rulers=[ruler([400,250,200,100],300)])
    assert workbench.profile_scale(workbench.Profile.model_validate(reverse))==workbench.profile_scale(profile)
    for bad in [ruler([100,100,300],300),ruler([100,300,200],300),ruler([100,200,300],90),
                ruler([100,200],300),ruler([100,200,300],300,0),
                {'points':[[100,200],[200,300],[300,400]],'step_m':1}]:
        with pytest.raises(ValidationError):workbench.Profile.model_validate(dict(data,metric_rulers=[bad]))


def test_ruler_measurement_is_invariant_under_preview_resizing():
    raw=np.zeros((500,600,3),np.uint8)
    profile=workbench.Profile(image_size=(600,500),polygon=RULER_POLY,
                               metric_rulers=[ruler([100,200,250,300,400],300)])
    result=workbench.render_raw(raw,workbench.FrameRequest(profile=profile,boxes=[(100,200,300,100)]),include_image=False)
    assert result['detections'][0]['length_m']==pytest.approx(4)
    # A genuinely resized source requires matching, scaled calibration coordinates.
    small=workbench.Profile(image_size=(300,250),polygon=np.asarray(RULER_POLY).astype(float)/2,
        metric_rulers=[ruler([50,100,125,150,200],150)])
    result2=workbench.render_raw(cv2.resize(raw,(300,250)),workbench.FrameRequest(profile=small,boxes=[(50,100,150,50)]),include_image=False)
    assert result2['detections'][0]['length_m']==pytest.approx(4)


def test_long_vehicle_warns_when_calibration_uses_short_reference():
    scale=fit_scale(POLY,REFS[:1])
    measured=measure_box([50,100,450,150],POLY,scale,(600,500))
    assert measured['length_m'] is not None
    assert measured['warnings']
