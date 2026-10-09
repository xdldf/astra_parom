from copy import deepcopy

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from test_outline import setup  # Shared corrected-frame / station fixture.
from vehicle_metrology.wheels import select_wheel_pair, wheel_features, predict_wheels, wheel_roi, verify_model
from web_app import workbench as wb, wheel_measurement as wm
from web_app.main import app


@pytest.fixture
def wheels(setup, monkeypatch):
    path, profile, mask, payload = setup
    detections = [dict(xyxy=[260, 200, 275, 215], score=.9), dict(xyxy=[320, 200, 335, 215], score=.85)]
    pair = select_wheel_pair(detections, payload['bbox'])
    from vehicle_metrology.outline import outline_features
    features = wheel_features(outline_features(mask, profile.polygon), pair, profile.polygon)
    calibration = {**profile.outline_calibration.model_dump(), 'calibration_id':'test-wheels',
        'mean':features.tolist(), 'scale':[1]*9, 'coefficient':[1,0,0,0,0,0,0,0,0],
        'intercept':4.3, 'feature_bounds':[[float(v)-1, float(v)+1] for v in features]}
    profile = wb.Profile.model_validate({**profile.model_dump(), 'wheel_calibration':calibration})
    payload = {**payload, 'profile':profile.model_dump()}
    details = dict(detections=detections, roi=[242, 142, 358, 243], image_space='lens_corrected_full_frame')
    monkeypatch.setattr(wm, 'infer_wheels', lambda *args:deepcopy(details))
    return path, profile, payload, details


def test_video_capture_uses_wheels_keeps_both_estimates_and_clean_full_corrected_frame(wheels):
    path, _, payload, _ = wheels
    response = TestClient(app).post('/api/station/capture', json=payload)
    assert response.status_code == 200, response.text
    record = response.json()
    measured = record['source']['measurement']
    assert record['length_m'] == record['measured_length_m'] == pytest.approx(4.3)
    assert measured['status'] == 'wheel_estimate'
    assert measured['wheels']['baseline_length_m'] == pytest.approx(4.2)
    assert measured['outline']['baseline_length_m'] == pytest.approx(5)
    assert measured['approximate'] and measured['accuracy_validated'] is False
    assert measured['wheels']['image_space'] == 'lens_corrected_full_frame'
    saved = cv2.imread(str(path/record['full_frame_photo']))
    assert saved.shape == (500, 600, 3)
    assert saved[20, 20, 1] == pytest.approx(180, abs=3)
    assert not saved[200, 260].any()  # Wheel drawing must remain separate metadata.


@pytest.mark.parametrize('problem', ['missing_model', 'no_pair', 'third_wheel', 'outside_range'])
def test_unavailable_refinement_retains_the_outline_and_photo(wheels, monkeypatch, problem):
    path, _, payload, details = wheels
    if problem == 'missing_model':
        def unavailable(*args):
            raise RuntimeError('model unavailable')
        monkeypatch.setattr(wm, 'infer_wheels', unavailable)
    elif problem == 'no_pair':
        details['detections'] = []
    elif problem == 'third_wheel':
        details['detections'].append(dict(xyxy=[290, 200, 305, 215], score=.95))
    else:
        payload['profile']['wheel_calibration']['feature_bounds'] = [[10, 11]]*9
    record = TestClient(app).post('/api/station/capture', json=payload).json()
    assert record['length_m'] == pytest.approx(4.2)
    evidence = record['source']['measurement']['wheels']
    assert evidence['status'] == 'fallback' and evidence['reason']
    assert record['source']['measurement']['status'] == 'outline_estimate'
    assert (path/record['full_frame_photo']).is_file()


def test_outline_review_and_ambiguous_identity_cannot_be_resurrected_by_wheels(wheels, monkeypatch):
    path, profile, payload, _ = wheels
    monkeypatch.setattr(wm, 'infer_wheels', lambda *args:pytest.fail('Review must not use wheel refinement'))
    payload['profile']['outline_calibration']['feature_bounds'] = [[10, 11]]*7
    record = TestClient(app).post('/api/station/capture', json=payload).json()
    assert record['length_m'] is None
    assert record['source']['measurement']['wheels']['status'] == 'not_applicable'
    assert (path/record['full_frame_photo']).is_file()
    measured = dict(status='capture_review', length_m=None, depth=.5,
                    temporal=dict(reasons=['ambiguous_vehicle_association']), outline=dict(status='unavailable'))
    assert wm.apply_wheels(profile, measured, None, label='car')['length_m'] is None


def test_ip_capture_uses_same_wheel_path(wheels, monkeypatch):
    import time
    from web_app import ip_cameras as ip
    path, profile, _, _ = wheels
    camera = ip.Station(ip.Settings(), profile)
    jpeg = cv2.imencode('.jpg', np.zeros((500, 600, 3), np.uint8))[1].tobytes()
    stamp = time.monotonic()-.3
    packets = [ip.Packet(i, stamp+.04*i, jpeg) for i in range(7)]
    camera.side.packets.extend(packets)
    monkeypatch.setattr(wb, 'detect_vehicles', lambda *args, **kwargs:[dict(bbox=[250,150,100,75], label='car')])
    record = camera.capture((packets[3], None, None), dict(bbox=[250,150,100,75], label='car'),
                            'car', temporal=True, epoch=camera.side.epoch)
    assert record['length_m'] == pytest.approx(4.3)
    assert record['source']['measurement']['temporal']['length_m'] == pytest.approx(5)
    assert (path/record['full_frame_photo']).is_file()


def test_invalid_or_stale_wheel_calibration_rejected(wheels):
    _, profile, _, _ = wheels
    for key, value in [('scale', [0]*9), ('mean', [0]*7), ('intercept', float('nan')),
                       ('geometry_signature', '0'*64), ('feature_bounds', [[2, 1]]*9)]:
        data = profile.model_dump()
        data['wheel_calibration'][key] = value
        with pytest.raises(ValueError):
            wb.Profile.model_validate(data)
    data = profile.model_dump()
    data['outline_calibration'] = None
    with pytest.raises(ValueError, match='fallback'):
        wb.Profile.model_validate(data)


def test_missing_wheels_are_never_zero_features_and_duplicate_proposals_are_not_extra_wheels(wheels):
    _, profile, payload, details = wheels
    with pytest.raises(ValueError, match='zero-filled'):
        wheel_features([0]*7, None, profile.polygon)
    detections = details['detections']
    pair = select_wheel_pair([*detections, {**detections[0], 'score':.7}], payload['bbox'])
    assert len(pair) == 2
    assert pair[0]['score'] == .9
    with pytest.raises(ValueError):
        select_wheel_pair(detections[:1], payload['bbox'])
    with pytest.raises(ValueError):
        predict_wheels([float('nan')]*9, profile.wheel_calibration.model_dump())
    assert wheel_roi([0, 0, 50, 60], (100, 100)) == (0, 0, 54, 75)


def test_model_loader_requires_all_pinned_files(tmp_path):
    (tmp_path/'model.safetensors').write_bytes(b'not the official weights')
    with pytest.raises(ValueError, match='unverified'):
        verify_model(tmp_path)


def test_family_exclusion_and_missing_wheels_do_not_leak_labels_into_fit():
    from vehicle_metrology.catalogue_calibration import nested_predictions
    rng = np.random.default_rng(42)
    features = rng.normal(size=(20, 9))
    groups = np.repeat(['a', 'b', 'c', 'd', 'e'], 4)
    targets = 4+.2*features[:, 0]
    features[0] = np.nan
    eligible = np.isfinite(features).all(axis=1)
    options = dict(training_eligible=eligible, feature_sets={'wheels':tuple(range(9))})
    from vehicle_metrology.wheels import FEATURE_NAMES
    before, folds = nested_predictions(features, targets, groups, feature_names=FEATURE_NAMES, **options)
    targets[groups == 'a'] += 500
    after, changed = nested_predictions(features, targets, groups, feature_names=FEATURE_NAMES, **options)
    assert np.isnan(before[0])  # Runtime must explicitly fall back, not invent features.
    np.testing.assert_allclose(before[1:4], after[1:4], atol=0, rtol=0)
    assert folds[0]['model'] == changed[0]['model']
    for fold in folds:
        assert fold['heldout_group'] not in fold['model']['training_groups']


def test_off_center_review_cannot_be_resurrected_by_wheel_models(wheels,monkeypatch):
    from web_app import outline_measurement as om
    _,_,payload,_=wheels
    monkeypatch.setattr(om,'infer_outline',lambda *a:pytest.fail('Off-center outline inference'))
    monkeypatch.setattr(wm,'infer_wheels',lambda *a:pytest.fail('Off-center wheel inference'))
    response=TestClient(app).post('/api/station/capture',json={**payload,
        'bbox':[150,150,100,75],'review_fallback':True})
    assert response.status_code==200
    record=response.json()
    assert record['length_m'] is None
    assert record['source']['measurement']['wheels']['status']=='not_applicable'
