from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from test_outline import setup
from test_wheels import wheels
from web_app import workbench as wb, outline_measurement as om, wheel_measurement as wm
from web_app.main import app
from vehicle_metrology.wheel_recovery import fit_frames, predict_frames, select_frame_model, select_passage_frames


@pytest.fixture
def recovery(wheels):
    path, profile, payload, details = wheels
    data = profile.model_dump()
    data['wheel_recovery_calibration'] = {**data['wheel_calibration'], 'calibration_id':'recovery',
                                        'intercept':4.35, 'training_frame_count':50}
    data['outline_calibration']['feature_bounds'] = [[10, 11]]*7
    profile = wb.Profile.model_validate(data)
    return path, profile, {**payload, 'profile':profile.model_dump()}, details


def test_only_feature_range_rejection_can_use_recovery(recovery):
    path, _, payload, _ = recovery
    response = TestClient(app).post('/api/station/capture', json=payload)
    assert response.status_code == 200, response.text
    record = response.json()
    measured = record['source']['measurement']
    assert record['length_m'] == pytest.approx(4.35)
    assert measured['wheels']['calibration_role'] == 'recovery'
    assert measured['wheels']['baseline_length_m'] is None
    assert measured['outline']['failure_code'] == 'outside_feature_range'
    assert measured['outline']['baseline_length_m'] == pytest.approx(5)
    assert 'outline_unavailable' not in measured['quality_reasons']
    assert om.UNAVAILABLE_WARNING not in measured['warnings']
    assert not measured['accuracy_validated']
    assert (path/record['full_frame_photo']).is_file()


@pytest.mark.parametrize('failure', ['model', 'ambiguous', 'unsupported_wheels'])
def test_recovery_never_invents_length_on_other_failures(recovery, monkeypatch, failure):
    path, profile, payload, _ = recovery
    if failure == 'model':
        def broken(*args):
            raise RuntimeError('model unavailable')
        monkeypatch.setattr(om, 'infer_outline', broken)
    elif failure == 'ambiguous':
        measured = dict(status='capture_review', length_m=None, depth=.5, bbox=payload['bbox'],
            outline=dict(status='unavailable', failure_code='outside_feature_range'),
            temporal=dict(reasons=['ambiguous_vehicle_association']))
        monkeypatch.setattr(wm, 'infer_wheels', lambda *args:pytest.fail('Must preserve identity rejection'))
        assert wm.apply_wheels(profile, measured, None, label='car')['length_m'] is None
        return
    else:
        payload['profile']['wheel_recovery_calibration']['feature_bounds'] = [[10, 11]]*9
    record = TestClient(app).post('/api/station/capture', json=payload).json()
    assert record['length_m'] is None
    assert (path/record['full_frame_photo']).is_file()


def test_primary_length_is_not_replaced_by_recovery(recovery):
    _, _, payload, _ = recovery
    payload['profile']['outline_calibration']['feature_bounds'] = [[-10, 10]]*7
    record = TestClient(app).post('/api/station/capture', json=payload).json()
    assert record['length_m'] == pytest.approx(4.3)
    assert record['source']['measurement']['wheels']['calibration_role'] == 'primary'


@pytest.mark.parametrize('accepted_outline', [False, True])
def test_saved_revoked_recovery_profile_cannot_assign_a_length(recovery, monkeypatch, accepted_outline):
    path, _, payload, _ = recovery
    payload['profile']['wheel_recovery_calibration']['reference_manifest_sha256'] = next(iter(wm.REVOKED_RECOVERY_REFERENCES))
    if accepted_outline:
        payload['profile']['outline_calibration']['feature_bounds'] = [[-10,10]]*7
    else:
        monkeypatch.setattr(wm,'infer_wheels',lambda *args:pytest.fail('Revoked coefficients must not run'))
    record = TestClient(app).post('/api/station/capture',json=payload).json()
    assert (path/record['full_frame_photo']).is_file()
    evidence = record['source']['measurement']['wheels']
    if accepted_outline:
        assert record['length_m'] == pytest.approx(4.3)
        assert evidence['calibration_role'] == 'primary'
    else:
        assert record['length_m'] is None
        assert evidence['failure_code'] == 'revoked_reference_manifest'
        assert 'wheel_recovery_unavailable' in record['source']['measurement']['quality_reasons']


def test_recovery_requires_primary_models_and_matching_geometry(recovery):
    _, profile, _, _ = recovery
    data = profile.model_dump()
    data['wheel_calibration'] = None
    with pytest.raises(ValueError, match='primary'):
        wb.Profile.model_validate(data)
    data = profile.model_dump()
    data['wheel_recovery_calibration']['geometry_signature'] = '0'*64
    with pytest.raises(ValueError, match='geometry'):
        wb.Profile.model_validate(data)


def test_near_identical_duplicate_labels_are_one_outline_but_different_masks_stay_ambiguous(monkeypatch):
    import torch
    image = np.zeros((500,600,3), np.uint8)
    mask = np.zeros((2,500,600), np.float32)
    mask[:,150:225,250:350] = 1
    result = SimpleNamespace(boxes=SimpleNamespace(xyxy=torch.tensor([[250,150,350,225],[251,150,350,225]]),
        cls=torch.tensor([2,7]), conf=torch.tensor([.8,.7])), masks=SimpleNamespace(data=torch.from_numpy(mask)))
    monkeypatch.setattr(om, 'load_model', lambda:SimpleNamespace(predict=lambda *args,**kwargs:[result]))
    actual, details = om.infer_outline(image, [250,150,100,75], device='cpu')
    assert details['duplicate_proposals_ignored'] == 1
    assert details['detector_class'] == 2
    np.testing.assert_array_equal(actual, mask[0])
    result.masks.data[1,150:190,250:350] = 0
    with pytest.raises(ValueError, match='Ambiguous'):
        om.infer_outline(image, [250,150,100,75], device='cpu')


def test_repeating_a_passages_frames_does_not_increase_its_training_weight():
    rng = np.random.default_rng(10)
    x = rng.normal(size=(12,9)); y = 4+.3*x[:,0]
    groups = np.repeat(['a','b','c','d'],3); ids = np.array([str(i) for i in range(12)])
    before = fit_frames(x,y,groups,ids,.1)
    repeated = np.r_[np.arange(12),np.repeat(0,20)]
    after = fit_frames(x[repeated],y[repeated],groups[repeated],ids[repeated],.1)
    np.testing.assert_allclose(predict_frames(before,x),predict_frames(after,x),atol=1e-12)
    assert after['training_count'] == 12 and after['training_frame_count'] == 32


def test_heldout_family_cannot_affect_its_fit_and_frame_selection_does_not_read_catalogue():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(20,9));y = 4+.2*x[:,0]
    groups = np.repeat(['a','b','c','d','e'],4);ids = np.array([str(i) for i in range(20)])
    train = groups != 'a';anchors = np.ones(20,bool)
    before, scores = select_frame_model(x[train],y[train],groups[train],ids[train],anchors[train])
    y[~train] += 500
    after, changed = select_frame_model(x[train],y[train],groups[train],ids[train],anchors[train])
    assert before == after and scores == changed
    row = dict(frame=100,bbox=[450,200,100,50],catalogue_length_range_m=[4,4])
    observations = [dict(frame=50,bbox=[300,200,100,50]),dict(frame=150,bbox=[600,200,100,50])]
    selected = select_passage_frames(row,observations,(1000,700))
    row['catalogue_length_range_m'] = [100,200]
    assert selected == select_passage_frames(row,observations,(1000,700))
    assert [s['frame'] for s in selected] == [50,100,150]
