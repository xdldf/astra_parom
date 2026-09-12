"""Synthetic calibration tests: these do NOT establish real-world accuracy."""
import importlib
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def calibration_module():
    assert importlib.util.find_spec('vehicle_metrology.calibration') is not None, 'calibration API missing'
    return importlib.import_module('vehicle_metrology.calibration')


def synthetic_survey(model='brown'):
    K = np.array([[700., 0, 480], [0, 710, 270], [0, 0, 1]])
    D = np.array([-0.12, 0.018, 0.001, -0.0008, 0.0]) if model == 'brown' else np.array([-0.08, 0.012, -0.002, 0.0002])
    R = np.array([[1., 0, 0], [0, -1/np.sqrt(5), -2/np.sqrt(5)], [0, 2/np.sqrt(5), -1/np.sqrt(5)]])
    C = np.array([0., -10, 5])
    t = -R @ C
    project = cv2.projectPoints if model == 'brown' else cv2.fisheye.projectPoints
    def points(xys, prefix):
        xyz = np.column_stack([xys, np.zeros(len(xys))]).astype(float)
        uv = project(xyz.reshape(-1, 1, 3), cv2.Rodrigues(R)[0], t, K, D)[0].reshape(-1, 2)
        return [dict(id=f'{prefix}{i}', xyz_m=p.tolist(), uv=q.tolist()) for i, (p, q) in enumerate(zip(xyz, uv))]
    intrinsic = dict(schema_version=1, image_size=[960, 540], model=model, K=K.tolist(), D=D.tolist())
    survey = dict(schema_version=1, image_size=[960, 540], units='m',
                  road_polygon=[[-4,-2],[4,-2],[4,8],[-4,8]],
                  control_points=points([[x,y] for y in [-1.,2.,6.] for x in [-3.,0.,3.]], 'c'),
                  check_points=points([[-2.,0.],[2.,3.],[-1.,5.],[1.,7.]], 'h'))
    return intrinsic, survey, R, t, C


@pytest.mark.parametrize('model', ['brown', 'fisheye'])
def test_synthetic_survey_recovers_physical_pose(model):
    api = calibration_module()
    intrinsic, survey, R, t, C = synthetic_survey(model)
    result = api.calibrate_survey(intrinsic, survey, calibration_id='synthetic-test', max_check_error_m=0.01)
    from vehicle_metrology.geometry import Camera
    camera = Camera.from_dict(result)
    np.testing.assert_allclose(camera.center, C, atol=1e-5)
    np.testing.assert_allclose(camera.Rcw, R, atol=1e-6)
    assert result['road_polygon'] == survey['road_polygon']
    assert result['diagnostics']['check_max_error_m'] < 1e-5
    assert result['diagnostics']['check_count'] == len(survey['check_points'])
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('defect, message', [
    ('size', 'image_size'), ('nonplanar', 'Z=0'), ('overlap', 'independent'),
    ('duplicate_coordinate', 'independent'), ('few_checks', 'check_points'),
    ('collinear', 'noncollinear'), ('nan', 'finite'), ('outside', 'image'),
    ('units', 'units'), ('bad_K', 'K'), ('bad_D', 'distortion'),
    ('threshold', 'max_check_error_m'), ('polygon', 'road_polygon'),
])
def test_survey_rejects_malformed_inputs(defect, message):
    api = calibration_module()
    intrinsic, survey, *_ = synthetic_survey()
    threshold = 0.1
    if defect == 'size': survey['image_size'] = [1920, 1080]
    if defect == 'nonplanar': survey['control_points'][0]['xyz_m'][2] = 0.1
    if defect == 'overlap': survey['check_points'][0]['id'] = survey['control_points'][0]['id']
    if defect == 'duplicate_coordinate': survey['check_points'][0]['xyz_m'] = survey['control_points'][0]['xyz_m']
    if defect == 'few_checks': survey['check_points'] = []
    if defect == 'collinear':
        for i, p in enumerate(survey['control_points']): p['xyz_m'] = [i, 0., 0.]
    if defect == 'nan': intrinsic['K'][0][0] = float('nan')
    if defect == 'outside': survey['control_points'][0]['uv'] = [-1, 50]
    if defect == 'units': survey['units'] = 'cm'
    if defect == 'bad_K': intrinsic['K'][0][1] = 2
    if defect == 'bad_D': intrinsic['D'] = [0]*8
    if defect == 'threshold': threshold = float('nan')
    if defect == 'polygon': survey['road_polygon'] = [[0,0],[1,0],[2,0]]
    with pytest.raises(ValueError, match=message):
        api.calibrate_survey(intrinsic, survey, calibration_id='test', max_check_error_m=threshold)


def synthetic_boards(model):
    intrinsic, *_ = synthetic_survey(model)
    K, D = np.array(intrinsic['K']), np.array(intrinsic['D'])
    xyz = np.zeros((48, 3), float)
    xyz[:,:2] = np.mgrid[0:8, 0:6].T.reshape(-1, 2) * 0.04
    xyz[:,:2] -= [0.14, 0.10]
    rng = np.random.default_rng(140)
    fn = cv2.projectPoints if model == 'brown' else cv2.fisheye.projectPoints
    views = []
    for i in range(20):
        r = rng.uniform([-0.6,-0.6,-0.4], [0.6,0.6,0.4])
        t = rng.uniform([-0.15,-0.08,0.7], [0.15,0.08,1.1])
        uv = fn(xyz.reshape(-1,1,3), r, t, K, D)[0].reshape(-1,2)
        views.append(dict(id=f'synthetic-board-{i}', object_points=xyz.copy(), image_points=uv))
    return intrinsic, views


@pytest.mark.parametrize('model', ['brown', 'fisheye'])
def test_synthetic_intrinsic_uses_explicit_model_and_heldout_views(model):
    api = calibration_module()
    assert hasattr(api, 'calibrate_intrinsic_views'), 'intrinsic calibration missing'
    truth, views = synthetic_boards(model)
    fit = api.calibrate_intrinsic_views(views[:16], views[16:], image_size=[960,540], model=model)
    np.testing.assert_allclose(np.array(fit['K'])[:2,:2], np.array(truth['K'])[:2,:2], atol=0.05)
    assert fit['model'] == model
    assert fit['diagnostics']['heldout_rmse_px'] < 0.001
    assert fit['diagnostics']['training_view_count'] == 16
    assert fit['diagnostics']['heldout_view_count'] == 4
    assert fit['diagnostics']['model_selection'] == 'explicit_user_choice_not_training_rms'
    json.dumps(fit, allow_nan=False)


def test_independent_checks_never_refine_pose_and_can_reject():
    api = calibration_module()
    intrinsic, survey, *_ = synthetic_survey()
    baseline = api.calibrate_survey(intrinsic, survey, calibration_id='test', max_check_error_m=1.)
    survey['check_points'][0]['uv'][0] += 12
    changed = api.calibrate_survey(intrinsic, survey, calibration_id='test', max_check_error_m=1.)
    np.testing.assert_array_equal(changed['Rcw'], baseline['Rcw'])
    np.testing.assert_array_equal(changed['tcw'], baseline['tcw'])
    assert changed['diagnostics']['check_max_error_m'] > 0.1
    with pytest.raises(ValueError, match='Independent check error'):
        api.calibrate_survey(intrinsic, survey, calibration_id='test', max_check_error_m=0.01)


def test_calibration_cli_survey_reads_files_and_intrinsics_extracts_recorded_board(tmp_path):
    assert (ROOT/'calibrate.py').exists(), 'calibration CLI missing'
    intrinsic,survey,*_ = synthetic_survey()
    i,s,out = tmp_path/'intrinsic.json',tmp_path/'survey.json',tmp_path/'camera.json'
    i.write_text(json.dumps(intrinsic)); s.write_text(json.dumps(survey))
    proc = subprocess.run([sys.executable,'calibrate.py','survey','--intrinsics',str(i),'--survey',str(s),
        '--output',str(out),'--calibration-id','synthetic-test','--max-check-error-m','0.01'],cwd=ROOT,capture_output=True,text=True)
    assert proc.returncode == 0, proc.stderr
    assert json.loads(out.read_text())['diagnostics']['check_max_error_m'] < .01
    from calibrate import extract_boards
    path = tmp_path/'SYNTHETIC_checkerboard.avi'
    writer = cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),10,(640,480))
    assert writer.isOpened()
    image = np.full((480,640,3),180,np.uint8)
    for y in range(7):
        for x in range(9):
            color=255 if (x+y)%2 else 0
            cv2.rectangle(image,(100+x*40,100+y*40),(139+x*40,139+y*40),(color,)*3,-1)
    for _ in range(3): writer.write(image)
    writer.release()
    views,size=extract_boards(path,8,6,.04,stride=1)
    assert size == (640,480)
    assert len(views) == 3
    assert np.asarray(views[0]['image_points']).shape == (48,2)
    with pytest.raises(ValueError,match='local'):
        extract_boards('rtsp://not-allowed',8,6,.04,stride=1)


def test_intrinsic_inputs_require_independent_diverse_view_sets():
    api=calibration_module()
    _,views=synthetic_boards('brown')
    with pytest.raises(ValueError,match='independent'):
        api.calibrate_intrinsic_views(views[:16],views[:4],image_size=[960,540],model='brown')
    with pytest.raises(ValueError,match='heldout'):
        api.calibrate_intrinsic_views(views[:16],[],image_size=[960,540],model='brown')
