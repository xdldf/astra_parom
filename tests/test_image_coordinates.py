"""Corrected-only annotation inputs must match the actual station image warp."""
import copy
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import pytest

from vehicle_metrology.image_coordinates import (
    CorrectedCoordinates, corrected_survey_to_raw, corrected_tracks_to_raw,
)
from vehicle_metrology.rig import Rig, measure_rig_component
from web_app.workbench import Lens, corrected


def profile(size, **overrides):
    lens=dict(k1=-.205,k2=.04,focal=.47,zoom=.91,cx=.538,cy=.473,tilt_deg=3.6)
    lens.update(overrides)
    return dict(image_size=list(size),lens=lens)


def display_pixels(raw, p):
    """Independent forward transform via OpenCV, used only for test fixtures."""
    w,h=p['image_size'];lens=p['lens'];f=max(w,h)*lens['focal']
    K=np.array([[f,0,w*lens['cx']],[0,f,h*lens['cy']],[0,0,1.]])
    new=K.copy();new[0,0]*=lens['zoom'];new[1,1]*=lens['zoom']
    uv=cv2.undistortPointsIter(np.asarray(raw,float).reshape(-1,1,2),K,
        np.array([lens['k1'],lens['k2'],0,0,0.]),None,new,
        (cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,100,1e-12)).reshape(-1,2)
    rotation=cv2.getRotationMatrix2D(((w-1)/2,(h-1)/2),-lens['tilt_deg'],1.)
    return uv@rotation[:,:2].T+rotation[:,2]


@pytest.mark.parametrize('size,overrides',[
    ((640,480),{}),((480,640),{}),((640,480),dict(k1=0,k2=0,zoom=1)),
    ((640,480),dict(k1=0,k2=0,zoom=1,tilt_deg=0)),
])
def test_inverse_annotation_map_matches_pixels_of_actual_corrected_image(size,overrides):
    p=profile(size,**overrides);transform=CorrectedCoordinates(p,size)
    yy,xx=np.indices((size[1],size[0]),dtype=np.float32)
    coordinates=np.stack([xx,yy],axis=2)
    displayed=corrected(coordinates,Lens(**p['lens']))
    pixels=np.array([[x,y] for y in (150,250,350) for x in (150,250,350)])
    raw,scale=transform.raw(pixels)
    # The renderer quantizes remap/warp interpolation fractions to 1/32 pixel.
    np.testing.assert_allclose(raw,displayed[pixels[:,1],pixels[:,0]],atol=.06)
    np.testing.assert_allclose(display_pixels(raw,p),pixels,atol=1e-8)
    assert np.all(scale > 0)


def test_pixel_uncertainty_uses_local_jacobian_not_a_constant_zoom_factor():
    p=profile((640,480));transform=CorrectedCoordinates(p,p['image_size'])
    uv=np.array([[180.,170.],[440.,360.]])
    _,scale=transform.raw(uv)
    step=1e-4
    jacobian=np.stack([(transform.raw(uv+np.eye(2)[i]*step)[0]-
                        transform.raw(uv-np.eye(2)[i]*step)[0])/(2*step) for i in range(2)],axis=2)
    np.testing.assert_allclose(scale,np.linalg.svd(jacobian,compute_uv=False)[:,0],atol=1e-8)
    assert abs(scale[0]-scale[1])>.01


@pytest.mark.parametrize('defect,message',[
    ('size','image_size'),('missing_lens','seven'),('rotation','rotation border'),
    ('remap','remap border'),('fold','folds'),('outside','inside'),
])
def test_ambiguous_profiles_black_borders_and_wrong_dimensions_are_rejected(defect,message):
    p=profile((640,480));pixel=[[200,200]]
    if defect=='size': p['image_size']=[320,240]
    if defect=='missing_lens': del p['lens']['tilt_deg']
    if defect=='rotation': pixel=[[0,0]]
    if defect=='remap': p['lens'].update(k1=.5,k2=.2,zoom=.5,tilt_deg=0);pixel=[[80,80]]
    if defect=='fold': p['lens'].update(k1=-.8,k2=0,zoom=.5,tilt_deg=0);pixel=[[80,80]]
    if defect=='outside': pixel=[[-1,10]]
    with pytest.raises(ValueError,match=message): CorrectedCoordinates(p,(640,480)).raw(pixel)


def test_survey_cli_accepts_only_explicit_corrected_profile_and_preserves_evidence(tmp_path):
    from test_calibration import spatial_survey
    from calibrate import main
    intrinsic,survey,_,_,center=spatial_survey()
    p=profile(intrinsic['image_size'])
    survey['coordinate_space']='corrected_full_resolution'
    for key in ('control_points','check_points','raised_control_points','raised_check_points'):
        for row in survey[key]: row['uv']=display_pixels([row['uv']],p)[0].tolist()
    original=copy.deepcopy(survey)
    converted=corrected_survey_to_raw(survey,p)
    assert survey==original and converted['coordinate_space']=='raw_distorted_pixels'
    paths={k:tmp_path/(k+'.json') for k in ('intrinsics','survey','profile','result')}
    for key,data in [('intrinsics',intrinsic),('survey',survey),('profile',p)]:
        paths[key].write_text(json.dumps(data))
    args=['survey','--intrinsics',str(paths['intrinsics']),'--survey',str(paths['survey']),
          '--calibration-id','corrected-fixture','--max-check-error-m','.03','--output',str(paths['result'])]
    with pytest.raises(SystemExit) as exc: main(args)
    assert exc.value.code==2 and not paths['result'].exists()
    main(args+['--display-profile',str(paths['profile'])])
    output=json.loads(paths['result'].read_text())
    np.testing.assert_allclose(output['diagnostics']['camera_center_m'],center,atol=1e-5)
    assert len(output['source_hashes']['display_profile'])==64
    assert json.loads(paths['survey'].read_text())==original
    # An explicitly corrected v1 survey must not bypass the coordinate check.
    survey['schema_version']=1
    from vehicle_metrology.calibration import calibrate_survey
    with pytest.raises(ValueError,match='coordinate_space'):
        calibrate_survey(intrinsic,survey,calibration_id='wrong',max_check_error_m=.03)


def test_corrected_multicamera_tracks_include_occlusions_and_component_annotations(tmp_path):
    from test_rig import scene
    rig_data,poses,endpoints=scene()
    rig=Rig.from_dict(rig_data)
    profiles={k:profile(c.image_size,tilt_deg=3.6 if k=='ST' else -2.) for k,c in rig.cameras.items()}
    for row in poses:
        for key in ('rear_contact_uv','front_contact_uv'):
            row[key]=display_pixels([row[key]],profiles['ST'])[0].tolist()
    for row in endpoints:
        row['uv']=display_pixels([row['uv']],profiles[row['camera_id']])[0].tolist()
    poses[1].update(visible=False,rear_contact_uv=None,front_contact_uv=None)
    endpoints[1].update(visible=False,uv=None)
    component=dict(component_id='body',pose_observations=poses,endpoints=endpoints)
    data=dict(schema_version=1,coordinate_space='corrected_full_resolution',rig_id=rig.rig_id,
              tracks=[dict(track_id='vehicle',components=[component],composition_verified=True)])
    saved=copy.deepcopy(data)
    converted=corrected_tracks_to_raw(data,profiles,rig)
    part=converted['tracks'][0]['components'][0]
    result=measure_rig_component(rig,part['pose_observations'],part['endpoints'])
    assert result['length_m']==pytest.approx(4.5,abs=1e-7)
    assert data==saved
    with pytest.raises(ValueError,match='every camera'): corrected_tracks_to_raw(data,{'ST':profiles['ST']},rig)
    rig_path=tmp_path/'rig.json';observations=tmp_path/'observations.json';output=tmp_path/'output'
    rig_path.write_text(json.dumps(rig_data));observations.write_text(json.dumps(data))
    command=[sys.executable,'scripts/reconstruct_camera_pair.py','--rig',str(rig_path),
             '--observations',str(observations),'--output-dir',str(output)]
    for key,p in profiles.items():
        path=tmp_path/(key+'.json');path.write_text(json.dumps(p))
        command+=['--display-profile',key+'='+str(path)]
    completed=subprocess.run(command,cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True)
    assert completed.returncode==0,completed.stderr
    result=json.loads((output/'results.json').read_text())
    assert result['tracks'][0]['length_m']==pytest.approx(4.5,abs=1e-7)
    assert set(json.loads((output/'manifest.json').read_text())['display_profile_sha256'])==set(profiles)
