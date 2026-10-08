"""Synthetic geometry checks do not establish real vehicle accuracy."""
import copy

import cv2
import numpy as np
import pytest

from vehicle_metrology.road_survey import fit_rulers,fit_ground_transfer,projective_span
from vehicle_metrology.bbox_scale import measure_box
from web_app.workbench import Profile,profile_scale


def synthetic():
    width,height=2592,1944
    K=np.array([[.47*width,0,width/2],[0,.47*width,height/2],[0,0,1.]])
    D=np.array([-.19,.02,0,0,0.])
    center=np.array([7,-10,6.])
    forward=np.array([0,13,-6.]);forward/=np.linalg.norm(forward)
    right=np.cross(forward,[0,0,1]);right/=np.linalg.norm(right)
    R=np.stack([right,np.cross(forward,right),forward]);t=-R@center
    def project(points):
        return cv2.projectPoints(np.array(points,float),cv2.Rodrigues(R)[0],t,K,D)[0][:,0]
    n=14;theta=-.02
    near=np.c_[np.arange(n),np.zeros(n),np.zeros(n)]
    far=np.c_[2.3+np.arange(n)*np.cos(theta),5.8+np.arange(n)*np.sin(theta),np.zeros(n)]
    rows=[dict(points_px=project(p).tolist(),distances_m=list(range(n)),sigma_px=1.) for p in [near,far]]
    widths=[]
    for i,j in [(2,0),(12,10)]:
        widths.append(dict(points_px=project([near[i],far[j]]).tolist(),distance_m=float(np.linalg.norm(near[i]-far[j]))))
    posts=[]
    for row in [near,far]:
        for i in [1,5,9,12]:
            top=row[i];foot=top+[0,0,-.7]
            posts.append(dict(name=str(len(posts)),points_px=project([top,foot]).tolist()))
    return dict(image_size=[width,height],rulers=rows,widths=widths,posts=posts,barrier_height_m=None)


def test_metric_rulers_and_raised_plane_on_independent_synthetic_projection():
    data=synthetic()
    fit=fit_rulers(data,heldout=[3,7,11,17,21,25])
    assert fit['optimizer_success'] and fit['radial_monotonic']
    assert fit['heldout_max_error_m']<.01
    assert fit['accuracy_validated'] is False
    ground=fit_ground_transfer(data,fit)
    assert ground['max_post_heldout_error_m']<.01
    # A misplaced/leaning support is reported by a held-out check.
    data['posts'][1]['points_px'][1][0]+=25
    bad=fit_ground_transfer(data,fit)
    assert bad['max_post_heldout_error_m']>.1


def test_invalid_rulers_do_not_reach_optimizer():
    data=synthetic()
    data['rulers'][0]['distances_m'][3]=1
    with pytest.raises(ValueError,match='ordered'):
        fit_rulers(data)
    data=synthetic()
    data['rulers'][0]['points_px'][0]=[-1,10]
    with pytest.raises(ValueError,match='inside'):
        fit_rulers(data)


def test_span_follows_projected_travel_direction():
    H=np.array([[40,12,50],[4,-20,420],[.01,.003,1.]])
    world=np.array([[[3.,2.],[7.,2.]]]);image=cv2.perspectiveTransform(world,H)[0]
    left,right=image[:,0];mid=(left+right)/2
    line=np.cross(np.r_[image[0],1],np.r_[image[1],1]);bottom=-(line[0]*mid+line[2])/line[1]
    length,ends=projective_span([left,bottom-30,right-left,30],H)
    assert length==pytest.approx(4)
    np.testing.assert_allclose(ends,world[0],atol=1e-9)


def profile(ground_error):
    return Profile(image_size=(600,500),lens={},polygon=[(50,420),(530,420),(590,300),(110,300)],
        measurement_line_x=280,survey_calibration=dict(
            projection=[[40,10,50],[0,-20,420],[0,0,1]],support_world=[[0,0],[12,0],[12,6],[0,6]],
            lens={},ruler_check_error_m=.02,ground_check_error_m=ground_error))


def test_failed_ground_fit_never_becomes_a_numeric_measurement():
    p=profile(.44);scale=profile_scale(p)
    result=measure_box([200,250,160,100],p.polygon,scale,p.image_size,280)
    assert result['length_m'] is None and result['status']=='calibration_review'
    assert result['candidate_length_m']==pytest.approx(4)
    p=profile(.02)
    result=measure_box([200,250,160,100],p.polygon,profile_scale(p),p.image_size,280)
    assert result['length_m']==pytest.approx(4)
    assert result['calibration_diagnostics']['accuracy_validated'] is False


def test_survey_cannot_be_reused_after_lens_changes_or_with_legacy_refs():
    data=profile(.02).model_dump()
    broken=copy.deepcopy(data);broken['lens']['k1']=-.2
    with pytest.raises(ValueError,match='different lens'):
        Profile.model_validate(broken)
    broken=copy.deepcopy(data);broken['references']=[dict(bbox=[200,250,160,100],length_m=4)]
    with pytest.raises(ValueError,match='cannot mix'):
        Profile.model_validate(broken)
    broken=copy.deepcopy(data);broken['survey_calibration']['projection']=[[40,10,50],[0,-20,420],[0,0,0]]
    with pytest.raises(ValueError,match='Singular'):
        Profile.model_validate(broken)


def test_vehicle_references_never_silently_reweight_a_survey(monkeypatch):
    from web_app import station
    from web_app import evaluation_review
    from web_app.calibration_references import merge
    def forbidden():
        raise AssertionError('Survey fit must not import vehicle reference scale')
    monkeypatch.setattr(station,'connect',forbidden)
    monkeypatch.setattr(evaluation_review,'samples_for',lambda profile:[])
    p=profile(.44)
    assert merge(p) == p  # Evaluation metadata may be copied; the survey fit is unchanged.
