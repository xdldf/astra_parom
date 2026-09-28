import copy

import cv2
import numpy as np
import pytest

from vehicle_metrology.projective_track import KEYS, RoadProjection, measure_projective_track


def fixture(y=2., angle=.1, span=7., error=.01, height_unit=.7):
    R = np.array([[1,0,0],[0,-1/np.sqrt(5),-2/np.sqrt(5)],
                  [0,2/np.sqrt(5),-1/np.sqrt(5)]])
    center = np.array([0.,-10.,5.])
    K = np.array([[650.,0,640],[0,650,360],[0,0,1]])
    P = K @ np.column_stack((R, -R@center))
    ground = P[:, [0,1,3]]
    raised = ground.copy(); raised[:,2] += height_unit*P[:,2]
    camera = RoadProjection.from_parallel_planes(ground, raised,
        [[-20,-4],[20,-4],[20,12],[-20,12]], [1280,720], error)
    # Independent physical projection: actual Z is metres, NOT barrier units.
    points = np.array([[0,0,0],[2.7,0,0],[-.9,.3,.6],[3.6,.4,.8]])
    c,s=np.cos(angle),np.sin(angle)
    A=np.array([[c,-s,0],[s,c,0],[0,0,1]])
    rows=[]
    for frame,x in enumerate(np.linspace(-span/2,span/2,12)):
        world=points@A.T+[x,y,0]
        uv=cv2.projectPoints(world,cv2.Rodrigues(R)[0],-R@center,K,np.zeros(5))[0][:,0]
        rows.append(dict(frame=frame,sigma_px=.5,**dict(zip(KEYS,uv.tolist()))))
    return camera,rows


@pytest.mark.parametrize('y,angle,height',[(0.,0.,.4),(6.,.2,.7),(2.,-.25,1.3)])
def test_recovers_physical_length_and_ground_distance_across_lanes(y,angle,height):
    camera,rows=fixture(y,angle,height_unit=height)
    result=measure_projective_track(camera,rows)
    assert result['status']=='accepted_conditional',result
    assert result['length_m']==pytest.approx(4.5,abs=1e-7)
    assert result['accuracy_validated'] is False
    np.testing.assert_allclose(camera.center[:2],[0,-10],atol=1e-8)
    assert camera.center[2]==pytest.approx(5/height)
    expected=np.linalg.norm([-3.5+1.35*np.cos(angle),10+y+1.35*np.sin(angle)])
    assert result['frames'][0]['horizontal_camera_distance_m']==pytest.approx(expected)
    assert result['diagnostics']['split_difference_m']<1e-7


def test_bad_survey_is_not_overridden_by_perfect_track():
    camera,rows=fixture(error=.438)
    result=measure_projective_track(camera,rows)
    assert result['length_m'] is None
    assert result['candidate_length_m']==pytest.approx(4.5)
    assert 'calibration_check_exceeds_tolerance' in result['reasons']


def test_occluded_contacts_are_not_inferred_as_ground_truth():
    camera,rows=fixture();rows[2]['occluded']=True
    assert measure_projective_track(camera,rows)['reasons']==['occluded_landmark']


def test_stationary_vehicle_and_independently_rescaled_planes_are_rejected():
    camera,rows=fixture(span=0)
    result=measure_projective_track(camera,rows)
    assert result['length_m'] is None
    ground=camera.H;raised=ground.copy();raised[:,2]+=camera.P[:,2]
    with pytest.raises(ValueError,match='share scale'):
        RoadProjection.from_parallel_planes(ground,raised*2,camera.support,camera.image_size,.01)


def test_repeated_frames_out_of_image_and_bad_noise_do_not_pass():
    camera,rows=fixture()
    with pytest.raises(ValueError,match='distinct'):
        measure_projective_track(camera,rows+[copy.deepcopy(rows[0])])
    rows[0]['front_uv']=[1280.,100.]
    with pytest.raises(ValueError,match='outside image'):
        measure_projective_track(camera,rows)
    camera,rows=fixture();rows[0]['sigma_px']=0
    with pytest.raises(ValueError,match='uncertainty'):
        measure_projective_track(camera,rows)


def test_sensitivity_is_reproducible_and_includes_failures():
    camera,rows=fixture()
    first=measure_projective_track(camera,rows,mc_samples=12,seed=8)
    assert first==measure_projective_track(camera,rows,mc_samples=12,seed=8)
    sensitivity=first['diagnostics']['pixel_sensitivity']
    assert sensitivity['samples_requested']==12
    assert sensitivity['conditional_p95_m'][0]<4.5<sensitivity['conditional_p95_m'][1]
    assert 'calibration_bias' in first['diagnostics']['excludes']


def test_inconsistent_physical_points_do_not_become_precise_lengths():
    camera,rows=fixture()
    for row in rows[::2]:
        row['front_uv'][0]+=25
    result=measure_projective_track(camera,rows)
    assert result['length_m'] is None
    assert set(result['reasons']) & {'high_reprojection_error','inconsistent_frame_subsets'}
