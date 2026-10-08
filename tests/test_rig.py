"""Independent projected scenes test implementation, not field accuracy."""
import copy

import cv2
import numpy as np
import pytest

from vehicle_metrology.rig import Rig, measure_rig_component, measure_rig_passage


def scene(stationary=False, offset=.12, body_shift=0., heading_shift=0., n=9):
    cameras={}
    for name,center in [('ST',[-3,-10,5]),('HiWatch',[8,-6,4])]:
        C=np.asarray(center,float)
        forward=np.array([0.,2.,0.])-C;forward/=np.linalg.norm(forward)
        right=np.cross(forward,[0,0,1]);right/=np.linalg.norm(right)
        down=np.cross(forward,right)
        R=np.array([right,down,forward]);t=-R@C
        calibration=dict(schema_version=1,image_size=[1920,1080],model='brown',
            K=[[900.,0,960],[0,900.,540],[0,0,1]],D=[-.08,.01,0,0,0],Rcw=R.tolist(),tcw=t.tolist(),
            road_polygon=[[-12,-4],[12,-4],[12,12],[-12,12]],calibration_id='synthetic-'+name,
            world_frame_id='synthetic-site',diagnostics=dict(survey_schema_version=2,
                check_points_used_for_fit=False,guarded_check_max_error_m=.002))
        cameras[name]=dict(calibration=calibration,time_offset_s=0 if name=='ST' else offset,
                           time_uncertainty_s=.001,synchronization_verified=True)
    data=dict(schema_version=1,coordinate_space='raw_distorted_pixels',world_frame_id='synthetic-site',
              rig_id='synthetic-rig',pose_camera='ST',cameras=cameras)
    def project(name,xyz):
        c=cameras[name]['calibration']
        return cv2.projectPoints(np.asarray(xyz,float),cv2.Rodrigues(np.asarray(c['Rcw']))[0],
            np.asarray(c['tcw']),np.asarray(c['K']),np.asarray(c['D']))[0].reshape(-1,2)
    body=np.array([[0.,0,0],[2.7,0,0],[-.9,.3,.6],[3.6,.4,.8]])
    poses=[];endpoints=[]
    for i,stamp in enumerate(np.linspace(1,2,n)):
        angle=(.2 if stationary else .2+.05*(stamp-1))+heading_shift
        R=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
        T=np.array([0 if stationary else 2*(stamp-1),2,0])+R[:,0]*body_shift
        world=body@R.T+T
        uv=project('ST',world)
        poses.append(dict(timestamp_s=float(stamp),visible=True,rear_contact_uv=uv[0].tolist(),front_contact_uv=uv[1].tolist()))
        for name in cameras:
            pixels=project(name,world)
            for key,point in zip(('rear_extreme','front_extreme'),pixels[2:]):
                endpoints.append(dict(camera_id=name,landmark_id=key,uv=point.tolist(),sigma_px=.2,
                    timestamp_s=float(stamp-cameras[name]['time_offset_s']),visible=True))
    return data,poses,endpoints


@pytest.mark.parametrize('stationary',[True,False])
def test_two_cameras_recover_elevated_endpoints_with_capture_time_offset(stationary):
    data,poses,endpoints=scene(stationary)
    result=measure_rig_component(Rig.from_dict(data),poses,endpoints,complete_vehicle=True)
    assert result['status']=='accepted_conditional',result
    assert result['length_m']==pytest.approx(4.5,abs=1e-7)
    assert result['diagnostics']['used_cameras']==['HiWatch','ST']
    assert result['accuracy_validated'] is False
    np.testing.assert_allclose(result['diagnostics']['endpoints']['rear_extreme']['body_xyz_m'],[-.9,.3,.6],atol=1e-7)


def test_second_view_resolves_stationary_single_camera_degeneracy():
    data,poses,endpoints=scene(stationary=True)
    result=measure_rig_component(Rig.from_dict(data),poses,[o for o in endpoints if o['camera_id']=='ST'])
    assert result['length_m'] is None
    assert 'weak_multiview_geometry' in result['reasons']


def test_occlusion_uses_supported_visible_observations_without_inventing_hidden_contacts():
    data,poses,endpoints=scene()
    for row in endpoints[:4]: row['visible']=False
    result=measure_rig_component(Rig.from_dict(data),poses,endpoints)
    assert result['length_m']==pytest.approx(4.5,abs=1e-7)
    assert result['diagnostics']['skipped_occluded_observations']==4
    poses[1]['visible']=False
    poses[1]['rear_contact_uv']=None;poses[1]['front_contact_uv']=None
    result=measure_rig_component(Rig.from_dict(data),poses,endpoints)
    assert result['length_m']==pytest.approx(4.5,abs=1e-7)
    assert result['diagnostics']['skipped_occluded_ground_poses']==1
    for row in poses: row['visible']=False
    assert measure_rig_component(Rig.from_dict(data),poses,endpoints)['reasons']==['insufficient_visible_ground_poses']


def test_timing_and_survey_budget_failures_withhold_numeric_measurement():
    data,poses,endpoints=scene()
    data['cameras']['HiWatch']['time_uncertainty_s']=.1
    result=measure_rig_component(Rig.from_dict(data),poses,endpoints)
    assert result['length_m'] is None
    assert 'synchronization_error_budget_exceeded' in result['reasons']
    assert result['diagnostics']['timing_displacement_m'] > .1
    data['cameras']['ST']['calibration']['diagnostics']['guarded_check_max_error_m']=.04
    assert measure_rig_component(Rig.from_dict(data),poses,endpoints)['reasons']==['calibration_error_budget_exceeded']


def test_no_pose_extrapolation_or_interpolation_through_long_missing_interval():
    data,poses,endpoints=scene()
    result=measure_rig_component(Rig.from_dict(data),poses[1:],endpoints)
    assert result['length_m']==pytest.approx(4.5,abs=1e-7)
    assert result['diagnostics']['unsupported_endpoint_observations']=={'endpoint_outside_pose_time_support':4}
    result=measure_rig_component(Rig.from_dict(data),[poses[0],poses[-1]],endpoints)
    assert result['length_m']==pytest.approx(4.5,abs=1e-7)
    assert result['diagnostics']['observation_count']==8
    assert result['diagnostics']['unsupported_endpoint_observations']=={'pose_interpolation_gap':28}
    assert measure_rig_component(Rig.from_dict(data),poses[1:],endpoints[:4])['reasons']==['insufficient_visible_endpoints']


def test_wrong_endpoint_identity_generates_reprojection_failure():
    data,poses,endpoints=scene()
    for row in endpoints:
        if row['camera_id']=='HiWatch' and row['landmark_id']=='rear_extreme': row['uv'][0]+=45
    result=measure_rig_component(Rig.from_dict(data),poses,endpoints)
    assert result['length_m'] is None
    assert 'high_reprojection_error' in result['reasons']


@pytest.mark.parametrize('change',[{'world_frame_id':'other'},
    {'diagnostics':{'survey_schema_version':1,'check_points_used_for_fit':False,'guarded_check_max_error_m':.001}}])
def test_incompatible_or_unsurveyed_cameras_cannot_be_combined(change):
    data,_,_=scene()
    data['cameras']['HiWatch']['calibration'].update(change)
    with pytest.raises(ValueError): Rig.from_dict(data)


def test_missing_sync_duplicate_observations_and_invalid_points_fail_explicitly():
    data,poses,endpoints=scene()
    bad=copy.deepcopy(data);bad['cameras']['HiWatch']['synchronization_verified']=False
    with pytest.raises(ValueError,match='synchronization'): Rig.from_dict(bad)
    rig=Rig.from_dict(data)
    with pytest.raises(ValueError,match='Duplicate'): measure_rig_component(rig,poses,endpoints+[endpoints[0]])
    endpoints[0]['uv']=[-1,20]
    with pytest.raises(ValueError,match='outside'): measure_rig_component(rig,poses,endpoints)


@pytest.mark.parametrize('shift,expected',[(-6.,10.5),(-3.,7.5)])
def test_complete_combination_includes_gap_without_double_counting_overlap(shift,expected):
    data,poses,endpoints=scene(stationary=True)
    _,second_poses,second_endpoints=scene(stationary=True,body_shift=shift)
    parts=[dict(component_id='tractor',pose_observations=poses,endpoints=endpoints),
           dict(component_id='trailer',pose_observations=second_poses,endpoints=second_endpoints)]
    result=measure_rig_passage(Rig.from_dict(data),parts,composition_verified=True)
    assert result['length_m']==pytest.approx(expected,abs=1e-7),result
    assert result['complete_vehicle'] is True
    assert result['accuracy_validated'] is False
    assert measure_rig_passage(Rig.from_dict(data),parts)['length_m'] is None
    parts[1]['endpoints']=[]
    result=measure_rig_passage(Rig.from_dict(data),parts,composition_verified=True)
    assert result['reasons']==['incomplete_component_measurements']
    assert result['length_m'] is None


def test_bent_combination_is_not_reported_as_straight_occupied_length():
    data,poses,endpoints=scene(stationary=True)
    _,p2,e2=scene(stationary=True,body_shift=-4,heading_shift=.15)
    parts=[dict(component_id='tractor',pose_observations=poses,endpoints=endpoints),
           dict(component_id='trailer',pose_observations=p2,endpoints=e2)]
    result=measure_rig_passage(Rig.from_dict(data),parts,composition_verified=True)
    assert result['reasons']==['no_shared_straight_interval']
    assert result['length_m'] is None


def test_camera_pair_cli_records_inputs_and_evaluates_only_independent_references(tmp_path):
    import json,subprocess,sys
    from pathlib import Path
    data,poses,endpoints=scene(stationary=True)
    rig,observations,truth=tmp_path/'rig.json',tmp_path/'observations.json',tmp_path/'truth.csv'
    rig.write_text(json.dumps(data))
    observations.write_text(json.dumps(dict(schema_version=1,coordinate_space='raw_distorted_pixels',
        rig_id='synthetic-rig',tracks=[dict(track_id='synthetic-case',complete_vehicle=True,
            pose_observations=poses,endpoints=endpoints)])))
    # This fixture validates CSV flow only, not physical metrology.
    truth.write_text('track_id,vehicle_id,length_m,uncertainty_m,reference_source,dataset_role\n'
                     'synthetic-case,fixture-body,4.5,.01,physical_measurement,validation\n')
    output=tmp_path/'output'
    p=subprocess.run([sys.executable,'scripts/reconstruct_camera_pair.py','--rig',str(rig),
        '--observations',str(observations),'--ground-truth',str(truth),'--output-dir',str(output)],
        cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True)
    assert p.returncode==0,p.stderr
    result=json.loads((output/'results.json').read_text())
    assert not result['accuracy_validated']
    assert result['tracks'][0]['length_m']==pytest.approx(4.5,abs=1e-7)
    assert json.loads((output/'evaluation.json').read_text())['acceptance']['passed_tracks']==1
    manifest=json.loads((output/'manifest.json').read_text())
    assert len(manifest['ground_truth_sha256'])==64


def noisy_scene(*,seed=0,sigma=1.,n=9,body_shift=0.):
    data,poses,endpoints=scene(n=n,body_shift=body_shift)
    rng=np.random.default_rng(seed)
    for row in poses:
        row['sigma_px']=sigma
        for key in ('rear_contact_uv','front_contact_uv'):
            row[key]=(np.asarray(row[key])+rng.normal(0,sigma,2)).tolist()
    for row in endpoints:
        row['sigma_px']=sigma
        row['uv']=(np.asarray(row['uv'])+rng.normal(0,sigma,2)).tolist()
    return data,poses,endpoints


def test_joint_passage_fit_handles_noisy_contacts_without_relaxing_pixel_checks():
    for seed in range(8):
        data,poses,endpoints=noisy_scene(seed=seed)
        result=measure_rig_component(Rig.from_dict(data),poses,endpoints)
        assert result['status']=='accepted_conditional',result
        assert result['diagnostics']['wheelbase_range_m']>.05  # Old exact-contact gate rejected these.
        assert abs(result['length_m']-4.5)<.025
        assert result['diagnostics']['bundle']['contact_reprojection_rmse_px']<1.5
        assert abs(result['diagnostics']['fitted_wheelbase_m']-2.7)<.025
        assert result['accuracy_validated'] is False


def test_joint_fit_rejects_wrong_contact_instead_of_hiding_it_in_mean_error():
    data,poses,endpoints=scene()
    poses[4]['front_contact_uv'][0]+=35
    result=measure_rig_component(Rig.from_dict(data),poses,endpoints)
    assert result['length_m'] is None
    assert 'ground_contact_outlier' in result['reasons']


def test_sparse_joint_fit_and_combination_use_refined_body_poses():
    data,poses,endpoints=noisy_scene(seed=9,n=48)
    _,p2,e2=noisy_scene(seed=10,n=48,body_shift=-6)
    rig=Rig.from_dict(data)
    parts=[dict(component_id='tractor',pose_observations=poses,endpoints=endpoints),
           dict(component_id='trailer',pose_observations=p2,endpoints=e2)]
    result=measure_rig_passage(rig,parts,composition_verified=True)
    assert result['length_m']==pytest.approx(10.5,abs=.03),result
    assert result['diagnostics']['length_range_m']<.05
    assert all(p['diagnostics']['bundle']['converged'] for p in result['components'])
