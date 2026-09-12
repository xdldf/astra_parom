import importlib.util
import numpy as np


def test_metric_ray_roundtrip_including_distorted_edges():
    assert importlib.util.find_spec('vehicle_metrology') is not None, 'geometry package not implemented'
    from vehicle_metrology.geometry import Camera
    R = np.array([[1,0,0],[0,-1/np.sqrt(5),-2/np.sqrt(5)],[0,2/np.sqrt(5),-1/np.sqrt(5)]])
    center = np.array([0,-10,5])
    points = np.array([[-7,0,0],[0,2,0],[7,5,0]], float)
    for model, D in [('brown',[-.15,.025,.001,-.002,0]),('fisheye',[-.04,.003,0,0])]:
        camera = Camera.from_dict(dict(schema_version=1,image_size=[960,540],model=model,
            K=[[700,0,480],[0,700,270],[0,0,1]], D=D,Rcw=R.tolist(),tcw=(-R@center).tolist(),
            road_polygon=[[-15,-3],[15,-3],[15,8],[-15,8]],calibration_id='synthetic'))
        np.testing.assert_allclose(camera.ground(camera.project(points)), points, atol=1e-7)


def synthetic_track(y=1.0, angle=0.0, span=8.0, n=16):
    from vehicle_metrology.geometry import Camera
    R = np.array([[1,0,0],[0,-1/np.sqrt(5),-2/np.sqrt(5)],[0,2/np.sqrt(5),-1/np.sqrt(5)]])
    camera = Camera.from_dict(dict(schema_version=1,image_size=[1280,720],model='fisheye',
        K=[[650,0,640],[0,650,360],[0,0,1]],D=[-.03,.001,0,0],Rcw=R.tolist(),
        tcw=(-R@np.array([0,-10,5])).tolist(),road_polygon=[[-20,-4],[20,-4],[20,12],[-20,12]],calibration_id='synthetic'))
    A = np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
    body = np.array([[0,0,0],[2.7,0,0],[-.9,.3,.6],[3.6,.4,.8]])
    observations = []
    for i, x in enumerate(np.linspace(-span/2,span/2,n)):
        uv = camera.project(body@A.T + [x,y,0])
        observations.append(dict(frame=i, **{k:v.tolist() for k,v in zip(
            ['rear_contact_uv','front_contact_uv','rear_uv','front_uv'],uv)}))
    return camera, observations


def test_multiview_recovers_elevated_extremes_across_lanes_and_headings():
    import vehicle_metrology.geometry as geometry
    assert hasattr(geometry,'measure_track'), 'multi-view measurement not implemented'
    for y,angle in [(0.,0.),(6.,0.),(2.,.25),(4.,-.2)]:
        camera, observations = synthetic_track(y,angle)
        result = geometry.measure_track(camera, observations)
        assert result['status'] == 'accepted_conditional', result
        assert abs(result['length_m']-4.5) < 1e-6
        assert result['diagnostics']['reprojection_rmse_px'] < 1e-6


def test_no_baseline_rejects_and_seeded_noise_reports_only_conditional_uncertainty():
    from vehicle_metrology.geometry import measure_track
    camera, observations = synthetic_track(span=0)
    rejected = measure_track(camera,observations)
    assert rejected['length_m'] is None
    assert 'weak_multiview_geometry' in rejected['reasons']
    camera, observations = synthetic_track()
    first = measure_track(camera,observations,mc_samples=12,seed=42)
    second = measure_track(camera,observations,mc_samples=12,seed=42)
    assert first == second
    uncertainty = first['diagnostics']['uncertainty']
    assert uncertainty['samples_successful'] == 12
    assert uncertainty['conditional_p95_m'][0] < 4.5 < uncertainty['conditional_p95_m'][1]
    assert 'calibration' in uncertainty['excludes']
    assert first['frames'][-1]['prefix_length_m'] is not None
    assert first['frames'][-1]['window_length_m'] is not None


def test_invalid_geometric_rays_reject_track_without_crashing():
    from vehicle_metrology.geometry import measure_track
    camera, observations = synthetic_track()
    for row in observations:
        row['rear_contact_uv'] = [640.,0.]
    result = measure_track(camera,observations)
    assert result['length_m'] is None
    assert 'invalid_ground_geometry' in result['reasons']
