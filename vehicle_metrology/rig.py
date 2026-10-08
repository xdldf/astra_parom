"""Conditional rigid-component reconstruction using calibrated camera passages.

Ground contacts define a time-varying body pose. The SAME physical endpoint is
then triangulated in body coordinates using rays from any calibrated camera.
Pixel coordinates are internal raw pixels; saved review images stay corrected.
This solver does not discover landmarks, synchronize cameras or certify accuracy.
"""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from .geometry import Camera, _poses


@dataclass
class Rig:
    cameras: dict
    pose_camera: str
    offsets_s: dict
    time_uncertainty_s: dict
    calibration_errors_m: dict
    world_frame_id: str
    rig_id: str

    @classmethod
    def from_dict(cls, data):
        if data.get('schema_version') != 1 or data.get('coordinate_space') != 'raw_distorted_pixels':
            raise ValueError('Rig requires schema_version 1 and raw_distorted_pixels')
        world=data.get('world_frame_id')
        ident=data.get('rig_id')
        if not isinstance(world,str) or not world.strip() or not isinstance(ident,str) or not ident.strip():
            raise ValueError('Rig and world frame IDs are required')
        entries=data.get('cameras',{})
        if not isinstance(entries,dict) or not 1 <= len(entries) <= 4 or data.get('pose_camera') not in entries:
            raise ValueError('Rig needs 1–4 named cameras and a pose_camera')
        cameras,offsets,uncertainties,errors={},{},{},{}
        for key,entry in entries.items():
            if not isinstance(key,str) or not key.strip():
                raise ValueError('Camera IDs must be nonempty strings')
            calibration=entry['calibration']
            if calibration.get('world_frame_id') != world:
                raise ValueError('Cameras must share the same surveyed world_frame_id')
            diagnostics=calibration.get('diagnostics',{})
            if diagnostics.get('survey_schema_version') != 2 or diagnostics.get('check_points_used_for_fit') is not False:
                raise ValueError('A v2 spatial survey with independent checks is required')
            error=diagnostics.get('guarded_check_max_error_m')
            offset=entry.get('time_offset_s')
            uncertainty=entry.get('time_uncertainty_s')
            if any(type(v) not in (float,int) or not np.isfinite(v) for v in (error,offset,uncertainty)):
                raise ValueError('Finite survey errors, time offsets and timing uncertainties required')
            if error < 0 or uncertainty < 0 or entry.get('synchronization_verified') is not True:
                raise ValueError('Verified synchronization and nonnegative uncertainties required')
            cameras[key]=Camera.from_dict(calibration)
            offsets[key]=float(offset)
            uncertainties[key]=float(uncertainty)
            errors[key]=float(error)
        return cls(cameras,data['pose_camera'],offsets,uncertainties,errors,world,ident)


def _timestamp(row):
    value=row.get('timestamp_s')
    if type(value) not in (int,float) or not np.isfinite(value) or value < 0:
        raise ValueError('Observation timestamp_s must be finite and nonnegative')
    return float(value)


def _pixels(camera, values):
    pixels=np.asarray(values,float)
    if pixels.ndim != 2 or pixels.shape[1] != 2 or not np.isfinite(pixels).all():
        raise ValueError('Finite Nx2 landmark pixels required')
    if np.any(pixels < 0) or np.any(pixels >= camera.image_size):
        raise ValueError('Landmark outside calibrated image')
    return pixels


def _rotation(angle):
    c,s=np.cos(angle),np.sin(angle)
    return np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])


def _pose_at(time, times, translations, headings, max_gap_s):
    if time < times[0]-1e-9 or time > times[-1]+1e-9:
        raise ValueError('endpoint_outside_pose_time_support')
    i=int(np.searchsorted(times,time))
    if i < len(times) and abs(times[i]-time) <= 1e-9:
        return translations[i],_rotation(headings[i])
    if i == 0 or i == len(times) or times[i]-times[i-1] > max_gap_s:
        raise ValueError('pose_interpolation_gap')
    alpha=(time-times[i-1])/(times[i]-times[i-1])
    return ((1-alpha)*translations[i-1]+alpha*translations[i],
            _rotation((1-alpha)*headings[i-1]+alpha*headings[i]))


def _fit_endpoint(views):
    directions=np.asarray([R.T@camera.rays([uv])[0] for camera,uv,T,R,sigma in views])
    origins=np.asarray([R.T@(camera.center-T) for camera,uv,T,R,sigma in views])
    weights=np.asarray([1/sigma**2 for camera,uv,T,R,sigma in views])
    projection=np.eye(3)[None]-directions[:,:,None]*directions[:,None,:]
    A=np.einsum('n,nij->ij',weights,projection)
    b=np.einsum('n,nij,nj->i',weights,projection,origins)
    condition=float(np.linalg.cond(A))
    angle=float(np.degrees(np.arccos(np.clip(np.min(directions@directions.T),-1,1))))
    initial=np.linalg.lstsq(A,b,rcond=None)[0]

    def residual(point):
        errors=[]
        for camera,uv,T,R,sigma in views:
            world=R@point+T
            if (camera.Rcw@world+camera.tcw)[2] <= 0:
                return np.full(2*len(views),1e6)
            errors.extend((camera.project([world])[0]-uv)/sigma)
        return np.asarray(errors)

    fit=least_squares(residual,initial,loss='soft_l1',max_nfev=200)
    errors=residual(fit.x).reshape(-1,2)*np.asarray([v[4] for v in views])[:,None]
    if any((c.Rcw@(R@fit.x+T)+c.tcw)[2] <= 0 for c,uv,T,R,s in views):
        raise ValueError('endpoint_behind_camera')
    return fit.x,dict(ray_angle_deg=angle,condition_number=condition if np.isfinite(condition) else None,
                     reprojection_rmse_px=float(np.sqrt(np.mean(errors**2))),converged=bool(fit.success))


def measure_rig_component(rig, pose_observations, endpoints, *, tolerance_m=.1,
                          complete_vehicle=False, max_pose_gap_s=.25):
    """Fit front/rear physical extremes of one rigid body over its whole passage.

    Each endpoint has landmark_id front_extreme/rear_extreme, camera_id,
    timestamp_s, uv, sigma_px and visible. Ground contacts refer to the SAME
    front/rear tyres on the pose camera in all frames. An articulated combination
    needs separately reconstructed components; it is not one rigid body.
    Time convention: rig_time = decoded_timestamp + camera.time_offset_s.
    """
    if not np.isfinite(tolerance_m) or tolerance_m <= 0 or not np.isfinite(max_pose_gap_s) or max_pose_gap_s <= 0:
        raise ValueError('Positive finite tolerance and interpolation gap required')
    result=dict(strategy='surveyed_multicamera_rigid_component',rig_id=rig.rig_id,
                calibration_ids={k:c.calibration_id for k,c in rig.cameras.items()},
                status='rejected',length_m=None,candidate_length_m=None,reasons=[],diagnostics={},
                complete_vehicle=complete_vehicle is True,accuracy_validated=False)
    if max(rig.calibration_errors_m.values()) > tolerance_m/3:
        result['reasons']=['calibration_error_budget_exceeded']
        return result
    rows=sorted(pose_observations,key=_timestamp)
    times=np.asarray([_timestamp(o)+rig.offsets_s[rig.pose_camera] for o in rows])
    if len(rows) < 2:
        result['reasons']=['insufficient_ground_poses']
        return result
    if np.any(np.diff(times) <= 0):
        raise ValueError('Ground pose timestamps must be distinct')
    if any(o.get('visible') is not True for o in rows):
        result['reasons']=['occluded_ground_contacts']
        return result
    pose_camera=rig.cameras[rig.pose_camera]
    _pixels(pose_camera,[o[k] for o in rows for k in ('rear_contact_uv','front_contact_uv')])
    try:
        translations,rotations,wheelbases,contacts=_poses(pose_camera,rows)
    except ValueError as exc:
        result['reasons']=['invalid_ground_geometry'];result['diagnostics']['detail']=str(exc)
        return result
    if not pose_camera.in_road(contacts.reshape(-1,3)):
        result['reasons']=['outside_calibrated_road']
        return result
    headings=np.unwrap(np.arctan2(rotations[:,1,0],rotations[:,0,0]))
    wheelbase_range=float(np.ptp(wheelbases))
    if wheelbase_range > tolerance_m/2:
        result['reasons']=['inconsistent_ground_anchors']
        result['diagnostics']['wheelbase_range_m']=wheelbase_range
        return result
    grouped={'front_extreme':[],'rear_extreme':[]}
    seen=set();used_cameras=set();skipped=0
    try:
        for row in endpoints:
            key=row.get('landmark_id');camera_id=row.get('camera_id')
            if key not in grouped or camera_id not in rig.cameras:
                raise ValueError('Unknown endpoint landmark_id or camera_id')
            stamp=_timestamp(row)
            identity=(key,camera_id,stamp)
            if identity in seen: raise ValueError('Duplicate endpoint observation')
            seen.add(identity)
            if row.get('visible') is not True:
                skipped+=1;continue
            sigma=row.get('sigma_px')
            if type(sigma) not in (int,float) or not np.isfinite(sigma) or sigma <= 0:
                raise ValueError('Positive finite endpoint sigma_px required')
            camera=rig.cameras[camera_id]
            uv=_pixels(camera,[row['uv']])[0]
            T,R=_pose_at(stamp+rig.offsets_s[camera_id],times,translations,headings,max_pose_gap_s)
            grouped[key].append((camera,uv,T,R,float(sigma)))
            used_cameras.add(camera_id)
    except ValueError as exc:
        if str(exc) not in ('endpoint_outside_pose_time_support','pose_interpolation_gap'): raise
        result['reasons']=[str(exc)]
        return result
    if min(map(len,grouped.values())) < 4:
        result['reasons']=['insufficient_visible_endpoints']
        return result
    speed=float(np.max(np.linalg.norm(np.diff(translations,axis=0),axis=1)/np.diff(times)))
    timing_uncertainty=max(0. if k==rig.pose_camera else
                           rig.time_uncertainty_s[k]+rig.time_uncertainty_s[rig.pose_camera] for k in used_cameras)
    try:
        fitted={key:_fit_endpoint(views) for key,views in grouped.items()}
    except (ValueError,np.linalg.LinAlgError) as exc:
        result['reasons']=['invalid_endpoint_geometry'];result['diagnostics']['detail']=str(exc)
        return result
    front,rear=fitted['front_extreme'][0],fitted['rear_extreme'][0]
    length=float(front[0]-rear[0])
    reasons=[]
    angular_speed=float(np.max(abs(np.diff(headings))/np.diff(times)))
    radius=max(np.linalg.norm(front[:2]),np.linalg.norm(rear[:2]))
    timing_displacement=float((speed+angular_speed*radius)*timing_uncertainty)
    if timing_displacement > tolerance_m/4: reasons.append('synchronization_error_budget_exceeded')
    if length <= 0 or min(front[2],rear[2]) < -.1: reasons.append('nonphysical_endpoints_or_wrong_heading')
    for point,diagnostic in fitted.values():
        if not diagnostic['converged']: reasons.append('optimizer_failed')
        if diagnostic['condition_number'] is None or diagnostic['condition_number'] > 1e5 or diagnostic['ray_angle_deg'] < 2:
            reasons.append('weak_multiview_geometry')
        if diagnostic['reprojection_rmse_px'] > 3: reasons.append('high_reprojection_error')
    result.update(status='rejected' if reasons else 'accepted_conditional',reasons=sorted(set(reasons)),
                  length_m=None if reasons else length,candidate_length_m=length)
    result['diagnostics']=dict(endpoints={key:dict(body_xyz_m=p.tolist(),**d) for key,(p,d) in fitted.items()},
        used_cameras=sorted(used_cameras),observation_count=sum(map(len,grouped.values())),
        skipped_occluded_observations=skipped,wheelbase_range_m=wheelbase_range,
        timing_displacement_m=timing_displacement,calibration_check_errors_m=rig.calibration_errors_m,
        reliability='conditional_geometry_only_not_validated_metrological_confidence',
        unbounded_errors=['contact_localization','correlated_landmark_bias','road_nonplanarity',
                          'endpoint_identity_and_visibility','camera_drift','rolling_shutter','pose_interpolation'])
    return result


def measure_rig_passage(rig, components, *, composition_verified=False, tolerance_m=.1,
                        max_pose_gap_s=.25):
    """Occupied longitudinal span of explicitly associated rigid components.

    Evaluate all components at common times when headings differ by at most one
    degree. Use the union of endpoint extents, so hitch gaps are included and
    overlapping tractor/trailer bodies are not double-counted. Association and
    completeness are supplied annotations, not claims made by this solver.
    """
    result=dict(strategy='surveyed_multicamera_passage',rig_id=rig.rig_id,status='rejected',
                length_m=None,reasons=[],accuracy_validated=False,complete_vehicle=False,
                components=[],diagnostics={})
    if not components or composition_verified is not True:
        result['reasons']=['unverified_vehicle_composition']
        return result
    ids=[part.get('component_id') for part in components]
    if any(not isinstance(i,str) or not i.strip() for i in ids) or len(set(ids)) != len(ids):
        raise ValueError('Components need distinct nonempty component_id values')
    poses=[]
    for part in components:
        measured=measure_rig_component(rig,part['pose_observations'],part['endpoints'],
                                      tolerance_m=tolerance_m,max_pose_gap_s=max_pose_gap_s)
        result['components'].append(dict(component_id=part['component_id'],**measured))
        if measured['length_m'] is None:
            continue
        rows=sorted(part['pose_observations'],key=_timestamp)
        T,R,_,_=_poses(rig.cameras[rig.pose_camera],rows)
        times=np.asarray([_timestamp(row)+rig.offsets_s[rig.pose_camera] for row in rows])
        headings=np.unwrap(np.arctan2(R[:,1,0],R[:,0,0]))
        ends=np.asarray([measured['diagnostics']['endpoints'][k]['body_xyz_m']
                         for k in ('rear_extreme','front_extreme')])
        poses.append((times,T,headings,ends))
    if len(poses) != len(components):
        result['reasons']=['incomplete_component_measurements']
        return result
    start=max(p[0][0] for p in poses);stop=min(p[0][-1] for p in poses)
    candidates=sorted({float(t) for times,*_ in poses for t in times if start <= t <= stop})
    spans=[]
    for time in candidates:
        try:
            states=[_pose_at(time,times,T,headings,max_pose_gap_s) for times,T,headings,ends in poses]
        except ValueError:
            continue
        axis=states[0][1][:,0]
        axes=np.asarray([R[:,0] for T,R in states])
        spread=float(np.degrees(np.arccos(np.clip(np.min(axes@axes.T),-1,1))))
        if spread > 1.: continue
        extremes=np.concatenate([(ends@R.T+T)@axis for (times,Ts,headings,ends),(T,R) in zip(poses,states)])
        spans.append(dict(timestamp_s=time,length_m=float(np.ptp(extremes)),heading_spread_deg=spread))
    if len(spans) < 2:
        result['reasons']=['no_shared_straight_interval']
        return result
    lengths=np.asarray([row['length_m'] for row in spans])
    result['diagnostics']=dict(straight_samples=spans,length_range_m=float(np.ptp(lengths)),
        max_heading_spread_deg=1.,note='Conditional reconstruction with explicitly annotated component membership and extreme points.')
    if np.ptp(lengths) > tolerance_m/2:
        result['reasons']=['inconsistent_occupied_span']
        return result
    result.update(status='accepted_conditional',length_m=float(np.median(lengths)),complete_vehicle=True)
    return result
