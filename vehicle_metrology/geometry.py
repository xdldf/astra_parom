"""Metric central-camera geometry on raw distorted pixels."""
from dataclasses import dataclass
import cv2
import numpy as np
from scipy.optimize import least_squares


@dataclass
class Camera:
    image_size: tuple
    model: str
    K: np.ndarray
    D: np.ndarray
    Rcw: np.ndarray
    tcw: np.ndarray
    road_polygon: np.ndarray
    calibration_id: str

    @classmethod
    def from_dict(cls, data):
        if data.get('schema_version') != 1:
            raise ValueError('Unsupported calibration schema_version')
        cam = cls(tuple(data['image_size']), data['model'], np.asarray(data['K'], float),
                  np.asarray(data['D'], float).reshape(-1), np.asarray(data['Rcw'], float),
                  np.asarray(data['tcw'], float).reshape(3),
                  np.asarray(data['road_polygon'], float), str(data['calibration_id']))
        if cam.model not in ("brown", "fisheye", "fisheye-corrected", "fisheye-corrected-rational"):
            raise ValueError("Model must be brown or fisheye; fisheye-corrected supports full image correction")
        if cam.K.shape != (3,3) or cam.Rcw.shape != (3,3):
            raise ValueError('K and Rcw must be 3x3')
        if len(cam.image_size) != 2 or any(int(v) != v or v <= 0 for v in cam.image_size):
            raise ValueError('Invalid image_size')
        if cam.road_polygon.ndim != 2 or cam.road_polygon.shape[1] != 2 or len(cam.road_polygon) < 3:
            raise ValueError('road_polygon must be Nx2, N>=3')
        if not all(np.all(np.isfinite(v)) for v in (cam.K,cam.D,cam.Rcw,cam.tcw,cam.road_polygon)):
            raise ValueError('Nonfinite calibration')
        if cam.D.size not in ((4,) if cam.model in ('fisheye', 'fisheye-corrected', 'fisheye-corrected-rational') else (4,5,8)):
            raise ValueError('Incorrect distortion coefficient count')
        if not np.allclose(cam.Rcw.T@cam.Rcw,np.eye(3),atol=1e-6) or not np.isclose(np.linalg.det(cam.Rcw),1):
            raise ValueError('Rcw must be a proper rotation')
        if cam.K[0,0] <= 0 or cam.K[1,1] <= 0 or not np.allclose(cam.K[2], [0,0,1]) or cam.K[0,1] != 0 or cam.K[1,0] != 0:
            raise ValueError('Expected positive focal lengths, zero skew standard K')
        if cam.center[2] <= 0:
            raise ValueError('Camera must be above the Z=0 road')
        return cam

    @property
    def center(self):
        return -self.Rcw.T @ self.tcw

    def to_dict(self):
        return dict(schema_version=1,image_size=list(self.image_size),model=self.model,
                    K=self.K.tolist(),D=self.D.tolist(),Rcw=self.Rcw.tolist(),tcw=self.tcw.tolist(),
                    road_polygon=self.road_polygon.tolist(),calibration_id=self.calibration_id)

    def project(self, points):
        points = np.ascontiguousarray(points,dtype=float).reshape(-1,3)
        if not np.isfinite(points).all() or np.any((points@self.Rcw.T+self.tcw)[:,2] <= 0):
            raise ValueError('Nonfinite points or points behind camera')
        rvec = cv2.Rodrigues(self.Rcw)[0]
        fn = (cv2.fisheye.projectPoints if self.model in ('fisheye', 'fisheye-corrected', 'fisheye-corrected-rational') else cv2.projectPoints)
        return fn(points.reshape(-1,1,3),rvec,self.tcw,self.K,self.D)[0].reshape(-1,2)

    def rays(self, pixels):
        pixels = np.ascontiguousarray(pixels,dtype=float).reshape(-1,1,2)
        if not np.isfinite(pixels).all():
            raise ValueError('Nonfinite pixels')
        if self.model in ('fisheye', 'fisheye-corrected', 'fisheye-corrected-rational'):
            uv = cv2.fisheye.undistortPoints(pixels,self.K,self.D).reshape(-1,2)
        else:
            uv = cv2.undistortPointsIter(pixels,self.K,self.D,None,None,
                (cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,100,1e-12)).reshape(-1,2)
        rays = np.column_stack((uv,np.ones(len(uv)))) @ self.Rcw
        rays /= np.linalg.norm(rays,axis=1)[:,None]
        check = self.project(self.center + rays)
        if not np.isfinite(rays).all() or np.max(np.abs(check-pixels.reshape(-1,2)),initial=0) > 0.01:
            raise ValueError('Distortion inverse failed roundtrip')
        return rays

    def ground(self, pixels):
        rays = self.rays(pixels)
        if np.any(rays[:,2] >= -1e-6):
            raise ValueError('Ray does not intersect usable road in front of camera')
        return self.center + (-self.center[2]/rays[:,2,None])*rays

    def in_road(self, points):
        polygon = self.road_polygon.astype(np.float32)
        return all(cv2.pointPolygonTest(polygon,tuple(map(float,p[:2])),False) >= 0 for p in points)


def _poses(camera, observations):
    contacts = camera.ground([o[k] for o in observations for k in ('rear_contact_uv','front_contact_uv')]).reshape(-1,2,3)
    delta = contacts[:,1]-contacts[:,0]
    wheelbases = np.linalg.norm(delta[:,:2],axis=1)
    if np.any(wheelbases < 0.2):
        raise ValueError('Ground anchors too close')
    angles = np.arctan2(delta[:,1],delta[:,0])
    c,s = np.cos(angles),np.sin(angles)
    rotations = np.zeros((len(angles),3,3))
    rotations[:,0,0] = rotations[:,1,1] = c
    rotations[:,1,0] = s
    rotations[:,0,1] = -s
    rotations[:,2,2] = 1
    return contacts[:,0],rotations,wheelbases,contacts


def _fit_endpoint(camera, pixels, translations, rotations, sigma):
    rays = camera.rays(pixels)
    directions = np.einsum('nji,nj->ni',rotations,rays)
    origins = np.einsum('nji,nj->ni',rotations,camera.center-translations)
    matrices = np.eye(3)[None]-directions[:,:,None]*directions[:,None,:]
    A = np.sum(matrices,axis=0)
    b = np.einsum('nij,nj->i',matrices,origins)
    initial = np.linalg.lstsq(A,b,rcond=None)[0]
    condition = float(np.linalg.cond(A))
    min_dot = np.clip(np.min(directions@directions.T),-1,1)
    angle = float(np.degrees(np.arccos(min_dot)))

    def residual(q):
        world = np.einsum('nij,j->ni',rotations,q)+translations
        return ((camera.project(world)-pixels)/sigma[:,None]).ravel()

    result = least_squares(residual,initial,loss='soft_l1',f_scale=1.,max_nfev=150)
    world = np.einsum('nij,j->ni',rotations,result.x)+translations
    errors = camera.project(world)-pixels
    return result.x,errors,condition,angle,bool(result.success)


def measure_track(camera, observations, min_frames=4, pixel_sigma=1., mc_samples=0, seed=0, *, _details=True):
    """Conditional rigid-body reconstruction; never a bounding-box conversion."""
    if min_frames < 4 or pixel_sigma <= 0 or not np.isfinite(pixel_sigma) or mc_samples < 0:
        raise ValueError('min_frames >= 4, positive pixel_sigma and nonnegative mc_samples required')
    observations = sorted(observations,key=lambda o:o['frame'])
    frames = [o['frame'] for o in observations]
    if any(not isinstance(f,int) or f < 0 for f in frames) or len(set(frames)) != len(frames):
        raise ValueError('Frame indices must be distinct nonnegative integers')
    result = dict(status='rejected',length_m=None,reasons=[],diagnostics={},frames=[])
    if len(observations) < min_frames:
        result['reasons'] = ['insufficient_frames']
        return result
    keys = ('rear_contact_uv','front_contact_uv','rear_uv','front_uv')
    pixels = np.asarray([[o[k] for k in keys] for o in observations],float)
    if pixels.shape != (len(observations),4,2) or not np.isfinite(pixels).all():
        raise ValueError('Each observation needs four finite 2D points')
    if np.any(pixels < 0) or np.any(pixels[:,:,0] >= camera.image_size[0]) or np.any(pixels[:,:,1] >= camera.image_size[1]):
        result['reasons'] = ['landmark_outside_image']
        return result
    sigma = np.asarray([o.get('sigma_px',pixel_sigma) for o in observations],float)
    if not np.isfinite(sigma).all() or np.any(sigma <= 0):
        raise ValueError('sigma_px must be positive and finite')
    try:
        translations,rotations,wheelbases,contacts = _poses(camera,observations)
    except ValueError as exc:
        result['reasons'] = ['invalid_ground_geometry']
        result['diagnostics']['detail'] = str(exc)
        return result
    if not camera.in_road(contacts.reshape(-1,3)):
        result['reasons'] = ['outside_calibrated_road']
        return result
    try:
        fitted = [_fit_endpoint(camera,pixels[:,k],translations,rotations,sigma) for k in (2,3)]
    except (ValueError,np.linalg.LinAlgError) as exc:
        result['reasons'] = ['invalid_endpoint_geometry']
        result['diagnostics']['detail'] = str(exc)
        return result
    rear,front = fitted[0][0],fitted[1][0]
    length = float(front[0]-rear[0])
    errors = np.stack([f[1] for f in fitted],axis=1)
    rmse = float(np.sqrt(np.mean(errors**2)))
    condition = max(f[2] for f in fitted)
    angle = min(f[3] for f in fitted)
    wheelbase_cv = float(np.std(wheelbases)/np.mean(wheelbases))
    reasons = []
    if not np.isfinite(condition) or condition > 1e5 or angle < 2.:
        reasons.append('weak_multiview_geometry')
    if not all(f[4] for f in fitted):
        reasons.append('optimizer_failed')
    if rmse > 3.:
        reasons.append('high_reprojection_error')
    if length <= 0 or np.any(np.array([rear[2],front[2]]) < -.1) or np.any(np.array([rear[2],front[2]]) >= camera.center[2]):
        reasons.append('nonphysical_endpoints_or_wrong_heading')
    if wheelbase_cv > .05:
        reasons.append('inconsistent_ground_anchors')
    result['diagnostics'] = dict(reprojection_rmse_px=rmse,ray_angle_deg=angle,
        condition_number=condition if np.isfinite(condition) else None,
        endpoint_world_body={'rear':rear.tolist(),'front':front.tolist()},
        wheelbase_mean_m=float(np.mean(wheelbases)),wheelbase_cv=wheelbase_cv,
        observation_count=len(observations),
        reliability='conditional_geometry_only_not_validated_metrological_confidence',
        thresholds={'max_rmse_px':3.,'min_ray_angle_deg':2.,'max_condition':1e5,'max_wheelbase_cv':.05})
    result['frames'] = [dict(frame=o['frame'],x_m=float(t[0]),y_m=float(t[1]),
        heading_deg=float(np.degrees(np.arctan2(r[1,0],r[0,0]))),
        wheelbase_m=float(w),reprojection_error_px=float(np.sqrt(np.mean(e**2))))
        for o,t,r,w,e in zip(observations,translations,rotations,wheelbases,errors)]
    result.update(reasons=reasons,status='rejected' if reasons else 'accepted_conditional',
                  length_m=None if reasons else length)
    if _details:
        for i, frame in enumerate(result['frames']):
            frame['prefix_length_m'] = None
            frame['window_length_m'] = None
            if i+1 >= min_frames:
                prefix = measure_track(camera,observations[:i+1],min_frames,pixel_sigma,_details=False)
                window = measure_track(camera,observations[max(0,i-7):i+1],min_frames,pixel_sigma,_details=False)
                frame['prefix_length_m'] = prefix['length_m']
                frame['window_length_m'] = window['length_m']
        values = [f['window_length_m'] for f in result['frames'] if f['window_length_m'] is not None]
        result['diagnostics']['window_consistency'] = dict(
            count=len(values),range_m=float(np.ptp(values)) if values else None,
            mad_m=float(np.median(np.abs(values-np.median(values)))) if values else None,
            note='Overlapping windows are correlated; consistency does not establish accuracy.')
    if mc_samples and not reasons:
        rng = np.random.default_rng(seed)
        lengths = []
        for _ in range(mc_samples):
            noisy = pixels+rng.normal(size=pixels.shape)*sigma[:,None,None]
            sample = [dict(frame=o['frame'],sigma_px=float(s),**dict(zip(keys,p.tolist())))
                      for o,s,p in zip(observations,sigma,noisy)]
            try:
                estimate = measure_track(camera,sample,min_frames,pixel_sigma,_details=False)
            except ValueError:
                continue  # A perturbed ray can leave the invertible/calibrated domain.
            if estimate['length_m'] is not None:
                lengths.append(estimate['length_m'])
        interval = np.quantile(lengths,[.025,.975]).tolist() if len(lengths) >= max(10,.8*mc_samples) else None
        result['diagnostics']['uncertainty'] = dict(samples_requested=mc_samples,
            samples_successful=len(lengths),conditional_p95_m=interval,
            model='Independent Gaussian pixel perturbations of all contacts and endpoints, fixed camera',
            excludes=['calibration','survey','road_nonplanarity','correlated_annotation_bias','shape_and_visibility','rolling_shutter'],
            warning='Not a validated confidence interval; sample failures are reported, not hidden.')
    return result
