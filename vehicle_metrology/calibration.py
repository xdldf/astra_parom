"""Offline central-camera calibration. All image coordinates are raw pixels."""
import cv2
import numpy as np
from scipy.optimize import least_squares

from .geometry import Camera


def _project(xyz, pose, K, D, model):
    fn = cv2.fisheye.projectPoints if model == 'fisheye' else cv2.projectPoints
    return fn(np.ascontiguousarray(xyz, dtype=float).reshape(-1, 1, 3), pose[:3], pose[3:], K, D)[0].reshape(-1, 2)


def _normalized(uv, K, D, model):
    uv = np.ascontiguousarray(uv, dtype=float).reshape(-1, 1, 2)
    if model == 'fisheye':
        result = cv2.fisheye.undistortPoints(uv, K, D)
    else:
        result = cv2.undistortPointsIter(uv, K, D, None, None,
                                      (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-12))
    return result.reshape(-1, 2)


def _pose(xyz, uv, K, D, model, above_road=False):
    """Initialize planar or spatial PnP, then refine original pixel residuals."""
    normalized = _normalized(uv, K, D, model)
    spatial = np.linalg.matrix_rank(xyz-xyz.mean(axis=0)) == 3
    solved = cv2.solvePnPGeneric(xyz, normalized, np.eye(3), None,
                                flags=cv2.SOLVEPNP_SQPNP if spatial else cv2.SOLVEPNP_IPPE)
    candidates = []
    for rvec, tvec in zip(solved[1], solved[2]):
        initial = np.r_[rvec.ravel(), tvec.ravel()]
        fit = least_squares(lambda p: (_project(xyz, p, K, D, model)-uv).ravel(), initial,
                            method='lm', max_nfev=300, ftol=1e-12, xtol=1e-12, gtol=1e-12)
        R = cv2.Rodrigues(fit.x[:3])[0]
        center = -R.T @ fit.x[3:]
        if fit.success and np.isfinite(fit.x).all() and np.all((xyz @ R.T + fit.x[3:])[:, 2] > 0):
            if not above_road or center[2] > 0:
                candidates.append((float(np.sum(fit.fun**2)), fit.x))
    if not candidates:
        raise ValueError('No positive-depth, physically valid pose; check survey axes and correspondences')
    return min(candidates, key=lambda item: item[0])[1]


def calibrate_intrinsic_views(training_views, heldout_views, *, image_size, model):
    """Fit K/D on training boards only; heldout board pose is a nuisance fit.

    Heldout RMS is conditional on each board's fitted six-DOF pose, not an
    independent metric accuracy certificate. Whole views, never corners, split.
    """
    size = _image_size(image_size)
    if model not in ('brown','fisheye'):
        raise ValueError('Explicit brown or fisheye model required')
    if len(training_views) < 10 or len(heldout_views) < 3:
        raise ValueError('Need at least 10 training and 3 heldout views')
    ids = [v['id'] for v in training_views+heldout_views]
    if len(set(ids)) != len(ids):
        raise ValueError('Training and heldout views must have independent unique ids')
    for view in training_views+heldout_views:
        xyz = _array(view['object_points'],'object_points')
        uv = _array(view['image_points'],'image_points')
        if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz)<6 or uv.shape != (len(xyz),2):
            raise ValueError('Invalid board point shape')
        if np.any(uv < 0) or np.any(uv >= size):
            raise ValueError('Board points outside image')
    objects = [np.asarray(v['object_points'], float) for v in training_views]
    images = [np.asarray(v['image_points'], float) for v in training_views]
    if model == 'fisheye':
        rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
            [p.reshape(-1,1,3) for p in objects], [p.reshape(-1,1,2) for p in images], size,
            np.eye(3), np.zeros((4,1)),
            flags=cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_CHECK_COND | cv2.fisheye.CALIB_FIX_SKEW,
            criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 200, 1e-10))
    else:
        rms, K, D, rvecs, tvecs = cv2.calibrateCamera(
            [p.astype(np.float32) for p in objects], [p.astype(np.float32) for p in images], size, None, None,
            criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 200, 1e-10))
    result = dict(schema_version=1, artifact_type='intrinsic', image_size=list(size), model=model,
                  K=K.tolist(), D=D.ravel().tolist())
    _intrinsic(result)
    per_view = []
    for view in heldout_views:
        xyz, uv = np.asarray(view['object_points'], float), np.asarray(view['image_points'], float)
        pose = _pose(xyz, uv, K, D, model)
        residual = _project(xyz, pose, K, D, model) - uv
        per_view.append(dict(id=view['id'], rmse_px=float(np.sqrt(np.mean(np.sum(residual**2, axis=1)))),
                             residuals_px=residual.tolist(), image_points=uv.tolist()))
    result['diagnostics'] = dict(training_view_count=len(training_views), heldout_view_count=len(heldout_views),
        training_rms_px=float(rms), heldout_rmse_px=float(np.sqrt(np.mean([p['rmse_px']**2 for p in per_view]))),
        heldout_views=per_view, training_view_ids=[v['id'] for v in training_views],
        model_selection='explicit_user_choice_not_training_rms',
        interpretation='Heldout whole-board views; board poses fitted with fixed K/D. Not a metric accuracy certificate.')
    return result


def _array(value, name, shape=None):
    try:
        result = np.asarray(value, float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} must be a numeric array') from exc
    if not np.isfinite(result).all():
        raise ValueError(f'{name} must be finite')
    if shape is not None and result.shape != shape:
        raise ValueError(f'{name} must have shape {shape}')
    return result


def _image_size(value):
    size = _array(value, 'image_size', (2,))
    if np.any(size <= 0) or np.any(size != np.floor(size)):
        raise ValueError('image_size must be positive integer [width,height]')
    return tuple(map(int, size))


def _intrinsic(data):
    if not isinstance(data, dict) or data.get('schema_version') != 1:
        raise ValueError('Intrinsic schema_version must be 1')
    size = _image_size(data.get('image_size'))
    K = _array(data.get('K'), 'K', (3, 3))
    D = _array(data.get('D'), 'distortion D')
    model = data.get('model')
    if model not in ('brown', 'fisheye'):
        raise ValueError('model must be explicitly brown or fisheye')
    if D.ndim != 1 or D.size not in ((4,) if model == 'fisheye' else (4, 5)):
        raise ValueError('Unsupported distortion: Brown needs 4/5 D coefficients; fisheye needs 4')
    if K[0,0] <= 0 or K[1,1] <= 0 or K[0,1] != 0 or K[1,0] != 0 or not np.array_equal(K[2], [0,0,1]):
        raise ValueError('K requires positive focal lengths, zero skew and last row [0,0,1]')
    return size, K, D, model


def _survey_points(value, name, minimum, size, *, raised=False):
    if not isinstance(value, list) or len(value) < minimum:
        raise ValueError(f'{name} requires at least {minimum} points with id, xyz_m and uv')
    if any(not isinstance(p, dict) or not isinstance(p.get('id'), str) or not p['id'] for p in value):
        raise ValueError(f'{name}: each point requires a nonempty string id')
    xyz = _array([p.get('xyz_m') for p in value], name + '.xyz_m', (len(value), 3))
    uv = _array([p.get('uv') for p in value], name + '.uv', (len(value), 2))
    if not raised and np.any(np.abs(xyz[:,2]) > 1e-9):
        raise ValueError(f'{name}: only surveyed Z=0 road points supported; do not flatten a nonplanar site')
    if raised and np.any(xyz[:,2] <= 0):
        raise ValueError(f'{name}: raised controls require measured positive heights')
    if np.linalg.matrix_rank(xyz[:,:2] - xyz[:,:2].mean(axis=0)) < 2:
        raise ValueError(f'{name} must be noncollinear and spatially spread')
    if np.any(uv < 0) or np.any(uv >= size):
        raise ValueError(f'{name}.uv outside image_size; use original raw pixels')
    return xyz, uv


def calibrate_survey(intrinsic, survey, *, calibration_id, max_check_error_m):
    """Fit controls only; v2 adds measured elevated targets and independent checks."""
    size, K, D, model = _intrinsic(intrinsic)
    if not isinstance(survey, dict) or survey.get('schema_version') not in (1,2):
        raise ValueError('Survey schema_version must be 1 or 2')
    if survey.get('coordinate_space','raw_distorted_pixels') != 'raw_distorted_pixels':
        raise ValueError('Survey coordinate_space must be raw_distorted_pixels; convert corrected annotations first')
    if _image_size(survey.get('image_size')) != size:
        raise ValueError('Survey image_size must match intrinsic image_size exactly; no resizing')
    if survey.get('units') != 'm':
        raise ValueError('Survey units must be m (meters)')
    if not isinstance(calibration_id, str) or not calibration_id.strip():
        raise ValueError('calibration_id must be a nonempty string')
    if not np.isfinite(max_check_error_m) or max_check_error_m <= 0:
        raise ValueError('max_check_error_m must be finite and positive, chosen from your error budget')
    xyz, uv = _survey_points(survey.get('control_points'), 'control_points', 6, size)
    check_xyz, check_uv = _survey_points(survey.get('check_points'), 'check_points', 3, size)
    spatial = survey['schema_version'] == 2
    controls = list(survey['control_points'])
    checks = list(survey['check_points'])
    ground_check_count = len(checks)
    if spatial:
        if survey.get('coordinate_space') != 'raw_distorted_pixels':
            raise ValueError('v2 requires explicit raw_distorted_pixels coordinate_space')
        if not isinstance(survey.get('world_frame_id'),str) or not survey['world_frame_id'].strip():
            raise ValueError('v2 requires world_frame_id shared by the camera pair')
        raised_xyz,raised_uv = _survey_points(survey.get('raised_control_points'),'raised_control_points',4,size,raised=True)
        raised_check_xyz,raised_check_uv = _survey_points(survey.get('raised_check_points'),'raised_check_points',3,size,raised=True)
        controls += survey['raised_control_points']
        checks += survey['raised_check_points']
        xyz,uv = np.vstack([xyz,raised_xyz]),np.vstack([uv,raised_uv])
        check_xyz,check_uv = np.vstack([check_xyz,raised_check_xyz]),np.vstack([check_uv,raised_check_uv])
        uncertainty = _array([p.get('uncertainty_m') for p in controls+checks],'survey uncertainty_m')
        if np.any(uncertainty <= 0) or np.any(uncertainty >= max_check_error_m):
            raise ValueError('Survey uncertainty_m must be positive and below the check error budget')
    points = controls + checks
    ids = [p['id'] for p in points]
    all_xyz = np.vstack([xyz, check_xyz])
    if len(set(ids)) != len(ids) or len(np.unique(all_xyz, axis=0)) != len(all_xyz):
        raise ValueError('Controls/checks must be independent: unique ids and surveyed coordinates')
    polygon = _array(survey.get('road_polygon'), 'road_polygon')
    if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3 or abs(cv2.contourArea(polygon.astype(np.float32))) < 1e-8:
        raise ValueError('road_polygon must be nondegenerate ordered Nx2 world meters, N>=3')
    pose = _pose(xyz, uv, K, D, model, above_road=True)
    result = dict(schema_version=1, image_size=intrinsic['image_size'], model=model,
                  K=K.tolist(), D=D.tolist(), Rcw=cv2.Rodrigues(pose[:3])[0].tolist(),
                  tcw=pose[3:].tolist(), road_polygon=survey['road_polygon'], calibration_id=calibration_id)
    camera = Camera.from_dict(result)
    control_residual = camera.project(xyz) - uv
    check_residual = camera.project(check_xyz) - check_uv
    check_world = np.vstack([camera.horizontal_plane([pixel],point[2])[0]
                             for pixel,point in zip(check_uv,check_xyz)])
    errors = np.linalg.norm(check_world - check_xyz, axis=1)
    uncertainty = np.asarray([p['uncertainty_m'] for p in checks]) if spatial else np.zeros(len(checks))
    guarded_errors = errors + uncertainty
    if guarded_errors.max() > max_check_error_m:
        raise ValueError(f'Independent check error including survey uncertainty {guarded_errors.max():.6g} m exceeds limit {max_check_error_m:.6g} m')
    if spatial:
        result['world_frame_id']=survey['world_frame_id']
    result['diagnostics'] = dict(control_count=len(xyz), check_count=len(check_xyz),
        control_reprojection_rmse_px=float(np.sqrt(np.mean(np.sum(control_residual**2, axis=1)))),
        check_reprojection_rmse_px=float(np.sqrt(np.mean(np.sum(check_residual**2, axis=1)))),
        check_rmse_m=float(np.sqrt(np.mean(errors**2))), check_max_error_m=float(errors.max()),
        check_errors_m=errors.tolist(), check_residuals_px=check_residual.tolist(),
        control_residuals_px=control_residual.tolist(), camera_center_m=camera.center.tolist(),
        max_check_error_m=float(max_check_error_m), check_points_used_for_fit=False,
        survey_schema_version=survey['schema_version'],ground_check_count=ground_check_count,
        raised_check_count=len(checks)-ground_check_count,check_point_ids=[p['id'] for p in checks],
        check_uncertainty_m=uncertainty.tolist(),guarded_check_max_error_m=float(guarded_errors.max()),
        vehicle_accuracy_validated=False)
    return result
