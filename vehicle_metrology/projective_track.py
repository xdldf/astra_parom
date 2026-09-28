"""Rigid landmark reconstruction from two parallel, metrically aligned planes.

X/Y are metres, Z is in barrier-height units. Unknown absolute barrier height
does not prevent longitudinal reconstruction when the two plane mappings are
known. It DOES prevent reporting Euclidean camera range or height in metres.
All pixels must use the same lens correction as the supplied projections.
"""
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import least_squares


KEYS = ('rear_contact_uv', 'front_contact_uv', 'rear_uv', 'front_uv')


@dataclass
class RoadProjection:
    P: np.ndarray
    support: np.ndarray
    image_size: tuple
    calibration_error_m: float

    def __post_init__(self):
        self.P = np.asarray(self.P, float)
        self.support = np.asarray(self.support, np.float32)
        if self.P.shape != (3, 4) or not np.isfinite(self.P).all():
            raise ValueError('Expected a finite 3x4 projection')
        if np.linalg.matrix_rank(self.P) != 3 or abs(np.linalg.det(self.H)) < 1e-10:
            raise ValueError('Degenerate projection')
        if self.support.ndim != 2 or self.support.shape[1] != 2 or len(self.support) < 3:
            raise ValueError('Expected a metric road support polygon')
        if not np.isfinite(self.support).all() or abs(cv2.contourArea(self.support)) < 1e-8:
            raise ValueError('Invalid support polygon')
        if len(self.image_size) != 2 or any(type(v) is not int or v <= 0 for v in self.image_size):
            raise ValueError('Invalid image size')
        if not np.isfinite(self.calibration_error_m) or self.calibration_error_m < 0:
            raise ValueError('A finite nonnegative calibration check error is required')
        if self.center[2] <= 0:
            raise ValueError('Camera must be above the road')

    @property
    def H(self):
        return self.P[:, [0, 1, 3]]

    @property
    def center(self):
        _, _, V = np.linalg.svd(self.P)
        if abs(V[-1, 3]) < 1e-10:
            raise ValueError('Camera centre is at infinity')
        return V[-1, :3] / V[-1, 3]

    @classmethod
    def from_parallel_planes(cls, ground, raised, support, image_size, calibration_error_m):
        # Both homographies MUST have the same scale/basis, not independent H[2,2]=1.
        ground, raised = np.asarray(ground, float), np.asarray(raised, float)
        if ground.shape != (3, 3) or raised.shape != (3, 3):
            raise ValueError('Expected two 3x3 homographies')
        if not np.allclose(ground[:, :2], raised[:, :2], rtol=1e-8, atol=1e-10):
            raise ValueError('Plane homographies must share scale and X/Y columns')
        P = np.column_stack((ground[:, :2], raised[:, 2]-ground[:, 2], ground[:, 2]))
        return cls(P, support, tuple(image_size), calibration_error_m)

    def ground(self, pixels):
        pixels = np.asarray(pixels, float).reshape(-1, 2)
        q = np.c_[pixels, np.ones(len(pixels))] @ np.linalg.inv(self.H).T
        if not np.isfinite(q).all() or np.any(abs(q[:, 2]) < 1e-9):
            raise ValueError('Ground ray at horizon')
        points = q[:, :2]/q[:, 2, None]
        if np.any((np.c_[points, np.ones(len(points))] @ self.H.T)[:, 2] <= 0):
            raise ValueError('Ground point behind camera')
        return points

    def project(self, points):
        points = np.asarray(points, float).reshape(-1, 3)
        q = np.c_[points, np.ones(len(points))] @ self.P.T
        if not np.isfinite(q).all() or np.any(q[:, 2] <= 1e-9):
            raise ValueError('Point behind camera')
        return q[:, :2]/q[:, 2, None]

    def contains(self, points):
        return all(cv2.pointPolygonTest(self.support, tuple(map(float, p[:2])), False) >= 0
                   for p in points)


def _fit(camera, pixels, sigmas):
    contacts = camera.ground(pixels[:, :2].reshape(-1, 2)).reshape(-1, 2, 2)
    delta = contacts[:, 1]-contacts[:, 0]
    wheelbases = np.linalg.norm(delta, axis=1)
    if np.any(wheelbases < .2):
        raise ValueError('Ground contacts too close')
    angles = np.arctan2(delta[:, 1], delta[:, 0])
    matrices = []
    for contact, angle in zip(contacts[:, 0], angles):
        c, s = np.cos(angle), np.sin(angle)
        body = np.eye(4)
        body[:2, :2] = [[c, -s], [s, c]]
        body[:2, 3] = contact
        matrices.append(camera.P @ body)
    matrices = np.asarray(matrices)
    endpoints, errors, conditions = [], [], []
    for k in (2, 3):
        uv = pixels[:, k]
        rows = (matrices[:, :2]-uv[:, :, None]*matrices[:, 2, None]).reshape(-1, 4)
        A, b = rows[:, :3], -rows[:, 3]
        norm = np.linalg.norm(A, axis=0)
        if np.any(norm < 1e-12):
            raise ValueError('Unobservable endpoint')
        condition = np.linalg.cond(A/norm)
        if not np.isfinite(condition) or condition > 1e5:
            raise ValueError('Unobservable endpoint')
        initial = np.linalg.lstsq(A, b, rcond=None)[0]

        def residual(point):
            q = np.einsum('nij,j->ni', matrices, np.r_[point, 1.])
            if np.any(abs(q[:, 2]) < 1e-9):
                return np.full(2*len(pixels), 1e9)
            return ((q[:, :2]/q[:, 2, None]-uv)/sigmas[:, None]).ravel()

        fit = least_squares(residual, initial, loss='soft_l1', max_nfev=200)
        if not fit.success:
            raise ValueError('Endpoint optimizer failed')
        q = np.einsum('nij,j->ni', matrices, np.r_[fit.x, 1.])
        if np.any(q[:, 2] <= 1e-9):
            raise ValueError('Endpoint behind camera')
        endpoints.append(fit.x)
        errors.append(residual(fit.x).reshape(-1, 2)*sigmas[:, None])
        conditions.append(condition)
    return dict(endpoints=np.asarray(endpoints), errors=np.asarray(errors), contacts=contacts,
                wheelbases=wheelbases, angles=angles, conditions=conditions)


def measure_projective_track(camera, observations, *, tolerance_m=.1, mc_samples=0, seed=0):
    """Diagnostic alternative to boxes. Never asserts validated vehicle accuracy.

    Tracks must identify the SAME visible physical bumper points in all frames;
    silhouette extrema, rotating wheel texture and inferred hidden contacts are
    not interchangeable with these landmarks. Pixel perturbations are conditional
    on the fixed camera model; calibration check failures always withhold length.
    """
    if not np.isfinite(tolerance_m) or tolerance_m <= 0 or type(mc_samples) is not int or mc_samples < 0:
        raise ValueError('Invalid tolerance or Monte Carlo sample count')
    result = dict(status='rejected', length_m=None, candidate_length_m=None,
                  accuracy_validated=False, reasons=[], diagnostics={}, frames=[])
    frames = [o['frame'] for o in observations]
    if any(type(f) is not int or f < 0 for f in frames) or len(set(frames)) != len(frames):
        raise ValueError('Frames must be distinct nonnegative integers')
    if len(observations) < 6:
        result['reasons'] = ['insufficient_frames']
        return result
    observations = sorted(observations, key=lambda o: o['frame'])
    pixels = np.asarray([[o[k] for k in KEYS] for o in observations], float)
    if pixels.shape != (len(observations), 4, 2) or not np.isfinite(pixels).all():
        raise ValueError('Four finite 2D landmarks required per observation')
    if np.any(pixels < 0) or np.any(pixels >= camera.image_size):
        raise ValueError('Landmark outside image')
    sigmas = np.asarray([o.get('sigma_px', 3.) for o in observations], float)
    if not np.isfinite(sigmas).all() or np.any(sigmas <= 0):
        raise ValueError('Positive landmark uncertainty required')
    if any(o.get('occluded', False) for o in observations):
        result['reasons'] = ['occluded_landmark']
        return result
    try:
        fitted = _fit(camera, pixels, sigmas)
    except (ValueError, np.linalg.LinAlgError) as exc:
        result['reasons'] = ['invalid_or_weak_geometry']
        result['diagnostics']['detail'] = str(exc)
        return result
    endpoints = fitted['endpoints']
    length = float(endpoints[1, 0]-endpoints[0, 0])
    rmse = float(np.sqrt(np.mean(fitted['errors']**2)))
    wheelbases = fitted['wheelbases']
    contacts = fitted['contacts']
    reasons = result['reasons']
    if not camera.contains(contacts.reshape(-1, 2)):
        reasons.append('outside_calibrated_road')
    footprints = []
    for contact,angle in zip(contacts[:,0],fitted['angles']):
        c,s=np.cos(angle),np.sin(angle)
        footprints.extend(endpoints[:,:2]@np.array([[c,s],[-s,c]])+contact)
    if not camera.contains(footprints) and 'outside_calibrated_road' not in reasons:
        reasons.append('outside_calibrated_road')
    if length <= 0 or np.any(endpoints[:, 2] < -.1) or np.any(endpoints[:, 2] >= camera.center[2]):
        reasons.append('nonphysical_endpoints')
    if rmse > 3.:
        reasons.append('high_reprojection_error')
    wheelbase_cv = float(np.std(wheelbases)/np.mean(wheelbases))
    if wheelbase_cv > .05:
        reasons.append('inconsistent_ground_contacts')
    baseline = float(np.max(np.linalg.norm(contacts[:, None, 0]-contacts[None, :, 0], axis=2)))
    if baseline < 1.:
        reasons.append('insufficient_motion_baseline')
    split_lengths = []
    for indices in (np.arange(0, len(pixels), 2), np.arange(1, len(pixels), 2)):
        try:
            part = _fit(camera, pixels[indices], sigmas[indices])['endpoints']
            split_lengths.append(float(part[1, 0]-part[0, 0]))
        except (ValueError, np.linalg.LinAlgError):
            pass
    split_range = float(np.ptp(split_lengths)) if len(split_lengths) == 2 else None
    if split_range is None or split_range > tolerance_m:
        reasons.append('inconsistent_frame_subsets')
    if camera.calibration_error_m > tolerance_m:
        reasons.append('calibration_check_exceeds_tolerance')
    result['candidate_length_m'] = length
    result['diagnostics'] = dict(reprojection_rmse_px=rmse, wheelbase_mean_m=float(np.mean(wheelbases)),
        wheelbase_cv=wheelbase_cv, baseline_m=baseline, endpoint_condition=max(fitted['conditions']),
        endpoint_body_xy_m_and_z_barrier_units=endpoints.tolist(),
        split_lengths_m=split_lengths, split_difference_m=split_range,
        camera_ground_xy_m=camera.center[:2].tolist(), calibration_check_error_m=camera.calibration_error_m,
        target_tolerance_m=tolerance_m, height_unit='unknown_barrier_height',
        excludes=['calibration_bias', 'wrong_landmark_identity', 'road_nonplanarity', 'bumper_shape'],
        note='Reconstruction consistency is not independent physical length validation.')
    for observation, contact, angle, wheelbase in zip(observations, contacts, fitted['angles'], wheelbases):
        result['frames'].append(dict(frame=observation['frame'], wheel_contacts_ground_xy_m=contact.tolist(),
            heading_deg=float(np.degrees(angle)), wheelbase_m=float(wheelbase),
            horizontal_camera_distance_m=float(np.linalg.norm(contact.mean(axis=0)-camera.center[:2]))))
    if mc_samples:
        rng = np.random.default_rng(seed)
        values = []
        for _ in range(mc_samples):
            noise = pixels+rng.normal(size=pixels.shape)*sigmas[:, None, None]
            try:
                fit = _fit(camera, noise, sigmas)
                e = fit['endpoints']; value = float(e[1, 0]-e[0, 0])
                if value > 0 and np.all(e[:, 2] >= -.1) and np.all(e[:, 2] < camera.center[2]):
                    values.append(value)
            except (ValueError, np.linalg.LinAlgError):
                pass
        interval = np.quantile(values, [.025, .975]).tolist() if len(values) >= max(10, .8*mc_samples) else None
        radius = max(abs(v-length) for v in interval) if interval else None
        result['diagnostics']['pixel_sensitivity'] = dict(samples_requested=mc_samples,
            samples_successful=len(values), conditional_p95_m=interval, radius_m=radius,
            note='Fixed calibration and independent Gaussian landmark noise; not a validated confidence interval.')
        if radius is None or radius > tolerance_m:
            reasons.append('landmark_sensitivity_exceeds_tolerance')
    if not reasons:
        result.update(status='accepted_conditional', length_m=length)
    return result
