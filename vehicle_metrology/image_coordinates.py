"""Map annotations on full corrected station images into internal camera pixels.

This reverses display rotation, zoom and Brown correction only. The display lens
is not a physical camera calibration. No images are read, saved or displayed.
"""
import copy

import cv2
import numpy as np


class CorrectedCoordinates:
    def __init__(self, profile, image_size):
        if list(profile.get('image_size',[])) != list(image_size):
            raise ValueError('Display profile image_size must match the calibrated full image; no resizing')
        self.size=np.asarray(image_size,float)
        if self.size.shape != (2,) or not np.isfinite(self.size).all() or np.any(self.size < 2):
            raise ValueError('Invalid full image_size')
        lens=profile.get('lens',{})
        keys=('k1','k2','focal','zoom','cx','cy','tilt_deg')
        if any(type(lens.get(k)) not in (int,float) or not np.isfinite(lens[k]) for k in keys):
            raise ValueError('Display profile must contain all seven finite lens settings used for the corrected image')
        self.k1,self.k2=lens['k1'],lens['k2']
        self.focal=max(self.size)*lens['focal']
        self.zoom=lens['zoom']
        if self.focal <= 0 or self.zoom <= 0:
            raise ValueError('Positive display focal and zoom required')
        self.center=self.size*np.asarray([lens['cx'],lens['cy']])
        rotation=cv2.getRotationMatrix2D(tuple((self.size-1)/2),-lens['tilt_deg'],1)
        self.inverse_rotation=cv2.invertAffineTransform(rotation)

    def _inside(self, points):
        # The final pixel centre, not the half-open pixel-cell boundary. Outside
        # this support, interpolation includes the black image border.
        return np.isfinite(points).all() and not (np.any(points < 0) or np.any(points > self.size-1))

    def raw(self, pixels):
        """Return raw Nx2 pixels and local maximum magnification of pixel error."""
        points=np.asarray(pixels,float)
        if points.ndim != 2 or points.shape[1] != 2 or not len(points) or not self._inside(points):
            raise ValueError('Expected finite full-resolution corrected pixels inside the image')
        unrotated=points@self.inverse_rotation[:,:2].T+self.inverse_rotation[:,2]
        if not self._inside(unrotated):
            raise ValueError('Corrected annotation lies in the rotation border')
        normalized=(unrotated-self.center)/(self.focal*self.zoom)
        r2=np.sum(normalized**2,axis=1)
        # A folding radial map can display duplicated content with ambiguous
        # coordinates. Check its derivative throughout the used radius.
        probes=[0.,float(r2.max())]
        if self.k2 > 0:
            probes.append(float(np.clip(-3*self.k1/(10*self.k2),0,r2.max())))
        if min(1+3*self.k1*r+5*self.k2*r*r for r in probes) <= 0:
            raise ValueError('Display distortion folds within the annotation radius')
        scale=1+self.k1*r2+self.k2*r2*r2
        raw=self.center+self.focal*normalized*scale[:,None]
        if not self._inside(raw):
            raise ValueError('Corrected annotation lies in the remap border')
        radial=(scale[:,None,None]*np.eye(2)+
                (2*self.k1+4*self.k2*r2)[:,None,None]*normalized[:,:,None]*normalized[:,None,:])
        jacobian=radial@self.inverse_rotation[:,:2]/self.zoom
        magnification=np.linalg.svd(jacobian,compute_uv=False)[:,0]
        return raw,magnification


def _convert_observation(row, keys, transform, *, sigma_required=False):
    if row.get('visible') is not True:
        return
    raw,scale=transform.raw([row[k] for k in keys])
    for key,value in zip(keys,raw):
        row[key]=value.tolist()
    if sigma_required or 'sigma_px' in row:
        sigma=row.get('sigma_px',1.)
        if type(sigma) not in (int,float) or not np.isfinite(sigma) or sigma <= 0:
            raise ValueError('Positive finite corrected sigma_px required')
        # A conservative isotropic bound after a non-isotropic coordinate map.
        # This is a residual weight, not a metrological confidence interval.
        row['sigma_px']=float(sigma*max(scale))


def corrected_survey_to_raw(survey, profile):
    if survey.get('coordinate_space') != 'corrected_full_resolution':
        raise ValueError('Display profile requires corrected_full_resolution survey coordinates')
    result=copy.deepcopy(survey)
    transform=CorrectedCoordinates(profile,result['image_size'])
    for key in ('control_points','check_points','raised_control_points','raised_check_points'):
        for point in result.get(key,[]):
            # Survey controls/checks must all be observed, never silently omitted.
            point['uv']=transform.raw([point['uv']])[0][0].tolist()
    result['coordinate_space']='raw_distorted_pixels'
    return result


def corrected_tracks_to_raw(data, profiles, rig):
    if data.get('coordinate_space') != 'corrected_full_resolution':
        raise ValueError('Display profiles require corrected_full_resolution track coordinates')
    if set(profiles) != set(rig.cameras):
        raise ValueError('Provide exactly one display profile for every camera in the rig')
    transforms={key:CorrectedCoordinates(profiles[key],camera.image_size) for key,camera in rig.cameras.items()}
    result=copy.deepcopy(data)
    for track in result['tracks']:
        for component in track.get('components',[track]):
            for pose in component['pose_observations']:
                _convert_observation(pose,('rear_contact_uv','front_contact_uv'),
                                     transforms[rig.pose_camera],sigma_required=True)
            for endpoint in component['endpoints']:
                key=endpoint['camera_id']
                if key not in transforms: raise ValueError('Unknown endpoint camera_id')
                if endpoint.get('visible') is True and 'sigma_px' not in endpoint:
                    raise ValueError('Endpoint sigma_px required in corrected pixels')
                _convert_observation(endpoint,('uv',),transforms[key],sigma_required=True)
    result['coordinate_space']='raw_distorted_pixels'
    return result
