"""Empirical bbox length approximation in corrected image coordinates."""
import cv2
import numpy as np


def road_depth(polygon, x, y):
    """0 = local far boundary; 1 = local near boundary at bottom-center x.

    A concave polygon may have multiple intervals; use the containing interval.
    """
    p = np.asarray(polygon, dtype=np.float32)
    if len(p) < 4 or cv2.pointPolygonTest(p, (float(x), float(y)), False) < 0:
        return None
    hits = []
    for a, b in zip(p, np.roll(p, -1, axis=0)):
        if a[0] != b[0] and min(a[0], b[0]) <= x < max(a[0], b[0]):
            hits.append(float(a[1] + (x-a[0]) * (b[1]-a[1])/(b[0]-a[0])))
    hits.sort()
    for lo, hi in zip(hits[::2], hits[1::2]):
        if lo <= y <= hi and hi-lo > 1:
            return float((y-lo)/(hi-lo))
    return None


def fit_scale(polygon, references):
    samples = []
    for r in references:
        x, y, w, h = r['bbox']
        t = road_depth(polygon, x+w/2, y+h)
        if t is None:
            raise ValueError('Every reference bbox bottom center must be inside the road.')
        samples.append((t, w/r['length_m']))
    if not samples:
        return None
    t, ppm = np.asarray(samples).T
    if len(samples) == 1:
        return dict(intercept=float(ppm[0]), slope=0., status='single_reference', depths=t.tolist())
    if np.ptp(t) < .08:
        raise ValueError('Reference cars are too close in road depth. Add one nearer or farther away.')
    slope, intercept = np.polyfit(t, ppm, 1)
    if slope <= 0 or intercept <= 0:
        raise ValueError('References imply an invalid perspective curve. Check lengths, boxes and road edges.')
    return dict(intercept=float(intercept), slope=float(slope), status='depth_calibrated', depths=t.tolist())


def measure_box(bbox, polygon, scale, image_size, line_x=None, line_tolerance_px=10):
    x, y, w, h = bbox
    t = road_depth(polygon, x+w/2, y+h)
    result = dict(bbox=bbox, bottom=[x+w/2, y+h], depth=t, length_m=None,
                  center=[x+w/2, y+h/2],
                  line_offset_px=None if line_x is None else x+w/2-line_x,
                  at_measurement_line=line_x is not None and abs(x+w/2-line_x) <= line_tolerance_px,
                  cm_per_px=None, coefficient=None, status='outside_road')
    if t is None:
        return result
    if min(x, y) <= 1 or x+w >= image_size[0]-1 or y+h >= image_size[1]-1:
        result['status'] = 'clipped'
        return result
    if line_x is not None and not result['at_measurement_line']:
        result['status'] = 'waiting_for_line'
        return result
    if scale is None:
        result['status'] = 'needs_reference'
        return result
    ppm = scale['intercept'] + scale['slope']*t
    result.update(length_m=float(w/ppm), cm_per_px=float(100/ppm),
                  coefficient=float((scale['intercept']+scale['slope'])/ppm),
                  status=scale['status'])
    if scale['status'] == 'depth_calibrated' and not min(scale['depths']) <= t <= max(scale['depths']):
        result['status'] = 'extrapolated'
    return result
