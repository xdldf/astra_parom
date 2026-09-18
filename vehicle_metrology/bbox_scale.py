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


def fit_scale(polygon, references, rulers=()):
    if rulers:
        return fit_rulers(polygon, rulers)
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
    width=max(r['bbox'][2] for r in references)
    if len(samples) == 1:
        return dict(intercept=float(ppm[0]), slope=0., status='single_reference', depths=t.tolist(), max_reference_width_px=width)
    if np.ptp(t) < .08:
        raise ValueError('Reference cars are too close in road depth. Add one nearer or farther away.')
    slope, intercept = np.polyfit(t, ppm, 1)
    if slope <= 0 or intercept <= 0:
        raise ValueError('References imply an invalid perspective curve. Check lengths, boxes and road edges.')
    return dict(intercept=float(intercept), slope=float(slope), status='depth_calibrated', depths=t.tolist(), max_reference_width_px=width)


def fit_rulers(polygon, rulers):
    """Piecewise metric coordinates along surveyed, equally spaced road marks.

    Horizontal intervals may have different pixel widths (residual lens distortion).
    Each ruler covers one narrow road-depth band; no extrapolation is allowed.
    """
    rows = []
    for ruler in rulers:
        points = np.asarray(ruler['points'], dtype=float)
        if points[0, 0] > points[-1, 0]:
            points = points[::-1]
        if np.any(np.diff(points[:, 0]) <= 2):
            raise ValueError('Ruler marks must run left to right (or right to left), at least 2 px apart.')
        depths = [road_depth(polygon, x, y) for x, y in points]
        if any(t is None for t in depths):
            raise ValueError('Every ruler mark must lie on the road.')
        if np.ptp(depths) > .10:
            raise ValueError('Draw each ruler along the vehicle travel direction at one road depth.')
        rows.append(dict(x=points[:, 0].tolist(), step_m=ruler['step_m'], depth=float(np.mean(depths))))
    rows.sort(key=lambda r: r['depth'])
    if any(b['depth']-a['depth'] < .08 for a,b in zip(rows, rows[1:])):
        raise ValueError('Use one continuous ruler per depth; separate near/far rulers by at least 8% road depth.')
    return dict(status='ruler_calibrated', rulers=rows, depths=[r['depth'] for r in rows])


def ruler_length(scale, left, right, depth):
    rows = scale['rulers']
    if len(rows) == 1:
        if abs(depth-rows[0]['depth']) > .05:
            return None
        nearby = rows
    else:
        exact = [r for r in rows if abs(depth-r['depth']) < 1e-7]
        if exact:
            nearby = exact
        elif depth < rows[0]['depth'] or depth > rows[-1]['depth']:
            return None
        else:
            index = min(len(rows)-2, max(0, int(np.searchsorted(scale['depths'], depth))-1))
            nearby = rows[index:index+2]
    ppm = []
    for row in nearby:
        xs = row['x']
        if left < xs[0] or right > xs[-1]:
            return None
        metres = np.arange(len(xs))*row['step_m']
        # Integrate across every interval, rather than sampling just the bbox center.
        length = np.interp(right, xs, metres)-np.interp(left, xs, metres)
        ppm.append((right-left)/length)
    local_ppm = np.interp(depth, [r['depth'] for r in nearby], ppm)
    return float((right-left)/local_ppm)


def measure_box(bbox, polygon, scale, image_size, line_x=None, line_tolerance_px=10):
    x, y, w, h = bbox
    t = road_depth(polygon, x+w/2, y+h)
    result = dict(bbox=bbox, bottom=[x+w/2, y+h], depth=t, length_m=None,
                  center=[x+w/2, y+h/2],
                  line_offset_px=None if line_x is None else x+w/2-line_x,
                  at_measurement_line=line_x is not None and abs(x+w/2-line_x) <= line_tolerance_px,
                  cm_per_px=None, coefficient=None, status='outside_road', warnings=[])
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
    if scale['status'] == 'ruler_calibrated':
        length = ruler_length(scale, x, x+w, t)
        result['status'] = 'outside_calibration' if length is None else 'ruler_calibrated'
        if length is not None:
            result.update(length_m=length, cm_per_px=100*length/w)
        return result
    ppm = scale['intercept'] + scale['slope']*t
    if w > 1.5*scale.get('max_reference_width_px',float('inf')):
        result['warnings'].append('Автомобиль значительно шире эталона в кадре. Масштаб по короткому автомобилю не проверяет искажение по всей длине состава. Проверьте длину по документам или мерным отметкам вдоль всей зоны измерения.')
    result.update(length_m=float(w/ppm), cm_per_px=float(100/ppm),
                  coefficient=float((scale['intercept']+scale['slope'])/ppm),
                  status=scale['status'])
    if scale['status'] == 'depth_calibrated' and not min(scale['depths']) <= t <= max(scale['depths']):
        result['status'] = 'extrapolated'
    return result
