"""Empirical bbox length approximation in corrected image coordinates."""
import cv2
import numpy as np


def unique_references(references):
    """One vote per annotated observation, even in legacy exported profiles."""
    unique = {}
    for reference in references:
        key = (reference.get('frame', 0), tuple(reference['bbox']))
        previous = unique.get(key)
        if previous is not None and previous['length_m'] != reference['length_m']:
            raise ValueError('The same reference observation has conflicting lengths.')
        if previous is None or reference.get('vehicle_id'):
            unique[key] = reference
    return list(unique.values())


def reference_diagnostics(scale, samples, references, input_count, tolerance_m):
    """Training residuals expose inconsistency; they do not validate accuracy."""
    depths, ppm = np.asarray(samples).T
    lengths = np.asarray([r['length_m'] for r in references])
    predicted = ppm * lengths / (scale['intercept'] + scale['slope'] * depths)
    errors = predicted - lengths
    return dict(input_reference_count=input_count, unique_reference_count=len(references),
                duplicates_ignored=input_count-len(references), target_tolerance_m=tolerance_m,
                reference_mae_m=float(np.mean(np.abs(errors))),
                reference_max_abs_error_m=float(np.max(np.abs(errors))),
                references_within_target=int(np.count_nonzero(np.abs(errors) <= tolerance_m+1e-12)),
                residuals=[dict(frame=r.get('frame', 0), length_m=r['length_m'],
                                predicted_m=float(v), error_m=float(e))
                           for r, v, e in zip(references, predicted, errors)],
                accuracy_validated=False,
                note='In-sample reference consistency only. Independent measured vehicles are required to validate accuracy.')


def road_cross_section(polygon, x, y):
    """Local far/near image boundaries enclosing the chosen contact proxy.

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
            return dict(far=[float(x),lo],near=[float(x),hi],
                        depth=float((y-lo)/(hi-lo)))
    return None


def road_depth(polygon, x, y):
    """Relative image depth only; NOT metric camera-to-car distance."""
    section=road_cross_section(polygon,x,y)
    return None if section is None else section['depth']


def fit_scale(polygon, references, rulers=(), *, tolerance_m=.05):
    if not np.isfinite(tolerance_m) or tolerance_m <= 0:
        raise ValueError('Tolerance must be finite and positive')
    if rulers:
        return fit_rulers(polygon, rulers)
    input_count = len(references)
    references = unique_references(references)
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
        scale = dict(intercept=float(ppm[0]), slope=0., status='verified_local' if references[0].get('vehicle_id') else 'single_reference', depths=t.tolist(), max_reference_width_px=width)
    elif np.ptp(t) < .08:
        verified = [i for i,r in enumerate(references) if r.get('vehicle_id')]
        if verified:
            # Repeated verified passes in one lane do not establish a depth curve.
            # Use their median local scale only within that observed lane band.
            scale = dict(intercept=float(np.median(ppm[verified])), slope=0.,
                        status='verified_local', depths=t[verified].tolist(),
                        max_reference_width_px=max(references[i]['bbox'][2] for i in verified))
        else:
            raise ValueError('Reference cars are too close in road depth. Add one nearer or farther away.')
    else:
        slope, intercept = np.polyfit(t, ppm, 1)
        if slope <= 0 or intercept <= 0:
            raise ValueError('References imply an invalid perspective curve. Check lengths, boxes and road edges.')
        scale = dict(intercept=float(intercept), slope=float(slope), status='depth_calibrated', depths=t.tolist(), max_reference_width_px=width)
    scale['diagnostics'] = reference_diagnostics(scale, samples, references, input_count, tolerance_m)
    return scale


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
    section = road_cross_section(polygon, x+w/2, y+h)
    t = None if section is None else section['depth']
    result = dict(bbox=bbox, bottom=[x+w/2, y+h], depth=t, length_m=None,
                  road_cross_section=section,
                  position_model='bbox_bottom_center_relative_road_depth',
                  camera_distance_m=None,
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
    if scale['status'] in {'survey_calibrated','survey_review'}:
        from .road_survey import projective_span
        diagnostics=scale['diagnostics']
        result['calibration_diagnostics']=diagnostics
        try:
            length,ends=projective_span(bbox,scale['projection'])
        except (ValueError,np.linalg.LinAlgError):
            result['status']='outside_calibration'
            return result
        result.update(position_model='ground_homography_at_bbox_bottom_center',
                      ground_span_xy_m=ends.tolist(),
                      projected_ground_span_px=cv2.perspectiveTransform(
                          ends[None],np.asarray(scale['projection'],float))[0].tolist())
        support=np.asarray(scale['support_world'],np.float32)
        if not all(cv2.pointPolygonTest(support,tuple(map(float,p)),False)>=0 for p in ends):
            result['status']='outside_calibration'
            return result
        result['warnings'].append('Метровые отметки расположены на барьере. Плоскость дороги оценена по стойкам; абсолютная точность длины автомобиля не подтверждена.')
        if scale['status']=='survey_review':
            result.update(status='calibration_review',candidate_length_m=length)
            result['warnings'].append('Проверка геометрии не уложилась в заданный допуск. Числовая длина не назначена.')
        else:
            result.update(status='projective_estimate',length_m=length,cm_per_px=100*length/w)
        return result
    if scale['status'] == 'ruler_calibrated':
        length = ruler_length(scale, x, x+w, t)
        result['status'] = 'outside_calibration' if length is None else 'ruler_calibrated'
        if length is not None:
            result.update(length_m=length, cm_per_px=100*length/w)
        return result
    diagnostics = scale.get('diagnostics', {})
    result['calibration_diagnostics'] = diagnostics
    tolerance = diagnostics.get('target_tolerance_m', .05)
    if diagnostics.get('reference_max_abs_error_m', 0) > tolerance:
        result['warnings'].append(f'Калибровка расходится с эталонными длинами более чем на {100*tolerance:g} см. Точность ±{100*tolerance:g} см не подтверждена; проверьте эталоны и геометрию.')
    if scale['status'] in {'verified_local', 'single_reference'}:
        if not min(scale['depths'])-.05 <= t <= max(scale['depths'])+.05:
            result['status']='outside_calibration'
            return result
        result['warnings'].append('Эталоны задают масштаб только в этой полосе. Для других глубин нужны дополнительные эталоны или мерные линии.')
    elif scale['status'] == 'depth_calibrated' and not min(scale['depths'])-1e-7 <= t <= max(scale['depths'])+1e-7:
        result['status'] = 'outside_calibration'
        return result
    ppm = scale['intercept'] + scale['slope']*t
    if w > 1.5*scale.get('max_reference_width_px',float('inf')):
        result['warnings'].append('Автомобиль значительно шире эталона в кадре. Масштаб по короткому автомобилю не проверяет искажение по всей длине состава. Проверьте длину по документам или мерным отметкам вдоль всей зоны измерения.')
    result.update(length_m=float(w/ppm), cm_per_px=float(100/ppm),
                  coefficient=float((scale['intercept']+scale['slope'])/ppm),
                  status=scale['status'])
    return result
