"""Robust local passage estimates; consistency is not physical accuracy."""
import math
import numpy as np

from .bbox_scale import measure_box
from .detection import box_iou


def _line_fit(x, y):
    # Median pairwise slopes tolerate one bad detector frame without pretending
    # that residual lens/perspective effects vanish across the local window.
    slopes = [(y[j]-y[i])/(x[j]-x[i]) for i in range(len(x)) for j in range(i+1, len(x))
              if abs(x[j]-x[i]) >= 8.]
    slope = float(np.median(slopes)) if slopes and np.ptp(x) >= 15 else 0.
    intercept = float(np.median(y-slope*x))
    return intercept, slope


def measure_passage(frames, anchor_box, polygon, scale, image_size, *, line_x,
                    line_tolerance_px=10., tolerance_m=.1, min_samples=5):
    """Associate a short neighbourhood with one box and fit length at the line.

    Input frames are decoded observations, not repeated browser polls. At least
    one actual box must satisfy the original measurement-line gate.
    Prefer the original narrow band. For fast passages with fewer than five
    samples, use up to 20% of the anchor width, capped at 120 original pixels.
    No ground-depth extrapolation, imputed observations or catalogue dimensions.
    """
    if not math.isfinite(tolerance_m) or tolerance_m <= 0 or min_samples < 5:
        raise ValueError('Positive tolerance_m and min_samples >= 5 required')
    result = dict(status='temporal_review', length_m=None, reasons=[],
                  method='robust_local_multiframe', accuracy_validated=False,
                  target_tolerance_m=tolerance_m, samples=[], diagnostics={})
    seen = set()
    rejected = []
    ambiguous = []
    half_band = min(60., max(30., 3*line_tolerance_px))
    extended_band = max(half_band, min(120., .2*anchor_box[2]))
    for frame in sorted(frames, key=lambda row: row['frame']):
        index = frame['frame']
        if type(index) is not int or index < 0 or index in seen:
            raise ValueError('Frame indices must be distinct nonnegative integers')
        seen.add(index)
        matches = sorted(((box_iou(anchor_box, d['bbox']), d) for d in frame['detections']),
                         key=lambda row: row[0], reverse=True)
        if not matches or matches[0][0] < .55:
            rejected.append(dict(frame=index, reason='lost_vehicle'))
            continue
        box = matches[0][1]['bbox']
        offset = 0. if line_x is None else box[0]+box[2]/2-line_x
        if abs(offset) > extended_band:
            rejected.append(dict(frame=index, reason='outside_temporal_band'))
            continue
        if len(matches) > 1 and matches[1][0] > .4:
            ambiguous.append(offset)
            rejected.append(dict(frame=index, reason='ambiguous_vehicle_association'))
            continue
        measured = measure_box(box, polygon, scale, image_size)
        if measured['length_m'] is None:
            rejected.append(dict(frame=index, reason=measured['status']))
            continue
        result['samples'].append(dict(frame=index, bbox=list(box),
                                      line_offset_px=offset, length_m=measured['length_m']))
    narrow = [s for s in result['samples'] if abs(s['line_offset_px']) <= half_band]
    expanded = len(narrow) < min_samples
    if not expanded:
        rejected.extend(dict(frame=s['frame'],reason='outside_temporal_band')
                        for s in result['samples'] if abs(s['line_offset_px']) > half_band)
        result['samples'] = narrow
    samples = result['samples']
    used_band = extended_band if expanded else half_band
    if any(abs(offset)<=used_band for offset in ambiguous):
        result['reasons'].append('ambiguous_vehicle_association')
    has_line = any(abs(s['line_offset_px']) <= line_tolerance_px for s in samples)
    result['diagnostics'].update(requested_frames=len(frames), used_frames=len(samples),
                                 temporal_band_px=extended_band if expanded else half_band,
                                 expanded_for_fast_passage=expanded and extended_band>half_band,
                                 excluded_frames=rejected,
                                 warning='Temporal consistency only; shared calibration and boundary bias remain unvalidated.')
    if len(samples) < min_samples:
        result['reasons'].append('insufficient_temporal_frames')
    if not has_line:
        result['reasons'].append('missing_line_evidence')
    if result['reasons']:
        return result
    x = np.asarray([s['line_offset_px'] for s in samples])
    y = np.asarray([s['length_m'] for s in samples])
    if not (min(x) <= 0 <= max(x)) and np.max(np.abs(x)) > line_tolerance_px:
        result['reasons'].append('line_not_bracketed')
        return result
    estimate, slope = _line_fit(x, y)
    residual = y-(estimate+slope*x)
    split = [_line_fit(x[i::2], y[i::2])[0] for i in (0, 1)]
    max_residual = float(np.max(np.abs(residual)))
    split_difference = abs(split[0]-split[1])
    result['diagnostics'].update(raw_range_m=float(np.ptp(y)), residual_max_m=max_residual,
                                 residual_mad_m=float(np.median(np.abs(residual))),
                                 interleaved_difference_m=split_difference,
                                 slope_m_per_px=slope, candidate_length_m=estimate)
    if estimate <= 0 or max_residual > tolerance_m or split_difference > tolerance_m:
        result['reasons'].append('unstable_temporal_length')
        return result
    result.update(status='temporal_consistent', length_m=estimate)
    return result


def apply_passage(measurement, passage):
    """Preserve the single-frame evidence alongside the independently recomputed fit."""
    result = dict(measurement, single_frame_length_m=measurement['length_m'], single_frame_status=measurement['status'],
                  temporal=passage, length_m=passage['length_m'])
    result['warnings'] = list(measurement.get('warnings', []))
    counts=passage.get('diagnostics',{})
    if 'requested_frames' in counts:
        result['warnings'].append(f"Проверено кадров: {counts['requested_frames']}; использовано для длины: {counts['used_frames']}.")
    if passage['length_m'] is None:
        result['status'] = 'temporal_review'
        result['warnings'].append('Недостаточно устойчивых кадров одного автомобиля. Длина не назначена; требуется проверка.')
    else:
        result['status'] = 'temporal_consistent'
        result['warnings'].append('Оценка по нескольким кадрам. Стабильность кадров не подтверждает абсолютную точность; проверьте калибровку.')
    return result
