"""Independent ground truth is consumed only after measurement."""
import csv
from collections import Counter, defaultdict
import math
import numpy as np


def metrics(rows):
    if not rows:
        return {'count':0}
    error = np.asarray([r['error_m'] for r in rows])
    absolute = np.abs(error)
    relative = absolute/np.asarray([r['truth_m'] for r in rows])
    return dict(count=len(rows),bias_m=float(np.mean(error)),mae_m=float(np.mean(absolute)),
        median_abs_error_m=float(np.median(absolute)),rmse_m=float(np.sqrt(np.mean(error**2))),
        max_abs_error_m=float(np.max(absolute)),p90_abs_error_m=float(np.quantile(absolute,.90)),
        p95_abs_error_m=float(np.quantile(absolute,.95)),p99_abs_error_m=float(np.quantile(absolute,.99)),
        mean_relative_abs_error=float(np.mean(relative)),median_relative_abs_error=float(np.median(relative)),
        tail_warning='Descriptive sample percentiles, not population guarantees; small samples cannot establish tails.')


def evaluate(tracks, ground_truth, tolerance_m=.1, *, calibration_vehicle_ids=(), calibration_track_ids=()):
    if not math.isfinite(tolerance_m) or tolerance_m <= 0:
        raise ValueError('tolerance_m must be finite and positive')
    tracks = list(tracks)
    seen = set()
    for track in tracks:
        key = track.get('track_id')
        if not key or key in seen:
            raise ValueError('Empty or duplicate prediction track_id')
        seen.add(key)
        length = track.get('length_m')
        if length is not None and (not math.isfinite(length) or length <= 0):
            raise ValueError('Prediction length_m must be finite and positive, or null for rejection')
    truth={}
    calibration_vehicle_ids=set(calibration_vehicle_ids)
    calibration_track_ids=set(calibration_track_ids)
    with open(ground_truth,newline='',encoding='utf-8-sig') as stream:
        for row in csv.DictReader(stream):
            key=row.get('track_id','')
            if not key or key in truth:
                raise ValueError('Empty or duplicate ground truth track_id')
            value=float(row['length_m'])
            if not math.isfinite(value) or value <= 0:
                raise ValueError('Ground truth length_m must be finite and positive')
            uncertainty=row.get('uncertainty_m','').strip()
            uncertainty=float(uncertainty) if uncertainty else None
            if uncertainty is not None and (not math.isfinite(uncertainty) or uncertainty <= 0):
                raise ValueError('Ground truth uncertainty_m must be positive and finite when provided')
            vehicle_id=row.get('vehicle_id','').strip()
            reference_reasons=[]
            if not vehicle_id: reference_reasons.append('missing_physical_vehicle_id')
            if uncertainty is None: reference_reasons.append('missing_reference_uncertainty')
            if row.get('reference_source') != 'physical_measurement': reference_reasons.append('reference_not_physically_measured')
            if row.get('dataset_role') != 'validation': reference_reasons.append('reference_not_reserved_for_validation')
            if vehicle_id in calibration_vehicle_ids or key in calibration_track_ids:
                reference_reasons.append('calibration_data_leakage')
            truth[key]={'length_m':value,'vehicle_id':vehicle_id or key,
                        'uncertainty_m':uncertainty,'reference_reasons':reference_reasons}
    rows=[]
    windows=[]
    spatial=defaultdict(list)
    orientation=defaultdict(list)
    vehicles=defaultdict(list)
    intervals=[]
    for track in tracks:
        key=track['track_id']
        if key not in truth or track.get('length_m') is None:
            continue
        gt=truth[key]
        error=float(track['length_m'])-gt['length_m']
        guarded_error=abs(error)+gt['uncertainty_m'] if gt['uncertainty_m'] is not None else None
        reasons=list(gt['reference_reasons'])
        if track.get('complete_vehicle') is not True: reasons.append('complete_vehicle_not_verified')
        # Numerical equality at the boundary is a failure: the goal is *lower*
        # than the tolerance, not <= rounded to a displayed centimetre value.
        within=guarded_error is not None and guarded_error < tolerance_m-1e-12 and not reasons
        row=dict(track_id=key,vehicle_id=gt['vehicle_id'],truth_m=gt['length_m'],
            prediction_m=track['length_m'],error_m=error,absolute_error_m=abs(error),
            relative_abs_error=abs(error)/gt['length_m'],within_tolerance=within,
            reference_uncertainty_m=gt['uncertainty_m'],guarded_absolute_error_m=guarded_error,
            reference_reasons=reasons)
        rows.append(row)
        vehicles[gt['vehicle_id']].append(row)
        interval=track.get('diagnostics',{}).get('uncertainty',{}).get('conditional_p95_m')
        if interval:
            intervals.append(interval[0] <= gt['length_m'] <= interval[1])
        visited_spatial=set()
        visited_orientation=set()
        for frame in track.get('frames',[]):
            if 'x_m' not in frame:
                continue
            cell=f"x[{5*math.floor(frame['x_m']/5)},{5*(math.floor(frame['x_m']/5)+1)}) y[{2*math.floor(frame['y_m']/2)},{2*(math.floor(frame['y_m']/2)+1)})"
            heading=(frame['heading_deg']+180)%360-180
            angle=f"heading[{15*math.floor(heading/15)},{15*(math.floor(heading/15)+1)})"
            if cell not in visited_spatial:
                spatial[cell].append(row)
                visited_spatial.add(cell)
            if angle not in visited_orientation:
                orientation[angle].append(row)
                visited_orientation.add(angle)
            if frame.get('window_length_m') is not None:
                windows.append(dict(track_id=key,frame=frame['frame'],truth_m=gt['length_m'],
                    error_m=frame['window_length_m']-gt['length_m'],x_m=frame['x_m'],y_m=frame['y_m'],heading_deg=heading))
    present={t['track_id'] for t in tracks}
    rejected=[t for t in tracks if t.get('length_m') is None]
    passed = sum(row['within_tolerance'] for row in rows)
    unmatched=sorted(present-set(truth))
    acceptance = dict(tolerance_m=tolerance_m, passed_tracks=passed,
        failed_track_ids=[row['track_id'] for row in rows if not row['within_tolerance']],
        fraction_of_measured_within_tolerance=passed/len(rows) if rows else None,
        fraction_of_ground_truth_within_tolerance=passed/len(truth) if truth else None,
        all_ground_truth_tracks_within_tolerance=bool(truth) and passed == len(truth) and not unmatched,
        criterion='abs(prediction-truth) + reference uncertainty < tolerance; independent physical references and complete vehicles required',
        invalid_reference_track_ids=sorted(key for key,value in truth.items() if value['reference_reasons']),
        note='Observed sample only. Missed, rejected, incomplete and unmatched vehicles do not count as passes; this is not a population guarantee.')
    return dict(metrics=metrics(rows),per_track=rows,per_vehicle={k:metrics(v) for k,v in sorted(vehicles.items())},
        acceptance=acceptance,
        coverage=dict(ground_truth_tracks=len(truth),detected_tracks=len(tracks),accepted_matched=len(rows),
            fraction_of_ground_truth_measured=len(rows)/len(truth) if truth else None,
            missed_track_ids=sorted(set(truth)-present),unmatched_prediction_ids=unmatched,
            rejected_tracks=len(rejected),rejection_reasons=dict(Counter(r for t in rejected for r in t.get('reasons',[])))),
        spatial_bins={k:metrics(v) for k,v in sorted(spatial.items())},
        orientation_bins={k:metrics(v) for k,v in sorted(orientation.items())},
        window_metrics=metrics(windows),window_errors=windows,
        conditional_interval_coverage=dict(count=len(intervals),fraction=float(np.mean(intervals)) if intervals else None),
        warnings=['Spatial bins contain one whole-track result per visited cell, correlated across cells. Use window_errors for trajectory-local errors.',
            'Window estimates overlap and are correlated; no independent-frame statistical claims.',
            'Repeated physical vehicles are grouped, not independent samples. No cluster-bootstrap population confidence supplied.',
            'Unmatched IDs are association diagnostics, not independently adjudicated false detections. Manual IDs do not validate automatic tracking.'])
