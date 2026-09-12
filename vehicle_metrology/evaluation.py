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


def evaluate(tracks, ground_truth):
    truth={}
    with open(ground_truth,newline='',encoding='utf-8-sig') as stream:
        for row in csv.DictReader(stream):
            key=row.get('track_id','')
            if not key or key in truth:
                raise ValueError('Empty or duplicate ground truth track_id')
            value=float(row['length_m'])
            if not math.isfinite(value) or value <= 0:
                raise ValueError('Ground truth length_m must be finite and positive')
            truth[key]={'length_m':value,'vehicle_id':row.get('vehicle_id') or key}
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
        row=dict(track_id=key,vehicle_id=gt['vehicle_id'],truth_m=gt['length_m'],
            prediction_m=track['length_m'],error_m=error,absolute_error_m=abs(error),
            relative_abs_error=abs(error)/gt['length_m'])
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
    return dict(metrics=metrics(rows),per_track=rows,per_vehicle={k:metrics(v) for k,v in sorted(vehicles.items())},
        coverage=dict(ground_truth_tracks=len(truth),detected_tracks=len(tracks),accepted_matched=len(rows),
            fraction_of_ground_truth_measured=len(rows)/len(truth) if truth else None,
            missed_track_ids=sorted(set(truth)-present),unmatched_prediction_ids=sorted(present-set(truth)),
            rejected_tracks=len(rejected),rejection_reasons=dict(Counter(r for t in rejected for r in t.get('reasons',[])))),
        spatial_bins={k:metrics(v) for k,v in sorted(spatial.items())},
        orientation_bins={k:metrics(v) for k,v in sorted(orientation.items())},
        window_metrics=metrics(windows),window_errors=windows,
        conditional_interval_coverage=dict(count=len(intervals),fraction=float(np.mean(intervals)) if intervals else None),
        warnings=['Spatial bins contain one whole-track result per visited cell, correlated across cells. Use window_errors for trajectory-local errors.',
            'Window estimates overlap and are correlated; no independent-frame statistical claims.',
            'Repeated physical vehicles are grouped, not independent samples. No cluster-bootstrap population confidence supplied.',
            'Unmatched IDs are association diagnostics, not independently adjudicated false detections. Manual IDs do not validate automatic tracking.'])
