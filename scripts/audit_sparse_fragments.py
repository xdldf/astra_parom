#!/usr/bin/env python3
"""Compare sparse and every-frame tracking on explicitly chosen fragment pairs.

This diagnoses association on bounded clips, not recall across an entire video.
Only corrected full-frame inputs are passed to the detector.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.select_video_audit import digest, save
from scripts.audit_passage_association import DETECTOR_SHA256


def replay(frames, fps, stride):
    from web_app.passage_tracking import match_track
    tracks, active = [], []
    for row in frames:
        if row['frame'] % stride:
            continue
        stamp = row['frame']/fps
        active = [t for t in active if stamp-t['stamp'] <= 3]
        used = set()
        for index, detection in enumerate(row['road']):
            track = match_track(active, row['road'], index, used, stamp)
            if track is None:
                track = dict(id=len(tracks)+1, box=detection['bbox'], stamp=stamp, observations=[])
                tracks.append(track)
                active.append(track)
            used.add(track['id'])
            track.update(box=detection['bbox'], stamp=stamp)
            track['observations'].append(dict(frame=row['frame'], **detection))
    return tracks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('selection', 'profile', 'weights', 'output-dir'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--pairs', nargs='+', required=True, help='Pairs of IDs, e.g. N1-003:N1-004')
    args = parser.parse_args()
    import cv2
    import torch
    from ultralytics import YOLO
    from vehicle_metrology.bbox_scale import measure_box
    from vehicle_metrology.detection import predict_vehicle_boxes, box_iou
    from web_app.temporal_capture import review_candidate
    from web_app.workbench import Profile, corrected, profile_scale
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    selection = json.loads(args.selection.read_text())
    profile = Profile.model_validate_json(args.profile.read_text())
    if digest(args.profile) != selection['profile_sha256'] or digest(args.weights) != DETECTOR_SHA256:
        raise ValueError('Profile/detector mismatch')
    rows = {r['id']: r for r in selection['rows']}
    model = YOLO(str(args.weights))
    scale = profile_scale(profile)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = dict(profile_sha256=digest(args.profile), detector_sha256=digest(args.weights),
        selection_sha256=digest(args.selection), script_sha256=digest(__file__),
        tracker_sha256=digest(ROOT/'web_app/passage_tracking.py'), confidence=.35, imgsz=640,
        device='cpu', scope='Two endpoint fragments per bounded clip; not a recall audit.', clips=[])
    for pair in args.pairs:
        first, last = [rows[ident] for ident in pair.split(':')]
        if first['video'] != last['video'] or first['frame'] >= last['frame']:
            raise ValueError('Pair must be chronologically ordered in one video')
        cap = cv2.VideoCapture(first['video'], cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])
        fps = cap.get(cv2.CAP_PROP_FPS)
        start, end = max(0, first['frame']-round(fps)), last['frame']+round(fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        frames, photos = [], []
        for number in range(start, end+1):
            ok, raw = cap.read()
            if not ok or raw.shape[1::-1] != profile.image_size:
                raise ValueError('Cannot decode frame')
            image = corrected(raw, profile.lens)
            detections = predict_vehicle_boxes(model, image, .35, device='cpu', imgsz=640)
            road = []
            for d in detections:
                position = measure_box(d['bbox'], profile.polygon, scale, profile.image_size,
                    profile.measurement_line_x, profile.line_tolerance_px,
                    estimate=profile.measurement_mode == 'estimate')
                d.update({k: position[k] for k in ('depth', 'status', 'line_offset_px')})
                if review_candidate(d):
                    road.append(d)
            frames.append(dict(frame=number, road=road, detections=detections))
            if number in {first['frame'], last['frame'], (first['frame']+last['frame'])//2}:
                path = args.output_dir/f'{first["id"]}-{number}-frame.jpg'
                if not cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 92]):
                    raise ValueError('Photo encoding failed')
                photos.append(dict(frame=number, file=path.name, sha256=digest(path)))
        cap.release()
        checks = {}
        for name, stride in [('every_frame', 1), ('one_second', round(fps))]:
            tracks = replay(frames, fps, stride)
            endpoint_tracks = [[t['id'] for t in tracks if any(
                o['frame'] == r['frame'] and box_iou(o['bbox'], r['bbox']) > .8
                for o in t['observations'])] for r in (first, last)]
            checks[name] = dict(track_count=len(tracks), endpoint_tracks=endpoint_tracks,
                endpoints_share_track=bool(set(endpoint_tracks[0]) & set(endpoint_tracks[1])), tracks=tracks)
        path = args.output_dir/(first['id']+'-detections.json')
        save(path, frames)
        report['clips'].append(dict(pair=pair, video=first['video'], video_sha256=digest(first['video']),
            first_frame=start, last_frame=end, fps=fps, decoded_frames=len(frames),
            road_detections_by_label=dict(Counter(d['label'] for f in frames for d in f['road'])),
            cache=path.name, cache_sha256=digest(path), photos=photos, checks=checks))
        save(args.output_dir/'report.json', report)
        print(pair, {k:(v['track_count'],v['endpoints_share_track']) for k,v in checks.items()}, flush=True)


if __name__ == '__main__':
    main()
