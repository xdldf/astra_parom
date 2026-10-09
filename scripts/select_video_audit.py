#!/usr/bin/env python3
"""Select temporally spread car observations without consulting length estimates.

Full-duration sparse scouting is not a vehicle census. Selected detections can
be replayed with audit_additional_captures.py, using its frozen-profile checks.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


SELECTION_RULE = ('First car track with a bounding box away from image edges by best-frame time in each equal-duration bin. '
    'Require a margin of 1 percent of each image dimension, minimum 3 px. Best frame minimizes distance from measurement line; '
    'no model identity, length or error in selection.')
ALL_RULE = ('Every on-road scout track, including edge fragments and non-car detector classes. '
    'Best frame minimizes distance from measurement line; no length or catalogue filtering. '
    'Track fragments may represent the same physical passage.')


def all_tracks(closed, source):
    return [dict(id=t['id'], video=str(source), frame=t['best_frame'],
        bbox=t['best_detection']['bbox'], image=t['image'], image_sha256=t['image_sha256'],
        image_space='corrected_full_resolution', detector_label=t['best_detection']['label'],
        scout_start_frame=t['start_frame'], scout_end_frame=t['end_frame'],
        observed_labels=sorted({d['label'] for d in t['observations']}),
        prior_model_candidate=None, prior_note='All-track coverage audit; fragments retained.')
        for t in sorted(closed, key=lambda t: (t['best_frame'], t['id']))]


def select_tracks(closed, source, frame_count, bins, image_size):
    """Do not trust waiting_for_line to distinguish a truncated detector box."""
    width, height = image_size
    choices, slots = [], []
    for slot in range(bins):
        first, last = frame_count * slot / bins, frame_count * (slot + 1) / bins
        candidates, clipped = [], []
        for t in closed:
            d = t['best_detection']
            if not first <= t['best_frame'] < last or d['label'] != 'car':
                continue
            x, y, w, h = d['bbox']
            if (min(x, width-x-w) < max(3, width*.01)
                    or min(y, height-y-h) < max(3, height*.01)
                    or min(w, h) <= 0 or d['status'] == 'clipped'):
                clipped.append(t['id'])
            else:
                candidates.append(t)
        candidates.sort(key=lambda t: (t['best_frame'], t['id']))
        slots.append(dict(slot=slot, first_frame=first, last_frame_exclusive=last,
            eligible_track_candidates=len(candidates), clipped_candidates=clipped,
            selected_id=candidates[0]['id'] if candidates else None))
        if candidates:
            t = candidates[0]
            choices.append(dict(id=t['id'], video=str(source), frame=t['best_frame'],
                bbox=t['best_detection']['bbox'], image=t['image'], image_sha256=t['image_sha256'],
                image_space='corrected_full_resolution', detector_label='car', slot=slot,
                scout_start_frame=t['start_frame'], scout_end_frame=t['end_frame'],
                prior_model_candidate=None, prior_note='New-video selection; no catalogue label at selection time.'))
    return choices, slots


def scout(job):
    number, source, profile_path, weights, output, bins = job
    import cv2
    import torch
    from ultralytics import YOLO
    from vehicle_metrology.bbox_scale import measure_box
    from vehicle_metrology.detection import predict_vehicle_boxes
    from web_app.passage_tracking import match_track
    from web_app.temporal_capture import review_candidate
    from web_app.workbench import Profile, corrected, profile_scale

    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    start = time.monotonic()
    profile = Profile.model_validate_json(profile_path.read_text())
    scale = profile_scale(profile)
    folder = output / source.stem
    folder.mkdir()
    detector = YOLO(str(weights))
    cap = cv2.VideoCapture(str(source), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])
    fps, count = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not cap.isOpened() or fps <= 0 or count <= 0:
        raise ValueError(f'Unreadable video: {source}')
    active, closed, sequence = [], [], 0

    def finish(track):
        image = folder / (track['id'] + '-frame.jpg')
        image.write_bytes(track.pop('jpeg'))
        track.update(image=str(image.relative_to(output)), image_sha256=digest(image))
        closed.append(track)

    indices = range(0, count, max(1, round(fps)))
    with (folder / 'detections.jsonl').open('w') as log:
        for step, index in enumerate(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, raw = cap.read()
            if not ok:
                raise ValueError(f'Cannot decode {source.name}, frame {index}')
            if raw.shape[1::-1] != profile.image_size:
                raise ValueError('Video/profile image size mismatch')
            image = corrected(raw, profile.lens)
            detections = predict_vehicle_boxes(detector, image, .35, device='cpu', imgsz=640)
            stamp = index / fps
            for track in list(active):
                if stamp - track['stamp'] > 3:
                    active.remove(track)
                    finish(track)
            road = []
            for d in detections:
                position = measure_box(d['bbox'], profile.polygon, scale, profile.image_size,
                    profile.measurement_line_x, profile.line_tolerance_px,
                    estimate=profile.measurement_mode == 'estimate')
                # Geometry determines membership; never persist or select by length.
                d.update({k: position[k] for k in ('depth', 'status', 'line_offset_px')})
                if review_candidate(d):
                    road.append(d)
            log.write(json.dumps(dict(frame=index, seconds=stamp, detections=detections)) + '\n')
            used = set()
            for i, d in enumerate(road):
                track = match_track(active, road, i, used, stamp)
                if track is None:
                    sequence += 1
                    track = dict(id=f'N{number + 1}-{sequence:03d}', box=d['bbox'], stamp=stamp,
                        start_frame=index, observations=[], score=float('inf'))
                    active.append(track)
                used.add(track['id'])
                track.update(box=d['bbox'], stamp=stamp, end_frame=index)
                track['observations'].append(dict(frame=index, **d))
                score = abs(d['line_offset_px']) + (5000 if d['status'] == 'clipped' else 0)
                if score < track['score']:
                    ok, jpeg = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 92])
                    if not ok:
                        raise ValueError('JPEG encoding failed')
                    track.update(score=score, best_frame=index, best_detection=d.copy(), jpeg=jpeg.tobytes())
            if step % 150 == 0:
                log.flush()
                print(json.dumps(dict(video=source.name, sampled=step + 1, total=len(indices),
                    track_candidates=sequence, elapsed_s=round(time.monotonic()-start))), flush=True)
    for track in active:
        finish(track)
    cap.release()
    save(folder / 'tracks.json', closed)
    choices, slots = select_tracks(closed, source, count, bins, profile.image_size)
    summary = dict(video=str(source), video_sha256=digest(source), fps=fps, frame_count=count,
        scout_samples=len(indices), track_candidates=len(closed), slots=slots,
        selected_count=len(choices), elapsed_s=round(time.monotonic()-start))
    save(folder / 'summary.json', summary)
    save(folder / 'selected.json', choices)
    print(json.dumps(dict(video=source.name, complete=True, selected=len(choices))), flush=True)
    return summary, choices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--videos', nargs='+', type=Path, required=True)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--bins', type=int, default=4)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--reuse-scout', action='store_true',
                        help='Reselect from an existing completed scan without inference; inputs must match')
    parser.add_argument('--all-tracks', action='store_true',
                        help='With --reuse-scout, save every track to all-selection.json, retaining fragments and all vehicle classes')
    args = parser.parse_args()
    if args.bins < 1 or args.workers < 1:
        parser.error('bins and workers must be positive')
    if args.all_tracks and not args.reuse_scout:
        parser.error('--all-tracks requires --reuse-scout')
    from scripts.audit_passage_association import DETECTOR_SHA256
    from web_app.workbench import Profile
    profile = Profile.model_validate_json(args.profile.read_text())
    if digest(args.weights) != DETECTOR_SHA256 or profile.detector_model != 'yolo26m' or profile.detector_imgsz != 640:
        parser.error('This audit requires the pinned YOLO26m detector and 640 input profile')
    sources = [p.resolve(strict=True) for p in args.videos]
    hashes = [digest(p) for p in sources]
    if len(set(hashes)) != len(hashes):
        parser.error('Duplicate video content')
    output = args.output_dir.resolve()
    if args.reuse_scout:
        manifest = json.loads((output/'manifest.json').read_text())
        if (manifest['profile_sha256'] != digest(args.profile)
                or manifest['detector_sha256'] != digest(args.weights)
                or manifest['videos'] != dict(zip(map(str, sources), hashes))
                or manifest['bins'] != args.bins):
            raise ValueError('Existing scout inputs differ')
        results = []
        for source in sources:
            folder = output/source.stem
            summary = json.loads((folder/'summary.json').read_text())
            tracks = json.loads((folder/'tracks.json').read_text())
            for t in tracks:
                if digest(output/t['image']) != t['image_sha256']:
                    raise ValueError('Saved scout image changed')
            rows, slots = select_tracks(tracks, source, summary['frame_count'], args.bins, profile.image_size)
            if args.all_tracks:
                rows = all_tracks(tracks, source)
            results.append((dict(summary, slots=slots, selected_count=len(rows)), rows))
        save(output/('all-selection.json' if args.all_tracks else 'selection.json'), dict(
            selection_rule=ALL_RULE if args.all_tracks else SELECTION_RULE,
            profile_sha256=manifest['profile_sha256'], scout_manifest_sha256=digest(output/'manifest.json'),
            selection_script_sha256=digest(__file__),
            scope='All on-road track candidates; not unique vehicle count.' if args.all_tracks else
                  'Temporally stratified new-video observations; provisional complete car detections.',
            rows=[r for _, rows in results for r in rows]))
        save(output/('all-selection-summaries.json' if args.all_tracks else 'selection-summaries.json'),
             [s for s, _ in results])
        return
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(profile_sha256=digest(args.profile), detector_sha256=digest(args.weights),
        script_sha256=digest(__file__), videos=dict(zip(map(str, sources), hashes)),
        bins=args.bins, device='cpu_explicit_offline', step_seconds=1,
        selection_rule=SELECTION_RULE,
        limitations=['Sparse track candidates are not a count of unique vehicles or a detector recall audit.',
            'Vehicle identity can switch in sparse tracking; selected full images need review.',
            'Same camera, new footage; physical vehicle identities may recur across recording dates.'])
    save(output / 'manifest.json', manifest)
    jobs = [(i, p, args.profile.resolve(), args.weights.resolve(), output, args.bins) for i, p in enumerate(sources)]
    with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs)),
                             mp_context=multiprocessing.get_context('spawn')) as pool:
        results = list(pool.map(scout, jobs))
    save(output / 'selection.json', dict(selection_rule=manifest['selection_rule'],
        profile_sha256=manifest['profile_sha256'], scout_manifest_sha256=digest(output/'manifest.json'),
        scope='Temporally stratified new-video observations; provisional car detections.',
        rows=[row for _, rows in results for row in rows]))
    save(output / 'summaries.json', [s for s, _ in results])


if __name__ == '__main__':
    main()
