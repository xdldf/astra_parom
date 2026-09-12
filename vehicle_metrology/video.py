"""Recorded-file diagnostic runner. Boxes are never metric measurements."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import platform

import cv2
import numpy as np
import scipy

DISCLAIMER = 'MOG2 foreground diagnostic, not semantic vehicle classification; boxes cannot determine meters.'


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def local_video(path):
    path = Path(path)
    if not path.is_file():
        raise ValueError('video must be an existing local recorded file (no camera or URL)')
    return path


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


class ForegroundTracker:
    """Deterministic greedy nearest-center association, not identity assurance."""
    def __init__(self):
        self.background = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=24, detectShadows=False)
        self.active = {}
        self.next_id = 1

    def update(self, frame, index):
        mask = self.background.apply(frame)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        height, width = frame.shape[:2]
        boxes = sorted(cv2.boundingRect(c) for c in contours if 30 <= cv2.contourArea(c) <= width * height * .8)
        self.active = {k: v for k, v in self.active.items() if index - v[1] <= 5}
        assigned = set()
        result = []
        for box in boxes:
            x, y, w, h = box
            center = np.array([x + w / 2, y + h / 2])
            candidates = sorted((float(np.linalg.norm(center - old)), key) for key, (old, _) in self.active.items() if key not in assigned)
            if candidates and candidates[0][0] <= 60:
                key = candidates[0][1]
            else:
                key = f'foreground-{self.next_id:04d}'
                self.next_id += 1
            assigned.add(key)
            self.active[key] = (center, index)
            result.append({'track_id': key, 'box_xywh': list(box)})
        return result


POINT_KEYS = ('rear_contact_uv', 'front_contact_uv', 'rear_uv', 'front_uv')


def load_observations(path, video_hash, size, frame_count):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(data, dict) or data.get('schema_version') != 1:
        raise ValueError('unsupported observation schema')
    if data.get('video_sha256') != video_hash:
        raise ValueError('observation video hash mismatch')
    if not isinstance(data.get('tracks'), list):
        raise ValueError('tracks must be a list')
    ids = set()
    for track in data['tracks']:
        if not isinstance(track, dict):
            raise ValueError('track must be an object')
        key = track.get('track_id')
        if not isinstance(key, str) or not key.strip() or key in ids:
            raise ValueError('track_id must be unique nonempty text')
        ids.add(key)
        rows = track.get('observations')
        if not isinstance(rows, list):
            raise ValueError('observations must be a list')
        seen = set()
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError('observation must be an object')
            frame = row.get('frame')
            if type(frame) is not int or not 0 <= frame < frame_count or frame in seen:
                raise ValueError('invalid or duplicate observation frame')
            seen.add(frame)
            for point in POINT_KEYS:
                uv = row.get(point)
                if not isinstance(uv, list) or len(uv) != 2 or any(type(v) not in (int, float) or not math.isfinite(v) for v in uv):
                    raise ValueError(f'{point} requires two finite pixels')
                if not 0 <= uv[0] < size[0] or not 0 <= uv[1] < size[1]:
                    raise ValueError(f'{point} outside video image')
            sigma = row.get('sigma_px', 1.0)
            if type(sigma) not in (int, float) or not math.isfinite(sigma) or sigma <= 0:
                raise ValueError('sigma_px must be positive and finite')
        rows.sort(key=lambda row: row['frame'])
    return data


def run_video(video, *, output_dir=None, headless=False, calibration=None, observations=None,
              output_video=None, ground_truth=None, seed=0, mc_samples=0):
    video = local_video(video)
    output = Path(output_dir) if output_dir else Path('runs') / video.stem
    sources = {Path(p).resolve() for p in (video,calibration,observations,ground_truth) if p is not None}
    destinations = [output/name for name in ('results.json','frames.json','frames.csv','measurements.csv','track_frames.csv','evaluation.json')]
    if output_video:
        destinations.append(Path(output_video))
    if any(p.resolve() in sources for p in destinations) or len({p.resolve() for p in destinations}) != len(destinations):
        raise ValueError('Output paths would overwrite an input artifact or another output')
    cv2.setNumThreads(1)
    cv2.setRNGSeed(int(seed))
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError('cannot decode local video')
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(fps) or fps <= 0:
        capture.release()
        raise ValueError('video has no usable frame rate')
    tracker = ForegroundTracker()
    frames, tracks = [], {}
    index, previous_time = 0, -1.0
    size = None
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            size = [frame.shape[1], frame.shape[0]]
            pts = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000
            good_pts = math.isfinite(pts) and pts >= 0 and pts > previous_time
            timestamp = pts if good_pts else index / fps
            previous_time = timestamp
            detections = tracker.update(frame, index)
            for detection in detections:
                key = detection['track_id']
                tracks.setdefault(key, {'track_id': key, 'status': 'rejected', 'length_m': None,
                                       'reasons': ['missing_calibration', 'missing_annotated_landmarks'],
                                       'diagnostics': {}, 'frames': []})['frames'].append({'frame': index})
            frames.append({'frame': index, 'timestamp_s': timestamp, 'timestamp_source': 'decoded_pts' if good_pts else 'frame_fps_fallback', 'detections': detections})
            index += 1
    finally:
        capture.release()
    if not frames:
        raise ValueError('video contains no decodable frames')
    camera = None
    if calibration:
        from .geometry import Camera
        camera = Camera.from_dict(json.loads(Path(calibration).read_text(encoding='utf-8')))
        if list(camera.image_size) != size:
            raise ValueError('Calibration image_size does not match recorded video')
        for track in tracks.values():
            track['reasons'] = ['missing_annotated_landmarks']
    video_hash = sha256_file(video)
    annotation_data = load_observations(observations, video_hash, size, index) if observations else None
    if annotation_data is not None:
        tracks = {track['track_id']: {'track_id': track['track_id'], 'status': 'rejected',
                  'length_m': None, 'reasons': ['missing_calibration'], 'diagnostics': {},
                  'frames': track['observations']} for track in annotation_data['tracks']}
        for row in frames:
            row['detections'] = []
            row['annotations'] = [{'track_id': track['track_id'], **observation}
                                  for track in annotation_data['tracks'] for observation in track['observations']
                                  if observation['frame'] == row['frame']]
        if camera is not None:
            from .geometry import measure_track
            for track in annotation_data['tracks']:
                measurement = measure_track(camera,track['observations'],seed=seed,mc_samples=mc_samples)
                tracks[track['track_id']] = {'track_id':track['track_id'],**measurement}
        by_frame = {}
        for track in tracks.values():
            for measurement in track['frames']:
                by_frame.setdefault(measurement['frame'],[]).append({'track_id':track['track_id'],**measurement})
        for row in frames:
            row['measurements'] = by_frame.get(row['frame'],[])
    result = {'schema_version': 1, 'video': {'sha256': video_hash, 'image_size': size,
              'fps': fps, 'decoded_frames': index}, 'detector': {'name': 'MOG2_greedy_center', 'disclaimer': DISCLAIMER},
              'config': {'seed': seed, 'mc_samples': mc_samples},
              'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__, 'opencv': cv2.__version__},
              'artifacts': {'observations_sha256': sha256_file(observations) if observations else None,
                            'calibration_sha256': sha256_file(calibration) if calibration else None},
              'tracks': list(tracks.values()), 'frames': frames}
    output = Path(output_dir) if output_dir else Path('runs') / video.stem
    output.mkdir(parents=True, exist_ok=True)
    if ground_truth:
        from .evaluation import evaluate
        result['evaluation'] = evaluate(result['tracks'],ground_truth)
        result['artifacts']['ground_truth_sha256'] = sha256_file(ground_truth)
        write_json(output/'evaluation.json',result['evaluation'])
    with (output/'measurements.csv').open('w',newline='',encoding='utf-8') as stream:
        writer = csv.DictWriter(stream,fieldnames=['track_id','status','length_m','reasons','diagnostics'])
        writer.writeheader()
        for track in result['tracks']:
            writer.writerow({k:json.dumps(track[k],allow_nan=False) if k in ('reasons','diagnostics') else track[k] for k in writer.fieldnames})
    with (output/'track_frames.csv').open('w',newline='',encoding='utf-8') as stream:
        fields=['track_id','frame','x_m','y_m','heading_deg','wheelbase_m','reprojection_error_px','prefix_length_m','window_length_m']
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore')
        writer.writeheader()
        for track in result['tracks']:
            for frame in track['frames']:
                writer.writerow({'track_id':track['track_id'],**frame})
    write_json(output / 'results.json', result)
    write_json(output / 'frames.json', frames)
    with (output / 'frames.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['frame', 'timestamp_s', 'timestamp_source', 'detections', 'annotations', 'measurements'])
        writer.writeheader()
        for row in frames:
            writer.writerow({**row, 'detections': json.dumps(row['detections'], allow_nan=False),
                             'annotations': json.dumps(row.get('annotations', []), allow_nan=False),
                             'measurements': json.dumps(row.get('measurements', []), allow_nan=False)})
    if output_video or not headless:
        render_video(video, result, output_video=output_video, display=not headless)
    return result


class Playback:
    """Shared keyboard state for replay and annotation; no capture side effects."""
    def __init__(self, count):
        self.count, self.index, self.paused, self.closed = count, 0, False, False

    def key(self, key):
        if key in (27, ord('q')):
            self.closed = True
        elif key == ord(' '):
            self.paused = not self.paused
        elif key in (ord('n'), ord('.')):
            self.index = min(self.count - 1, self.index + 1)
            self.paused = True
        elif key in (ord('b'), ord(',')):
            self.index = max(0, self.index - 1)
            self.paused = True
        elif key in (ord('r'), ord('R')):
            self.index = 0
            self.paused = True

    def advance(self):
        if not self.paused:
            if self.index + 1 < self.count:
                self.index += 1
            else:
                self.paused = True


def overlay_frame(frame, row, tracks):
    image = frame.copy()
    cv2.putText(image, f"frame {row['frame']}  {row['timestamp_s']:.3f}s", (4, 16),
                cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 255, 255), 1)
    cv2.putText(image, 'Conditional geometry / diagnostic detections', (4, image.shape[0]-12), cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 255, 255), 1)
    for detection in row['detections']:
        x, y, w, h = detection['box_xywh']
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 180, 255), 1)
        cv2.putText(image, detection['track_id'] + ': no meters', (x, max(12, y - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, .35, (0, 180, 255), 1)
    colors = [(255, 100, 0), (0, 200, 0), (255, 0, 255), (0, 255, 255)]
    for annotation in row.get('annotations', []):
        for key, color in zip(POINT_KEYS, colors):
            point = tuple(np.rint(annotation[key]).astype(int))
            cv2.circle(image, point, 4, color, -1)
            cv2.putText(image, key, point, cv2.FONT_HERSHEY_SIMPLEX, .35, color, 1)
        track = tracks[annotation['track_id']]
        length = track['length_m']
        label = f"{track['track_id']} whole-track: " + (f'{length:.3f} m' if length is not None else 'null ' + ','.join(track['reasons']))
        measured = next((m for m in row.get('measurements',[]) if m['track_id'] == track['track_id']),{})
        if measured.get('window_length_m') is not None:
            label += f" | window {measured['window_length_m']:.3f} m"
        label_y = 65 + 22*list(tracks).index(track['track_id'])
        cv2.putText(image, label, (8,label_y), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
    return image


def render_video(video, result, *, output_video=None, display=True):
    capture = cv2.VideoCapture(str(video))
    tracks = {track['track_id']: track for track in result['tracks']}
    writer = None
    window = 'Recorded video | space pause | n/b step | R replay | q quit'
    try:
        if output_video:
            target = Path(output_video)
            target.parent.mkdir(parents=True, exist_ok=True)
            codec = 'MJPG' if target.suffix.lower() == '.avi' else 'mp4v'
            writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*codec), result['video']['fps'], tuple(result['video']['image_size']))
            if not writer.isOpened():
                raise ValueError('cannot create output video')
            for row in result['frames']:
                ok, frame = capture.read()
                if not ok:
                    raise ValueError('video decode failed during overlay export')
                writer.write(overlay_frame(frame, row, tracks))
            writer.release()
            writer = None
        if display:
            state = Playback(len(result['frames']))
            cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
            while not state.closed:
                capture.set(cv2.CAP_PROP_POS_FRAMES, state.index)
                ok, frame = capture.read()
                if not ok:
                    raise ValueError('video seek/decode failed during replay')
                cv2.imshow(window, overlay_frame(frame, result['frames'][state.index], tracks))
                key = cv2.waitKey(30 if state.paused else max(1, round(1000 / result['video']['fps']))) & 0xff
                if key == 255:
                    state.advance()
                else:
                    state.key(key)
                if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if display:
            cv2.destroyAllWindows()
