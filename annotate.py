"""Manually annotate original pixels of recorded video, never box-derived meters.

Click rear contact, front contact (same side), rear extreme, front extreme.
Use SAME rigid physical extreme points across frames, not silhouette tangents.
Wheel-road contacts are not tread marks; occluded landmarks must not be guessed.
"""
import argparse
import math
from pathlib import Path

import cv2

from vehicle_metrology.video import POINT_KEYS, Playback, load_observations, local_video, sha256_file, write_json


class AnnotationSession:
    def __init__(self, video, output, track_id='vehicle-1', sigma_px=1.0):
        self.video, self.output = local_video(video), Path(output)
        if not track_id.strip() or not math.isfinite(sigma_px) or sigma_px <= 0:
            raise ValueError('track ID and positive sigma_px required')
        capture = cv2.VideoCapture(str(self.video))
        count = 0
        self.size = None
        self.fps = capture.get(cv2.CAP_PROP_FPS)
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                self.size = [frame.shape[1], frame.shape[0]]
                count += 1
        finally:
            capture.release()
        if not count:
            raise ValueError('no decodable video frames')
        self.playback = Playback(count)
        self.playback.paused = True
        self.data = load_observations(output, sha256_file(video), self.size, count) if self.output.exists() else {
            'schema_version': 1, 'video_sha256': sha256_file(video), 'tracks': []}
        self.track_id, self.sigma_px = track_id, sigma_px
        self.data['tracks'] = self.data['tracks']
        if not any(t['track_id'] == track_id for t in self.data['tracks']):
            self.data['tracks'].append({'track_id': track_id, 'observations': []})
        self.points = []

    @property
    def track(self):
        return next(t for t in self.data['tracks'] if t['track_id'] == self.track_id)

    def add_point(self, x, y):
        if not self.playback.paused:
            return
        if len(self.points) >= 4:
            raise ValueError('four points already selected; commit or clear')
        if not 0 <= x < self.size[0] or not 0 <= y < self.size[1]:
            raise ValueError('point outside original image')
        self.points.append([float(x), float(y)])

    def commit(self):
        if len(self.points) != 4:
            raise ValueError('all four points required')
        self.delete()
        self.track['observations'].append({'frame': self.playback.index,
                                          'sigma_px': self.sigma_px, **dict(zip(POINT_KEYS, self.points))})
        self.track['observations'].sort(key=lambda row: row['frame'])
        self.points = []

    def delete(self):
        self.track['observations'] = [row for row in self.track['observations'] if row['frame'] != self.playback.index]

    def save(self):
        self.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.output.with_name(self.output.name + '.tmp')
        write_json(temporary, self.data)
        temporary.replace(self.output)


def annotation_ui(session):
    window = 'Annotation: original pixels (see console controls)'
    capture = cv2.VideoCapture(str(session.video))
    status = ''
    print(__doc__)
    print('space play/pause | n/b frame | R replay | click 4 points | Enter commit+save | u undo | c clear | d delete+save | Tab switch track | s save | q save+quit')
    print('Add another identity by rerunning with --track-id; existing tracks are preserved.')
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)

    def mouse(event, x, y, flags, param):
        nonlocal status
        if event == cv2.EVENT_LBUTTONDOWN:
            try:
                session.add_point(x, y)
                status = ''
            except ValueError as exc:
                status = str(exc)
    cv2.setMouseCallback(window, mouse)
    try:
        while not session.playback.closed:
            capture.set(cv2.CAP_PROP_POS_FRAMES, session.playback.index)
            ok, image = capture.read()
            if not ok:
                raise ValueError('failed to seek/decode video')
            current = next((row for row in session.track['observations'] if row['frame'] == session.playback.index), None)
            marked = session.points if session.points else ([current[k] for k in POINT_KEYS] if current else [])
            for i, point in enumerate(marked):
                pixel = tuple(round(v) for v in point)
                cv2.circle(image, pixel, 4, (0, 255, 255), -1)
                cv2.putText(image, POINT_KEYS[i], pixel, cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 255, 255), 1)
            next_point = POINT_KEYS[len(session.points)] if len(session.points) < 4 else 'Enter commits'
            lines = [f'{session.track_id} frame {session.playback.index}', f'next: {next_point}', status]
            for i, text in enumerate(lines):
                cv2.putText(image, text, (4, 16 + i * 18), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 200, 255), 1)
            cv2.imshow(window, image)
            key = cv2.waitKey(30 if session.playback.paused else max(1, round(1000 / max(1, session.fps)))) & 0xff
            old_index = session.playback.index
            try:
                if key in (10, 13):
                    session.commit()
                    session.save()
                    status = 'saved'
                elif key == ord('u'):
                    session.points = session.points[:-1]
                elif key == ord('c'):
                    session.points = []
                elif key == ord('d'):
                    session.delete()
                    session.save()
                elif key == ord('s'):
                    session.save()
                elif key == 9:
                    ids = [t['track_id'] for t in session.data['tracks']]
                    session.track_id = ids[(ids.index(session.track_id) + 1) % len(ids)]
                    session.points = []
                elif key == 255:
                    session.playback.advance()
                else:
                    session.playback.key(key)
                if old_index != session.playback.index:
                    session.points = []
                if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except ValueError as exc:
                status = str(exc)
        session.save()
    finally:
        capture.release()
        cv2.destroyAllWindows()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video')
    parser.add_argument('--output', required=True)
    parser.add_argument('--track-id', default='vehicle-1')
    parser.add_argument('--sigma-px', type=float, default=1.0)
    args = parser.parse_args(argv)
    try:
        annotation_ui(AnnotationSession(args.video, args.output, args.track_id, args.sigma_px))
    except (ValueError, OSError) as exc:
        parser.exit(2, str(exc) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
