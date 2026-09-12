"""Real codec/file tests on explicitly synthetic moving rectangles, not traffic."""
import json
from pathlib import Path

import cv2
import numpy as np
import pytest


def synthetic_video(path, count=18):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 10, (160, 120))
    assert writer.isOpened()
    for i in range(count):
        frame = np.zeros((120, 160, 3), np.uint8)
        if i >= 3:
            cv2.rectangle(frame, (8 + i * 3, 45), (30 + i * 3, 65), (255, 255, 255), -1)
        writer.write(frame)
    writer.release()
    return path


def test_real_video_without_calibration_never_claims_meters(tmp_path):
    from vehicle_metrology.video import run_video
    video = synthetic_video(tmp_path / 'SYNTHETIC_rectangles.avi')
    output = tmp_path / 'run'
    result = run_video(video, output_dir=output, headless=True)
    assert result['video']['decoded_frames'] == 18
    assert result['tracks']
    assert all(t['length_m'] is None for t in result['tracks'])
    assert all('missing_calibration' in t['reasons'] for t in result['tracks'])
    assert 'not semantic' in result['detector']['disclaimer']
    assert len(result['frames']) == 18
    assert json.loads((output / 'results.json').read_text()) == result
    assert (output / 'frames.csv').is_file()
    again = run_video(video, output_dir=tmp_path / 'again', headless=True)
    assert result == again


def test_cli_saves_decodable_overlays_and_playback_controls(tmp_path):
    import subprocess
    import sys
    from vehicle_metrology.video import Playback
    state = Playback(18)
    assert state.paused is False
    state.key(ord(' '))
    assert state.paused
    state.key(ord('n'))
    assert state.index == 1 and state.paused
    state.key(ord('b'))
    assert state.index == 0
    state.index = 17
    state.key(ord('r'))
    assert state.index == 0
    state.key(ord('q'))
    assert state.closed
    video = synthetic_video(tmp_path / 'SYNTHETIC.avi')
    overlay = tmp_path / 'overlay.avi'
    proc = subprocess.run([sys.executable, 'test.py', str(video), '--headless',
                           '--output-dir', str(tmp_path / 'run'), '--output-video', str(overlay)],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    cap = cv2.VideoCapture(str(overlay))
    decoded = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        decoded.append(frame)
    cap.release()
    assert len(decoded) == 18
    assert np.count_nonzero(decoded[0]) > 0  # genuine text drawn over the blank synthetic frame


def test_calibrated_annotations_use_parent_geometry_and_export_frame_drift(tmp_path):
    from vehicle_metrology.geometry import Camera
    from vehicle_metrology.video import run_video, sha256_file
    rotation = np.array([[1, 0, 0], [0, -1 / np.sqrt(5), -2 / np.sqrt(5)], [0, 2 / np.sqrt(5), -1 / np.sqrt(5)]])
    camera = Camera.from_dict({'schema_version': 1, 'image_size': [160, 120], 'model': 'brown',
        'K': [[80, 0, 80], [0, 80, 60], [0, 0, 1]], 'D': [0, 0, 0, 0, 0],
        'Rcw': rotation.tolist(), 'tcw': (-rotation @ np.array([0, -10, 5])).tolist(),
        'road_polygon': [[-15, -3], [15, -3], [15, 8], [-15, 8]], 'calibration_id': 'SYNTHETIC'})
    video = synthetic_video(tmp_path / 'SYNTHETIC.avi')
    points = np.array([[0, 0, 0], [2.7, 0, 0], [-.9, .3, .6], [3.6, .4, .8]])
    rows = []
    for i, x in enumerate(np.linspace(-4, 4, 18)):
        uv = camera.project(points + [x, 1, 0])
        rows.append({'frame': i, **dict(zip(['rear_contact_uv', 'front_contact_uv', 'rear_uv', 'front_uv'], uv.tolist()))})
    sidecar = tmp_path / 'manual.json'
    sidecar.write_text(json.dumps({'schema_version': 1, 'video_sha256': sha256_file(video), 'tracks': [{'track_id': 'synthetic-only', 'observations': rows}]}))
    calibration = tmp_path / 'calibration.json'
    calibration.write_text(json.dumps(camera.to_dict()))
    result = run_video(video, output_dir=tmp_path / 'run', headless=True, observations=sidecar, calibration=calibration)
    assert result['tracks'][0]['length_m'] == pytest.approx(4.5, abs=1e-6)
    assert result['frames'][-1]['measurements'][0]['window_length_m'] == pytest.approx(4.5, abs=1e-6)
    truth = tmp_path / 'truth.csv'
    truth.write_text('track_id,vehicle_id,length_m\nsynthetic-only,synthetic-body,4.5\n')
    with_truth = run_video(video, output_dir=tmp_path / 'validated', headless=True,
                          observations=sidecar, calibration=calibration, ground_truth=truth)
    assert with_truth['evaluation']['metrics']['mae_m'] < 1e-6
    assert with_truth['tracks'] == result['tracks']
    assert (tmp_path / 'validated' / 'measurements.csv').exists()
    assert (tmp_path / 'validated' / 'track_frames.csv').exists()
    proc = __import__('subprocess').run([__import__('sys').executable, 'test.py', '--video', str(video),
          '--headless','--output-dir',str(tmp_path/'cli-flag')],capture_output=True,text=True)
    assert proc.returncode == 0, proc.stderr
    changed = camera.to_dict()
    changed['image_size'] = [320, 240]
    calibration.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='image_size'):
        run_video(video, output_dir=tmp_path / 'bad', headless=True, calibration=calibration)


def test_manual_annotation_edits_roundtrip_real_video(tmp_path):
    from annotate import AnnotationSession
    from vehicle_metrology.video import load_observations, sha256_file
    video = synthetic_video(tmp_path / 'SYNTHETIC.avi')
    output = tmp_path / 'manual.json'
    session = AnnotationSession(video, output, track_id='car-A')
    session.playback.index = 4
    session.add_point(10, 60)
    with pytest.raises(ValueError, match='four'):
        session.commit()
    for point in [(30, 60), (8, 40), (35, 40)]:
        session.add_point(*point)
    session.commit()
    session.save()
    loaded = load_observations(output, sha256_file(video), [160, 120], 18)
    assert loaded['tracks'][0]['observations'][0]['front_uv'] == [35, 40]
    resumed = AnnotationSession(video, output, track_id='car-A')
    resumed.playback.index = 4
    resumed.delete()
    resumed.save()
    assert not json.loads(output.read_text())['tracks'][0]['observations']


def test_annotations_require_valid_schema_hash_and_frame_coordinates(tmp_path):
    from vehicle_metrology.video import run_video, sha256_file
    video = synthetic_video(tmp_path / 'SYNTHETIC.avi')
    sidecar = tmp_path / 'observations.json'
    observation = {'frame': 4, 'rear_contact_uv': [10, 60], 'front_contact_uv': [30, 60],
                   'rear_uv': [8, 40], 'front_uv': [35, 40]}
    data = {'schema_version': 1, 'video_sha256': sha256_file(video),
            'tracks': [{'track_id': 'manual-1', 'observations': [observation]}]}
    sidecar.write_text(json.dumps(data))
    result = run_video(video, output_dir=tmp_path / 'run', headless=True, observations=sidecar)
    assert [t['track_id'] for t in result['tracks']] == ['manual-1']
    assert result['tracks'][0]['length_m'] is None
    assert result['tracks'][0]['reasons'] == ['missing_calibration']
    assert result['artifacts']['observations_sha256'] == sha256_file(sidecar)
    for change, message in [({'video_sha256': '0' * 64}, 'hash'), ({'schema_version': 2}, 'schema')]:
        sidecar.write_text(json.dumps({**data, **change}))
        with pytest.raises(ValueError, match=message):
            run_video(video, output_dir=tmp_path / 'bad', headless=True, observations=sidecar)
    for change in [{'frame': 18}, {'frame': 4.5}, {'front_uv': [160, 40]}, {'rear_uv': [float('nan'), 4]}]:
        data['tracks'][0]['observations'] = [{**observation, **change}]
        sidecar.write_text(json.dumps(data))
        with pytest.raises(ValueError):
            run_video(video, output_dir=tmp_path / 'bad', headless=True, observations=sidecar)


def test_output_paths_never_overwrite_input_artifacts(tmp_path):
    from vehicle_metrology.video import run_video, sha256_file
    video = synthetic_video(tmp_path/'SYNTHETIC.avi')
    before = sha256_file(video)
    with pytest.raises(ValueError,match='overwrite'):
        run_video(video,headless=True,output_video=video,output_dir=tmp_path/'run')
    assert sha256_file(video) == before
    calibration = tmp_path/'results.json'
    calibration.write_text('{}')
    with pytest.raises(ValueError,match='overwrite'):
        run_video(video,headless=True,calibration=calibration,output_dir=tmp_path)
