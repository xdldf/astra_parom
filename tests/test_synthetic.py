import importlib.util
import json
import cv2


def test_generated_video_has_metric_annotations_and_is_decodable(tmp_path):
    assert importlib.util.find_spec('vehicle_metrology.synthetic') is not None, 'Synthetic experiment missing'
    from vehicle_metrology.synthetic import generate
    from vehicle_metrology.geometry import Camera, measure_track
    paths = generate(tmp_path)
    cap = cv2.VideoCapture(str(paths['video']))
    count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        assert frame.shape[:2] == (720,1280)
        count += 1
    cap.release()
    annotations = json.loads(paths['observations'].read_text())
    assert count == 120
    cam = Camera.from_dict(json.loads(paths['calibration'].read_text()))
    for track in annotations['tracks']:
        m = measure_track(cam,track['observations'])
        assert m['status'] == 'accepted_conditional', m
        assert abs(m['length_m']-4.5) < .1
