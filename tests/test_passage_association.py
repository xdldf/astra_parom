from copy import deepcopy
import json
from pathlib import Path

import pytest

from vehicle_metrology.passage_association import associate_view


def detection(x, label='car', confidence=.9):
    return dict(bbox=[x, 200, 200, 80], label=label, confidence=confidence)


@pytest.mark.parametrize('velocity', [-10, 0, 10])
def test_continuous_car_can_move_either_way_or_wait(velocity):
    frames = {i:[detection(400+velocity*i)] for i in range(10)}
    anchor = dict(frame=4, bbox=frames[4][0]['bbox'])
    for end in (0, 9):
        candidate = dict(frame=end, bbox=frames[end][0]['bbox'])
        result = associate_view(anchor, candidate, frames, (1600,900))
        assert result['status'] == 'accepted'
        assert len(result['forward']['trace']) == abs(end-4)+1


@pytest.mark.parametrize('failure', ['missing', 'lost', 'competitor', 'switch', 'clipping'])
def test_uncertain_identity_rejects_extra_view(failure):
    frames = {i:[detection(400+10*i)] for i in range(10)}
    anchor = dict(frame=0, bbox=frames[0][0]['bbox'])
    candidate = dict(frame=9, bbox=frames[9][0]['bbox'])
    if failure == 'missing':
        del frames[5]
    elif failure == 'lost':
        frames[5] = []
    elif failure == 'competitor':
        frames[5].append(detection(490, confidence=.8))
    elif failure == 'switch':
        frames[9].append(detection(900))
        candidate['bbox'] = frames[9][1]['bbox']
    else:
        frames[5][0]['bbox'][0] = 1
    assert associate_view(anchor, candidate, frames, (1600,900))['status'] == 'rejected'


def test_cross_class_duplicate_is_not_a_second_physical_car():
    frames = {i:[detection(400+10*i), detection(401+10*i, 'truck', .8)] for i in range(10)}
    anchor = dict(frame=0,bbox=frames[0][0]['bbox'])
    candidate = dict(frame=9,bbox=frames[9][0]['bbox'])
    assert associate_view(anchor,candidate,frames,(1600,900))['status'] == 'accepted'


def test_actual_sparse_track_switch_to_following_pickup_is_rejected():
    data = json.loads((Path(__file__).parent/'fixtures/passage_switch.json').read_text())
    frames = {int(k):v for k,v in data['frames'].items()}
    result = associate_view(data['anchor'],data['candidate'],frames,data['image_size'])
    assert result['status'] == 'rejected'
    assert result['forward']['reason'] == 'clipped_vehicle'
    assert result['backward']['reason'] == 'discontinuous_motion_or_size'
    assert result['forward']['frame'] != result['backward']['frame']


def test_benchmark_rejects_unaudited_views_and_modified_decisions(tmp_path):
    from scripts.audit_passage_association import DETECTOR_SHA256
    from scripts.benchmark_wheel_recovery import checked_associations, ROOT
    from vehicle_metrology.wheels import file_digest
    from types import SimpleNamespace

    def write(name, value):
        path = tmp_path/name
        path.write_text(json.dumps(value))
        return path

    frames = {i:dict(detections=[detection(400+10*i)],pixels_sha256=str(i)) for i in range(3)}
    anchor = dict(frame=0,bbox=frames[0]['detections'][0]['bbox'])
    candidate = dict(frame=2,bbox=frames[2]['detections'][0]['bbox'])
    comparison, profile = write('comparison.json',{}), write('profile.json',{})
    extraction = write('extraction.json',dict(rows={'passage':dict(selected_frames=[anchor,candidate],samples=[
        dict(frame=i,inference_pixels_sha256=str(i)) for i in (0,2)])}))
    signature = dict(extraction_sha256=file_digest(extraction),comparison_sha256=file_digest(comparison),
        profile_sha256=file_digest(profile),detector_sha256=DETECTOR_SHA256,imgsz=640,confidence=.3,
        image_space='lens_corrected_full_frame', device='cpu',
        association_implementation_sha256=file_digest(ROOT/'vehicle_metrology/passage_association.py'),
        detector_implementation_sha256=file_digest(ROOT/'vehicle_metrology/detection.py'),
        script_sha256=file_digest(ROOT/'scripts/audit_passage_association.py'))
    cache = write('detections.json',dict(signature=signature,id='passage',video='video.mp4',frames=frames))
    association = associate_view(anchor,candidate,{k:v['detections'] for k,v in frames.items()},(1600,900))
    checks = [dict(frame=0,expected_box=anchor['bbox'],status='anchor'),
              dict(frame=2,expected_box=candidate['bbox'],**association)]
    audit = dict(**signature,rows={'passage':dict(anchor=anchor,checks=checks,
        detections_file=cache.name,detections_sha256=file_digest(cache))})
    path = write('audit.json',audit)
    args = (path,extraction,comparison,profile,[dict(id='passage',video='video.mp4',**anchor)],
            SimpleNamespace(image_size=(1600,900)))
    assert checked_associations(*args)['passage'][2]['status'] == 'accepted'
    changed = deepcopy(audit)
    changed['rows']['passage']['checks'][1]['status'] = 'rejected'
    write('audit.json',changed)
    with pytest.raises(ValueError,match='do not reproduce'):
        checked_associations(*args)
    changed['rows'] = {}
    write('audit.json',changed)
    with pytest.raises(ValueError,match='incomplete'):
        checked_associations(*args)
