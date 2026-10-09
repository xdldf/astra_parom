from scripts.select_video_audit import all_tracks, select_tracks


def track(ident, frame, box, length=0):
    return dict(id=ident, best_frame=frame, start_frame=frame, end_frame=frame,
                image=ident+'.jpg', image_sha256='test', best_detection=dict(
                    label='car', bbox=box, status='waiting_for_line', length_m=length))


def test_edge_fragment_waiting_for_line_does_not_replace_complete_car():
    tracks = [track('fragment', 2, [95, 30, 5, 20]),
              track('complete', 4, [20, 30, 40, 20])]
    rows, slots = select_tracks(tracks, 'video.mp4', 100, 2, (100, 100))
    assert [r['id'] for r in rows] == ['complete']
    assert slots[0]['clipped_candidates'] == ['fragment']
    assert slots[1]['selected_id'] is None


def test_selection_uses_time_not_estimated_length_or_input_order():
    tracks = [track('later', 20, [20, 30, 40, 20], 4.5),
              track('first', 10, [20, 30, 40, 20], None)]
    rows, _ = select_tracks(tracks, 'video.mp4', 100, 1, (100, 100))
    assert [r['id'] for r in rows] == ['first']
    assert 'length_m' not in rows[0]


def test_detector_margin_of_several_pixels_can_still_be_a_truncated_car():
    tracks = [track('edge', 2, [2432.83, 1119.17, 153.99, 197.81]),
              track('body', 4, [954.1, 1118.93, 589.06, 213.02])]
    rows, slots = select_tracks(tracks, 'video.mp4', 100, 1, (2592, 1944))
    assert [r['id'] for r in rows] == ['body']
    assert slots[0]['clipped_candidates'] == ['edge']


def test_all_selection_keeps_fragments_and_class_changes_without_length_filtering():
    fragment = track('fragment', 2, [95, 30, 5, 20], None)
    bus = track('bus', 4, [20, 30, 40, 20], None)
    bus['best_detection']['label'] = 'bus'
    fragment['observations'] = [fragment['best_detection']]
    bus['observations'] = [dict(label='car'), bus['best_detection']]
    rows = all_tracks([bus, fragment], 'video.mp4')
    assert [r['id'] for r in rows] == ['fragment', 'bus']
    assert rows[1]['detector_label'] == 'bus'
    assert rows[1]['observed_labels'] == ['bus', 'car']
    assert all('length_m' not in r for r in rows)
