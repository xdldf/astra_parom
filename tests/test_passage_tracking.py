"""Passage identity must outlive inference delays and partial exit detections."""
import pytest

from web_app.passage_tracking import associate, update_tracks, capture_ready, station_profile
from web_app.workbench import Profile


def detection(x, width=100, y=150):
    return dict(bbox=[x,y,width,75])


def test_latency_jump_preserves_identity_without_overlap():
    track=dict(id='car',box=[50,100,100,60],stamp=1,sent=True)
    d=dict(bbox=[200,102,100,62])
    assert associate([track],[d],2)=={0:track}
    assert not associate([track],[d],4)


def test_ambiguous_neighbours_and_other_lanes_are_not_associated():
    track=dict(id='car',box=[50,100,100,60],stamp=1)
    detections=[dict(bbox=[200,102,100,62]),dict(bbox=[210,104,100,60])]
    assert not associate([track],detections,2)
    other=dict(id='other',box=[60,102,100,60],stamp=1)
    assert not associate([track,other],detections[:1],2)
    assert not associate([track],[dict(bbox=[200,200,100,60])],2)


def test_latency_is_not_a_new_passage_and_short_detection_gap_keeps_identity():
    tracks,rows=update_tracks([], [detection(270)], 0)
    car=rows[0]
    assert not capture_ready(car,detection(270))
    tracks,rows=update_tracks(tracks,[detection(250)],3)
    assert rows[0] is car and capture_ready(car,detection(250))
    car['sent']=True
    tracks,_=update_tracks(tracks,[],6)
    tracks,rows=update_tracks(tracks,[detection(248)],7)
    assert rows[0] is car and not capture_ready(car,detection(248))
    tracks,rows=update_tracks(tracks,[detection(245)],12)
    assert rows[0] is car and not capture_ready(car,detection(245))


def test_rear_fragment_keeps_counted_identity_even_below_overlap_threshold():
    tracks,rows=update_tracks([], [detection(220,200)],0)
    car=rows[0];car['sent']=True
    tracks,rows=update_tracks(tracks,[detection(295,25)],.5)
    assert rows[0] is car
    assert not capture_ready(car,detection(295,25))


def test_full_car_and_nested_fragment_share_counted_identity():
    tracks,rows=update_tracks([], [detection(200,200)],0)
    car=rows[0];car['sent']=True
    tracks,rows=update_tracks(tracks,[detection(285,30),detection(180,200)],.5)
    assert rows[0] is car and rows[1] is car
    assert len(tracks)==1
    assert car['box']==[180,150,200,75]


def test_shrinking_unclaimed_box_at_line_cannot_be_measured():
    tracks,_=update_tracks([], [detection(200,200)],0)
    tracks,rows=update_tracks(tracks,[detection(280,40)],.2)
    assert not capture_ready(rows[0],detection(280,40))


def test_following_car_has_independent_capture_without_time_cooldown():
    tracks,rows=update_tracks([], [detection(250),detection(390)],0)
    leading,following=rows;leading['sent']=True
    tracks,rows=update_tracks(tracks,[detection(170),detection(310)],.3)
    assert rows==[leading,following]
    tracks,rows=update_tracks(tracks,[detection(100),detection(250)],.6)
    assert rows==[leading,following]
    assert not capture_ready(leading,detection(100))
    assert capture_ready(following,detection(250))


def test_confirmed_absence_allows_a_later_car():
    tracks,rows=update_tracks([], [detection(250)],0)
    car=rows[0];car['sent']=True
    tracks,_=update_tracks(tracks,[],.1)
    tracks,rows=update_tracks(tracks,[detection(270)],3)
    assert rows[0]['id']!=car['id'] and not rows[0]['sent']


@pytest.mark.parametrize('line',[None,280])
def test_station_has_a_gate_even_if_old_calibration_disabled_it(line):
    profile=Profile(image_size=(600,500),measurement_line_x=line)
    assert station_profile(profile).measurement_line_x==(300 if line is None else line)
    assert profile.measurement_line_x==line
