from web_app.passage_tracking import match_track


def test_latency_jump_preserves_identity_without_box_overlap():
    track=dict(id='car',box=[50,100,100,60],stamp=1,sent=True)
    d=dict(bbox=[200,102,100,62])
    assert match_track([track],[d],0,set(),2) is track
    assert match_track([track],[d],0,{'car'},2) is None
    assert match_track([track],[d],0,set(),4) is None


def test_motion_fallback_does_not_merge_ambiguous_neighbours_or_lanes():
    track=dict(id='car',box=[50,100,100,60],stamp=1)
    detections=[dict(bbox=[200,102,100,62]),dict(bbox=[210,104,100,60])]
    assert match_track([track],detections,0,set(),2) is None
    other=dict(id='other',box=[60,102,100,60],stamp=1)
    assert match_track([track,other],detections[:1],0,set(),2) is None
    assert match_track([track],[dict(bbox=[200,200,100,60])],0,set(),2) is None
