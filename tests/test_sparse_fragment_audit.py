from scripts.audit_sparse_fragments import replay


def test_dense_tracking_connects_an_entering_fragment_that_sparse_tracking_splits():
    frames = [dict(frame=i, road=[dict(bbox=[100, 40, 5+i, 20], label='car')])
              for i in range(26)]
    dense = replay(frames, fps=25, stride=1)
    sparse = replay(frames, fps=25, stride=25)
    assert len(dense) == 1
    assert [o['frame'] for o in dense[0]['observations']] == list(range(26))
    assert len(sparse) == 2
    assert [t['observations'][0]['frame'] for t in sparse] == [0, 25]


def test_sparse_replay_keeps_original_video_sampling_phase():
    frames = [dict(frame=i, road=[dict(bbox=[100, 40, 20, 20], label='car')])
              for i in range(23, 29)]
    tracks = replay(frames, fps=25, stride=25)
    assert [o['frame'] for o in tracks[0]['observations']] == [25]
