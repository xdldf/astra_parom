import importlib.util
import numpy as np
import pytest


def test_errors_cover_rejections_positions_and_repeated_vehicles(tmp_path):
    assert importlib.util.find_spec('vehicle_metrology.evaluation') is not None, 'Evaluation not implemented'
    from vehicle_metrology.evaluation import evaluate
    truth = tmp_path/'truth.csv'
    truth.write_text('track_id,vehicle_id,length_m\na,car,4\nb,car,4\nc,truck,8\nd,missed,5\n')
    tracks = [dict(track_id=k,length_m=l,reasons=[] if l else ['occluded'],frames=[dict(frame=0,x_m=x,y_m=2,heading_deg=15,window_length_m=l)]) for k,l,x in [('a',4.1,0),('b',3.8,10),('c',None,0)]]
    report=evaluate(tracks,truth)
    assert report['coverage']['accepted_matched'] == 2
    assert report['coverage']['ground_truth_tracks'] == 4
    assert report['coverage']['missed_track_ids'] == ['d']
    assert report['metrics']['mae_m'] == pytest.approx(.15)
    assert report['metrics']['bias_m'] == pytest.approx(-.05)
    assert report['metrics']['max_abs_error_m'] == pytest.approx(.2)
    assert report['metrics']['mean_relative_abs_error'] == pytest.approx(.15/4)
    assert report['per_vehicle']['car']['count'] == 2
    assert len(report['spatial_bins']) == 2
    assert report['orientation_bins']
    assert report['window_metrics']['count'] == 2
    assert report['coverage']['rejection_reasons']['occluded'] == 1
    truth.write_text('track_id,length_m\na,4\na,5\n')
    with pytest.raises(ValueError,match='duplicate'):
        evaluate(tracks,truth)
