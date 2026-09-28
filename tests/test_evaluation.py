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


def test_five_centimetre_acceptance_counts_missed_and_rejected_vehicles(tmp_path):
    from vehicle_metrology.evaluation import evaluate
    truth=tmp_path/'truth.csv'
    truth.write_text('track_id,length_m\na,4\nb,4\nc,4\nd,4\n')
    tracks=[dict(track_id='a',length_m=4.05),dict(track_id='b',length_m=4.051),
            dict(track_id='c',length_m=None,reasons=['occluded'])]
    report=evaluate(tracks,truth)
    assert report['acceptance']['passed_tracks']==1
    assert report['acceptance']['failed_track_ids']==['b']
    assert report['acceptance']['fraction_of_measured_within_tolerance']==.5
    assert report['acceptance']['fraction_of_ground_truth_within_tolerance']==.25
    assert not report['acceptance']['all_ground_truth_tracks_within_tolerance']
    assert evaluate([dict(track_id=x,length_m=4) for x in 'abcd'],truth)['acceptance']['all_ground_truth_tracks_within_tolerance']
    with pytest.raises(ValueError,match='duplicate prediction'):
        evaluate([tracks[0],tracks[0]],truth)
    for value in [float('nan'),float('inf'),0,-1]:
        with pytest.raises(ValueError,match='Prediction'):
            evaluate([dict(track_id='a',length_m=value)],truth)
        with pytest.raises(ValueError,match='tolerance_m'):
            evaluate([],truth,tolerance_m=value)
    truth.write_text('track_id,length_m\n')
    assert not evaluate([],truth)['acceptance']['all_ground_truth_tracks_within_tolerance']
