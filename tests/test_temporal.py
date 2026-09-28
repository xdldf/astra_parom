import numpy as np
import pytest

from vehicle_metrology.bbox_scale import fit_scale
from vehicle_metrology.temporal import measure_passage, apply_passage


POLYGON = [(10,100),(590,100),(590,450),(10,450)]
SCALE = fit_scale(POLYGON, [dict(bbox=[100,200,200,100],length_m=4)])


def passage(errors=None, offsets=None):
    errors = errors if errors is not None else [-.02,.01,.03,-.01,0.,-.03,.02]
    offsets = offsets if offsets is not None else [24,16,8,0,-8,-16,-24]
    rows=[]
    for i,(error,offset) in enumerate(zip(errors,offsets)):
        width=(4 + .001*offset + error)*50
        rows.append(dict(frame=i,detections=[dict(bbox=[300+offset-width/2,200,width,100])]))
    return rows


def measure(rows):
    return measure_passage(rows,[200,200,200,100],POLYGON,SCALE,(600,500),line_x=300)


def test_local_fit_recovers_length_without_changing_original_line_gate():
    result=measure(passage())
    assert result['length_m']==pytest.approx(4,abs=.025)
    assert result['status']=='temporal_consistent'
    assert result['accuracy_validated'] is False
    assert len(result['samples'])==7
    assert result['diagnostics']['interleaved_difference_m'] < .1
    assert measure(passage(offsets=[25,24,23,22,21,20,19]))['length_m'] is None


def test_bad_frame_is_exposed_instead_of_discarded_to_pass_tolerance():
    result=measure(passage(errors=[0,0,0,.4,0,0,0]))
    assert result['length_m'] is None
    assert 'unstable_temporal_length' in result['reasons']
    assert result['diagnostics']['raw_range_m'] > .3
    old=dict(length_m=4.4,status='single_reference',warnings=[])
    saved=apply_passage(old,result)
    assert saved['length_m'] is None and saved['single_frame_length_m']==4.4
    assert old['length_m']==4.4 and old['warnings']==[]


def test_duplicate_polls_missing_frames_and_ambiguous_neighbours():
    rows=passage()
    with pytest.raises(ValueError,match='distinct'):
        measure(rows+[rows[0]])
    assert 'insufficient_temporal_frames' in measure(rows[:3])['reasons']
    rows[3]['detections'].append(dict(bbox=[220,200,200,100]))
    result=measure(rows)
    assert 'ambiguous_vehicle_association' in result['reasons']
    assert result['length_m'] is None


def test_stationary_vehicle_has_no_undefined_slope():
    result=measure(passage(offsets=[2]*7))
    assert np.isfinite(result['length_m'])
    assert result['diagnostics']['slope_m_per_px']==0
