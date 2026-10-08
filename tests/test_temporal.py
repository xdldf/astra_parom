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
    estimate=apply_passage(old,result,allow_estimate=True)
    assert estimate['length_m']==pytest.approx(4,abs=.03)
    assert estimate['approximate'] and estimate['status']=='temporal_estimate'
    assert 'unstable_temporal_length' in estimate['quality_reasons']
    assert estimate['temporal']['length_m'] is None  # Failed check is not re-labelled as success.


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


def test_temporal_input_can_be_a_single_pass_iterator():
    rows=passage()
    assert measure(iter(reversed(rows)))==measure(rows)


def test_fast_passage_uses_bounded_neighbours_but_still_requires_line_evidence():
    rows=passage(errors=[0]*5,offsets=[36,18,0,-18,-36])
    result=measure(rows)
    assert result['length_m']==pytest.approx(4,abs=.01)
    assert len(result['samples'])==5
    assert result['diagnostics']['expanded_for_fast_passage']
    assert result['diagnostics']['temporal_band_px']==40
    # No extrapolation from five observations on the same side of the line.
    assert measure(passage(errors=[0]*5,offsets=[39,35,30,25,20]))['length_m'] is None
    assert measure(passage(errors=[0]*5,offsets=[80,40,0,-40,-80]))['length_m'] is None


def test_extended_window_keeps_bad_frames_visible():
    result=measure(passage(errors=[0,0,.4,0,0],offsets=[36,18,0,-18,-36]))
    assert result['length_m'] is None
    assert 'unstable_temporal_length' in result['reasons']


@pytest.mark.parametrize('offset',[35,70])
def test_ambiguity_outside_used_band_does_not_erase_clean_line_measurements(offset):
    rows=passage()
    box=[200+offset,200,200,100]
    rows.append(dict(frame=10,detections=[dict(bbox=box),dict(bbox=[box[0]+2,*box[1:]])]))
    result=measure(rows)
    assert result['length_m']==measure(passage())['length_m']
    assert result['diagnostics']['used_frames']==7
    assert result['diagnostics']['excluded_frames'][-1]['frame']==10


@pytest.mark.parametrize('motion', [-1,0,1])
def test_isolated_roof_proposal_needs_agreement_with_both_adjacent_frames(motion):
    rows=passage(errors=[0]*7,offsets=[motion*v for v in [24,16,8,0,-8,-16,-24]])
    x,y,w,h=rows[3]['detections'][0]['bbox']
    rows[3]['detections'].append(dict(bbox=[x,y-60,w,h+60]))
    result=measure(rows)
    assert result['length_m']==pytest.approx(4,abs=.01)
    assert len(result['samples'])==7
    resolved=result['diagnostics']['resolved_proposals']
    assert len(resolved)==1 and resolved[0]['frame']==3
    assert resolved[0]['neighbour_frames']==[2,4]
    assert resolved[0]['accepted_prediction_iou']>=.9
    assert resolved[0]['rejected_prediction_iou']<=.75


@pytest.mark.parametrize('case', ['missing_neighbour','ambiguous_neighbour','size_jump','different_sides','different_bottom','third_proposal'])
def test_roof_resolution_does_not_relax_uncertain_identity(case):
    rows=passage(errors=[0]*7)
    x,y,w,h=rows[3]['detections'][0]['bbox']
    roof=dict(bbox=[x,y-60,w,h+60]);rows[3]['detections'].append(roof)
    if case=='missing_neighbour':
        rows.pop(2)
    elif case=='ambiguous_neighbour':
        rows[2]['detections'].append(dict(bbox=[x,y-60,w,h+60]))
    elif case=='size_jump':
        rows[2]['detections'][0]['bbox'][3]*=1.2
    elif case=='different_sides':
        roof['bbox'][0]-=.03*w
    elif case=='different_bottom':
        roof['bbox'][3]+=.03*h
    else:
        rows[3]['detections'].append(dict(bbox=[x-30,y,w,h]))
    result=measure(rows)
    assert result['length_m'] is None
    assert 'ambiguous_vehicle_association' in result['reasons']


def test_actual_loaded_ipsum_has_one_trajectory_consistent_body_proposal():
    import json
    from pathlib import Path
    from vehicle_metrology.detection import box_iou
    from vehicle_metrology.temporal import _resolve_roof_proposal
    data=json.loads((Path(__file__).parent/'fixtures/roof_proposals.json').read_text())
    matches={f['frame']:sorted([(box_iou(data['anchor_bbox'],d['bbox']),d) for d in f['detections']],
                               key=lambda item:item[0],reverse=True) for f in data['frames']}
    box,evidence=_resolve_roof_proposal(26382,matches)
    assert box==data['frames'][1]['detections'][0]['bbox']
    assert evidence['accepted_prediction_iou']==pytest.approx(.9703279311905165)
    assert evidence['rejected_prediction_iou']==pytest.approx(.6902930455810292)
