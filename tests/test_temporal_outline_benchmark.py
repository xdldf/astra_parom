from copy import deepcopy
import hashlib
import json

import numpy as np
import pytest

from scripts.benchmark_temporal_outline import aggregate, select_frames
from vehicle_metrology.catalogue_labels import corrected_rows


def test_frame_selection_is_independent_of_catalogue_and_keeps_anchor_and_singletons():
    row = dict(frame=14, bbox=[10,20,100,80], measurement=dict(temporal=dict(
        samples=[dict(frame=i,bbox=[i,20,100,80]) for i in range(20)])))
    selected = select_frames(row)
    assert len(selected) == 5 and len({s['frame'] for s in selected}) == 5
    assert 14 in [s['frame'] for s in selected]
    changed = deepcopy(row)
    changed.update(model_candidate='anything', catalogue_length_range_m=[1,90])
    assert select_frames(changed) == selected
    assert select_frames(dict(frame=14,bbox=row['bbox'])) == [dict(frame=14,bbox=row['bbox'])]


def test_local_aggregation_recovers_anchor_with_one_boundary_outlier():
    offsets = np.array([-30., -15., 0., 15., 30.])
    values = np.array([np.ones(7)+x*.002 for x in offsets])
    values[:,3] = .5+offsets/1000
    values[-1,0] += .5
    samples = [dict(frame=i,status='ok',features=v.tolist()) for i,v in enumerate(values)]
    expected = np.ones(7); expected[3] = .5
    np.testing.assert_allclose(aggregate(samples,2,'local_fit',1000),expected)
    samples[1]['status'] = 'failed'
    with pytest.raises(ValueError,match='All selected'):
        aggregate(samples,2,'median',1000)


def test_catalogue_revision_is_bound_to_exact_audit_and_image(tmp_path):
    comparison = tmp_path/'comparison.json'
    image = tmp_path/'corrected.jpg'
    image.write_bytes(b'full corrected evidence')
    row = dict(id='A',image=image.name,bbox=[1,2,3,4],frame=5,
               model_candidate='old',source_ids=['spacio_early'],catalogue_length_range_m=[4.24,4.275])
    comparison.write_text(json.dumps(dict(rows=[row])))
    change = dict(id='A',previous_model_candidate='old',previous_source_ids=['spacio_early'],
        corrected_image_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
        model_candidate='Wish AE10',source_ids=['wish_ae10'],review_note='Visual body comparison',
        stock_body_uncertain=True,condition_note='Damaged bumper')
    corrections = tmp_path/'corrections.json'
    data = dict(version=1,base_comparison_sha256=hashlib.sha256(comparison.read_bytes()).hexdigest(),
        review_method='Visual',visual_references=[],sources=dict(wish_ae10=dict(length_m=[4.55,4.56])),
        corrections=[change])
    corrections.write_text(json.dumps(data))
    result = corrected_rows([row],comparison,corrections)
    assert len(result)==1 and result[0]['stock_body_uncertain']
    assert result[0]['catalogue_length_range_m']==[4.55,4.56]
    assert result[0]['bbox']==row['bbox'] and row['source_ids']==['spacio_early']
    image.write_bytes(b'different image')
    with pytest.raises(ValueError,match='frame has changed'):
        corrected_rows([row],comparison,corrections)
