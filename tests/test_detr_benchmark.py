from scripts.benchmark_detr import coverage_summary, target_measurement_rows
from scripts.compare_vehicle_detectors import summarize
from web_app.workbench import Profile


def test_missing_target_detections_remain_in_coverage_denominator():
    target=dict(bbox=[200,100,200,100],depth=.5,line_offset_px=0)
    ref=[dict(case='car',frame=i,modes={'m640':dict(detections=[target])}) for i in range(2)]
    good=dict(target,confidence=.9,length_m=None)
    unrelated=dict(bbox=[900,100,200,100],confidence=.99,length_m=None)
    rows=[dict(case='car',frame=0,modes={'detr':dict(detections=[good])}),
          dict(case='car',frame=1,modes={'detr':dict(detections=[unrelated])})]
    result=coverage_summary(rows,ref)['results'][0]
    assert result['reference_frames']==2
    assert result['associated_frames']==1
    assert result['ambiguous_frames']==0


def test_empty_model_output_is_not_a_zero_error_measurement():
    profile=Profile(image_size=(1200,900),measurement_line_x=600)
    row=dict(case='car',frame=10,seconds=.4,modes={'empty':dict(seconds=.1,detections=[])})
    result=summarize([row],profile)['results'][0]
    assert result['raw_range_m'] is None
    assert result['temporal']['length_m'] is None
    assert result['temporal']['reasons']==['missing_line_evidence']


def test_neighbour_at_line_cannot_replace_selected_target():
    target = dict(bbox=[200,100,200,100],depth=.5,line_offset_px=80,
                  length_m=4.1,confidence=.9)
    neighbour = dict(bbox=[600,100,200,100],depth=.5,line_offset_px=0,
                     length_m=4.7,confidence=.95)
    references = [dict(case='car',frame=i,modes={'m640':dict(detections=[target])})
                  for i in range(2)]
    rows = [dict(case='car',frame=0,seconds=0,modes={'detr':dict(
                seconds=.1,detections=[target,neighbour])}),
            dict(case='car',frame=1,seconds=.04,modes={'detr':dict(
                seconds=.1,detections=[neighbour])})]
    filtered = target_measurement_rows(rows, references)
    assert filtered[0]['modes']['detr']['detections'] == [target]
    assert filtered[1]['modes']['detr']['detections'] == []
    assert len(rows[0]['modes']['detr']['detections']) == 2
    result = summarize(filtered, Profile(image_size=(1200,900),measurement_line_x=600))
    assert result['results'][0]['temporal']['reasons'] == ['missing_line_evidence']
