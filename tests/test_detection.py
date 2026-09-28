from vehicle_metrology.detection import suppress_duplicate_boxes


def test_end_to_end_duplicate_classes_do_not_create_two_cars():
    detections = [dict(bbox=[10, 50, 500, 250], confidence=.7, label='truck'),
                  dict(bbox=[12, 51, 499, 248], confidence=.9, label='car'),
                  dict(bbox=[400, 50, 500, 250], confidence=.8, label='car')]
    assert suppress_duplicate_boxes(detections) == [detections[1], detections[2]]
    assert len(detections) == 3


def test_empty_and_same_class_duplicates():
    assert suppress_duplicate_boxes([]) == []
    box = dict(bbox=[100, 200, 80, 90], confidence=.8, label='car')
    assert suppress_duplicate_boxes([box, dict(box, confidence=.6)]) == [box]
