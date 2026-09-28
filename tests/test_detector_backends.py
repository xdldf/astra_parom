from types import SimpleNamespace

import numpy as np
import pytest

from vehicle_metrology.detector_backends import predict_rfdetr_boxes, DetectorBackend


def test_rf_uses_rgb_sparse_coco_ids_and_original_pixel_coordinates():
    frame=np.full((2,3,3),[10,20,30],np.uint8)
    class Model:
        def predict(self, image, **kwargs):
            np.testing.assert_array_equal(image,np.full_like(frame,[30,20,10]))
            assert kwargs==dict(threshold=.3,include_source_image=False)
            return SimpleNamespace(xyxy=np.array([[100,200,600,400],[20,30,50,70],[0,0,8,8]]),
                confidence=np.array([.9,.8,.95]),class_id=np.array([3,8,2]))
    result=predict_rfdetr_boxes(Model(),frame)
    assert [(r['bbox'],r['label']) for r in result]==[([100.,200.,500.,200.],'car'),([20.,30.,30.,40.],'truck')]
    np.testing.assert_array_equal(frame,np.full_like(frame,[10,20,30]))


def test_rf_duplicate_classes_invalid_boxes_and_low_scores_are_handled():
    prediction=SimpleNamespace(xyxy=np.array([[0,0,100,50],[1,1,101,51],[0,0,2,2],
                                              [3,3,1,1],[0,0,np.nan,4],[300,100,350,200]]),
        confidence=np.array([.9,.8,.2,.9,.9,.95]),class_id=np.array([3,8,3,3,3,6]))
    model=SimpleNamespace(predict=lambda *args,**kwargs: prediction)
    result=predict_rfdetr_boxes(model,np.zeros((10,10,3),np.uint8))
    assert len(result)==2
    assert [r['label'] for r in result]==['bus','car']
    assert result[1]['bbox'][0]==0  # Clipping remains observable by metrology.


def test_no_implicit_checkpoint_download(tmp_path):
    with pytest.raises(ValueError,match='existing local'):
        DetectorBackend('rtdetr',tmp_path/'missing.pt')
    checkpoint=tmp_path/'exists.pt';checkpoint.write_bytes(b'not loaded')
    with pytest.raises(ValueError,match='Unknown'):
        DetectorBackend('untrusted',checkpoint)
