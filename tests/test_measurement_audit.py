import csv
import json

import cv2
import numpy as np
import pytest

from scripts import audit_measurements as audit
from web_app.workbench import Profile


def test_audit_preserves_rejected_samples_and_leaves_truth_blank(tmp_path, monkeypatch):
    path=tmp_path/'video.avi'
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),2,(600,500))
    assert writer.isOpened()
    for _ in range(4):
        writer.write(np.zeros((500,600,3),np.uint8))
    writer.release()
    profile=Profile(image_size=(600,500),polygon=[(20,200),(580,200),(580,450),(20,450)],
                    references=[dict(bbox=[100,150,100,75],length_m=5),
                                dict(bbox=[100,300,200,125],length_m=5)], measurement_line_x=150)
    def detect(*args,**kwargs):
        return [dict(bbox=[100,150,100,75],label='car',confidence=.9),
                dict(bbox=[300,150,100,75],label='car',confidence=.8)]
    monkeypatch.setattr(audit,'predict_vehicle_boxes',detect)
    output=tmp_path/'report'
    summary, samples=audit.audit([path],profile,None,output,device='cpu',step_seconds=.5,compare_imgsz=1280)
    assert summary['sampled_frames']==4
    assert summary['detection_samples']==8
    assert summary['measured_samples']==4
    assert summary['statuses']['waiting_for_line']==4
    assert summary['accuracy_status']=='unvalidated_no_independent_ground_truth'
    assert summary['resolution_comparison']['median_abs_width_delta_px']==0
    assert len({s['track_id'] for s in samples})==8
    assert json.loads((output/'summary.json').read_text())['videos'][0]['sampled_frames']==[0,1,2,3]
    with (output/'ground_truth_template.csv').open() as stream:
        truth=list(csv.DictReader(stream))
    assert len(truth)==4
    assert all(r['length_m']=='' and r['vehicle_id']=='' for r in truth)
    # Do not destroy independent annotations on rerun.
    (output/'ground_truth_template.csv').write_text('keep my annotations')
    audit.audit([path],profile,None,output,device='cpu',step_seconds=1)
    assert (output/'ground_truth_template.csv').read_text()=='keep my annotations'


def test_audit_rejects_invalid_sample_schedule(tmp_path):
    for step in (0,-1,float('nan'),float('inf')):
        with pytest.raises(ValueError,match='step_seconds'):
            audit.audit([],None,None,tmp_path,device='cpu',step_seconds=step)
