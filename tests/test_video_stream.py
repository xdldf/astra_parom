import numpy as np

from web_app import video_stream as stream, workbench as wb


def test_state_retains_intermediate_detections_for_each_browser_cursor(monkeypatch):
    monkeypatch.setattr(wb,'read_frame',lambda *a:np.zeros((500,600,3),np.uint8))
    monkeypatch.setattr(wb,'require_gpu',lambda:0)
    monkeypatch.setitem(wb.media,'clip',dict(kind='video',path='unused.avi',fps=25,frames=1000))
    camera=stream.Camera(stream.Start(media_id='clip',profile=wb.Profile(image_size=(600,500))))
    monkeypatch.setitem(stream.sessions,camera.id,camera)
    for index in (10,12,14):camera.publish_result(dict(detections=[dict(frame=index)]),index)
    a=stream.state(camera.id,0)
    assert a['detector']==dict(model='yolo26n',imgsz=640)
    assert [r['frame'] for r in a['results']]==[10,12,14]
    assert a['result']['frame']==14 and not a['result_gap']
    assert [r['frame'] for r in stream.state(camera.id,1)['results']]==[12,14]
    assert stream.state(camera.id,3)['results']==[]
    # Another viewer has an independent cursor; reading does not drain the buffer.
    assert len(stream.state(camera.id,0)['results'])==3
    for index in range(130):camera.publish_result(dict(detections=[]),20+index)
    a=stream.state(camera.id,0)
    assert len(a['results'])==128 and a['result_gap']
