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


def test_completed_video_keeps_final_detections_until_browser_closes(tmp_path,monkeypatch):
    import cv2
    video=tmp_path/'last-frame.avi'
    writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'MJPG'),25,(600,500))
    writer.write(np.zeros((500,600,3),np.uint8));writer.release()
    monkeypatch.setattr(wb,'require_gpu',lambda:0)
    monkeypatch.setitem(wb.media,'finished',dict(kind='video',path=video,fps=25,frames=1))
    camera=stream.Camera(stream.Start(media_id='finished',profile=wb.Profile(image_size=(600,500))))
    monkeypatch.setitem(stream.sessions,camera.id,camera)
    camera.publish_result(dict(detections=[dict(bbox=[100,100,100,50])]),0)
    camera.produce()
    state=stream.state(camera.id)
    assert state['ended'] and len(state['results'])==1
    stream.close(camera.id)
    assert camera.id not in stream.sessions


def test_browsers_share_passage_identity_and_concurrent_capture_is_idempotent(tmp_path,monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from fastapi.testclient import TestClient
    from web_app.main import app
    from web_app import station as st
    monkeypatch.setattr(st,'DATA',tmp_path)
    monkeypatch.setattr(st,'DB',tmp_path/'station.sqlite3')
    monkeypatch.setattr(wb,'read_frame',lambda *a:np.zeros((500,600,3),np.uint8))
    monkeypatch.setattr(wb,'require_gpu',lambda:0)
    monkeypatch.setitem(wb.media,'clip',dict(kind='video',path='unused.avi',fps=25,frames=1000))
    profile=wb.Profile(image_size=(600,500),polygon=[(20,200),(580,200),(580,450),(20,450)],
                       references=[dict(bbox=[250,150,100,75],length_m=5)])
    camera=stream.Camera(stream.Start(media_id='clip',profile=profile))
    monkeypatch.setitem(stream.sessions,camera.id,camera)
    for index,x in [(10,280),(12,250),(14,248)]:
        camera.publish_result(dict(detections=[dict(bbox=[x,150,100,75])]),index)
    rows=stream.state(camera.id,0)['results']
    passage=rows[0]['detections'][0]['passage_id']
    assert not rows[0]['detections'][0]['capture_ready']
    assert all(row['detections'][0]['passage_id']==passage for row in rows)
    requests=[dict(media_id='clip',profile=profile.model_dump(),frame=row['frame'],
                   bbox=row['detections'][0]['bbox'],capture_session_id=camera.id)
              for row in rows[1:]]
    client=TestClient(app)
    measured=[];render=wb.render_raw
    def counted_render(*args,**kwargs):
        measured.append(1)
        return render(*args,**kwargs)
    monkeypatch.setattr(wb,'render_raw',counted_render)
    # Different original frames and bboxes, submitted by independent viewers.
    with ThreadPoolExecutor(2) as pool:
        responses=list(pool.map(lambda body:client.post('/api/station/capture',json=body),requests))
    assert all(r.status_code==200 for r in responses),[r.text for r in responses]
    assert responses[0].json()['id']==responses[1].json()['id']
    assert responses[0].json()['source']['passage_id']==passage
    assert len(measured)==1
    with st.connect() as db:assert db.execute('SELECT COUNT(*) FROM vehicles').fetchone()[0]==1
    assert len(list(tmp_path.glob('*.jpg')))==2
    camera.publish_result(dict(detections=[dict(bbox=[295,150,25,75])]),16)
    tail=camera.result['detections'][0]
    assert tail['passage_id']==passage and tail['passage_captured'] and not tail['capture_ready']
    camera.publish_result(dict(detections=[]),20)
    camera.publish_result(dict(detections=[]),100)
    # A delayed browser still resolves the original server passage after exit.
    assert client.post('/api/station/capture',json=requests[0]).json()['id']==responses[0].json()['id']
    assert len(measured)==1
