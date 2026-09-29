import time

import cv2
import numpy as np
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from web_app import workbench as wb, station, ip_cameras as ip
from web_app.temporal_capture import video_frames, crossing_match


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(station, 'DATA', tmp_path)
    monkeypatch.setattr(station, 'DB', tmp_path/'station.sqlite3')
    profile = wb.Profile(image_size=(600,500), polygon=[(20,200),(580,200),(580,450),(20,450)],
                         references=[dict(bbox=[250,150,100,75],length_m=5)],
                         measurement_line_x=300, detector_model='yolo26m',measurement_mode='strict')
    return tmp_path, profile


def test_saved_video_uses_original_neighbours_and_preserves_anchor(setup, monkeypatch):
    path, profile = setup
    video = path/'passage.avi'
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'),25,(600,500))
    assert writer.isOpened()
    for i in range(50):
        writer.write(np.full((500,600,3),i,np.uint8))
    writer.release()
    item = dict(path=video,kind='video',fps=25,frames=50)
    monkeypatch.setitem(wb.media,'passage',item)
    seen=[]
    def detect(image, confidence, **options):
        i=round(float(image.mean()))
        seen.append((i,options))
        width=100+.2*np.sin(i)
        return [dict(bbox=[300+2*(20-i)-width/2,150,width,75],label='car')]
    monkeypatch.setattr(wb,'detect_vehicles',detect)
    payload=station.Capture(media_id='passage',profile=profile,frame=20,bbox=[250,150,100,75],
                            source='yolo26m',temporal=True)
    record=station.capture(payload)
    evidence=record['source']['measurement']
    assert record['length_m']==pytest.approx(5,abs=.015)
    assert record['source']['frame']==20 and record['source']['detector']=='yolo26m'
    assert evidence['single_frame_length_m']==pytest.approx(5)
    assert evidence['temporal']['accuracy_validated'] is False
    assert evidence['temporal']['anchor_frame']==20
    indices=[s['frame'] for s in evidence['temporal']['samples']]
    assert min(indices)<20<max(indices) and len(indices)==len(set(indices))>=5
    assert all(o==dict(detector_model='yolo26m',imgsz=640) for _,o in seen)
    assert (path/record['side_photo']).is_file()
    # High-rate recordings still have bounded inference work, including anchor.
    high_rate=dict(item,fps=120)
    frames=list(video_frames(high_rate,20))
    assert len(frames)<=31 and 20 in [i for i,_ in frames]


@pytest.mark.parametrize('count,expected',[(7,5.),(3,None)])
def test_ip_capture_requires_enough_actual_frames(setup,monkeypatch,count,expected):
    _,profile=setup
    camera=ip.Station(ip.Settings(),profile)
    jpeg=cv2.imencode('.jpg',np.zeros((500,600,3),np.uint8))[1].tobytes()
    stamp=time.monotonic()-.45
    packets=[ip.Packet(i,stamp+.04*i,jpeg) for i in range(count)]
    camera.side.packets.extend(packets)
    monkeypatch.setattr(wb,'detect_vehicles',lambda *a,**k:[dict(bbox=[250,150,100,75],label='car')])
    record=camera.capture((packets[count//2],None,None),dict(bbox=[250,150,100,75],label='car',length_m=99),
                          'vehicle',temporal=True,epoch=camera.side.epoch)
    assert record['length_m']==expected
    evidence=record['source']['measurement']['temporal']
    assert len(evidence['samples'])==count
    if expected is None:
        assert evidence['reasons']==['insufficient_temporal_frames']
        assert record['status']=='Требует проверки'
        from web_app.main import app
        row=TestClient(app).get('/api/station/vehicles').json()['rows'][0]
        assert row['measurement_reason']=='insufficient_temporal_frames'
        assert 'source' not in row  # Queue diagnostics do not send entire calibration/images.


@pytest.mark.parametrize('count',[3,7])
def test_working_estimate_mode_preserves_numeric_length_and_failed_checks(setup,monkeypatch,count):
    _,profile=setup
    profile=profile.model_copy(update={'measurement_mode':'estimate'})
    camera=ip.Station(ip.Settings(),profile);now=time.monotonic()
    jpeg=cv2.imencode('.jpg',np.zeros((500,600,3),np.uint8))[1].tobytes()
    packets=[ip.Packet(i,now-.3+.04*i,jpeg) for i in range(count)]
    camera.side.packets.extend(packets)
    # Entire sequence outside the supported depth: it must be an estimate,
    # even if the actual detector boxes are perfectly consistent.
    box=[250,300,100,75]
    monkeypatch.setattr(wb,'detect_vehicles',lambda *a,**k:[dict(bbox=box,label='car')])
    record=camera.capture((packets[count//2],None,None),dict(bbox=box,label='car'),'car',temporal=True)
    measured=record['source']['measurement']
    assert record['length_m']==pytest.approx(5)
    assert record['status']=='Требует проверки'
    assert measured['approximate'] and measured['status']=='temporal_estimate'
    assert 'outside_calibration' in measured['quality_reasons']
    assert measured['temporal']['accuracy_validated'] is False
    assert len(measured['temporal']['samples'])==count
    if count==3:assert 'insufficient_temporal_frames' in measured['quality_reasons']
    from web_app.main import app
    row=TestClient(app).get('/api/station/vehicles').json()['rows'][0]
    assert row['length_m']==pytest.approx(5) and row['measurement_approximate']


def test_ip_reconnect_cannot_mix_passages(setup,monkeypatch):
    _,profile=setup
    camera=ip.Station(ip.Settings(),profile)
    packet=ip.Packet(1,time.monotonic(),cv2.imencode('.jpg',np.zeros((500,600,3),np.uint8))[1].tobytes())
    camera.side.packets.append(packet)
    def detect(*a,**k):
        camera.side.epoch+=1
        return [dict(bbox=[250,150,100,75])]
    monkeypatch.setattr(wb,'detect_vehicles',detect)
    with pytest.raises(HTTPException,match='переподключением'):
        camera.capture((packet,None,None),dict(bbox=[250,150,100,75],label='car'),'vehicle',temporal=True)
    assert not camera.saved


def test_ip_capture_freezes_neighbours_before_anchor_render(setup,monkeypatch):
    _,profile=setup
    camera=ip.Station(ip.Settings(),profile)
    jpeg=cv2.imencode('.jpg',np.zeros((500,600,3),np.uint8))[1].tobytes()
    now=time.monotonic()
    packets=[ip.Packet(i,now-.3+.04*i,jpeg) for i in range(7)]
    camera.side.packets.extend(packets)
    original=wb.render_raw
    def render(*args,**kwargs):
        # The receiver keeps moving while the detector/rectification is busy.
        camera.side.packets.clear()
        camera.side.packets.append(ip.Packet(99,now,jpeg))
        return original(*args,**kwargs)
    monkeypatch.setattr(wb,'render_raw',render)
    monkeypatch.setattr(wb,'detect_vehicles',lambda *a,**k:[dict(bbox=[250,150,100,75])])
    record=camera.capture((packets[3],None,None),dict(bbox=[250,150,100,75],label='car'),
                          'vehicle',temporal=True)
    assert record['length_m']==pytest.approx(5)
    evidence=record['source']['measurement']['temporal']
    assert [s['frame'] for s in evidence['samples']]==list(range(7))
    assert evidence['diagnostics']['buffered_frames']==7


def test_pending_ip_passage_keeps_past_and_future_after_rolling_eviction(setup,monkeypatch):
    _,profile=setup
    camera=ip.Station(ip.Settings(),profile)
    jpeg=cv2.imencode('.jpg',np.zeros((500,600,3),np.uint8))[1].tobytes()
    now=time.monotonic();start=now-.8
    packets=[ip.Packet(i,start+.04*i,jpeg) for i in range(23)]
    # A snapshot taken before slow inference contains the pre-line frames.
    for p in packets[:12]:camera.side.append(p)
    earlier=camera.side.snapshot()
    with camera.side.lock:camera.side.packets.clear();camera.side.size=0
    camera.side.append(packets[11])
    window=camera.side.retain_passage(packets[11],camera.side.epoch,earlier)
    # Future frames are collected by the receiver even if infer is occupied.
    for p in packets[12:]:camera.side.append(p)
    for i in range(100):camera.side.append(ip.Packet(23+i,now+.12+.01*i,jpeg))
    assert camera.side.snapshot()[0].seq>packets[-1].seq
    monkeypatch.setattr(wb,'detect_vehicles',lambda *a,**k:[dict(bbox=[250,150,100,75])])
    track=dict(id='vehicle',sent=False,pending=((packets[11],None,None),
        dict(bbox=[250,150,100,75],label='car'),camera.side.epoch),passage_frames=window)
    camera.tracks=[track]
    camera.flush_pending(start+1.1)
    record=camera.saved['vehicle']
    assert record['length_m']==pytest.approx(5)
    assert track['sent'] and 'passage_frames' not in track
    evidence=record['source']['measurement']['temporal']
    assert [s['frame'] for s in evidence['samples']]==list(range(23))
    assert evidence['diagnostics']['pinned_passage'] is True
    assert not camera.side.windows
    # A reconnect invalidates even the retained originals.
    camera.side.epoch+=1
    with pytest.raises(HTTPException,match='переподключилась'):
        camera.capture((packets[11],None,None),dict(bbox=[250,150,100,75],label='car'),
                       'new',temporal=True,epoch=camera.side.epoch,passage_frames=window)


def test_passage_retention_is_bounded_and_always_keeps_anchor(setup):
    _,profile=setup
    camera=ip.Station(ip.Settings(),profile)
    anchor=ip.Packet(50,10,b'jpeg')
    window=camera.side.retain_passage(anchor,0)
    for i in range(100):camera.side.append(ip.Packet(100+i,9.55+.009*i,b'jpeg'))
    assert len(window.snapshot())==31
    assert anchor in window.snapshot()
    assert len({p.seq for p in window.snapshot()})==31
    assert camera.side.size<=48*1024*1024
    for i in range(3):camera.side.retain_passage(ip.Packet(i,11,b'jpeg'),0)
    with pytest.raises(HTTPException,match='одновременных'):
        camera.side.retain_passage(ip.Packet(4,11,b'jpeg'),0)


def test_pending_capture_waits_for_post_line_frames(setup,monkeypatch):
    _,profile=setup
    camera=ip.Station(ip.Settings(),profile)
    packet=ip.Packet(1,10,b'')
    pending=((packet,None,None),dict(bbox=[250,150,100,75]),camera.side.epoch)
    track=dict(id='vehicle',sent=False,pending=pending)
    camera.tracks=[track]
    calls=[]
    monkeypatch.setattr(camera,'capture',lambda *a,**k:calls.append((a,k)))
    camera.flush_pending(10.2)
    assert not calls
    camera.flush_pending(10.5)
    assert len(calls)==1 and calls[0][0][0][0] is packet
    assert calls[0][1]['temporal'] is True and track['sent']
    camera.flush_pending(10.6)
    assert len(calls)==1
    track.update(sent=False,pending=pending)
    camera.flush_pending(14)
    assert len(calls)==1 and not track.get('pending')


def test_uncalibrated_crossing_is_reviewed_without_relaxing_line_gate(setup,monkeypatch):
    _,profile=setup
    camera=ip.Station(ip.Settings(),profile)
    packet=ip.Packet(1,time.monotonic(),cv2.imencode('.jpg',np.zeros((500,600,3),np.uint8))[1].tobytes())
    camera.side.packets.append(packet)
    monkeypatch.setattr(wb,'detect_vehicles',lambda *a,**k:[dict(bbox=[250,325,100,75])])
    record=camera.capture((packet,None,None),dict(bbox=[250,325,100,75],label='truck'),'truck',temporal=True)
    assert record['length_m'] is None and record['side_photo']
    assert 'anchor_outside_calibration' in record['source']['measurement']['temporal']['reasons']
    assert not record['source']['measurement']['temporal']['samples']
    assert record['source']['measurement']['single_frame_status']=='outside_calibration'
    with pytest.raises(HTTPException):
        camera.capture((packet,None,None),dict(bbox=[350,325,100,75],label='truck'),'off-line',temporal=True)


@pytest.mark.parametrize('source',['ip','video'])
def test_outside_anchor_uses_only_calibrated_neighbours_at_the_line(setup,monkeypatch,source):
    path,profile=setup
    anchor=[250,164,100,75]  # Bottom just beyond local reference support.
    raw=[np.full((500,600,3),i*20,np.uint8) for i in range(7)]
    def detect(image,*args,**kwargs):
        i=round(float(image.mean())/20)
        return [dict(bbox=anchor if i==3 else [250+4*(i-3),150,100,75],label='car')]
    monkeypatch.setattr(wb,'detect_vehicles',detect)
    if source=='ip':
        camera=ip.Station(ip.Settings(),profile);now=time.monotonic()
        packets=[ip.Packet(i,now-.24+.04*i,cv2.imencode('.jpg',image)[1].tobytes()) for i,image in enumerate(raw)]
        camera.side.packets.extend(packets)
        record=camera.capture((packets[3],None,None),dict(bbox=anchor,label='car'),'car',temporal=True)
    else:
        video=path/'depth.avi';writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'MJPG'),25,(600,500))
        assert writer.isOpened()
        for image in raw:writer.write(image)
        writer.release()
        monkeypatch.setitem(wb.media,'depth',dict(path=video,kind='video',fps=25,frames=7))
        record=station.capture(station.Capture(media_id='depth',profile=profile,frame=3,bbox=anchor,temporal=True))
    evidence=record['source']['measurement']
    assert record['length_m']==pytest.approx(5)
    assert evidence['single_frame_length_m'] is None
    assert evidence['single_frame_status']=='outside_calibration'
    assert {s['frame'] for s in evidence['temporal']['samples']}=={0,1,2,4,5,6}
    assert evidence['temporal']['accuracy_validated'] is False
    assert dict(frame=3,reason='outside_calibration') in evidence['temporal']['diagnostics']['excluded_frames']
    assert record['status']=='Требует проверки'


def test_crossing_search_recovers_real_frame_when_interpolation_misses_line(setup,monkeypatch):
    path,profile=setup
    video=path/'crossing.avi'
    writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'MJPG'),25,(600,500))
    assert writer.isOpened()
    for i in range(50):writer.write(np.full((500,600,3),i,np.uint8))
    writer.release()
    monkeypatch.setitem(wb.media,'crossing',dict(path=video,kind='video',fps=25,frames=50))
    def detect(image,*args,**kwargs):
        i=round(float(image.mean()))
        center=400-(i-10)*100/12 if i<=22 else 300-(i-22)*100/8
        return [dict(bbox=[center-100,150,200,75],label='car')]
    monkeypatch.setattr(wb,'detect_vehicles',detect)
    payload=station.CrossingCapture(media_id='crossing',profile=profile,
        before_frame=10,before_bbox=[300,150,200,75],frame=30,bbox=[100,150,200,75])
    from web_app.main import app
    response=TestClient(app).post('/api/station/capture-crossing',json=payload.model_dump())
    assert response.status_code==200
    recovered=response.json()
    assert recovered['captured']
    assert recovered['predicted_frame']==20
    assert 20<recovered['frame']<=23  # MJPG levels may round by one; predicted frame still misses.
    assert recovered['examined_frames']>1
    record=recovered['record']
    assert record['length_m']==pytest.approx(10)
    assert record['source']['frame']==recovered['frame']
    assert record['source']['measurement']['at_measurement_line']
    assert len(record['source']['measurement']['temporal']['samples'])>=5
    assert record['source']['measurement']['temporal']['accuracy_validated'] is False
    assert station.capture_crossing(payload)['record']['id']==record['id']
    with pytest.raises(HTTPException):
        station.capture_crossing(payload.model_copy(update={'before_frame':31}))


def test_crossing_search_does_not_replace_target_with_neighbour():
    expected=[200,150,200,100]
    target=dict(bbox=expected,length_m=4,at_measurement_line=True,status='depth_calibrated')
    neighbour=dict(target,bbox=[200,350,200,100])
    assert crossing_match([neighbour],expected) is None
    assert crossing_match([target,neighbour],expected) is target
    assert crossing_match([target,dict(target,bbox=[210,150,200,100])],expected) is None
    assert crossing_match([dict(target,at_measurement_line=False)],expected) is None


def test_ip_search_recovers_crossing_beyond_three_nearest_packets(setup,monkeypatch):
    _,profile=setup
    camera=ip.Station(ip.Settings(),profile)
    start=time.monotonic()-.8
    jpeg=cv2.imencode('.jpg',np.zeros((500,600,3),np.uint8))[1].tobytes()
    packets=[ip.Packet(i+1,start+.04*i,jpeg) for i in range(21)]
    camera.side.packets.extend(packets)
    arrivals=iter([packets[0],packets[-1]])
    class Stop:
        count=0
        def wait(self,*args):
            self.count+=1
            return self.count>2
        def is_set(self):return self.count>2
    camera.stop=Stop()
    monkeypatch.setattr(camera,'fresh',lambda side:next(arrivals))
    monkeypatch.setattr(camera,'paired',lambda:None)
    seen=[]
    def render(raw,request,**kwargs):
        from vehicle_metrology.bbox_scale import measure_box
        seen.append(request.frame)
        center={1:360,21:240,8:300}.get(request.frame,360)
        measured=measure_box([center-100,150,200,75],profile.polygon,wb.profile_scale(profile),
            profile.image_size,profile.measurement_line_x,profile.line_tolerance_px)
        return dict(detections=[dict(measured,label='car')])
    monkeypatch.setattr(wb,'render_raw',render)
    camera.infer()
    assert len(camera.tracks)==1
    pending=camera.tracks[0]['pending']
    assert pending[0][0].seq==8 and pending[1]['at_measurement_line']
    assert len(seen)>5  # Two live observations plus more than three recovery attempts.
