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
    assert len(calls)==2 and not track.get('pending')
    assert calls[-1][1]['review_fallback'] and track['sent']


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


@pytest.mark.parametrize('source',['ip','video'])
@pytest.mark.parametrize('mode',['strict','estimate'])
def test_slow_truck_waits_for_actual_crossing_and_reuses_original_evidence(setup,monkeypatch,source,mode):
    path,profile=setup
    profile=profile.model_copy(update={'measurement_mode':mode})
    # A 19.33 m synthetic truck enters the 10 px line gate. Its centre only
    # crosses 0.67 s later, beyond the old 0.45 s capture window.
    frames=[np.full((500,600,3),i*2,np.uint8) for i in range(101)]
    def box(i):return [300+8-12*(i-50)/25-193.3,150,386.6,75]
    calls=[]
    def detect(image,*args,**kwargs):
        i=round(float(image.mean())/2);calls.append(i)
        return [dict(bbox=box(i),label='truck')]
    monkeypatch.setattr(wb,'detect_vehicles',detect)
    initial_frames=None
    if source=='video':
        video=path/'slow-truck.avi'
        writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'MJPG'),25,(600,500))
        assert writer.isOpened()
        for image in frames:writer.write(image)
        writer.release()
        item=dict(path=video,kind='video',fps=25,frames=len(frames))
        monkeypatch.setitem(wb.media,'slow',item)
        initial_frames={index for index,_ in video_frames(item,50)}
        record=station.capture(station.Capture(media_id='slow',profile=profile,frame=50,
            bbox=box(50),label='truck',temporal=True))
    else:
        camera=ip.Station(ip.Settings(),profile)
        # Isolated clock: preserve realistic packet freshness while advancing
        # the receiver without waiting two wall-clock seconds in the test.
        clock=[100.]
        monkeypatch.setattr(ip.time,'monotonic',lambda:clock[0])
        packets=[ip.Packet(i,98+i/25,cv2.imencode('.jpg',image)[1].tobytes())
                 for i,image in enumerate(frames)]
        for p in packets[:51]:camera.side.append(p)
        window=camera.side.retain_passage(packets[50],camera.side.epoch)
        track=dict(id='slow',sent=False,pending=((packets[50],None,None),
            dict(bbox=box(50),label='truck'),camera.side.epoch),passage_frames=window)
        camera.tracks=[track]
        for p in packets[51:63]:camera.side.append(p)
        clock[0]=packets[62].stamp
        camera.flush_pending(clock[0])
        assert not camera.saved and not track['sent']
        assert window.passage['reasons']==['line_not_bracketed']
        initial_frames={row['frame'] for row in window.observations}
        initial_calls=len(calls)
        camera.flush_pending(clock[0]+.1)
        assert len(calls)==initial_calls  # Pending polls do not rerun inference.
        for p in packets[63:]:camera.side.append(p)
        clock[0]=packets[-1].stamp
        camera.flush_pending(clock[0])
        assert track['sent'] and not camera.side.windows
        record=camera.saved['slow']
        camera.flush_pending(clock[0]+.1)
        assert len(camera.saved)==1
    evidence=record['source']['measurement']['temporal']
    assert record['source']['frame']==50 and record['source']['bbox']==box(50)
    assert record['length_m']==pytest.approx(19.33,abs=.002)
    assert evidence['status']=='temporal_consistent' and not evidence['reasons']
    assert evidence['accuracy_validated'] is False
    assert evidence['window_seconds']==2
    assert evidence['diagnostics']['extended_time_window'] is True
    assert evidence['diagnostics']['initial_review_reasons']==['line_not_bracketed']
    assert len(calls)==evidence['diagnostics']['requested_frames']<=61
    offsets=[s['line_offset_px'] for s in evidence['samples']]
    assert min(offsets)<0<max(offsets)
    included={s['frame'] for s in evidence['samples']}
    excluded={s['frame'] for s in evidence['diagnostics']['excluded_frames']}
    assert initial_frames <= included|excluded


def test_extended_retention_keeps_both_time_edges_with_count_and_byte_limits():
    from web_app.temporal_capture import MAX_EXTENDED_FRAMES
    anchor=ip.Packet(500,10,b'anchor')
    window=ip.PassageFrames(anchor,0)
    # High receive rate must not crowd out the sparse outer reserve.
    window.extend(ip.Packet(i,8+i*.004,b'jpeg') for i in range(1001))
    packets=window.snapshot(extended=True)
    assert len(packets)<=MAX_EXTENDED_FRAMES and anchor in packets
    assert packets[0].stamp<8.15 and packets[-1].stamp>11.85
    assert len(window.snapshot())<=31
    large=b'x'*(2*1024*1024)
    window=ip.PassageFrames(anchor,0)
    window.extend(ip.Packet(i,8+i*.05,large) for i in range(81))
    assert 46*1024*1024<=sum(len(p.jpeg) for p in window.snapshot(extended=True))<=48*1024*1024
    assert anchor in window.snapshot(extended=True)


@pytest.mark.parametrize('failure',['unstable_temporal_length','ambiguous_vehicle_association','missing_line_evidence','unstable_after_extension'])
def test_failed_checks_remain_visible_without_discarding_original_frames(setup,monkeypatch,failure):
    path,profile=setup
    profile=profile.model_copy(update={'measurement_mode':'estimate'})
    video=path/'failed-check.avi'
    writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'MJPG'),25,(600,500))
    assert writer.isOpened()
    for i in range(101):writer.write(np.full((500,600,3),i*2,np.uint8))
    writer.release()
    monkeypatch.setitem(wb.media,'failed',dict(path=video,kind='video',fps=25,frames=101))
    calls=[]
    def detect(image,*args,**kwargs):
        i=round(float(image.mean())/2);calls.append(i)
        offset=8-.48*(i-50) if failure=='unstable_after_extension' else 2*(i-50)
        if failure=='missing_line_evidence':offset=20
        width=106 if i==50 and failure.startswith('unstable') else 100
        d=dict(bbox=[300+offset-width/2,150,width,75],label='car')
        return [d,d] if failure=='ambiguous_vehicle_association' else [d]
    monkeypatch.setattr(wb,'detect_vehicles',detect)
    record=station.capture(station.Capture(media_id='failed',profile=profile,frame=50,
        bbox=[258 if failure=='unstable_after_extension' else 250,150,100,75],temporal=True))
    temporal=record['source']['measurement']['temporal']
    expected='unstable_temporal_length' if failure=='unstable_after_extension' else failure
    assert expected in temporal['reasons'] and temporal['length_m'] is None
    assert record['status']=='Требует проверки'
    assert temporal['diagnostics']['extended_time_window']==(failure=='unstable_after_extension')
    if failure=='unstable_after_extension':
        assert temporal['diagnostics']['initial_review_reasons']==['line_not_bracketed']
        assert 50 in {s['frame'] for s in temporal['samples']}
        assert temporal['diagnostics']['residual_max_m']>.1
        assert 31<len(calls)<=61
    else:
        indices={s['frame'] for s in temporal['samples']}|{s['frame'] for s in temporal['diagnostics']['excluded_frames']}
        assert len(calls)==len(indices)<=31 and min(indices)>=39 and max(indices)<=61


@pytest.mark.parametrize('mode',['strict','estimate'])
def test_slow_vehicle_that_never_crosses_still_requires_review(setup,monkeypatch,mode):
    _,profile=setup
    profile=profile.model_copy(update={'measurement_mode':mode})
    camera=ip.Station(ip.Settings(),profile)
    clock=[100.];monkeypatch.setattr(ip.time,'monotonic',lambda:clock[0])
    images=[np.full((500,600,3),2*i,np.uint8) for i in range(101)]
    # Approaches then stops 8 px before the centre line. Earlier samples are
    # outside tolerance: the system must not invent a crossing after waiting.
    def detect(image,*args,**kwargs):
        i=round(float(image.mean())/2)
        return [dict(bbox=[250+max(8,8-(i-50)*.5),150,100,75],label='car')]
    monkeypatch.setattr(wb,'detect_vehicles',detect)
    packets=[ip.Packet(i,98+i/25,cv2.imencode('.jpg',raw)[1].tobytes()) for i,raw in enumerate(images)]
    for p in packets[:51]:camera.side.append(p)
    window=camera.side.retain_passage(packets[50],0)
    track=dict(id='stopped',sent=False,pending=((packets[50],None,None),dict(bbox=[258,150,100,75],label='car'),0),passage_frames=window)
    camera.tracks=[track]
    for p in packets[51:63]:camera.side.append(p)
    clock[0]=100.48;camera.flush_pending(clock[0])
    assert not camera.saved
    for p in packets[63:]:camera.side.append(p)
    clock[0]=102;camera.flush_pending(clock[0])
    record=camera.saved['stopped'];evidence=record['source']['measurement']
    assert evidence['temporal']['reasons']==['line_not_bracketed']
    assert record['length_m']==(5 if mode=='estimate' else None)
    assert record['status']=='Требует проверки'
    assert evidence['temporal']['diagnostics']['extended_time_window']
    assert not camera.side.windows


@pytest.mark.parametrize('cancel',['reconnect','automatic_off'])
def test_waiting_slow_passage_is_discarded_when_invalidated(setup,monkeypatch,cancel):
    _,profile=setup
    camera=ip.Station(ip.Settings(),profile)
    anchor=ip.Packet(1,10,b'jpeg')
    window=camera.side.retain_passage(anchor,0)
    window.passage=dict(reasons=['line_not_bracketed'],diagnostics=dict(extended_time_window=False))
    track=dict(id='slow',sent=False,pending=((anchor,None,None),dict(bbox=[258,150,100,75]),0),passage_frames=window)
    camera.tracks=[track]
    if cancel=='reconnect':camera.side.epoch+=1
    if cancel=='automatic_off':camera.cfg.auto_measure=False
    calls=[];monkeypatch.setattr(camera,'capture',lambda *a,**k:calls.append(1))
    camera.flush_pending(15.1 if cancel=='expired' else 11)
    assert not calls and not camera.saved and not track.get('pending')
    assert 'passage_frames' not in track and not camera.side.windows


@pytest.mark.parametrize('mode',['strict','estimate'])
def test_ip_keeps_off_line_observation_even_after_live_buffer_expires(setup,monkeypatch,mode):
    path,profile=setup
    profile=profile.model_copy(update={'measurement_mode':mode})
    camera=ip.Station(ip.Settings(),profile)
    monkeypatch.setattr(ip.time,'monotonic',lambda:30.)
    image=np.full((500,600,3),90,np.uint8)
    packet=ip.Packet(3,10,cv2.imencode('.jpg',image)[1].tobytes())
    detection=dict(bbox=[150,150,100,75],label='car')
    track=dict(id='missed',sent=False,stamp=10,box=detection['bbox'],
               review=((packet,None,None),detection,camera.side.epoch,100))
    camera.tracks=[track]
    camera.flush_reviews(10.5)
    assert not camera.saved
    camera.flush_reviews(30)
    record=camera.saved['missed']
    assert record['length_m'] is None
    assert record['source']['frame']==3 and record['source']['bbox']==detection['bbox']
    assert cv2.imread(str(path/record['full_frame_photo'])).shape==image.shape
    assert track['sent'] and not track.get('review')
    camera.flush_reviews(31)
    assert len(camera.saved)==1


def test_video_crossing_without_any_centred_box_saves_real_endpoint(setup,monkeypatch):
    path,profile=setup
    profile=profile.model_copy(update={'measurement_mode':'estimate'})
    video=path/'jump.avi'
    writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'MJPG'),25,(600,500))
    for i in range(10):writer.write(np.full((500,600,3),i*20,np.uint8))
    writer.release()
    monkeypatch.setitem(wb.media,'jump',dict(path=video,kind='video',fps=25,frames=10))
    monkeypatch.setattr(wb,'detect_vehicles',lambda image,*a,**k:[
        dict(bbox=[180 if image.mean()<100 else 310,150,100,75],label='car')])
    result=station.capture_crossing(station.CrossingCapture(media_id='jump',profile=profile,
        before_frame=0,before_bbox=[180,150,100,75],frame=9,bbox=[310,150,100,75]))
    assert result['captured'] and result['review_fallback'] and result['frame']==9
    record=result['record']
    assert record['length_m'] is None
    assert record['full_frame_photo']
    assert record['source']['measurement']['line_offset_px']==60
    assert not record['source']['measurement']['at_measurement_line']


def test_rectified_border_is_clipped_even_inside_canvas(setup):
    from web_app.temporal_capture import capture_geometry_reasons
    _,profile=setup
    profile=profile.model_copy(update={'lens':wb.Lens(tilt_deg=10), 'measurement_line_x':50})
    assert capture_geometry_reasons(profile,[10,10,80,75])==['clipped']
    assert not capture_geometry_reasons(profile.model_copy(update={'measurement_line_x':300}),[250,150,100,75])


def test_center_recovery_stops_at_lost_or_ambiguous_identity(setup):
    from web_app.temporal_capture import centered_observation
    _,profile=setup
    def detection(x):
        return dict(bbox=[x,150,100,75],label='car',depth=.1,status='waiting_for_line')
    rows=[dict(frame=i,detections=[detection(150+10*i)]) for i in range(11)]
    found=centered_observation(profile,0,rows[0]['detections'][0]['bbox'],rows)
    assert found['frame']==10
    for detections in ([],[detection(200),detection(205)]):
        broken=[dict(row) for row in rows];broken[5]=dict(frame=5,detections=detections)
        assert centered_observation(profile,0,rows[0]['detections'][0]['bbox'],broken) is None
    assert centered_observation(profile,0,rows[0]['detections'][0]['bbox'],rows[:5]+rows[6:]) is None


@pytest.mark.parametrize('anchor',[0,20])
def test_video_review_recovers_actual_center_and_updates_photo_and_source(setup,monkeypatch,anchor):
    path,profile=setup
    profile=profile.model_copy(update={'measurement_mode':'estimate'})
    video=path/'center.avi'
    writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'MJPG'),25,(600,500))
    for i in range(21):writer.write(np.full((500,600,3),i*10,np.uint8))
    writer.release()
    monkeypatch.setitem(wb.media,'center',dict(path=video,kind='video',fps=25,frames=21))
    def detect(image,*args,**kwargs):
        i=round(float(image.mean())/10)
        return [dict(bbox=[150+10*i,150,100,75],label='car')]
    monkeypatch.setattr(wb,'detect_vehicles',detect)
    record=station.capture(station.Capture(media_id='center',profile=profile,frame=anchor,
        bbox=[150+10*anchor,150,100,75],review_fallback=True))
    assert record['length_m']==pytest.approx(5)
    assert record['source']['frame']==10
    assert record['source']['bbox']==[250,150,100,75]
    measured=record['source']['measurement']
    assert measured['at_measurement_line'] and measured['line_offset_px']==0
    assert measured['center_recovery']['anchor_frame']==anchor
    assert cv2.imread(str(path/record['full_frame_photo'])).mean()==pytest.approx(100,abs=2)


def test_ip_visible_off_center_car_does_not_expire_review(setup,monkeypatch):
    _,profile=setup
    camera=ip.Station(ip.Settings(),profile)
    packet=ip.Packet(1,10,b'')
    track=dict(id='waiting',sent=False,stamp=15,box=[150,150,100,75],
        review=((packet,None,None),dict(bbox=[150,150,100,75]),camera.side.epoch,100))
    camera.tracks=[track]
    monkeypatch.setattr(camera,'capture',lambda *a,**k:pytest.fail('Still visible before the centre'))
    camera.flush_reviews(15)
    camera.flush_reviews(16)
    assert not track['sent']


def test_ip_review_searches_original_buffer_for_center(setup,monkeypatch):
    path,profile=setup
    camera=ip.Station(ip.Settings(),profile)
    packets=[ip.Packet(i,10+i*.1,cv2.imencode('.jpg',np.full((500,600,3),i*10,np.uint8))[1].tobytes())
             for i in range(21)]
    for packet in packets:camera.side.append(packet)
    monkeypatch.setattr(wb,'detect_vehicles',lambda image,*a,**k:[dict(
        bbox=[150+10*round(float(image.mean())/10),150,100,75],label='car')])
    camera.tracks=[dict(id='missed',sent=False,stamp=12,box=[350,150,100,75],
        review=((packets[0],None,None),dict(bbox=[150,150,100,75],label='car'),camera.side.epoch,100))]
    camera.flush_reviews(14)
    record=camera.saved['missed']
    assert record['source']['frame']==10 and record['length_m']==pytest.approx(5)
    assert record['source']['measurement']['at_measurement_line']
    assert cv2.imread(str(path/record['full_frame_photo'])).mean()==pytest.approx(100,abs=2)


def test_ip_review_pins_center_frames_after_receiver_eviction(setup,monkeypatch):
    _,profile=setup
    camera=ip.Station(ip.Settings(),profile)
    packets=[ip.Packet(i,10+i*.1,cv2.imencode('.jpg',np.full((500,600,3),i*10,np.uint8))[1].tobytes())
             for i in range(21)]
    camera.side.append(packets[0])
    window=camera.side.retain_passage(packets[0],0,review=True)
    for packet in packets[1:]:camera.side.append(packet)
    # Enough later packets to evict all actual passage images from the receiver.
    for i in range(100):camera.side.append(ip.Packet(100+i,20+i*.1,packets[-1].jpeg))
    assert camera.side.snapshot()[0].seq==100
    monkeypatch.setattr(wb,'detect_vehicles',lambda image,*a,**k:[dict(
        bbox=[150+10*round(float(image.mean())/10),150,100,75],label='car')])
    camera.tracks=[dict(id='pinned-review',sent=False,stamp=12,box=[350,150,100,75],review_frames=window,
        review=((packets[0],None,None),dict(bbox=[150,150,100,75],label='car'),0,100))]
    camera.flush_reviews(30)
    record=camera.saved['pinned-review']
    assert record['source']['frame']==10 and record['length_m']==pytest.approx(5)
    assert 'review_frames' not in camera.tracks[0]


def test_center_frame_retention_is_bounded_and_preserves_consecutive_evidence():
    anchor=ip.Packet(100,10,b'x'*(1024*1024))
    window=ip.CenterFrames(anchor,0)
    window.extend(ip.Packet(i,10+(i-100)/30,anchor.jpeg) for i in range(241))
    packets=window.snapshot()
    assert anchor in packets
    assert len(packets)<=241 and sum(len(p.jpeg) for p in packets)<=48*1024*1024
    assert [p.seq for p in packets]==list(range(packets[0].seq,packets[-1].seq+1))
