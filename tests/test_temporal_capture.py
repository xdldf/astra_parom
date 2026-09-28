import time

import cv2
import numpy as np
import pytest
from fastapi import HTTPException

from web_app import workbench as wb, station, ip_cameras as ip
from web_app.temporal_capture import video_frames


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(station, 'DATA', tmp_path)
    monkeypatch.setattr(station, 'DB', tmp_path/'station.sqlite3')
    profile = wb.Profile(image_size=(600,500), polygon=[(20,200),(580,200),(580,450),(20,450)],
                         references=[dict(bbox=[250,150,100,75],length_m=5)],
                         measurement_line_x=300, detector_model='yolo26m')
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


def test_uncalibrated_crossing_is_reviewed_without_relaxing_line_gate(setup):
    _,profile=setup
    camera=ip.Station(ip.Settings(),profile)
    packet=ip.Packet(1,time.monotonic(),cv2.imencode('.jpg',np.zeros((500,600,3),np.uint8))[1].tobytes())
    camera.side.packets.append(packet)
    record=camera.capture((packet,None,None),dict(bbox=[250,325,100,75],label='truck'),'truck',temporal=True)
    assert record['length_m'] is None and record['side_photo']
    assert record['source']['measurement']['temporal']['reasons']==['anchor_outside_calibration']
    assert record['source']['measurement']['single_frame_status']=='outside_calibration'
    with pytest.raises(HTTPException):
        camera.capture((packet,None,None),dict(bbox=[350,325,100,75],label='truck'),'off-line',temporal=True)
