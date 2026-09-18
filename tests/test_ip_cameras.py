import time
import threading
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from web_app import ip_cameras as ip, station, plates, workbench as wb
from web_app.main import app


def packet(seq,stamp):return ip.Packet(seq,stamp,b'jpeg')


def test_pairing_offset_freshness_and_skew():
    a=[packet(1,10),packet(2,10.1)]
    b=[packet(3,10.2),packet(4,10.3)]
    assert ip.pair_packets(a,b,.2,.04,10.4)[0].seq==2
    assert ip.pair_packets(a,b,0,.04,10.4) is None
    assert ip.pair_packets(a,b,.2,.04,12) is None
    assert ip.pair_packets([],b,0,.2,10.4) is None


def test_configuration_secret_redaction_and_scheme(tmp_path,monkeypatch):
    monkeypatch.setattr(ip,'CONFIG',tmp_path/'ip.json')
    monkeypatch.setattr(ip,'active',None)
    client=TestClient(app)
    cfg={'side_url':'rtsp://user:password@192.168.1.2/path','front_url':'http://user:secret@192.168.1.3/mjpeg'}
    result=client.post('/api/ip/configuration',json=cfg)
    assert result.status_code==200
    assert 'password' not in result.text and 'secret' not in result.text
    assert 'side_url' not in client.get('/api/ip/configuration').json()
    assert client.post('/api/ip/configuration',json={'offset_seconds':.1}).status_code==200
    assert ip.settings().side_url==cfg['side_url']
    with pytest.raises(ValidationError):ip.Settings(side_url='file:///secret')
    with pytest.raises(ValidationError):ip.Settings(side_url='C:/video.mp4')


def test_ip_calibration_is_separate_and_start_is_singleton(tmp_path,monkeypatch):
    monkeypatch.setattr(ip,'CONFIG',tmp_path/'ip.json')
    monkeypatch.setattr(ip,'active',None)
    monkeypatch.setattr(wb,'require_gpu',lambda:0)
    calls=[]
    monkeypatch.setattr(ip.Station,'start',lambda self:calls.append(self.id))
    monkeypatch.setattr(ip.Station,'close',lambda self:None)
    client=TestClient(app)
    cfg={'side_url':'rtsp://localhost/side','front_url':'rtsp://localhost/front'}
    assert client.post('/api/ip/configuration',json=cfg).status_code==200
    profile={'image_size':[600,500],'polygon':[[20,200],[580,200],[580,450],[20,450]],'references':[{'bbox':[100,150,100,75],'length_m':5}]}
    assert client.post('/api/ip/calibration',json=profile).status_code==200
    monkeypatch.setattr(wb,'operator_calibration',lambda:pytest.fail('Must use separate IP calibration'))
    a=client.post('/api/ip/start').json();b=client.post('/api/ip/start').json()
    assert a['id']==b['id'] and len(calls)==1
    assert client.post('/api/ip/configuration',json=cfg).status_code==409
    assert client.post('/api/ip/automatic?enabled=false').json()=={'enabled':False}
    assert not ip.settings().auto_measure
    assert client.post('/api/ip/stop').json()=={'running':False}
    assert not client.get('/api/ip/state').json()['running']


def test_receiver_reconnect_clears_old_frames():
    stop=threading.Event();opens=[]
    class Fake:
        def isOpened(self):return True
        def read(self):
            if len(opens)==1:return False,None
            stop.set();return True,np.zeros((20,30,3),np.uint8)
        def release(self):pass
    def open_(url):opens.append(url);return Fake()
    receiver=ip.Receiver('rtsp://test',stop,open_)
    t=threading.Thread(target=receiver.run);t.start();t.join(4)
    assert not t.is_alive() and len(opens)==2
    assert receiver.snapshot()==[]


def test_ip_capture_survives_buffer_loss(tmp_path,monkeypatch):
    monkeypatch.setattr(station,'DATA',tmp_path)
    monkeypatch.setattr(station,'DB',tmp_path/'db.sqlite')
    monkeypatch.setattr(plates,'enqueue',lambda _:None)
    profile=wb.Profile(image_size=(600,500),polygon=[(20,200),(580,200),(580,450),(20,450)],references=[{'bbox':[100,150,100,75],'length_m':5}])
    camera=ip.Station(ip.Settings(),profile)
    raw=np.zeros((500,600,3),np.uint8);jpeg=cv2.imencode('.jpg',raw)[1].tobytes();stamp=time.monotonic()
    a=ip.Packet(1,stamp,jpeg);b=ip.Packet(2,stamp,jpeg)
    camera.side.packets.append(a);camera.front.packets.append(b)
    d={'bbox':[100,150,100,75],'label':'car'}
    result=camera.capture((a,b,0),d,'track-1')
    assert result['source']['front_camera']['kind']=='ip'
    assert 'media_id' not in result['source']['front_camera']
    assert (tmp_path/result['source']['front_samples'][0]['photo']).exists()
    assert camera.capture((a,b,0),d,'track-1')['id']==result['id']
    camera.side.packets.clear();camera.front.packets.clear()
    monkeypatch.setattr(plates,'front_vehicle',lambda _: [0,0,600,500])
    monkeypatch.setattr(plates,'recognize',lambda image,reference_box=None:[])
    plates.process_record(result['id'])
    with station.connect() as db:assert station.find(db,result['id'])['plate_ocr']['state']=='not_found'
    with pytest.raises(Exception):camera.capture((a,b,0),d,'track-2')


def test_side_measurement_persists_without_front(tmp_path,monkeypatch):
    monkeypatch.setattr(station,'DATA',tmp_path)
    monkeypatch.setattr(station,'DB',tmp_path/'db.sqlite3')
    profile=wb.Profile(image_size=(600,500),polygon=[(20,200),(580,200),(580,450),(20,450)],references=[{'bbox':[100,150,100,75],'length_m':5}])
    c=ip.Station(ip.Settings(),profile)
    a=ip.Packet(1,time.monotonic(),cv2.imencode('.jpg',np.zeros((500,600,3),np.uint8))[1].tobytes())
    c.side.packets.append(a)
    assert c.paired() is None
    result=c.capture((a,None,None),{'bbox':[100,150,100,75],'label':'car'},'one')
    assert result['length_m']==pytest.approx(5)
    assert result['front_photo'] is None and result['plate']==''
    assert result['side_photo'] and result['source']['camera_note']
    assert 'plate_ocr' not in result


def test_front_recognition_runs_without_side(monkeypatch):
    c=ip.Station(ip.Settings(),wb.Profile(image_size=(600,500)))
    b=ip.Packet(1,time.monotonic(),cv2.imencode('.jpg',np.zeros((500,600,3),np.uint8))[1].tobytes())
    c.front.packets.append(b)
    monkeypatch.setattr(plates,'recognize',lambda image:[{'text':'А123ВС14','bbox':[100,300,200,350]}])
    t=threading.Thread(target=c.read_plates);t.start()
    try:
        deadline=time.monotonic()+2
        while c.plates is None and time.monotonic()<deadline:time.sleep(.02)
        assert c.plates['candidates'][0]['text']=='А123ВС14'
        assert c.paired() is None
        monkeypatch.setattr(ip,'active',c)
        status=ip.state()
        assert status['front_ready'] and not status['side_ready']
        assert status['plates'] and status['result'] is None
    finally:c.stop.set();t.join(2)



def test_many_viewers_share_encoded_frames_without_blocking_worker_threads():
    import asyncio
    c=ip.Station(ip.Settings(),wb.Profile(image_size=(600,500)))
    c.jpeg=b'shared-side';c.front_jpeg=b'shared-front';c.sequence=1
    async def view():
        viewers=[c.frames(bool(i%2)) for i in range(48)]
        first=await asyncio.gather(*(anext(viewer) for viewer in viewers))
        assert all((b'shared-front' if i%2 else b'shared-side') in frame for i,frame in enumerate(first))
        next_frames=[asyncio.create_task(anext(viewer)) for viewer in viewers]
        await asyncio.sleep(.06)
        assert not any(task.done() for task in next_frames)  # unchanged buffers are not resent
        c.jpeg=b'new-side';c.front_jpeg=b'new-front';c.sequence=2
        second=await asyncio.wait_for(asyncio.gather(*next_frames),1)
        assert all((b'new-front' if i%2 else b'new-side') in frame for i,frame in enumerate(second))
        c.stop.set()
        await asyncio.gather(*(viewer.aclose() for viewer in viewers))
    asyncio.run(view())


def test_concurrent_starts_share_one_camera_station(tmp_path,monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    monkeypatch.setattr(ip,'CONFIG',tmp_path/'ip.json');monkeypatch.setattr(ip,'active',None)
    monkeypatch.setattr(wb,'require_gpu',lambda:0)
    started=[]
    monkeypatch.setattr(ip.Station,'start',lambda self:started.append(self.id))
    profile=wb.Profile(image_size=(600,500),polygon=[(20,200),(580,200),(580,450),(20,450)],
                      metric_rulers=[{'points':[(100,300),(200,300),(300,300)],'step_m':1}])
    ip.CONFIG.write_text(ip.Settings(side_url='rtsp://localhost/side',front_url='rtsp://localhost/front',profile=profile).model_dump_json())
    with ThreadPoolExecutor(max_workers=8) as pool:
        results=list(pool.map(lambda _:ip.start(),range(8)))
    assert len(started)==1
    assert {r['id'] for r in results}==set(started)
