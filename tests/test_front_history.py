import pytest
from web_app.front_history import FrontHistory
from web_app.ip_cameras import Packet


BOX=[100,100,500,300]
CANDIDATE=dict(text='А123ВС14',confidence=.95,detection_confidence=.9,bbox=[200,250,300,280])


def observe(history, second, candidates=(), box=BOX, epoch=1):
    packet=Packet(round(second*10),second,b'jpeg')
    history.observe(packet,box,list(candidates),epoch)
    return packet


def test_earlier_cab_retained_through_trailer_without_plate():
    history=FrontHistory()
    observe(history,0,[CANDIDATE])
    for second in range(1,9):
        packet=observe(history,second)
    frames=history.select(packet,BOX,1)
    assert len(frames)==1 and frames[0]['packet'].stamp==0
    assert not history.select(packet,[1000,0,50,100],1)
    assert not history.select(packet,BOX,2)
    assert not history.select(Packet(100,10,b''),BOX,1)


@pytest.mark.parametrize('gap,box,epoch',[(2,BOX,1),(1,None,1),(1,[1000,0,50,100],1),(1,BOX,2)])
def test_gap_vehicle_loss_or_reconnect_drops_earlier_plate(gap,box,epoch):
    history=FrontHistory()
    observe(history,0,[CANDIDATE])
    packet=observe(history,gap,box=box,epoch=epoch)
    assert not history.select(packet,BOX,epoch)


def test_following_number_is_not_mixed_and_history_has_hard_limits():
    history=FrontHistory(seconds=12,max_frames=5,max_bytes=12)
    for index in range(40):
        packet=observe(history,index*.3,[CANDIDATE])
    assert len(history.select(packet,BOX,1))<=3
    other=dict(CANDIDATE,text='В456ЕЕ14')
    packet=observe(history,12,[other])
    assert all(f['candidates'][0]['text']==other['text'] for f in history.select(packet,BOX,1))
    for second in range(13,26):packet=observe(history,second)
    assert not history.select(packet,BOX,1)


def test_capture_selects_its_own_time_after_front_processing_moves_ahead():
    history=FrontHistory()
    cab=observe(history,0,[CANDIDATE])
    for second in range(1,11):observe(history,second)
    anchor=Packet(80,8,b'')
    assert [f['packet'] for f in history.select(anchor,BOX,1)]==[cab]
    # Also match a capture between real observations of the same passage.
    assert history.select(Packet(85,8.5,b''),BOX,1)
    assert not history.select(Packet(85,8.5,b''),[1000,0,50,100],1)


def test_late_truck_capture_does_not_take_following_vehicle_number():
    history=FrontHistory()
    cab=observe(history,0,[CANDIDATE])
    for second in range(1,10):observe(history,second)
    other=dict(CANDIDATE,text='В456ЕЕ14',confidence=.99)
    for i in range(6):latest=observe(history,10+i*.2,[other])
    assert [f['packet'] for f in history.select(Packet(80,8,b''),BOX,1)]==[cab]
    assert all(f['candidates'][0]['text']==other['text'] for f in history.select(latest,BOX,1))
    # The unobserved transition between vehicles is not assigned by overlap.
    assert not history.select(Packet(95,9.5,b''),BOX,1)


def test_slow_truck_cab_survives_twenty_four_seconds_and_capture_delay():
    history=FrontHistory()
    cab=observe(history,0,[CANDIDATE])
    for second in range(1,34):observe(history,second)
    assert [f['packet'] for f in history.select(Packet(290,29,b''),BOX,1)]==[cab]
    assert not history.select(Packet(310,31,b''),BOX,1)
    # Epoch and a recording rewind invalidate historical associations too.
    observe(history,34,epoch=2)
    assert not history.select(Packet(290,29,b''),BOX,1)
    observe(history,1,epoch=2)
    assert not history.select(Packet(290,29,b''),BOX,2)


def test_archived_passages_and_geometry_are_bounded():
    history=FrontHistory(max_frames=5,max_bytes=40)
    for i in range(1000):
        candidate=dict(CANDIDATE,text=f'А{i//6:03}ВС14')
        packet=observe(history,i*.01,[candidate])
    assert len(history.observations)<=512
    assert len({f['passage'] for f in history.frames})<=3
    assert len(history.frames)<=15
    assert sum(len(f['packet'].jpeg) for f in history.frames)<=40
    assert all('packet' not in row for row in history.observations)
    count=len(history.frames)
    history.observe(packet,BOX,[candidate],1)
    assert len(history.frames)==count
