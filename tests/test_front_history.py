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
    history=FrontHistory(max_frames=5,max_bytes=12)
    for index in range(40):
        packet=observe(history,index*.3,[CANDIDATE])
    assert len(history.select(packet,BOX,1))<=3
    other=dict(CANDIDATE,text='В456ЕЕ14')
    packet=observe(history,12,[other])
    assert all(f['candidates'][0]['text']==other['text'] for f in history.select(packet,BOX,1))
    for second in range(13,26):packet=observe(history,second)
    assert not history.select(packet,BOX,1)
