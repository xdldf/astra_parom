import asyncio
from contextlib import asynccontextmanager
import socket
import struct
import sys
import threading
import time
from types import SimpleNamespace
import urllib.request

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import pytest

from web_app import server as station_server


def test_windows_launcher_survives_abrupt_video_disconnects(monkeypatch):
    # Exercise the Windows launcher branch on every platform. On Windows this
    # also tests the native Selector socket implementation (no Proactor mocks).
    monkeypatch.setattr(station_server,'sys',SimpleNamespace(platform='win32'))
    instances=[];errors=[];state=dict(active=0,closed=0)
    original_server=station_server.uvicorn.Server
    def make_server(config):
        instance=original_server(config);instances.append(instance)
        return instance
    monkeypatch.setattr(station_server.uvicorn,'Server',make_server)

    @asynccontextmanager
    async def lifespan(app):
        loop=asyncio.get_running_loop()
        state['loop']=loop
        loop.set_exception_handler(lambda loop,context:errors.append(context))
        yield
        state['shutdown']=True
    app=FastAPI(lifespan=lifespan)

    @app.get('/health')
    def health():return {'online':True}

    @app.get('/video')
    async def video():
        async def frames():
            state['active']+=1
            try:
                while True:
                    yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\nframe\r\n'
                    await asyncio.sleep(.01)
            finally:
                state['active']-=1;state['closed']+=1
        return StreamingResponse(frames(),media_type='multipart/x-mixed-replace; boundary=frame')

    with socket.socket() as probe:
        probe.bind(('127.0.0.1',0));port=probe.getsockname()[1]
    def run():
        try:station_server.run_server(app,host='127.0.0.1',port=port)
        except BaseException as exc:errors.append(exc)
    thread=threading.Thread(target=run,daemon=True);thread.start()
    try:
        deadline=time.monotonic()+5
        while time.monotonic()<deadline and not (instances and instances[0].started):
            time.sleep(.01)
        assert instances and instances[0].started,errors
        assert isinstance(state['loop'],asyncio.SelectorEventLoop)
        for count in range(1,6):
            with socket.create_connection(('127.0.0.1',port),timeout=2) as viewer:
                viewer.sendall(b'GET /video HTTP/1.1\r\nHost: localhost\r\n\r\n')
                data=b''
                while b'--frame\r\n' not in data:
                    chunk=viewer.recv(4096)
                    assert chunk,'Server closed before sending a video frame'
                    data+=chunk
                # Reset rather than gracefully completing the streaming body.
                linger=struct.pack('HH' if sys.platform=='win32' else 'ii',1,0)
                viewer.setsockopt(socket.SOL_SOCKET,socket.SO_LINGER,linger)
            deadline=time.monotonic()+2
            while time.monotonic()<deadline and state['closed']<count:time.sleep(.01)
            assert state['closed']==count and state['active']==0
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=2) as response:
                assert response.status==200
        assert not errors
    finally:
        if instances:instances[0].should_exit=True
        thread.join(timeout=5)
    assert not thread.is_alive() and state.get('shutdown')
    assert state['loop'].is_closed() and not errors


def test_windows_startup_failure_still_reaches_launcher(monkeypatch):
    monkeypatch.setattr(station_server,'sys',SimpleNamespace(platform='win32'))
    @asynccontextmanager
    async def lifespan(app):
        raise RuntimeError('A real startup failure')
        yield
    with pytest.raises((RuntimeError,SystemExit)) as failure:
        station_server.run_server(FastAPI(lifespan=lifespan),host='127.0.0.1',port=0)
    if isinstance(failure.value,SystemExit):
        assert failure.value.code==3  # Uvicorn's startup failure status.
    else:
        assert 'did not complete startup' in str(failure.value)


def test_other_platforms_keep_uvicorn_default_loop(monkeypatch):
    monkeypatch.setattr(station_server,'sys',SimpleNamespace(platform='linux'))
    calls=[]
    monkeypatch.setattr(station_server.uvicorn,'run',lambda app,**options:calls.append((app,options)))
    station_server.run_server()
    assert calls==[('web_app.main:app',dict(host='0.0.0.0',port=8000,access_log=False,workers=1))]
