"""Single-process station server, including Windows stream-disconnect handling."""
import asyncio
import sys

import uvicorn


def selector_loop():
    loop=asyncio.SelectorEventLoop()
    asyncio.set_event_loop(loop)
    return loop


def run_server(app='web_app.main:app', *, host='0.0.0.0', port=8000, access_log=False):
    options=dict(host=host,port=port,access_log=access_log,workers=1)
    if sys.platform!='win32':
        return uvicorn.run(app,**options)
    # The Proactor transport can raise WinError 10054 while shutting down a
    # socket reset by a video viewer. Selector closes that socket directly.
    # Own the loop explicitly: recent Uvicorn versions override loop policies.
    # Camera decoding/inference use threads, not asyncio subprocess transports.
    server=uvicorn.Server(uvicorn.Config(app,loop='none',**options))
    try:
        with asyncio.Runner(loop_factory=selector_loop) as runner:
            runner.run(server.serve())
    finally:
        asyncio.set_event_loop(None)
    if not server.started:
        raise RuntimeError('Station server did not complete startup')
