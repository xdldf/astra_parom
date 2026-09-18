"""Launch locally; open the browser only after the server answers."""
from pathlib import Path
import os
import sys
import threading
import time
import urllib.request
import webbrowser

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
URL='http://127.0.0.1:8000'


def ready():
    try:
        import json
        with urllib.request.urlopen(URL+'/api/station/health',timeout=1) as response:
            return json.load(response).get('online') is True
    except Exception:return False


def open_when_ready():
    for _ in range(90):
        if ready():
            webbrowser.open(URL)
            return
        time.sleep(1)


if __name__=='__main__':
    os.chdir(ROOT)
    if ready():
        print('Station is already running.')
        webbrowser.open(URL)
    else:
        try:
            import uvicorn
            threading.Thread(target=open_when_ready,daemon=True).start()
            print('Keep this window open. Press Ctrl+C to stop.')
            uvicorn.run('web_app.main:app',host='0.0.0.0',port=8000,access_log=False)
        except KeyboardInterrupt:pass
        except Exception as exc:
            print('Startup failed:',exc,'Run INSTALL.cmd if dependencies are missing.',file=sys.stderr)
            sys.exit(1)
