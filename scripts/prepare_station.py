"""First-run model download and supplied installation calibration."""
import os
from pathlib import Path
import shutil
import sys
import socket
import time
from urllib.error import URLError

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    from web_app import workbench as wb
    from web_app.plates import engine
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('NVIDIA GPU unavailable. Install/update the NVIDIA driver; CPU mode is disabled.')
    print('GPU:',torch.cuda.get_device_name(0),flush=True)
    wb.DATA.mkdir(parents=True,exist_ok=True)
    from ultralytics import YOLO
    previous=Path.cwd()
    try:
        os.chdir(wb.DATA)
        model=YOLO('yolo26n.pt')
        import numpy as np
        model.predict(np.zeros((640,640,3),np.uint8),device=0,verbose=False)
    finally:
        os.chdir(previous)
    socket.setdefaulttimeout(30)
    for attempt in range(3):
        try:
            engine()
            break
        except (URLError, TimeoutError, ConnectionError):
            if attempt==2:raise
            print('Model download interrupted; retrying...',flush=True)
            time.sleep(2)
    supplied=ROOT/'config'/'st-calibration.json'
    target=wb.DATA.parent/'operator-calibration.json'
    if supplied.exists() and not target.exists():
        wb.Profile.model_validate_json(supplied.read_text(encoding='utf-8'))
        shutil.copyfile(supplied,target)
        print('ST calibration copied. Recalibrate for another camera position/resolution.')
    from web_app.station import connect
    with connect():pass
    print('Ready. Run START.cmd.',flush=True)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        print('Preparation failed:',exc,file=sys.stderr)
        sys.exit(1)
