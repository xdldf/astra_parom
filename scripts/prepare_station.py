"""Install bundled models, verify GPU inference and supply installation calibration."""
from pathlib import Path
import shutil
import sys
import socket
import time
from urllib.error import URLError

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    from scripts.install_model_assets import install_models
    install_models()
    from web_app import workbench as wb
    from web_app.plates import engine
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('NVIDIA GPU unavailable. Install/update the NVIDIA driver; CPU mode is disabled.')
    print('GPU:',torch.cuda.get_device_name(0),flush=True)
    wb.DATA.mkdir(parents=True,exist_ok=True)
    supplied=ROOT/'config'/'st-calibration.json'
    target=wb.DATA.parent/'operator-calibration.json'
    profile_path=target if target.exists() else supplied
    profile=wb.Profile.model_validate_json(profile_path.read_text(encoding='utf-8'))
    import numpy as np
    # N still serves front-camera association; initialize the configured side model too.
    for name in dict.fromkeys(['yolo26n',profile.detector_model]):
        print('Checking detector:',name,flush=True)
        wb.detect_vehicles(np.zeros((640,640,3),np.uint8),detector_model=name,imgsz=profile.detector_imgsz)
    socket.setdefaulttimeout(30)
    for attempt in range(3):
        try:
            engine()
            break
        except (URLError, TimeoutError, ConnectionError):
            if attempt==2:raise
            print('Model download interrupted; retrying...',flush=True)
            time.sleep(2)
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
