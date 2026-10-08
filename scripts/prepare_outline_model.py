#!/usr/bin/env python3
"""Install the optional, checksum-pinned official YOLO26-M segmentation model."""
import hashlib
from pathlib import Path
import sys
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vehicle_metrology.outline import WEIGHTS_NAME, WEIGHTS_SHA256, WEIGHTS_URL


def main():
    destination = ROOT/'web_app/data'/WEIGHTS_NAME
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and hashlib.sha256(destination.read_bytes()).hexdigest() == WEIGHTS_SHA256:
        print(f'Already installed: {destination}')
        return
    temporary = destination.with_suffix('.'+uuid.uuid4().hex+'.tmp')
    try:
        with urllib.request.urlopen(WEIGHTS_URL, timeout=60) as source, temporary.open('wb') as target:
            while block := source.read(1024*1024):
                target.write(block)
        if hashlib.sha256(temporary.read_bytes()).hexdigest() != WEIGHTS_SHA256:
            raise ValueError('Official checkpoint changed; checksum mismatch. No model installed.')
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    print(f'Installed: {destination}')


if __name__ == '__main__':
    main()
