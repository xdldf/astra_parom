#!/usr/bin/env python3
"""Install the optional official wheel detector from a pinned, verified snapshot."""
import argparse
from pathlib import Path
import shutil
import sys
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vehicle_metrology.wheels import MODEL_DIRECTORY, MODEL_ID, MODEL_REVISION, MODEL_FILES, file_digest, verify_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, help='Optional existing official snapshot; every file is still verified')
    args = parser.parse_args()
    destination = ROOT/'web_app/data'/MODEL_DIRECTORY
    destination.mkdir(parents=True, exist_ok=True)
    for name, expected in MODEL_FILES.items():
        target = destination/name
        if target.is_file() and file_digest(target) == expected:
            continue
        temporary = destination/(name+'.'+uuid.uuid4().hex+'.tmp')
        try:
            if args.source_dir:
                shutil.copyfile(args.source_dir/name, temporary)
            else:
                url = f'https://huggingface.co/{MODEL_ID}/resolve/{MODEL_REVISION}/{name}'
                with urllib.request.urlopen(url, timeout=60) as source, temporary.open('wb') as output:
                    shutil.copyfileobj(source, output)
            if file_digest(temporary) != expected:
                raise ValueError(f'Official model checksum mismatch: {name}')
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
    verify_model(destination)
    print(f'Installed: {destination}')


if __name__ == '__main__':
    main()
