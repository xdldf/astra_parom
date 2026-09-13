"""Install the bundled, checksum-verified models into the local runtime cache."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def install_models(root=ROOT):
    root = Path(root)
    bundle = root / 'models'
    manifest = json.loads((bundle / 'manifest.json').read_text(encoding='utf-8'))
    # Validate the entire bundle before modifying the runtime cache.
    for entry in manifest['files']:
        source = bundle / entry['path']
        if hashlib.sha256(source.read_bytes()).hexdigest() != entry['sha256']:
            raise RuntimeError(f"Model checksum mismatch: {source}. Restore it with git.")
    for entry in manifest['files']:
        target = root / 'web_app' / 'data' / entry['path']
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(bundle / entry['path'], target)
            print(f"Installed bundled model asset: {entry['path']}", flush=True)


if __name__ == '__main__':
    install_models()
