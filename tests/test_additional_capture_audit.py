from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from scripts import audit_additional_captures as audit


@pytest.mark.parametrize('label', ['car', 'bus', 'truck'])
def test_capture_failure_keeps_full_corrected_frame(tmp_path, label):
    """A failed measurement must remain reviewable, using corrected pixels."""
    raw = np.zeros((24, 32, 3), dtype=np.uint8)
    corrected = np.full_like(raw, (10, 80, 160))
    video = SimpleNamespace(isOpened=lambda: True, get=lambda _: 25, release=lambda: None)
    row = dict(id='sample', video='source.mp4', frame=3, bbox=[2, 4, 16, 8], detector_label=label)
    profile = SimpleNamespace(image_size=(32, 24), lens=None)
    with patch.object(audit.cv2, 'VideoCapture', return_value=video), \
         patch.object(audit.wb, 'read_frame', return_value=raw), \
         patch.object(audit.wb, 'corrected', return_value=corrected), \
         patch.object(audit.st, 'Capture', side_effect=lambda **kwargs: kwargs), \
         patch.object(audit.st, 'capture', side_effect=RuntimeError('No unambiguous car')) as capture:
        result = audit.capture_one(row, profile, tmp_path)
    assert capture.call_args.args[0]['label'] == label
    assert result['capture_status'] == 'failed'
    assert result['length_m'] is None
    assert result['reason'] == 'No unambiguous car'
    photo = tmp_path/result['corrected_image']
    expected = cv2.imencode('.jpg', corrected, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes()
    assert photo.read_bytes() == expected
    assert cv2.imread(str(photo)).shape == raw.shape
    assert 'sample' not in audit.wb.media
    audit.verify_saved(result, tmp_path)
    photo.write_bytes(b'changed')
    with pytest.raises(ValueError, match='Changed saved artifact'):
        audit.verify_saved(result, tmp_path)


def test_resume_checks_station_photo_not_only_audit_copy(tmp_path):
    (tmp_path/'captures').mkdir()
    (tmp_path/'frame.jpg').write_bytes(b'corrected photo')
    (tmp_path/'captures/station.jpg').write_bytes(b'wrong photo')
    record = tmp_path/'captures/record.json'
    audit.save(record, dict(full_frame_photo='station.jpg'))
    entry = dict(capture_status='saved', corrected_image='frame.jpg',
                 corrected_image_sha256=audit.file_digest(tmp_path/'frame.jpg'),
                 record_file='captures/record.json', record_sha256=audit.file_digest(record))
    with pytest.raises(ValueError, match='Station full-frame photo changed'):
        audit.verify_saved(entry, tmp_path)
