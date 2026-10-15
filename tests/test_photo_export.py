import csv
import io
import json
import zipfile

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from web_app import station
from web_app.main import app


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(station, 'DATA', tmp_path / 'station')
    monkeypatch.setattr(station, 'DB', station.DATA / 'test.sqlite3')
    return TestClient(app)


def saved_car(stamp='2026-10-13T10:00:00+09:00', kinds=('full', 'side', 'plate'), **fields):
    record = station.new_record(station.Fields(**fields))
    record['created_at'] = stamp
    station.DATA.mkdir(parents=True, exist_ok=True)
    for kind in kinds:
        field, suffix = {'full': ('full_frame_photo', 'frame'), 'side': ('side_photo', 'side'),
                         'plate': ('front_photo', 'front')}[kind]
        record[field] = f'{record["id"]}-{suffix}.jpg'
        # Byte preservation matters: the export must not resize/re-encode originals.
        (station.DATA / record[field]).write_bytes(b'original-jpeg-' + kind.encode())
    with station.connect() as db:
        db.execute('INSERT INTO vehicles VALUES(?,?,?,?)', (record['id'], None, 1, json.dumps(record)))
    return record


def archive_files(response):
    assert response.status_code == 200, response.text if response.status_code != 200 else ''
    assert response.headers['content-type'] == 'application/zip'
    assert int(response.headers['content-length']) == len(response.content)
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        assert archive.testzip() is None
        return {name: archive.read(name) for name in archive.namelist()}


def captured_car(client, monkeypatch, bbox=(100.5, 150.5, 100.25, 75.25)):
    monkeypatch.setattr(station.workbench, 'read_frame', lambda *_: np.full((500, 600, 3), 30, np.uint8))
    response = client.post('/api/station/capture', json=dict(
        media_id='photo-export-test', frame=7, bbox=bbox, source='yolo26n',
        profile=dict(image_size=[600, 500], polygon=[[2, 100], [598, 100], [598, 450], [2, 450]],
                     measurement_line_x=bbox[0]+bbox[2]/2,
                     references=[dict(bbox=bbox, length_m=5)])))
    assert response.status_code == 200, response.text
    return response.json()


@pytest.mark.parametrize('variant', ['original', 'boxes', 'both'])
@pytest.mark.parametrize('left', [10.5, 100.5])
def test_export_draws_saved_model_box_in_full_frame_and_crop_without_changing_evidence(client, monkeypatch, variant, left):
    record = captured_car(client, monkeypatch, bbox=(left, 150.5, 100.25, 75.25))
    origins = {'full': (0, 0), 'side': (max(0, int(left)-20), 130)}
    assert record['source']['side_crop_origin'] == list(origins['side'])
    originals = {kind: (station.DATA / record[field]).read_bytes()
                 for kind, field in [('full', 'full_frame_photo'), ('side', 'side_photo')]}
    params = [('photo_types', 'full'), ('photo_types', 'side'), ('photo_variant', variant)]
    preview = client.get('/api/station/photos.zip', params=params + [('preview', 'true')]).json()
    files = archive_files(client.get('/api/station/photos.zip', params=params))
    assert len(files) == preview['count'] == (4 if variant == 'both' else 2)
    assert preview['missing'] == 0
    if variant != 'original':
        assert preview['missing_boxes'] == 0
    for kind, field in [('full', 'full_frame_photo'), ('side', 'side_photo')]:
        basename = record[field]
        if variant in {'original', 'both'}:
            assert next(data for name, data in files.items() if name.endswith('/'+basename)) == originals[kind]
        if variant in {'boxes', 'both'}:
            data = next(data for name, data in files.items() if name.endswith('/'+basename[:-4]+'-boxes.jpg'))
            image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            original = cv2.imdecode(np.frombuffer(originals[kind], np.uint8), cv2.IMREAD_COLOR)
            assert image.shape == original.shape
            # Detect the rendered green frame; independent of JPEG byte changes.
            blue, green, red = image.astype(int).transpose(2, 0, 1)
            ys, xs = np.where((green > 140) & (green > red+60) & (green > blue+60))
            ox, oy = origins[kind]
            assert xs.min() == pytest.approx(left-ox, abs=3)
            assert xs.max() == pytest.approx(left+100.25-ox, abs=3)
            assert ys.min() == pytest.approx(150.5-oy, abs=3)
            assert ys.max() == pytest.approx(225.75-oy, abs=3)
            assert image[-3, -3].tolist() == pytest.approx([30, 30, 30], abs=3)
        assert (station.DATA / basename).read_bytes() == originals[kind]
    with station.connect() as db:
        assert station.find(db, record['id']) == record


@pytest.mark.parametrize('length_m,expected', [(4.72564, '4.726 m'), (None, None), (0, None)])
def test_export_labels_full_frame_and_crop_with_calculated_length_not_operator_edit(client, monkeypatch, length_m, expected):
    record = captured_car(client, monkeypatch)
    record['measured_length_m'] = length_m
    record['source']['measured_length_m'] = length_m
    record['source']['measurement']['length_m'] = length_m
    record['length_m'] = 8.5  # An operator edit must not replace the model result on the image.
    with station.connect() as db:
        db.execute('UPDATE vehicles SET data=? WHERE id=?', (json.dumps(record), record['id']))
    labels = []
    original_put_text = cv2.putText
    def capture_label(image, text, origin, font, scale, color, thickness, line_type):
        (width, height), baseline = cv2.getTextSize(text, font, scale, thickness)
        x, y = origin
        assert 0 <= x and x+width <= image.shape[1]
        assert 0 <= y-height and y+baseline <= image.shape[0]
        labels.append((text, image.shape))
        return original_put_text(image, text, origin, font, scale, color, thickness, line_type)
    monkeypatch.setattr(cv2, 'putText', capture_label)
    files = archive_files(client.get('/api/station/photos.zip', params=[
        ('photo_types', 'full'), ('photo_types', 'side'), ('photo_variant', 'both')]))
    assert labels == ([(expected, (500, 600, 3)), (expected, (115, 140, 3))] if expected else [])
    for field in ('full_frame_photo', 'side_photo'):
        filename = record[field]
        original = (station.DATA / filename).read_bytes()
        assert next(data for name, data in files.items() if name.endswith('/'+filename)) == original
        annotated = next(data for name, data in files.items() if name.endswith('/'+filename[:-4]+'-boxes.jpg'))
        image = cv2.imdecode(np.frombuffer(annotated, np.uint8), cv2.IMREAD_COLOR)
        # White lettering is present in the actual JPEG only for a known measurement.
        assert bool(np.any(np.min(image, axis=2) > 180)) == bool(expected)


def test_legacy_crop_without_saved_origin_gets_same_box(client, monkeypatch):
    record = captured_car(client, monkeypatch)
    params = dict(photo_types='side', photo_variant='boxes')
    current = archive_files(client.get('/api/station/photos.zip', params=params))
    record['source'].pop('side_crop_origin')
    record['source'].pop('image_space')
    with station.connect() as db:
        db.execute('UPDATE vehicles SET data=? WHERE id=?', (json.dumps(record), record['id']))
    assert archive_files(client.get('/api/station/photos.zip', params=params)) == current


@pytest.mark.parametrize('variant', ['boxes', 'both'])
@pytest.mark.parametrize('source', [None, {}, {'bbox': [1, 2, -3, 4]}, {'bbox': [1, 2, 3]},
                                   {'bbox': [1, 2, None, 4]}, {'bbox': 'bad'},
                                   {'bbox': [1, 2, 3, 4], 'image_space': 'unknown'}])
def test_missing_or_invalid_boxes_preserve_original_once_and_report_fallback(client, variant, source):
    record = saved_car()
    record['source'] = source
    with station.connect() as db:
        db.execute('UPDATE vehicles SET data=? WHERE id=?', (json.dumps(record), record['id']))
    params = [('photo_types', kind) for kind in ('full', 'side', 'plate')] + [('photo_variant', variant)]
    assert client.get('/api/station/photos.zip', params=params + [('preview', 'true')]).json() == {
        'count': 3, 'missing': 0, 'missing_boxes': 2}
    files = archive_files(client.get('/api/station/photos.zip', params=params))
    assert len(files) == 3
    assert set(files.values()) == {b'original-jpeg-full', b'original-jpeg-side', b'original-jpeg-plate'}


def test_front_photo_remains_original_when_exporting_both_variants(client, monkeypatch):
    record = captured_car(client, monkeypatch)
    record['front_photo'] = record['id']+'-front.jpg'
    (station.DATA / record['front_photo']).write_bytes(b'original-jpeg-plate')
    with station.connect() as db:
        db.execute('UPDATE vehicles SET data=? WHERE id=?', (json.dumps(record), record['id']))
    params = [('photo_types', kind) for kind in ('full', 'side', 'plate')] + [('photo_variant', 'both')]
    assert client.get('/api/station/photos.zip', params=params + [('preview', 'true')]).json() == {
        'count': 5, 'missing': 0, 'missing_boxes': 0}
    files = archive_files(client.get('/api/station/photos.zip', params=params))
    assert len(files) == 5
    assert next(data for name, data in files.items() if name.endswith('-front.jpg')) == b'original-jpeg-plate'


def test_invalid_photo_variant_is_rejected(client):
    assert client.get('/api/station/photos.zip?photo_variant=unknown').status_code == 422


def test_default_exports_only_original_full_frames_and_all_pages(client):
    records = [saved_car() for _ in range(251)]
    files = archive_files(client.get('/api/station/photos.zip'))
    assert len(files) == 251
    assert {name.rsplit('/', 1)[1] for name in files} == {r['full_frame_photo'] for r in records}
    assert set(files.values()) == {b'original-jpeg-full'}
    assert all(name.startswith('2026-10-13T10_00_00_') for name in files)


@pytest.mark.parametrize('kinds', [('full',), ('side',), ('plate',), ('full', 'side'),
                                  ('full', 'plate'), ('side', 'plate'), ('full', 'side', 'plate')])
def test_archive_contains_exactly_the_selected_photo_types(client, kinds):
    saved_car()
    params = [('photo_types', kind) for kind in kinds]
    preview = client.get('/api/station/photos.zip', params=params + [('preview', 'true')]).json()
    assert preview == {'count': len(kinds), 'missing': 0}
    files = archive_files(client.get('/api/station/photos.zip', params=params))
    assert len(files) == len(kinds)
    assert set(files.values()) == {b'original-jpeg-' + kind.encode() for kind in kinds}


def test_archive_csv_and_report_share_inclusive_registration_time_filters(client):
    records = [saved_car(stamp, category='motorcycle') for stamp in (
        '2026-10-12T23:59:59+09:00', '2026-10-13T09:59:59+09:00',
        '2026-10-13T10:00:00+09:00', '2026-10-13T10:30:00+09:00',
        '2026-10-13T10:30:01+09:00', '2026-10-14T00:00:00+09:00')]
    saved_car(category='car')
    params = dict(date_from='2026-10-13', date_to='2026-10-13', time_from='10:00', time_to='10:30',
                  category='motorcycle', status='Требует проверки')
    # Use the actual initial status, also checking that status filtering is honored.
    params['status'] = records[0]['status']
    files = archive_files(client.get('/api/station/photos.zip', params=params))
    assert {name.rsplit('/', 1)[1] for name in files} == {r['full_frame_photo'] for r in records[2:4]}
    report = client.get('/api/station/vehicles', params=params).json()
    assert {r['id'] for r in report['rows']} == {r['id'] for r in records[2:4]}
    csv_response = client.get('/api/station/reports.csv', params=params)
    rows = list(csv.reader(io.StringIO(csv_response.content.decode('utf-8-sig')), delimiter=';'))
    assert [r[1] for r in rows[1:]] == ['10:30:00', '10:00:00']
    params['status'] = 'Оплачен'
    assert client.get('/api/station/photos.zip', params=params).status_code == 404


def test_dates_without_times_include_whole_days(client):
    for stamp in ('2026-10-12T23:59:59+09:00', '2026-10-13T00:00:00+09:00',
                  '2026-10-13T23:59:59+09:00', '2026-10-14T00:00:00+09:00'):
        saved_car(stamp)
    files = archive_files(client.get('/api/station/photos.zip', params=dict(date_from='2026-10-13', date_to='2026-10-13')))
    assert len(files) == 2


def test_missing_full_frames_are_skipped_without_substituting_crops(client):
    saved_car(kinds=('side', 'plate'))
    missing = saved_car()
    (station.DATA / missing['full_frame_photo']).unlink()
    assert client.get('/api/station/photos.zip').status_code == 404
    assert client.get('/api/station/photos.zip?preview=true').status_code == 404
    complete = saved_car()
    assert client.get('/api/station/photos.zip?preview=true').json() == {'count': 1, 'missing': 2}
    files = archive_files(client.get('/api/station/photos.zip'))
    assert [name.rsplit('/', 1)[1] for name in files] == [complete['full_frame_photo']]


@pytest.mark.parametrize('params', [
    {'date_from': '2026-10-14', 'date_to': '2026-10-13'},
    {'date_from': '2026-10-13', 'date_to': '2026-10-13', 'time_from': '12:00', 'time_to': '11:00'},
    {'time_from': '10:00'}, {'time_to': '10:00'}, {'date_from': 'not-a-date'},
    {'date_from': '2026-10-13', 'time_from': '10:00:00Z'},
])
def test_invalid_period_is_rejected_consistently(client, params):
    for endpoint in ('photos.zip', 'reports.csv', 'vehicles'):
        assert client.get('/api/station/' + endpoint, params=params).status_code == 422


@pytest.mark.parametrize('kind', ['', 'unknown', '../station.sqlite3'])
def test_invalid_photo_selection_is_rejected(client, kind):
    assert client.get('/api/station/photos.zip', params={'photo_types': kind}).status_code == 422


def test_exports_do_not_read_paths_outside_photo_directory(client, tmp_path):
    record = saved_car()
    outside = tmp_path / 'outside.jpg'
    outside.write_bytes(b'private')
    path = station.DATA / record['full_frame_photo']
    path.unlink()
    path.symlink_to(outside)
    assert client.get('/api/station/photos.zip').status_code == 404
    record['full_frame_photo'] = '../outside.jpg'
    with station.connect() as db:
        db.execute('UPDATE vehicles SET data=? WHERE id=?', (json.dumps(record), record['id']))
    assert client.get('/api/station/photos.zip').status_code == 404


def test_temporary_archive_is_closed_after_download_and_empty_result(client, monkeypatch):
    opened = []
    original = station.tempfile.TemporaryFile
    def tracked_file():
        handle = original()
        opened.append(handle)
        return handle
    monkeypatch.setattr(station.tempfile, 'TemporaryFile', tracked_file)
    record = saved_car()
    archive_files(client.get('/api/station/photos.zip'))
    assert opened[-1].closed
    assert client.get('/api/station/photos.zip?date_from=2999-01-01').status_code == 404
    assert opened[-1].closed
    with station.connect() as db:
        stored = db.execute('SELECT data FROM vehicles WHERE id=?', (record['id'],)).fetchone()[0]
    assert json.loads(stored) == record
