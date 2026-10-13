import csv
import io
import json
import zipfile

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
