"""Reproducible queue benchmark. Uses only a temporary database, never station data.

Run: python scripts/benchmark_station.py --cars 10000
"""
import argparse
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient
from web_app.main import app
from web_app import station


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cars',type=int,default=10000)
    args=parser.parse_args()
    if args.cars<1:parser.error('--cars must be positive')
    with tempfile.TemporaryDirectory(prefix='ferry-benchmark-') as directory:
        station.DATA=Path(directory);station.DB=station.DATA/'station.sqlite3'
        with station.connect() as db:
            for index in range(args.cars):
                record=station.new_record(station.Fields(category='car',length_m=4.5,plate=str(index)),
                                          source={'calibration':{'fixture':'x'*4096}})
                db.execute('INSERT INTO vehicles VALUES(?,?,?,?)',(record['id'],None,1,json.dumps(record)))
        client=TestClient(app)
        # Reproduce the former unfiltered history scan/serialization for comparison.
        start=time.perf_counter()
        with station.connect() as db:
            old=[json.loads(row['data']) for row in db.execute('SELECT data FROM vehicles ORDER BY rowid DESC')]
        old_body=json.dumps(old).encode()
        old_ms=(time.perf_counter()-start)*1000
        timings=[]
        for _ in range(7):
            start=time.perf_counter();response=client.get('/api/station/vehicles')
            response.raise_for_status();timings.append((time.perf_counter()-start)*1000)
        unchanged=[]
        for _ in range(7):
            start=time.perf_counter()
            hit=client.get('/api/station/vehicles',headers={'If-None-Match':response.headers['etag']})
            assert hit.status_code==304
            unchanged.append((time.perf_counter()-start)*1000)
        print(json.dumps(dict(cars=args.cars,old_scan_and_serialize_ms=round(old_ms,2),
            old_response_bytes=len(old_body),page_rows=len(response.json()['rows']),
            page_response_bytes=len(response.content),page_median_ms=round(statistics.median(timings),2),
            unchanged_response_bytes=len(hit.content),unchanged_median_ms=round(statistics.median(unchanged),2)),indent=2))


if __name__=='__main__':main()
