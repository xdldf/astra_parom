"""SQLite projection for bounded queue reads, maintained atomically by triggers."""
import sqlite3
import threading

SCHEMA_LOCK = threading.Lock()


class Connection(sqlite3.Connection):
    def __exit__(self, *args):
        try:
            return super().__exit__(*args)
        finally:
            self.close()


def connect_database(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=15, factory=Connection)
    db.row_factory = sqlite3.Row
    try:
        if db.execute('PRAGMA user_version').fetchone()[0] < 1:
            with SCHEMA_LOCK:
                if db.execute('PRAGMA user_version').fetchone()[0] < 1:
                    initialize(db)
        return db
    except Exception:
        db.close()
        raise


def initialize(db):
    # Readers keep a snapshot while an operator or camera commits a write.
    db.execute('PRAGMA journal_mode=WAL')
    db.executescript('''
        BEGIN IMMEDIATE;
        CREATE TABLE IF NOT EXISTS vehicles(id TEXT PRIMARY KEY, source_key TEXT UNIQUE, version INTEGER NOT NULL, data TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS audit(id INTEGER PRIMARY KEY, vehicle_id TEXT, timestamp TEXT, actor TEXT, action TEXT, reason TEXT, before_json TEXT, after_json TEXT);
        CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY, value TEXT);
        CREATE INDEX IF NOT EXISTS audit_vehicle ON audit(vehicle_id, id DESC);
        CREATE TABLE vehicle_list(
            seq INTEGER PRIMARY KEY, id TEXT UNIQUE NOT NULL, version INTEGER NOT NULL,
            created_at TEXT, plate TEXT, category TEXT, status TEXT, length_m REAL,
            amount_rub INTEGER, code TEXT, actor TEXT, ocr_state TEXT, ocr_text TEXT,
            load_capacity_t REAL
        );
        CREATE INDEX vehicle_list_status ON vehicle_list(status, seq DESC);
        CREATE INDEX vehicle_list_category ON vehicle_list(category, seq DESC);
        CREATE INDEX vehicle_list_date ON vehicle_list(created_at);
        CREATE INDEX vehicle_list_length ON vehicle_list(length_m);
        INSERT OR IGNORE INTO settings VALUES('vehicle_revision','0');
    ''')
    fields = ('created_at', 'plate', 'category', 'status', 'length_m',
              'tariff.amount_rub', 'tariff.code', 'actor', 'plate_ocr.state',
              'plate_ocr.candidates[0].text', 'load_capacity_t')

    def projection(alias):
        return ','.join([f'{alias}.rowid', f'{alias}.id', f'{alias}.version'] +
                        [f"json_extract({alias}.data, '$.{field}')" for field in fields])

    # Old records and audit data are preserved; no historical repricing.
    db.execute('INSERT INTO vehicle_list SELECT '+projection('v')+' FROM vehicles v')
    for action in ('INSERT', 'UPDATE'):
        db.execute(f'''CREATE TRIGGER vehicle_list_{action.lower()} AFTER {action} ON vehicles BEGIN
            INSERT OR REPLACE INTO vehicle_list VALUES({projection('NEW')});
            UPDATE settings SET value=CAST(value AS INTEGER)+1 WHERE key='vehicle_revision';
        END''')
    db.execute('''CREATE TRIGGER vehicle_list_delete AFTER DELETE ON vehicles BEGIN
        DELETE FROM vehicle_list WHERE id=OLD.id;
        UPDATE settings SET value=CAST(value AS INTEGER)+1 WHERE key='vehicle_revision';
    END''')
    db.execute('PRAGMA user_version=1')
    db.commit()


def summary(row):
    r = dict(row)
    r.pop('seq')
    r['tariff'] = dict(amount_rub=r.pop('amount_rub'), code=r.pop('code'))
    state, text = r.pop('ocr_state'), r.pop('ocr_text')
    if state:
        r['plate_ocr'] = dict(state=state, candidates=[dict(text=text)] if text else [])
    return r
