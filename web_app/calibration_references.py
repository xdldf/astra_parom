"""Human-verified vehicle references, independent of camera estimates."""
import hashlib
import json
from functools import lru_cache
from fastapi import HTTPException

PREFIX = 'calibration_reference:'
APPROVED = {'Подтвержден', 'Оплачен'}
revision = 0


def invalidate():
    global revision
    revision += 1
    approved_snapshot.cache_clear()


@lru_cache(maxsize=128)
def approved_snapshot(database, generation, ids):
    from web_app import station
    placeholders=','.join('?' for _ in ids)
    with station.connect() as db:
        rows=db.execute(f'SELECT value FROM settings WHERE key IN ({placeholders})',
                        tuple(PREFIX+i for i in ids)).fetchall()
        result={}
        for row in rows:
            item=json.loads(row['value']);ident=item['reference']['vehicle_id']
            record=station.find(db,ident)
            if record['status'] in APPROVED and record.get('calibration_reference')==item:
                result[ident]=item
        return result


def eligible(profile):
    """Cheap cached eligibility check for running streams; never add new references."""
    from web_app import station
    references=profile.references
    ids=tuple(sorted({r.vehicle_id for r in references if r.vehicle_id}))
    if not ids:
        return references
    allowed=approved_snapshot(str(station.DB),revision,ids)
    key=geometry_key(profile)
    return [r for r in references if not r.vehicle_id or
            allowed.get(r.vehicle_id)==dict(geometry=key,reference=r.model_dump(mode='json'))]


def geometry_key(profile):
    # References change scale, not the coordinate system. Lens/road changes do.
    return hashlib.sha256(json.dumps(dict(image_size=profile.image_size,lens=profile.lens.model_dump(),
        polygon=profile.polygon), sort_keys=True).encode()).hexdigest()


def merge(profile):
    from web_app import station, workbench
    references = [r for r in profile.references if not r.vehicle_id]
    key = geometry_key(profile)
    with station.connect() as db:
        rows = db.execute("SELECT value FROM settings WHERE key LIKE ?", (PREFIX+'%',)).fetchall()
        for row in rows:
            item = json.loads(row['value'])
            if item['geometry'] != key:
                continue
            record = station.find(db, item['reference']['vehicle_id'])
            # A stale export cannot resurrect a withdrawn or no-longer-approved reference.
            if record['status'] not in APPROVED or record.get('calibration_reference') != item:
                continue
            references.append(workbench.Reference.model_validate(item['reference']))
    if len(references) > 100:
        raise HTTPException(409, 'Больше 100 эталонов. Удалите лишние проверенные эталоны перед сохранением.')
    data = profile.model_dump()
    data['references'] = [r.model_dump() for r in references]
    try:
        return workbench.Profile.model_validate(data)
    except ValueError as exc:
        raise HTTPException(422, 'Проверенные эталоны несовместимы с геометрией калибровки: '+str(exc))
