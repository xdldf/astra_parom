"""Auditable visual-reference corrections, separate from frozen image evidence."""
import copy
import hashlib
import json
from pathlib import Path


def corrected_rows(rows, comparison_path, correction_path):
    if correction_path is None:
        return rows
    correction_path, comparison_path = Path(correction_path), Path(comparison_path)
    data = json.loads(correction_path.read_text())
    if data['version'] != 1 or data['base_comparison_sha256'] != hashlib.sha256(comparison_path.read_bytes()).hexdigest():
        raise ValueError('Catalogue correction does not match the frozen comparison')
    updated = copy.deepcopy(rows)
    indexed = {r['id']: r for r in updated}
    seen = set()
    for change in data['corrections']:
        ident = change['id']
        if ident in seen or ident not in indexed:
            raise ValueError('Duplicate or missing corrected passage')
        seen.add(ident)
        row = indexed[ident]
        if row['source_ids'] != change['previous_source_ids'] or row['model_candidate'] != change['previous_model_candidate']:
            raise ValueError('Previous catalogue identity differs from reviewed evidence')
        image_path = comparison_path.parent/row['image']
        if hashlib.sha256(image_path.read_bytes()).hexdigest() != change['corrected_image_sha256']:
            raise ValueError('Reviewed corrected frame has changed')
        sources = [data['sources'][key] for key in change['source_ids']]
        row.update(model_candidate=change['model_candidate'], source_ids=change['source_ids'],
                   catalogue_sources=sources,
                   catalogue_length_range_m=[min(s['length_m'][0] for s in sources), max(s['length_m'][1] for s in sources)],
                   review_note=change['review_note'], stock_body_uncertain=change['stock_body_uncertain'],
                   condition_note=change['condition_note'],
                   reference_correction=dict(previous_model_candidate=change['previous_model_candidate'],
                       previous_source_ids=change['previous_source_ids'],
                       correction_sha256=hashlib.sha256(correction_path.read_bytes()).hexdigest(),
                       review_method=data['review_method'], visual_references=data['visual_references']))
    return updated
