#!/usr/bin/env python3
"""Benchmark cached corrected-frame silhouettes and export a station profile.

Input masks must come from the pinned YOLO26-M-seg model at 1280 with retina
masks. This script checks their dimensions and mask provenance hashes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vehicle_metrology.catalogue_calibration import (
    catalogue_family, comparison_metrics, nested_predictions, select_model,
)
from vehicle_metrology.outline import FEATURE_NAMES, WEIGHTS_SHA256, geometry_signature, outline_features
from web_app.workbench import Profile


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--masks', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.comparison.read_text())
    profile = Profile.model_validate(json.loads(args.manifest.read_text())['profile'])
    rows = [r for r in data['rows'] if r.get('comparison_eligible')]
    if len({row['id'] for row in rows}) != len(rows):
        raise ValueError('Duplicate catalogue passage IDs')
    provenance = json.loads((args.masks/'mask-manifest.json').read_text())
    if (provenance['model_sha256'] != WEIGHTS_SHA256 or provenance['imgsz'] != 1280 or not provenance.get('retina_masks')
            or provenance['comparison_sha256'] != digest(args.comparison)
            or provenance['coordinate_manifest_sha256'] != digest(args.manifest)):
        raise ValueError('Mask provenance does not match this benchmark')
    features = []
    for row in rows:
        path = args.masks/(row['id']+'.npz')
        entry = provenance['masks'].get(row['id'], {})
        if entry.get('status') != 'ok':
            raise ValueError('Missing/failed outline for '+row['id']+'; no cases may be silently omitted')
        if entry['sha256'] != digest(path):
            raise ValueError('Mask hash mismatch: '+row['id'])
        mask = np.load(path, allow_pickle=False)['mask']
        if mask.shape[::-1] != profile.image_size:
            raise ValueError('Expected full-resolution corrected mask')
        features.append(outline_features(mask, profile.polygon))
    features = np.asarray(features)
    intervals = np.asarray([r['catalogue_length_range_m'] for r in rows], float)
    groups = np.asarray([catalogue_family(r['source_ids']) for r in rows])
    targets = intervals.mean(axis=1)
    options = dict(feature_sets={'outline_body': tuple(range(7))}, feature_names=FEATURE_NAMES)
    predictions, folds = nested_predictions(features, targets, groups, **options)
    baseline = np.asarray([r['length_m'] for r in rows])
    guarded = predictions.copy()
    unsupported = []
    for fold in folds:
        indices = fold.pop('heldout_indices')
        training = groups != fold['heldout_group']
        lo, hi = features[training].min(axis=0), features[training].max(axis=0)
        margin = .1*(hi-lo)
        for i in indices:
            ambiguous = 'ambiguous_vehicle_association' in rows[i]['measurement'].get('temporal', {}).get('reasons', [])
            if ambiguous or np.any(features[i] < lo-margin) or np.any(features[i] > hi+margin):
                guarded[i] = np.nan
                unsupported.append(rows[i]['id'])
        fold['heldout_ids'] = [rows[i]['id'] for i in indices]
        fold['training_ids'] = [r['id'] for r, yes in zip(rows, training) if yes]
    fitted, scores = select_model(features, targets, groups, **options)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    references = dict(reference_source='catalogue', physical_accuracy_validated=False,
        image_space='lens_corrected_full_frame',
        comparison_sha256=digest(args.comparison),
        rows=[dict(id=r['id'], family=str(g), model_candidate=r['model_candidate'],
                   video=Path(r['video']).name, frame=r['frame'], bbox=r['bbox'],
                   catalogue_interval_m=r['catalogue_length_range_m'], sources=r['catalogue_sources'])
              for r, g in zip(rows, groups)])
    reference_path = args.output_dir/'references.json'
    reference_path.write_text(json.dumps(references, indent=2, ensure_ascii=False)+'\n')
    calibration = dict(version=1, calibration_id='st-catalogue-outline-20261008',
        geometry_signature=geometry_signature(profile.model_dump(mode='json')), reference_source='catalogue',
        **{k: fitted[k] for k in ('mean', 'scale', 'coefficient', 'intercept')},
        feature_bounds=np.stack([features.min(axis=0), features.max(axis=0)], axis=1).tolist(),
        training_count=len(rows), training_family_count=len(set(groups)),
        reference_manifest_sha256=digest(reference_path))
    exported = Profile.model_validate({**profile.model_dump(), 'outline_calibration': calibration})
    (args.output_dir/'calibration.json').write_text(exported.model_dump_json(indent=2)+'\n')
    available = np.isfinite(guarded)
    guarded_metrics = comparison_metrics(guarded[available], intervals[available], groups[available])
    guarded_metrics.update(total_passages=len(rows), review_without_length=int((~available).sum()),
        failures_or_variant_dependent=int((~available).sum())+guarded_metrics['outside_10cm_even_optimistically']+guarded_metrics['variant_dependent'],
        note='Numeric error statistics describe the reported subset only. Review records count as unsuccessful measurements.')
    result = dict(accuracy_validated=False, dataset_role='development_catalogue_check',
        summary=dict(existing=comparison_metrics(baseline, intervals, groups),
                     outline=comparison_metrics(predictions, intervals, groups),
                     guarded_runtime=guarded_metrics),
        feature_names=FEATURE_NAMES, folds=folds, selected_model=fitted, inner_scores=scores,
        heldout_review_ids=unsupported,
        limitations=[
            'Catalogue identities and trims are provisional; no physical length guarantee.',
            'Outline features were developed on these videos after observing earlier errors; this is development evidence, not a fresh final test.',
            'Nested family exclusion selects regularization without the excluded family labels.',
            'Single corrected capture frame; the retained temporal baseline is a separate estimate.',
            'Guarded runtime saves a review without numeric length outside training feature ranges or with ambiguous temporal identity; it does not silently restore the old estimate.',
            'All 78 eligible passages are included; missed/unidentified/custom/loaded vehicles remain outside catalogue scoring.'
        ],
        inputs_sha256=dict(comparison=digest(args.comparison), coordinate_manifest=digest(args.manifest),
                           masks=digest(args.masks/'mask-manifest.json'),
                           script=digest(Path(__file__)), implementation=digest(ROOT/'vehicle_metrology/outline.py')),
        rows=[dict(id=r['id'], catalogue_interval_m=r['catalogue_length_range_m'],
                   baseline_m=float(a), heldout_outline_m=float(b), heldout_runtime_m=float(c) if np.isfinite(c) else None)
              for r, a, b, c in zip(rows, baseline, predictions, guarded)])
    (args.output_dir/'results.json').write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps(result['summary'], indent=2))


if __name__ == '__main__':
    main()
