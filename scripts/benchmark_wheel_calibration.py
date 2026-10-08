#!/usr/bin/env python3
"""Family-heldout development benchmark, including actual capture fallback guards.

All catalogue passages stay in the denominator. Absent wheel pairs are excluded
from wheel fitting and fall back to the independently family-heldout outline.
This is catalogue comparison, not validation of physical measurement accuracy.
"""
import argparse
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vehicle_metrology.catalogue_calibration import catalogue_family, comparison_metrics, nested_predictions, select_model
from vehicle_metrology.catalogue_labels import corrected_rows
from vehicle_metrology.outline import FEATURE_NAMES as OUTLINE_NAMES, WEIGHTS_SHA256, geometry_signature, outline_features
from vehicle_metrology.wheels import (FEATURE_NAMES, MODEL_ID, MODEL_REVISION, MODEL_FILES,
    PROMPT, THRESHOLD, file_digest, select_wheel_pair, wheel_features)
from web_app.workbench import Profile
from web_app.outline_measurement import apply_outline
from web_app.wheel_measurement import apply_wheels


def model_calibration(model, features, profile, reference_hash, name):
    return dict(version=1, calibration_id=name, geometry_signature=geometry_signature(profile.model_dump(mode='json')),
        reference_source='catalogue', **{k: model[k] for k in ('mean', 'scale', 'coefficient', 'intercept')},
        feature_bounds=np.stack([features.min(0), features.max(0)], axis=1).tolist(),
        training_count=model['training_count'], training_family_count=len(model['training_groups']),
        reference_manifest_sha256=reference_hash)


def guarded_metrics(predictions, intervals, groups):
    available = np.isfinite(predictions)
    metrics = comparison_metrics(predictions[available], intervals[available], groups[available]) if available.any() else {}
    metrics.update(total_passages=len(predictions), review_without_length=int((~available).sum()),
        failures_or_variant_dependent=len(predictions)-metrics.get('within_10cm_for_entire_interval', 0),
        note='Numeric errors describe the reported subset. Reviews remain unsuccessful measurements in the denominator.')
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ('comparison', 'manifest', 'masks', 'detections', 'outline-profile', 'catalogue-corrections', 'output-dir'):
        parser.add_argument('--'+option, type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.comparison.read_text())
    rows = corrected_rows([r for r in data['rows'] if r.get('comparison_eligible')], args.comparison, args.catalogue_corrections)
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate catalogue passage IDs')
    profile = Profile.model_validate_json(args.outline_profile.read_text())
    source_profile = Profile.model_validate(json.loads(args.manifest.read_text())['profile'])
    if geometry_signature(profile.model_dump(mode='json')) != geometry_signature(source_profile.model_dump(mode='json')):
        raise ValueError('Profile differs from the corrected audit geometry')
    masks = json.loads((args.masks/'mask-manifest.json').read_text())
    wheels = json.loads(args.detections.read_text())
    common = dict(comparison_sha256=file_digest(args.comparison), coordinate_manifest_sha256=file_digest(args.manifest))
    expected_masks = dict(**common, model_sha256=WEIGHTS_SHA256, imgsz=1280, retina_masks=True)
    expected_wheels = dict(**common, model=MODEL_ID, revision=MODEL_REVISION,
        model_sha256=MODEL_FILES['model.safetensors'], text=PROMPT, threshold=THRESHOLD, text_threshold=THRESHOLD)
    for actual, expected in ((masks, expected_masks), (wheels, expected_wheels)):
        if any(actual.get(k) != v for k, v in expected.items()):
            raise ValueError('Detector provenance does not match this benchmark')
    outline, features, pairs = [], [], {}
    for row in rows:
        ident = row['id']
        mask_path = args.masks/(ident+'.npz')
        entry, detected = masks['masks'].get(ident, {}), wheels['rows'].get(ident, {})
        image_hash = file_digest(args.comparison.parent/row['image'])
        if (row.get('image_space') != 'corrected_full_resolution' or entry.get('status') != 'ok'
                or detected.get('status') != 'ok' or entry.get('sha256') != file_digest(mask_path)
                or entry.get('corrected_image_sha256') != image_hash or detected.get('corrected_image_sha256') != image_hash
                or detected.get('vehicle_bbox') != row['bbox'] or detected.get('frame') != row['frame']
                or detected.get('image_space') != 'lens_corrected_full_frame'):
            raise ValueError('Missing, mismatched or failed input: '+ident+'; no passages may be omitted')
        mask = np.load(mask_path, allow_pickle=False)['mask']
        if mask.shape[::-1] != profile.image_size:
            raise ValueError('Expected full corrected mask: '+ident)
        base = outline_features(mask, profile.polygon)
        outline.append(base)
        try:
            pair = select_wheel_pair(detected['detections'], row['bbox'])
            value = wheel_features(base, pair, profile.polygon)
            pairs[ident] = dict(status='ok', pair=pair)
        except ValueError as exc:
            value = np.full(9, np.nan)
            pairs[ident] = dict(status='unavailable', reason=str(exc))
        features.append(value)
    outline, features = np.asarray(outline), np.asarray(features)
    intervals = np.asarray([r['catalogue_length_range_m'] for r in rows], float)
    groups = np.asarray([catalogue_family(r['source_ids']) for r in rows])
    targets = intervals.mean(1)
    outline_training = np.asarray([not r.get('stock_body_uncertain', False) for r in rows])
    available = np.isfinite(features).all(axis=1)
    training = outline_training & available
    options = dict(feature_sets={'outline_wheels': tuple(range(9))}, feature_names=FEATURE_NAMES)
    outline_options = dict(feature_sets={'outline_body': tuple(range(7))}, feature_names=OUTLINE_NAMES)
    baseline, base_folds = nested_predictions(outline, targets, groups, training_eligible=outline_training, **outline_options)
    predictions, folds = nested_predictions(features, targets, groups, training_eligible=training, **options)
    predictions[~available] = baseline[~available]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    references = dict(reference_source='catalogue', physical_accuracy_validated=False,
        image_space='lens_corrected_full_frame', comparison_sha256=file_digest(args.comparison),
        corrections_sha256=file_digest(args.catalogue_corrections),
        rows=[dict(id=r['id'], family=str(g), model_candidate=r['model_candidate'], video=Path(r['video']).name,
            frame=r['frame'], bbox=r['bbox'], catalogue_interval_m=r['catalogue_length_range_m'],
            sources=r['catalogue_sources'], training_eligible=bool(yes), stock_body_uncertain=r.get('stock_body_uncertain', False),
            wheel_status=pairs[r['id']]['status'], reference_correction=r.get('reference_correction'), condition_note=r.get('condition_note'))
            for r, g, yes in zip(rows, groups, training)])
    ref_path = args.output_dir/'references.json'
    ref_path.write_text(json.dumps(references, indent=2, ensure_ascii=False)+'\n')
    ref_hash = file_digest(ref_path)
    guarded, guarded_base = np.full(len(rows), np.nan), np.full(len(rows), np.nan)
    runtime_rows = {}
    image = np.zeros((*profile.image_size[::-1], 3), np.uint8)
    for fold, base_fold in zip(folds, base_folds):
        if fold['heldout_group'] != base_fold['heldout_group']:
            raise ValueError('Mismatched validation folds')
        train = (groups != fold['heldout_group']) & training
        base_train = (groups != fold['heldout_group']) & outline_training
        calibration = model_calibration(fold['model'], features[train], profile, ref_hash, 'wheel-heldout')
        base_calibration = model_calibration(base_fold['model'], outline[base_train], profile, ref_hash, 'outline-heldout')
        heldout = Profile.model_validate({**profile.model_dump(), 'outline_calibration':base_calibration, 'wheel_calibration':calibration})
        indices = fold.pop('heldout_indices')
        fold.update(heldout_ids=[rows[i]['id'] for i in indices], training_ids=[r['id'] for r, yes in zip(rows, train) if yes],
                    calibration=calibration, outline_fallback_calibration=base_calibration)
        for i in indices:
            row, ident = rows[i], rows[i]['id']
            mask = np.load(args.masks/(ident+'.npz'), allow_pickle=False)['mask']
            details = {k:wheels['rows'][ident][k] for k in ('detections', 'roi', 'image_space')}
            with patch('web_app.outline_measurement.infer_outline', return_value=(mask, {})), \
                 patch('web_app.wheel_measurement.infer_wheels', return_value=details):
                measured = apply_outline(heldout, row['measurement'], image, label=row['detector_label'])
                guarded_base[i] = measured['length_m'] if measured['length_m'] is not None else np.nan
                measured = apply_wheels(heldout, measured, image, label=row['detector_label'])
            guarded[i] = measured['length_m'] if measured['length_m'] is not None else np.nan
            evidence = measured.get('wheels', {})
            if evidence.get('status') == 'estimated' and abs(guarded[i]-predictions[i]) > 1e-10:
                raise ValueError('Runtime differs from the heldout regression: '+ident)
            runtime_rows[ident] = dict(status=measured['status'], wheel_status=evidence.get('status'), reason=evidence.get('reason'))
    fitted, scores = select_model(features[training], targets[training], groups[training], **options)
    calibration = model_calibration(fitted, features[training], profile, ref_hash, 'st-catalogue-wheels-20261008')
    exported = Profile.model_validate({**profile.model_dump(), 'wheel_calibration':calibration})
    (args.output_dir/'calibration.json').write_text(exported.model_dump_json(indent=2)+'\n')
    result = dict(accuracy_validated=False, dataset_role='development_catalogue_check', feature_names=FEATURE_NAMES,
        summary=dict(outline=comparison_metrics(baseline, intervals, groups),
                     wheels_with_outline_fallback=comparison_metrics(predictions, intervals, groups),
                     guarded_outline=guarded_metrics(guarded_base, intervals, groups),
                     guarded_runtime=guarded_metrics(guarded, intervals, groups)),
        folds=folds, selected_model=fitted, inner_scores=scores, pairs=pairs,
        training_ineligible_ids=[r['id'] for r, yes in zip(rows, training) if not yes],
        limitations=['Catalogue identities and trims remain provisional; no physical accuracy guarantee.',
            'Features were developed after examining errors in these same six videos; not a fresh independent final test.',
            'Regularization and coefficients exclude the entire heldout family; identity is never an inference input.',
            'Damaged body and unavailable wheel pairs do not train the wheel model but remain in all comparison counts.',
            'Visible-wheel image proxies do not reconstruct ground contacts hidden by the barrier.',
            'Unidentified, missed, custom and loaded vehicles are outside catalogue scoring; detector recall is not established.'],
        inputs_sha256=dict(comparison=file_digest(args.comparison), manifest=file_digest(args.manifest),
            masks=file_digest(args.masks/'mask-manifest.json'), detections=file_digest(args.detections),
            corrections=file_digest(args.catalogue_corrections), outline_profile=file_digest(args.outline_profile),
            script=file_digest(Path(__file__)), features=file_digest(ROOT/'vehicle_metrology/wheels.py'),
            runtime=file_digest(ROOT/'web_app/wheel_measurement.py')),
        rows=[dict(id=r['id'], catalogue_interval_m=r['catalogue_length_range_m'], heldout_outline_m=float(b),
                   heldout_wheel_m=float(p), heldout_runtime_m=float(g) if np.isfinite(g) else None, **runtime_rows[r['id']])
              for r, b, p, g in zip(rows, baseline, predictions, guarded)])
    (args.output_dir/'results.json').write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    print(json.dumps(result['summary'], indent=2))


if __name__ == '__main__':
    main()
