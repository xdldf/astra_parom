#!/usr/bin/env python3
"""Fit broad-position wheel features and replay the real primary/recovery policy.

Catalogue families are excluded from both coefficient fitting and regularization
selection. The primary result is preserved; only its outline feature-range
rejections may use recovery. Every eligible passage stays in the denominator.
"""
import argparse
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vehicle_metrology.catalogue_calibration import catalogue_family, comparison_metrics
from vehicle_metrology.catalogue_labels import corrected_rows
from vehicle_metrology.outline import outline_features, geometry_signature, WEIGHTS_SHA256
from vehicle_metrology.wheels import select_wheel_pair, wheel_features, file_digest, MODEL_FILES, MODEL_REVISION
from vehicle_metrology.wheel_recovery import select_passage_frames, select_frame_model
from web_app.workbench import Profile
from web_app.outline_measurement import apply_outline
from web_app.wheel_measurement import apply_wheels


def summary(values, intervals, groups):
    available = np.isfinite(values)
    result = comparison_metrics(values[available], intervals[available], groups[available])
    result.update(total_passages=len(values), review_without_length=int((~available).sum()),
                  unsuccessful_or_variant_dependent=len(values)-result['within_10cm_for_entire_interval'])
    return result


def calibration(model, profile, reference_hash, ident):
    return dict(version=1, calibration_id=ident, geometry_signature=geometry_signature(profile.model_dump(mode='json')),
        reference_source='catalogue', reference_manifest_sha256=reference_hash,
        **{k:v for k,v in model.items() if k != 'alpha'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ('comparison','manifest','masks','detections','primary-profile','primary-results',
                   'passage-extraction','catalogue-corrections','output-dir'):
        parser.add_argument('--'+option, type=Path, required=True)
    args = parser.parse_args()
    rows = corrected_rows([r for r in json.loads(args.comparison.read_text())['rows'] if r.get('comparison_eligible')],
                          args.comparison, args.catalogue_corrections)
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate passage IDs')
    profile = Profile.model_validate_json(args.primary_profile.read_text())
    extraction = json.loads(args.passage_extraction.read_text())
    primary = json.loads(args.primary_results.read_text())
    masks = json.loads((args.masks/'mask-manifest.json').read_text())
    wheels = json.loads(args.detections.read_text())
    geometry = geometry_signature(profile.model_dump(mode='json'))
    if (extraction['comparison_sha256'] != file_digest(args.comparison)
            or extraction['profile_sha256'] != file_digest(args.primary_profile)
            or geometry != geometry_signature(json.loads(args.manifest.read_text())['profile'])
            or primary['inputs_sha256']['comparison'] != file_digest(args.comparison)
            or primary['inputs_sha256']['masks'] != file_digest(args.masks/'mask-manifest.json')
            or primary['inputs_sha256']['detections'] != file_digest(args.detections)
            or primary['inputs_sha256']['corrections'] != file_digest(args.catalogue_corrections)
            or masks['model_sha256'] != WEIGHTS_SHA256 or masks['imgsz'] != 1280 or not masks['retina_masks']
            or wheels['model_sha256'] != MODEL_FILES['model.safetensors'] or wheels['revision'] != MODEL_REVISION):
        raise ValueError('Input provenance does not match the primary benchmark and passage images')
    directory = args.passage_extraction.parent
    tracks = {}
    features, targets, groups, passages, anchors = [], [], [], [], []
    failed_passages, no_pairs, frames_by_id = [], [], {}
    for row in rows:
        ident = row['id']
        if row['video'] not in tracks:
            track_path = args.comparison.parent/Path(row['video']).stem/'scout_tracks.json'
            tracks[row['video']] = {t['id']:t for t in json.loads(track_path.read_text())}
        selected = select_passage_frames(row, tracks[row['video']][ident]['observations'], profile.image_size)
        entry = extraction['rows'][ident]
        if (entry['selected_frames'] != selected or len(entry['samples']) != len(selected)
                or entry['anchor_frame'] != row['frame']):
            raise ValueError('Missing or changed frame selection: '+ident)
        failed = any(s['status'] != 'ok' for s in entry['samples'])
        if failed:
            failed_passages.append(ident)
        frame_rows = []
        for sample, expected in zip(entry['samples'], selected):
            if sample['frame'] != expected['frame'] or sample['expected_box'] != expected['bbox']:
                raise ValueError('Mismatched frame/box: '+ident)
            frame_rows.append(dict(frame=sample['frame'], bbox=sample['expected_box'], status=sample['status'],
                                   wheel_status=sample.get('wheel_status'), initial_failure=sample.get('initial_failure')))
            if sample['status'] != 'ok':
                continue
            if (file_digest(directory/sample['mask']) != sample['mask_sha256']
                    or file_digest(directory/sample['corrected_image']) != sample['corrected_image_sha256']):
                raise ValueError('Changed corrected image or mask: '+ident)
            mask = np.load(directory/sample['mask'], allow_pickle=False)['mask']
            if mask.shape[::-1] != profile.image_size:
                raise ValueError('Expected a full corrected mask')
            base = outline_features(mask, profile.polygon)
            np.testing.assert_allclose(base, sample['outline_features'], rtol=0, atol=1e-12)
            try:
                pair = select_wheel_pair(sample['wheels']['detections'], sample['expected_box'])
                value = wheel_features(base, pair, profile.polygon)
            except ValueError:
                if sample.get('wheel_status') != 'unavailable':
                    raise ValueError('Wheel availability does not reproduce: '+ident)
                no_pairs.append(dict(id=ident, frame=sample['frame']))
                continue
            if sample.get('wheel_status') != 'ok':
                raise ValueError('Unexpected accepted wheel pair: '+ident)
            np.testing.assert_allclose(value, sample['features'], rtol=0, atol=1e-12)
            if failed or row.get('stock_body_uncertain', False):
                continue
            features.append(value)
            targets.append(np.mean(row['catalogue_length_range_m']))
            groups.append(catalogue_family(row['source_ids']))
            passages.append(ident)
            anchors.append(sample['frame'] == row['frame'])
        frames_by_id[ident] = frame_rows
    features, targets, groups, passages, anchors = [np.asarray(v) for v in (features, targets, groups, passages, anchors)]
    references = dict(reference_source='catalogue', physical_accuracy_validated=False, image_space='lens_corrected_full_frame',
        comparison_sha256=file_digest(args.comparison), corrections_sha256=file_digest(args.catalogue_corrections),
        extraction_sha256=file_digest(args.passage_extraction),
        rows=[dict(id=r['id'], family=catalogue_family(r['source_ids']), model_candidate=r['model_candidate'],
            video=Path(r['video']).name, anchor_frame=r['frame'], catalogue_interval_m=r['catalogue_length_range_m'],
            sources=r['catalogue_sources'], training_eligible=bool(np.any(passages == r['id'])),
            stock_body_uncertain=r.get('stock_body_uncertain', False), frames=frames_by_id[r['id']]) for r in rows])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ref_path = args.output_dir/'references.json'
    ref_path.write_text(json.dumps(references, indent=2, ensure_ascii=False)+'\n')
    ref_hash = file_digest(ref_path)
    primary_folds = {f['heldout_group']:f for f in primary['folds']}
    old_rows = {r['id']:r for r in primary['rows']}
    image = np.zeros((*profile.image_size[::-1], 3), np.uint8)
    output, baseline, folds = {}, {}, []
    for group in sorted(set(groups)):
        train = groups != group
        model, scores = select_frame_model(features[train], targets[train], groups[train], passages[train], anchors[train])
        recovery = calibration(model, profile, ref_hash, 'wheel-recovery-heldout')
        primary_fold = primary_folds[group]
        heldout = Profile.model_validate({**profile.model_dump(),
            'outline_calibration':primary_fold['outline_fallback_calibration'],
            'wheel_calibration':primary_fold['calibration'], 'wheel_recovery_calibration':recovery})
        original = heldout.model_copy(update={'wheel_recovery_calibration':None})
        fold_rows = [r for r in rows if catalogue_family(r['source_ids']) == group]
        folds.append(dict(heldout_group=str(group), heldout_ids=[r['id'] for r in fold_rows],
            training_ids=sorted(set(passages[train])), model=model, calibration=recovery, inner_scores=scores))
        for row in fold_rows:
            ident = row['id']
            mask_path = args.masks/(ident+'.npz')
            entry, detected = masks['masks'][ident], wheels['rows'][ident]
            photo_hash = file_digest(args.comparison.parent/row['image'])
            if (entry['status'] != 'ok' or detected['status'] != 'ok' or entry['sha256'] != file_digest(mask_path)
                    or entry['corrected_image_sha256'] != photo_hash or detected['corrected_image_sha256'] != photo_hash
                    or detected['vehicle_bbox'] != row['bbox'] or detected['frame'] != row['frame']):
                raise ValueError('Changed primary anchor evidence: '+ident)
            mask = np.load(mask_path, allow_pickle=False)['mask']
            details = {k:detected[k] for k in ('detections','roi','image_space')}
            with patch('web_app.outline_measurement.infer_outline', return_value=(mask, {})), \
                 patch('web_app.wheel_measurement.infer_wheels', return_value=details):
                outline = apply_outline(heldout, row['measurement'], image, label=row['detector_label'])
                old = apply_wheels(original, outline, image, label=row['detector_label'])
                measured = apply_wheels(heldout, outline, image, label=row['detector_label'])
            expected = old_rows[ident]['heldout_runtime_m']
            if (expected is None) != (old['length_m'] is None) or (expected is not None and abs(old['length_m']-expected) > 1e-10):
                raise ValueError('Primary runtime did not reproduce: '+ident)
            baseline[ident] = old['length_m']
            output[ident] = dict(id=ident, catalogue_interval_m=row['catalogue_length_range_m'],
                previous_runtime_m=old['length_m'], runtime_m=measured['length_m'], status=measured['status'],
                wheel_status=measured.get('wheels', {}).get('status'),
                calibration_role=measured.get('wheels', {}).get('calibration_role'),
                reason=measured.get('wheels', {}).get('reason'))
            if old['length_m'] is not None and measured['length_m'] != old['length_m']:
                raise ValueError('Recovery changed an existing primary result')
    fitted, scores = select_frame_model(features, targets, groups, passages, anchors)
    exported_calibration = calibration(fitted, profile, ref_hash, 'st-wheel-recovery-20261008')
    exported = Profile.model_validate({**profile.model_dump(), 'wheel_recovery_calibration':exported_calibration})
    (args.output_dir/'calibration.json').write_text(exported.model_dump_json(indent=2)+'\n')
    values = np.asarray([output[r['id']]['runtime_m'] if output[r['id']]['runtime_m'] is not None else np.nan for r in rows])
    base_values = np.asarray([baseline[r['id']] if baseline[r['id']] is not None else np.nan for r in rows])
    intervals = np.asarray([r['catalogue_length_range_m'] for r in rows])
    row_groups = np.asarray([catalogue_family(r['source_ids']) for r in rows])
    result = dict(accuracy_validated=False, dataset_role='development_catalogue_check',
        summary=dict(primary=summary(base_values, intervals, row_groups), recovery=summary(values, intervals, row_groups)),
        recovered_ids=[r['id'] for r in rows if baseline[r['id']] is None and output[r['id']]['runtime_m'] is not None],
        failed_passages=failed_passages, frames_without_wheel_pair=no_pairs,
        training_count=fitted['training_count'], training_frame_count=fitted['training_frame_count'],
        folds=folds, selected_model=fitted, inner_scores=scores,
        limitations=['Catalogue identities/variants remain provisional; this does not prove physical accuracy.',
            'Recovery policy and image features were developed on these videos after examining their errors.',
            'Every catalogue passage remains in the denominator, including missing wheels, reviews and damaged bodies.',
            'Family exclusion prevents tested family labels entering coefficient fitting or regularization selection.',
            'Missing wheel anchors do not enter inner selection scores; they retain runtime primary/review fallback.',
            'This is anchor inference using a model trained on broader views, not a new temporal runtime estimator.',
            'No claim of complete detector recall or accuracy on the 258 unscored audit candidates.'],
        inputs_sha256=dict(comparison=file_digest(args.comparison), manifest=file_digest(args.manifest),
            extraction=file_digest(args.passage_extraction), masks=file_digest(args.masks/'mask-manifest.json'),
            detections=file_digest(args.detections), primary_results=file_digest(args.primary_results),
            primary_profile=file_digest(args.primary_profile), corrections=file_digest(args.catalogue_corrections),
            script=file_digest(Path(__file__)), fit=file_digest(ROOT/'vehicle_metrology/wheel_recovery.py'),
            outline_runtime=file_digest(ROOT/'web_app/outline_measurement.py'), wheel_runtime=file_digest(ROOT/'web_app/wheel_measurement.py')),
        rows=[output[r['id']] for r in rows])
    (args.output_dir/'results.json').write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    print(json.dumps(result['summary'], indent=2))


if __name__ == '__main__':
    main()
