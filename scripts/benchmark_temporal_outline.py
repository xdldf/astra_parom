#!/usr/bin/env python3
"""Compare single and multi-frame silhouettes without changing the station.

Uses frozen, already associated temporal observations. Catalogue identities and
lengths do not affect frame selection, segmentation, or feature aggregation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vehicle_metrology.catalogue_calibration import catalogue_family, comparison_metrics, nested_predictions
from vehicle_metrology.outline import FEATURE_NAMES, WEIGHTS_SHA256, outline_features
from vehicle_metrology.temporal import _line_fit
from web_app import workbench as wb
from web_app.outline_measurement import infer_outline
from vehicle_metrology.catalogue_labels import corrected_rows


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def select_frames(row):
    """At most five real observations spanning the accepted band, with anchor."""
    observations = {s['frame']: dict(frame=s['frame'], bbox=s['bbox'])
                    for s in row.get('measurement', {}).get('temporal', {}).get('samples', [])}
    observations[row['frame']] = dict(frame=row['frame'], bbox=row['bbox'])
    indices = sorted(observations)
    chosen = sorted({indices[int(round(i))] for i in np.linspace(0, len(indices)-1, min(5, len(indices)))})
    if row['frame'] not in chosen:
        nearest = min(range(len(chosen)), key=lambda i: abs(chosen[i]-row['frame']))
        chosen[nearest] = row['frame']
    return [observations[i] for i in sorted(chosen)]


def aggregate(samples, anchor_frame, method, image_width):
    """Missing frames are errors, never an opportunity to select a better subset."""
    if not samples or any(s.get('status') != 'ok' for s in samples):
        raise ValueError('All selected silhouettes must be available')
    if len({s['frame'] for s in samples}) != len(samples):
        raise ValueError('Repeated frame')
    anchor = next(s for s in samples if s['frame'] == anchor_frame)
    values = np.asarray([s['features'] for s in samples], float)
    if values.shape != (len(samples), 7) or not np.isfinite(values).all():
        raise ValueError('Expected seven finite features for every frame')
    if method == 'anchor':
        return np.asarray(anchor['features'], float)
    if method == 'mean':
        return values.mean(axis=0)
    if method == 'median':
        return np.median(values, axis=0)
    if method == 'local_fit':
        offsets = (values[:, 3] - anchor['features'][3]) * image_width
        return np.asarray([_line_fit(offsets, values[:, i])[0] for i in range(7)])
    raise ValueError('Unknown aggregation')


def extract(args, rows, profile):
    import torch
    cv2.setNumThreads(2)
    torch.set_num_threads(4)
    destination = args.output_dir
    destination.mkdir(parents=True, exist_ok=True)
    metadata_path = destination/'extraction.json'
    sources = sorted({r['video'] for r in rows})
    signature = dict(comparison=digest(args.comparison), coordinates=digest(args.manifest),
                     model=WEIGHTS_SHA256, script=digest(__file__),
                     outline_features=digest(ROOT/'vehicle_metrology/outline.py'),
                     runtime_inference=digest(ROOT/'web_app/outline_measurement.py'),
                     videos={p: digest(p) for p in sources})
    saved = dict(inputs_sha256=signature, device=args.device, image_space='lens_corrected_full_frame',
                 imgsz=1280, retina_masks=True, rows={})
    if metadata_path.exists():
        saved = json.loads(metadata_path.read_text())
        if saved['inputs_sha256'] != signature:
            raise ValueError('Cached extraction uses different inputs or implementation; use a new output directory')
    capture = None
    current_video = None
    try:
        for count, row in enumerate(rows, 1):
            if row['id'] in saved['rows']:
                continue
            if current_video != row['video']:
                if capture is not None:
                    capture.release()
                capture = cv2.VideoCapture(row['video'])
                current_video = row['video']
            selected = select_frames(row)
            wanted = {s['frame']: s for s in selected}
            samples = []
            capture.set(cv2.CAP_PROP_POS_FRAMES, min(wanted))
            for index in range(min(wanted), max(wanted)+1):
                ok, raw = capture.read()
                if index not in wanted:
                    continue
                sample = dict(frame=index, expected_box=wanted[index]['bbox'], status='failed')
                try:
                    if not ok or raw.shape[1::-1] != profile.image_size:
                        raise ValueError('Cannot decode original full-resolution video frame')
                    corrected = wb.corrected(raw, profile.lens)
                    mask, details = infer_outline(corrected, sample['expected_box'], device=args.device)
                    features = outline_features(mask, profile.polygon)
                    stem = f"{row['id']}-{index}"
                    mask_path, image_path = destination/(stem+'.npz'), destination/(stem+'.jpg')
                    np.savez_compressed(mask_path, mask=mask)
                    if not cv2.imwrite(str(image_path), corrected, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                        raise ValueError('Could not save corrected evidence')
                    sample.update(status='ok', features=features.tolist(), **details,
                                  mask=mask_path.name, mask_sha256=digest(mask_path),
                                  corrected_image=image_path.name, corrected_image_sha256=digest(image_path),
                                  inference_pixels_sha256=hashlib.sha256(corrected.tobytes()).hexdigest())
                except Exception as exc:
                    sample['reason'] = str(exc)
                samples.append(sample)
            saved['rows'][row['id']] = dict(anchor_frame=row['frame'], selected_frames=selected, samples=samples)
            write_json(metadata_path, saved)
            print(f"{count}/{len(rows)} {row['id']}: {sum(s['status']=='ok' for s in samples)}/{len(samples)} outlines", flush=True)
    finally:
        if capture is not None:
            capture.release()
    return saved


def benchmark(args, rows, profile):
    extraction = json.loads((args.output_dir/'extraction.json').read_text())
    inputs = extraction['inputs_sha256']
    if (inputs['comparison'] != digest(args.comparison) or inputs['coordinates'] != digest(args.manifest)
            or inputs['model'] != WEIGHTS_SHA256 or inputs['outline_features'] != digest(ROOT/'vehicle_metrology/outline.py')):
        raise ValueError('Mismatched extraction provenance')
    for row in rows:
        cached = extraction['rows'][row['id']]
        if cached['selected_frames'] != select_frames(row):
            raise ValueError('Frame selection has changed')
        if [s['frame'] for s in cached['samples']] != [s['frame'] for s in cached['selected_frames']]:
            raise ValueError('Missing selected frames')
        for sample in cached['samples']:
            if sample['status'] != 'ok':
                raise ValueError('Selected mask failed; retain the failure and do not silently exclude this passage')
            for key, hash_key in [('mask', 'mask_sha256'), ('corrected_image', 'corrected_image_sha256')]:
                if digest(args.output_dir/sample[key]) != sample[hash_key]:
                    raise ValueError('Changed evidence file')
            with np.load(args.output_dir/sample['mask'], allow_pickle=False) as stored:
                mask = stored['mask']
                if mask.shape[1::-1] != profile.image_size:
                    raise ValueError('Mask is not full resolution')
                if not np.array_equal(outline_features(mask, profile.polygon), sample['features']):
                    raise ValueError('Features do not reproduce from mask')
    rows = corrected_rows(rows, args.comparison, args.catalogue_corrections)
    intervals = np.asarray([r['catalogue_length_range_m'] for r in rows], float)
    groups = np.asarray([catalogue_family(r['source_ids']) for r in rows])
    training_eligible = np.asarray([not r.get('stock_body_uncertain', False) for r in rows])
    options = dict(feature_sets={'outline_body': tuple(range(7))}, feature_names=FEATURE_NAMES)
    output = dict(accuracy_validated=False, dataset_role='development_catalogue_check',
                  corrections_sha256=digest(args.catalogue_corrections) if args.catalogue_corrections else None,
                  training_ineligible_ids=[r['id'] for r,e in zip(rows,training_eligible) if not e],
                  extraction_sha256=digest(args.output_dir/'extraction.json'),
                  benchmark_script_sha256=digest(__file__), methods={},
                  limitations=['Catalogue identities/variants are provisional; no physical accuracy guarantee.',
                      'All methods use the same 78 cases and split families. Method comparison is development evidence.',
                      'Ten passages have no accepted temporal neighbours; they remain as singletons, not exclusions.',
                      'Existing temporal association and applicability reviews remain unsuccessful automatic measurements.'])
    for method in ('anchor', 'mean', 'median', 'local_fit'):
        features = np.asarray([aggregate(extraction['rows'][r['id']]['samples'], r['frame'], method, profile.image_size[0]) for r in rows])
        predictions, folds = nested_predictions(features, intervals.mean(axis=1), groups, training_eligible=training_eligible, **options)
        reviews = []
        for fold in folds:
            indices = fold.pop('heldout_indices')
            training = (groups != fold['heldout_group']) & training_eligible
            lo, hi = features[training].min(axis=0), features[training].max(axis=0)
            margin = .1*(hi-lo)
            for i in indices:
                ambiguous = 'ambiguous_vehicle_association' in rows[i]['measurement'].get('temporal', {}).get('reasons', [])
                if ambiguous or np.any(features[i] < lo-margin) or np.any(features[i] > hi+margin):
                    reviews.append(rows[i]['id'])
            fold['heldout_ids'] = [rows[i]['id'] for i in indices]
        available = np.asarray([r['id'] not in reviews for r in rows])
        summary = comparison_metrics(predictions, intervals, groups)
        guarded = comparison_metrics(predictions[available], intervals[available], groups[available])
        guarded.update(total_passages=len(rows), review_without_length=len(reviews))
        output['methods'][method] = dict(summary=summary, guarded_runtime=guarded, review_ids=reviews, folds=folds,
            rows=[dict(id=r['id'], catalogue_interval_m=r['catalogue_length_range_m'],
                       predicted_m=float(p), runtime_m=float(p) if a else None,
                       sample_count=len(extraction['rows'][r['id']]['samples'])) for r,p,a in zip(rows,predictions,available)])
        print(method, json.dumps(summary), flush=True)
    write_json(args.output_dir/'results.json', output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['extract', 'benchmark'])
    parser.add_argument('--comparison', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda:0'], default='cuda:0')
    parser.add_argument('--catalogue-corrections', type=Path)
    args = parser.parse_args()
    rows = [r for r in json.loads(args.comparison.read_text())['rows'] if r.get('comparison_eligible')]
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate passage IDs')
    profile = wb.Profile.model_validate(json.loads(args.manifest.read_text())['profile'])
    if args.command == 'extract':
        extract(args, rows, profile)
    else:
        benchmark(args, rows, profile)


if __name__ == '__main__':
    main()
