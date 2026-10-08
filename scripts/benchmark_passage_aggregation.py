#!/usr/bin/env python3
"""Replay the frozen 2026-10-08 passage aggregation experiment, offline only.

Requires the local audit caches listed in the associated report. This script
does not install coefficients or write to the station database. Catalogue
agreement is not an independent measurement of physical accuracy.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.benchmark_wheel_recovery import checked_associations, summary
from vehicle_metrology.catalogue_calibration import ALPHAS, catalogue_family, fit_ridge, predict
from vehicle_metrology.catalogue_labels import corrected_rows
from vehicle_metrology.outline import outline_features
from vehicle_metrology.wheels import FEATURE_NAMES, file_digest, select_wheel_pair, wheel_features
from web_app.workbench import Profile


def supported(model, features):
    low, high = np.array(model['feature_bounds']).T
    margin = .1 * (high - low)
    return (np.isfinite(features).all()
            and np.all(features >= low - margin)
            and np.all(features <= high + margin))


def aggregate(samples, reducer):
    if not samples:
        return np.full(9, np.nan)
    features = np.array([s['features'] for s in samples])
    visibility = np.array([s['visibility'] for s in samples])
    if reducer == 'mean':
        return features.mean(0)
    if reducer == 'coordinate_median':
        return np.median(features, axis=0)
    if reducer == 'wheel_visibility_weighted':
        return np.average(features, axis=0, weights=visibility ** 2)
    if reducer == 'clearest_wheels':
        return features[np.argmax(visibility)]
    raise ValueError('Unknown reducer: ' + reducer)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output == ROOT / 'web_app/data' or ROOT / 'web_app/data' in output.parents:
        parser.error('Use an offline output directory outside web_app/data')

    paths = {key: ROOT / path for key, path in {
        'comparison': 'runs/catalogue_verify_20261008/comparison.json',
        'profile': 'config/st-wheel-calibration.json',
        'extraction': 'runs/passage_wheels_20261008/extraction.json',
        'association': 'runs/association_audit_20261008/dense/association.json',
        'baseline': 'runs/temporal_roof_20261008/current-final/results.json',
        'references': 'runs/association_audit_20261008/recovery/references.json',
        'corrections': 'config/st-catalogue-corrections.json',
        'plan': 'docs/passage-aggregation-plan-2026-10-08.json',
        'newrefs': 'docs/additional-cars-catalogue-2026-10-08.json',
        'additional': 'runs/additional_cars_20261008/results.json',
    }.items()}
    paths['script'] = Path(__file__)
    inputs = {key: file_digest(path) for key, path in paths.items()}
    recorded = json.loads((ROOT / 'docs/passage-aggregation-2026-10-08.json').read_text())
    for key, digest in inputs.items():
        if key != 'script' and digest != recorded['inputs_sha256'][key]:
            raise ValueError('Changed frozen experiment input: ' + key)

    def read(key):
        return json.loads(paths[key].read_text())

    rows = corrected_rows(
        [row for row in read('comparison')['rows'] if row.get('comparison_eligible')],
        paths['comparison'], paths['corrections'])
    profile = Profile.model_validate_json(paths['profile'].read_text())
    associations = checked_associations(
        paths['association'], paths['extraction'], paths['comparison'],
        paths['profile'], rows, profile)
    extraction = read('extraction')
    baseline = read('baseline')['rows']
    references = {row['id']: row for row in read('references')['rows']}
    intervals = np.array([row['catalogue_length_range_m'] for row in rows])
    targets = intervals.mean(1)
    groups = np.array([catalogue_family(row['source_ids']) for row in rows])
    row_ids = [row['id'] for row in rows]
    frames, selection, fit_eligible = {}, {}, []
    for row in rows:
        ident = row['id']
        samples = []
        reference = references[ident]
        for sample in extraction['rows'][ident]['samples']:
            if (associations[ident][sample['frame']]['status'] not in ('anchor', 'accepted')
                    or sample['status'] != 'ok' or sample.get('wheel_status') != 'ok'):
                continue
            for field in ('mask', 'corrected_image'):
                path = paths['extraction'].parent / sample[field]
                if file_digest(path) != sample[field + '_sha256']:
                    raise ValueError('Changed ' + field + ': ' + ident)
            with np.load(paths['extraction'].parent / sample['mask'], allow_pickle=False) as saved:
                mask = saved['mask']
            pair = select_wheel_pair(sample['wheels']['detections'], sample['expected_box'])
            features = wheel_features(outline_features(mask, profile.polygon), pair, profile.polygon)
            np.testing.assert_allclose(features, sample['features'], rtol=0, atol=1e-12)
            visibility = float(np.clip(min(w['bbox'][3] / w['bbox'][2] for w in pair), .05, 1))
            samples.append(dict(frame=sample['frame'], features=features.tolist(), visibility=visibility))
        frames[ident] = samples
        selection[ident] = dict(frames=[s['frame'] for s in samples],
                                training_eligible=reference['training_eligible'])
        fit_eligible.append(bool(reference['training_eligible'] and samples))
    fit_eligible = np.array(fit_eligible)

    def fit(features, train, alpha):
        model = fit_ridge(features[train], targets[train], groups[train], 'wheels', alpha,
                          feature_sets={'wheels': tuple(range(9))}, feature_names=FEATURE_NAMES)
        model['feature_bounds'] = np.stack([features[train].min(0), features[train].max(0)], 1).tolist()
        return model

    def select(features, training, criterion):
        scores = []
        for alpha in ALPHAS:
            errors, worst = [], []
            for group in sorted(set(groups[training])):
                test = training & (groups == group)
                train = training & ~test
                model = fit(features, train, alpha)
                predictions = predict(model, features[test])
                errors.append(float(np.mean(abs(predictions - targets[test]))))
                worst.extend(np.max(abs(predictions[:, None] - intervals[test]), axis=1))
            scores.append(dict(alpha=alpha, family_mae=float(np.mean(errors)),
                               worst_interval=float(max(worst))))
        selected = min(scores, key=lambda score: score[criterion])
        return fit(features, training, selected['alpha']), scores

    additional, newrefs = read('additional')['rows'], read('newrefs')['rows']
    result = dict(accuracy_validated=False, inputs_sha256=inputs, selection=selection,
                  baseline_summary=summary(np.array([baseline[i]['runtime_m'] for i in row_ids]),
                                           intervals, groups), methods={})
    for reducer in read('plan')['reducers']:
        features = np.array([aggregate(frames[ident], reducer) for ident in row_ids])
        for criterion in ('family_mae', 'worst_interval'):
            values = np.array([baseline[i]['runtime_m'] for i in row_ids])
            records, folds = [], []
            for group in sorted(set(groups)):
                train = (groups != group) & fit_eligible
                model, scores = select(features, train, criterion)
                folds.append(dict(heldout_group=str(group), model=model, inner_scores=scores,
                                  training_ids=[i for i, include in zip(row_ids, train) if include]))
                for index in np.flatnonzero(groups == group):
                    ident = row_ids[index]
                    reasons = baseline[ident]['measurement'].get('temporal', {}).get('reasons', [])
                    eligible = bool(supported(model, features[index])
                                    and 'ambiguous_vehicle_association' not in reasons)
                    raw = (float(predict(model, features[index:index + 1])[0])
                           if np.isfinite(features[index]).all() else None)
                    if eligible:
                        values[index] = raw
                    records.append(dict(
                        id=ident, available_views=len(frames[ident]), supported=eligible,
                        features=features[index].tolist() if np.isfinite(features[index]).all() else None,
                        raw_m=raw, runtime_m=float(values[index]), baseline_m=baseline[ident]['runtime_m'],
                        catalogue_interval_m=intervals[index].tolist()))
            model, scores = select(features, fit_eligible, criterion)
            new = []
            for ref in newrefs:
                ident = ref['id']
                anchor = np.array(additional[ident]['measurement'].get('wheels', {}).get('features', []))
                valid = len(anchor) == 9 and supported(model, anchor)
                prediction = float(predict(model, anchor[None, :])[0]) if len(anchor) == 9 else None
                new.append(dict(id=ident, raw_single_view_m=prediction, supported=bool(valid),
                                baseline_m=additional[ident]['length_m'],
                                catalogue_interval_m=ref['catalogue_interval_m'],
                                conditional_stock_body_comparison=ref['conditional_stock_body_comparison']))
            key = reducer + '_' + criterion
            stats = summary(values, intervals, groups)
            result['methods'][key] = dict(summary=stats, folds=folds,
                rows=sorted(records, key=lambda record: record['id']), all_fitted_model=model,
                new_anchor_stress_check=new)
            print(key, json.dumps(stats), flush=True)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'results.json').write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
