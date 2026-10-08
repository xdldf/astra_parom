#!/usr/bin/env python3
"""Fit/check an image-only estimator using internet dimensions, without a board.

Writes an offline experiment, never a station profile or physical validation.
"""
import argparse
import hashlib
from html import escape
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vehicle_metrology.catalogue_calibration import (
    catalogue_family, comparison_metrics, image_features, nested_predictions,
    select_model,
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_html(result, output):
    """Static, script-free report; every image is the original corrected frame."""
    from urllib.parse import quote
    summary = result['summary']
    table = []
    for name, label in [('existing', 'Текущий метод'), ('nested_catalogue', 'Обучение по каталогам')]:
        item = summary[name]
        table.append(f'<tr><td>{label}</td><td>{100*item["midpoint_mae_m"]:.1f} см</td>'
                     f'<td>{item["outside_10cm_even_optimistically"]} / {item["count"]}</td>'
                     f'<td>{100*item["optimistic_interval_max_m"]:.1f} см</td></tr>')
    cases = []
    for row in result['rows']:
        lo, hi = row['catalogue_interval_m']
        url = quote(os.path.relpath(row['image'], output.parent), safe='/')
        sources = ' · '.join(f'<a href="{escape(s["url"], quote=True)}">{escape(s["model"])}</a>'
                             for s in row['sources'])
        cases.append(f'<details><summary>{escape(row["id"])} — {escape(row["model_candidate"])}'
                     f' — {row["existing_length_m"]:.3f} → {row["heldout_length_m"]:.3f} м; '
                     f'каталог {lo:.3f}–{hi:.3f} м</summary>'
                     f'<p>Семейство {escape(row["family"])} полностью исключено из обучения этой оценки. '
                     f'{sources}</p><a href="{url}"><img loading="lazy" src="{url}" '
                     f'alt="Полный исправленный кадр {escape(row["id"])}"></a></details>')
    limitations = ''.join(f'<li>{escape(item)}</li>' for item in result['limitations'])
    introduction = (f'<p>Без калибровочной доски и высот отбойников. {summary["existing"]["count"]} проездов, '
                    f'{summary["existing"]["family_count"]} семейств. Для каждой проверяемой машины всё её '
                    'семейство исключено и из обучения, и из выбора настроек.</p>'
                    '<p>Сравнение с заводскими размерами предполагаемых моделей. Точность реальных '
                    'габаритов не подтверждена. '
                    f'{summary["nested_catalogue"]["outside_10cm_even_optimistically"]} случаев всё ещё '
                    'расходятся с каталогом более чем на 10 см. Прототип не подключён к рабочим измерениям.</p>')
    output.write_text('''<!doctype html><html lang="ru"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Обучение по каталожным длинам</title><style>
body{font:17px/1.5 system-ui;margin:30px auto;padding:0 20px;max-width:1100px;color:#17202a;background:#f7f9fb}
h1{font-size:28px}table{border-collapse:collapse;width:100%;margin:20px 0}td,th{padding:12px;text-align:left;border-bottom:1px solid #cdd6df}
details{background:white;border:1px solid #dce3e9;border-radius:8px;padding:12px;margin:10px 0}summary{cursor:pointer}
img{display:block;width:100%;height:auto}a{color:#075fab}li{margin:6px 0}
</style><h1>Обучение по каталожным длинам</h1>'''
        + introduction + '''<table><thead><tr><th>Метод</th><th>Среднее абсолютное отклонение от середины диапазона</th>
<th>Более 10 см от всего диапазона</th><th>Максимальное расстояние до диапазона</th></tr></thead><tbody>'''
        + ''.join(table) + '</tbody></table><p>Кадры раскрываются целиком, после коррекции объектива.</p>'
        + ''.join(cases) + '<h2>Условия эксперимента</h2><ul>' + limitations + '</ul></html>', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True,
                        help='Frozen scan manifest containing the exact image profile')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    comparison = json.loads(args.comparison.read_text())
    profile = json.loads(args.manifest.read_text())['profile']
    rows = [r for r in comparison['rows'] if r.get('comparison_eligible')]
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate passage IDs')
    if any(r.get('image_space') != 'corrected_full_resolution' for r in rows):
        raise ValueError('Only full-resolution corrected-image observations supported')
    intervals = np.asarray([r['catalogue_length_range_m'] for r in rows], float)
    if intervals.shape != (len(rows), 2) or not np.isfinite(intervals).all() or (
            np.any(intervals[:, 0] <= 0) or np.any(intervals[:, 1] < intervals[:, 0])):
        raise ValueError('Invalid candidate catalogue ranges')
    groups = np.asarray([catalogue_family(r['source_ids']) for r in rows])
    features = np.asarray([image_features(r, profile['polygon'], profile['image_size'][0]) for r in rows])
    target = intervals.mean(axis=1)
    predictions, folds = nested_predictions(features, target, groups)
    for fold in folds:
        fold['heldout_ids'] = [rows[i]['id'] for i in fold.pop('heldout_indices')]
        fold['training_ids'] = [r['id'] for r, g in zip(rows, groups) if g != fold['heldout_group']]
    model, selection = select_model(features, target, groups)
    baseline = np.asarray([r['length_m'] for r in rows], float)
    result = dict(schema_version=1, artifact_type='catalogue_learning_benchmark',
        method='Nested leave-one-model-family-out; image-only ridge regression; interval midpoints as training targets',
        constraints=dict(calibration_board=False, barrier_heights=False, physical_vehicle_lengths=False),
        accuracy_validated=False, automatic_deployment=False,
        input_sha256=dict(comparison=digest(args.comparison), manifest=digest(args.manifest),
                          script=digest(Path(__file__)),
                          implementation=digest(ROOT/'vehicle_metrology/catalogue_calibration.py')),
        limitations=[
            'Provisional visual identity and catalogue intervals, not physical ground truth.',
            'Midpoints are declared training targets, not claims about exact filmed variants.',
            'All passages of related model families stay together; uncertain identities can still cause unnoticed leakage.',
            'Nested heldout predictions evaluate this fixed small-data learning procedure. Future model development needs fresh video.',
            'Existing corrected boxes, polygon and temporal sample selection are frozen; camera/lens/quality gates are not refitted.',
            f'Only eligible body candidates are scored; missed cars, trailers, loads and {len(comparison["rows"])-len(rows)} unscored tracks are not solved.',
            'Existing live estimates may include legacy reference exposure and are a descriptive comparator.',
            'The catalogue range is a convex envelope; interior values need not correspond to an actual trim.',
            'Image path slope and box height are proxies, not recovered 3D heading or height.',
            'The model fitted on all rows is an offline candidate; its training residuals are not reported as validation.'
        ],
        summary=dict(existing=comparison_metrics(baseline, intervals, groups),
                     nested_catalogue=comparison_metrics(predictions, intervals, groups)),
        folds=folds,
        rows=[dict(id=r['id'], family=str(g), model_candidate=r['model_candidate'],
                   image=str((args.comparison.parent/r['image']).resolve()),
                   sources=r['catalogue_sources'], catalogue_interval_m=interval.tolist(),
                   existing_length_m=float(old), heldout_length_m=float(new),
                   image_features=f.tolist())
              for r, g, interval, old, new, f in zip(rows, groups, intervals, baseline, predictions, features)])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir/'results.json').write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
    candidate = dict(schema_version=1, artifact_type='offline_catalogue_regression',
                     coordinate_profile={k: profile[k] for k in ('image_size', 'lens', 'polygon')},
                     reference_source='catalogue', accuracy_validated=False,
                     station_profile_compatible=False, model=model, selection=selection,
                     training_ids=[r['id'] for r in rows],
                     training_reference_file=str((args.output_dir/'results.json').resolve()),
                     benchmark_sha256=digest(args.output_dir/'results.json'))
    (args.output_dir/'candidate.json').write_text(json.dumps(candidate, indent=2, ensure_ascii=False)+'\n')
    write_html(result, args.output_dir/'report.html')
    print(json.dumps(result['summary'], indent=2))


if __name__ == '__main__':
    main()
