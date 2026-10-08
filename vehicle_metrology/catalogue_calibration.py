"""Offline catalogue learning experiment, not physical camera calibration.

Image-only features predict length. Catalogue identity is used only to keep
related passages in the same validation fold, never as a prediction feature.
"""
from collections import Counter

import numpy as np

from .bbox_scale import road_cross_section
from .temporal import _line_fit


FEATURE_SETS = {
    'width_depth': (0, 1),
    'width_depth_height': (0, 1, 2),
    'position_height': (0, 1, 2, 3),
    'perspective_interactions': (0, 1, 2, 3, 4, 5),
    'motion': (0, 1, 2, 3, 4, 5, 6, 7, 8),
}
FEATURE_NAMES = (
    'width_over_road_span', 'relative_road_depth', 'height_over_road_span',
    'horizontal_position', 'width_times_depth', 'height_times_depth',
    'bottom_path_slope', 'horizontal_direction', 'motion_available',
)
ALPHAS = (.01, .1, 1., 10., 100.)


def image_features(row, polygon, image_width):
    """Aggregate saved corrected boxes without reading their estimated lengths.

    Accepted temporal samples are a frozen upstream selection. For captures
    without them, use the anchor, including off-line anchors. No row is dropped
    for being difficult. Motion is a projected path proxy, not measured yaw.
    """
    if not np.isfinite(image_width) or image_width <= 0:
        raise ValueError('Positive image width required')
    temporal = row.get('measurement', {}).get('temporal', {})
    samples = temporal.get('samples', []) if temporal.get('length_m') is not None else []
    boxes = [s['bbox'] for s in samples] if samples else [row['bbox']]
    offsets = np.asarray([s['line_offset_px'] for s in samples], float)
    features = []
    for x, y, w, h in boxes:
        if not np.isfinite([x, y, w, h]).all() or min(w, h) <= 0:
            raise ValueError('Expected finite positive corrected boxes')
        section = road_cross_section(polygon, x+w/2, y+h)
        if section is None:
            raise ValueError('A saved box has no road cross-section')
        span = section['near'][1] - section['far'][1]
        width, height, depth = w/span, h/span, section['depth']
        features.append([width, depth, height, (x+w/2)/image_width,
                         width*depth, height*depth])
    values = np.asarray(features)
    aggregate = (np.asarray([_line_fit(offsets, values[:, i])[0] for i in range(6)])
                 if samples else values[0])
    centers = np.asarray([x+w/2 for x, y, w, h in boxes])
    bottoms = np.asarray([y+h for x, y, w, h in boxes])
    available = bool(samples and len(samples) >= 3 and np.ptp(centers) >= 15)
    slope = _line_fit(centers, bottoms)[1] if available else 0.
    direction = float(np.sign(centers[-1]-centers[0])) if available else 0.
    return np.r_[aggregate, slope, direction, float(available)]


def catalogue_family(source_ids):
    """Conservative families for the reviewed ST dataset; fail on unknowns.

    All generations/facelifts of a model stay together. Noah/Voxy, Probox/
    Succeed and Vitz/Yaris are grouped across badges to avoid obvious leakage.
    This is not proof of independent physical identity for uncertain labels.
    """
    mapping = {'vitz': 'vitz_yaris', 'yaris': 'vitz_yaris',
               'noah': 'noah_voxy', 'voxy': 'noah_voxy',
               'probox': 'probox_succeed', 'succeed': 'probox_succeed'}
    supported = {'fielder', 'fit', 'freed', 'ipsum', 'patrol', 'prado',
                 'ractis', 'rav4', 'rush', 'spacio', 'wish'}
    families = set()
    for source in source_ids:
        prefix = source.split('_')[0]
        if source == 'yaris_cross':
            family = 'yaris_cross'
        elif prefix in mapping:
            family = mapping[prefix]
        elif prefix in supported:
            family = prefix
        else:
            raise ValueError(f'Assign an explicit split family for {source}')
        families.add(family)
    if len(families) != 1:
        raise ValueError('Candidate identities span different split families')
    return families.pop()


def fit_ridge(features, target, groups, feature_set, alpha, *, feature_sets=None, feature_names=None):
    """Standardization, target and all coefficients use training rows only."""
    columns = list((FEATURE_SETS if feature_sets is None else feature_sets)[feature_set])
    names = FEATURE_NAMES if feature_names is None else feature_names
    x = np.asarray(features, float)[:, columns]
    y = np.asarray(target, float)
    if len(x) != len(y) or len(groups) != len(y) or not len(y):
        raise ValueError('Training arrays must be nonempty and aligned')
    if not np.isfinite(x).all() or not np.isfinite(y).all() or not np.isfinite(alpha) or alpha <= 0:
        raise ValueError('Finite training values and positive regularization required')
    counts = Counter(groups)
    weights = np.asarray([1/counts[g] for g in groups], float)
    weights /= weights.sum()
    mean = weights @ x
    scale = np.sqrt(weights @ ((x-mean)**2))
    scale = np.where(scale > 1e-8, scale, 1.)
    z = (x-mean)/scale
    intercept = float(weights @ y)
    # Each family has total weight one. Alpha is an equivalent family count.
    fit_weights = weights*len(counts)
    coefficient = np.linalg.solve(z.T @ (z*fit_weights[:, None]) + alpha*np.eye(z.shape[1]),
                                  z.T @ ((y-intercept)*fit_weights))
    return dict(feature_set=feature_set, alpha=float(alpha), columns=columns,
                feature_names=[names[i] for i in columns], mean=mean.tolist(),
                scale=scale.tolist(), coefficient=coefficient.tolist(), intercept=intercept,
                training_groups=sorted(counts), training_count=len(y))


def predict(model, features):
    x = np.asarray(features, float)[:, model['columns']]
    return ((x-model['mean'])/model['scale']) @ model['coefficient'] + model['intercept']


def select_model(features, target, groups, *, feature_sets=None, feature_names=None):
    """Inner leave-one-family-out selection using family-balanced midpoint MAE."""
    groups = np.asarray(groups)
    if len(set(groups)) < 3:
        raise ValueError('Need at least three training families for inner validation')
    scores = []
    options = dict(feature_sets=feature_sets, feature_names=feature_names)
    for feature_set in (FEATURE_SETS if feature_sets is None else feature_sets):
        for alpha in ALPHAS:
            errors = []
            for group in sorted(set(groups)):
                test = groups == group
                model = fit_ridge(features[~test], target[~test], groups[~test], feature_set, alpha, **options)
                errors.append(float(np.mean(np.abs(predict(model, features[test])-target[test]))))
            scores.append(dict(feature_set=feature_set, alpha=alpha,
                               family_balanced_mae_m=float(np.mean(errors))))
    # Stable tie-break uses the declared order, favouring simpler feature sets.
    best = min(scores, key=lambda item: item['family_balanced_mae_m'])
    return fit_ridge(features, target, groups, best['feature_set'], best['alpha'], **options), scores


def nested_predictions(features, target, groups, *, feature_sets=None, feature_names=None, training_eligible=None):
    """No outer-fold label enters inner selection or the final fold model."""
    features, target, groups = np.asarray(features), np.asarray(target), np.asarray(groups)
    training_eligible = np.ones(len(target), dtype=bool) if training_eligible is None else np.asarray(training_eligible, dtype=bool)
    if training_eligible.shape != target.shape:
        raise ValueError('Training eligibility must match every passage')
    if len(set(groups)) < 4:
        raise ValueError('Need at least four families for nested evaluation')
    predictions = np.full(len(target), np.nan)
    folds = []
    for group in sorted(set(groups)):
        test = groups == group
        training = (~test) & training_eligible
        model, scores = select_model(features[training], target[training], groups[training],
                                    feature_sets=feature_sets, feature_names=feature_names)
        predictions[test] = predict(model, features[test])
        folds.append(dict(heldout_group=str(group), heldout_indices=np.flatnonzero(test).tolist(),
                          model=model, inner_scores=scores))
    return predictions, folds


def comparison_metrics(predicted, intervals, groups):
    predicted, intervals, groups = np.asarray(predicted), np.asarray(intervals), np.asarray(groups)
    lo, hi = intervals.T
    optimistic = np.maximum.reduce([lo-predicted, predicted-hi, np.zeros(len(predicted))])
    conservative = np.maximum(np.abs(predicted-lo), np.abs(predicted-hi))
    midpoint_error = np.abs(predicted-intervals.mean(axis=1))
    return dict(count=len(predicted), family_count=len(set(groups)),
                midpoint_mae_m=float(midpoint_error.mean()),
                family_balanced_midpoint_mae_m=float(np.mean([
                    midpoint_error[groups == g].mean() for g in sorted(set(groups))])),
                optimistic_interval_mae_m=float(optimistic.mean()),
                optimistic_interval_max_m=float(optimistic.max()),
                outside_10cm_even_optimistically=int(np.sum(optimistic > .1+1e-12)),
                within_10cm_for_entire_interval=int(np.sum(conservative <= .1+1e-12)),
                variant_dependent=int(np.sum((optimistic <= .1+1e-12) & (conservative > .1+1e-12))))
