"""Offline frame-balanced fitting for cars outside primary outline support."""
from collections import Counter

import numpy as np

from .catalogue_calibration import ALPHAS


def select_passage_frames(row, observations, image_size):
    """Anchor plus two position-based views, without consulting length labels."""
    width, height = image_size
    selected = [dict(frame=row['frame'], bbox=row['bbox'])]
    candidates = [o for o in observations if abs(o['frame']-row['frame']) <= 75
        and o['bbox'][0] > 2 and o['bbox'][1] > 2 and o['bbox'][0]+o['bbox'][2] < width-2
        and o['bbox'][1]+o['bbox'][3] < height-2]
    for target in (.35, .65):
        if not candidates:
            continue
        match = min(candidates, key=lambda o: abs((o['bbox'][0]+o['bbox'][2]/2)/width-target))
        center = (match['bbox'][0]+match['bbox'][2]/2)/width
        if abs(center-target) > .12 or any(abs(center-(s['bbox'][0]+s['bbox'][2]/2)/width) < .07 for s in selected):
            continue
        selected.append(dict(frame=match['frame'], bbox=match['bbox']))
    return sorted(selected, key=lambda s: s['frame'])


def fit_frames(features, target, groups, passages, alpha):
    x, y = np.asarray(features, float), np.asarray(target, float)
    groups, passages = np.asarray(groups), np.asarray(passages)
    if (x.ndim != 2 or x.shape[1] != 9 or not len(y) or y.shape != (len(x),)
            or groups.shape != y.shape or passages.shape != y.shape
            or not np.isfinite(x).all() or not np.isfinite(y).all()
            or not np.isfinite(alpha) or alpha <= 0):
        raise ValueError('Expected aligned finite nine-feature frames and positive regularization')
    if any(len(set(groups[passages == p])) != 1 for p in set(passages)):
        raise ValueError('A passage cannot belong to multiple catalogue families')
    if any(np.ptp(y[passages == p]) > 1e-10 for p in set(passages)):
        raise ValueError('A passage must have the same target across its frames')
    counts = Counter(passages)
    family_passages = {g: len(set(passages[groups == g])) for g in set(groups)}
    # A long or slow passage must not gain weight by supplying extra frames.
    weights = np.asarray([1/(counts[p]*family_passages[g]) for p, g in zip(passages, groups)])
    weights /= weights.sum()
    mean = weights @ x
    scale = np.sqrt(weights @ ((x-mean)**2))
    scale = np.where(scale > 1e-8, scale, 1.)
    z = (x-mean)/scale
    intercept = float(weights @ y)
    fit_weights = weights*len(family_passages)
    coefficient = np.linalg.solve(z.T @ (z*fit_weights[:, None]) + alpha*np.eye(9),
                                  z.T @ ((y-intercept)*fit_weights))
    return dict(mean=mean.tolist(), scale=scale.tolist(), coefficient=coefficient.tolist(), intercept=intercept,
        alpha=float(alpha), feature_bounds=np.stack([x.min(0), x.max(0)], axis=1).tolist(),
        training_count=len(counts), training_frame_count=len(y), training_family_count=len(family_passages))


def predict_frames(model, features):
    return ((np.asarray(features)-model['mean'])/model['scale']) @ model['coefficient'] + model['intercept']


def select_frame_model(features, target, groups, passages, anchors):
    """Select regularization by family-heldout anchor MAE, fitting all views.

Only observed wheel pairs train this model. Missing anchor pairs are absent
from every candidate's inner score, never filled with made-up features. Full
benchmark denominators must include them via the primary/review fallback.
"""
    features, target, groups, passages, anchors = [np.asarray(v) for v in (features, target, groups, passages, anchors)]
    if len(set(groups)) < 3 or anchors.shape != target.shape or anchors.dtype != bool:
        raise ValueError('Need at least three families and aligned boolean anchor flags')
    if any(np.count_nonzero(anchors & (passages == p)) > 1 for p in set(passages)):
        raise ValueError('A passage has multiple anchor frames')
    scores = []
    for alpha in ALPHAS:
        errors = []
        for group in sorted(set(groups)):
            train = groups != group
            test = (groups == group) & anchors
            if not np.any(test):
                raise ValueError('Every validation family must have an observed wheel anchor')
            model = fit_frames(features[train], target[train], groups[train], passages[train], alpha)
            errors.append(float(np.mean(np.abs(predict_frames(model, features[test])-target[test]))))
        scores.append(dict(alpha=alpha, family_balanced_anchor_mae_m=float(np.mean(errors))))
    selected = min(scores, key=lambda item: item['family_balanced_anchor_mae_m'])
    return fit_frames(features, target, groups, passages, selected['alpha']), scores
