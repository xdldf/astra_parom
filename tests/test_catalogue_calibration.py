"""Protect the independence of catalogue learning, not field accuracy."""
from copy import deepcopy

import numpy as np
import pytest

from vehicle_metrology.catalogue_calibration import (
    catalogue_family, comparison_metrics, fit_ridge, image_features,
    nested_predictions, predict,
)


def test_test_family_labels_cannot_change_its_predictions_or_selected_model():
    rng = np.random.default_rng(431)
    x = rng.normal(size=(18, 9))
    groups = np.repeat(['a', 'b', 'c', 'd', 'e', 'f'], 3)
    y = 4.5 + .3*x[:, 0] + .12*x[:, 2] + rng.normal(0, .02, 18)
    original, folds = nested_predictions(x, y, groups)
    changed = y.copy()
    changed[groups == 'a'] += 10
    after, changed_folds = nested_predictions(x, changed, groups)
    np.testing.assert_array_equal(after[groups == 'a'], original[groups == 'a'])
    assert folds[0] == changed_folds[0]
    for fold in folds:
        assert fold['heldout_group'] not in fold['model']['training_groups']
        assert all(groups[i] == fold['heldout_group'] for i in fold['heldout_indices'])


def test_aliases_and_generations_cannot_leak_between_families():
    assert catalogue_family(['vitz_2010']) == catalogue_family(['vitz_2017', 'yaris'])
    assert catalogue_family(['yaris_cross']) != catalogue_family(['yaris'])
    assert catalogue_family(['prado_2013']) == catalogue_family(['prado_2017'])
    assert catalogue_family(['noah_2014']) == catalogue_family(['voxy_2014', 'voxy_aero'])
    assert catalogue_family(['fielder_2012']) == catalogue_family(['fielder_2006'])
    with pytest.raises(ValueError, match='explicit split family'):
        catalogue_family(['unknown'])
    with pytest.raises(ValueError, match='different split families'):
        catalogue_family(['prado_2017', 'vitz_2017'])


def test_features_ignore_catalogue_identity_and_old_length_estimates():
    row = dict(bbox=[60, 110, 80, 40], length_m=4., model_candidate='car',
               catalogue_length_range_m=[4., 4.1], measurement={'temporal': {
                   'length_m': 4., 'samples': [
                       dict(bbox=[x, 110, 80, 40], frame=i, line_offset_px=x-60, length_m=4.)
                       for i, x in enumerate([40, 60, 80]) ]}})
    polygon = [[0, 100], [400, 100], [400, 200], [0, 200]]
    original = image_features(row, polygon, 400)
    changed = deepcopy(row)
    changed.update(length_m=900, model_candidate='some other car', catalogue_length_range_m=[9, 10])
    changed['measurement']['temporal']['length_m'] = 900
    for sample in changed['measurement']['temporal']['samples']:
        sample['length_m'] = 900
    np.testing.assert_array_equal(original, image_features(changed, polygon, 400))
    np.testing.assert_allclose(original, [.8, .5, .4, .25, .4, .2, 0., 1., 1.])


def test_repeating_a_family_does_not_increase_its_training_weight():
    rng = np.random.default_rng(981)
    x = rng.normal(size=(8, 9))
    y = rng.uniform(3.5, 5., 8)
    groups = np.repeat(['a', 'b', 'c', 'd'], 2)
    model = fit_ridge(x, y, groups, 'position_height', .1)
    repeated = fit_ridge(np.r_[x, x[:2]], np.r_[y, y[:2]], np.r_[groups, groups[:2]], 'position_height', .1)
    np.testing.assert_allclose(predict(model, x), predict(repeated, x), atol=1e-12)


def test_ambiguous_intervals_do_not_count_as_unconditional_passes():
    metrics = comparison_metrics([4.1, 4.1, 4.5], [[4.1, 4.1], [4., 4.3], [4., 4.2]], ['a', 'b', 'c'])
    assert metrics['within_10cm_for_entire_interval'] == 1
    assert metrics['variant_dependent'] == 1
    assert metrics['outside_10cm_even_optimistically'] == 1
    assert metrics['optimistic_interval_max_m'] == pytest.approx(.3)
