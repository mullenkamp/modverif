"""Tests for modverif.crossval."""
import numpy as np
import pytest

from modverif.crossval import loo_cluster_pred, loo_global_mean


def test_loo_cluster_pred_excludes_the_point_itself():
    z = np.array([1.0, 3.0, 5.0, 100.0])
    labels = np.array([0, 0, 0, 1])
    pred = loo_cluster_pred(z, labels)
    assert pred[0] == pytest.approx(4.0), 'point 0 must be predicted from points 1 and 2 only'
    assert pred[3] == pytest.approx(3.0), 'a singleton falls back to the leave-one-out global mean'


def test_loo_global_mean_is_the_mean_of_the_others():
    z = np.array([1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(loo_global_mean(z), [3.0, 8 / 3, 7 / 3, 2.0])
