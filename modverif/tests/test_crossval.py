"""Tests for modverif.crossval."""
import numpy as np
import pytest
from scipy.stats import theilslopes

from modverif.crossval import cluster_holdout, holdout_high, loo_cluster_pred, loo_global_mean, loo_theilsen
from modverif.metrics import compute_residual_skill_score


def test_loo_cluster_pred_excludes_the_point_itself():
    z = np.array([1.0, 3.0, 5.0, 100.0])
    labels = np.array([0, 0, 0, 1])
    pred = loo_cluster_pred(z, labels)
    assert pred[0] == pytest.approx(4.0), 'point 0 must be predicted from points 1 and 2 only'
    assert pred[3] == pytest.approx(3.0), 'a singleton falls back to the leave-one-out global mean'


def test_loo_global_mean_is_the_mean_of_the_others():
    z = np.array([1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(loo_global_mean(z), [3.0, 8 / 3, 7 / 3, 2.0])


# ------------------------------------------------------------------- out-of-sample assessment
def test_loo_theilsen_recovers_a_linear_relationship():
    gen = np.random.default_rng(0)
    x = gen.uniform(0, 1000, 60)
    y = 2.0 + 0.003 * x + gen.normal(0, 0.05, 60)
    pred = loo_theilsen(x, y)
    assert np.corrcoef(pred, y)[0, 1] > 0.9


def test_holdout_high_withholds_the_TOP_of_the_covariate_range():
    """The whole point: it is an extrapolation test, not a random split."""
    x = np.arange(50, dtype=float)
    y = 0.1 * x
    _, (_, _, _, _, max_train, held) = holdout_high(x, y, 0.2)
    assert set(held) == set(np.arange(40, 50)), 'the highest 20% must be the held-out set'
    assert max_train == pytest.approx(39.0), 'training support must end below the held-out range'


def test_holdout_high_always_withholds_at_least_two_points():
    x = np.arange(5, dtype=float)
    _, (_, _, _, _, _, held) = holdout_high(x, 0.1 * x, 0.01)
    assert len(held) == 2


def test_holdout_high_reports_negative_skill_when_the_slope_does_not_extrapolate():
    """The informative failure: a relationship that is real in-sample but reverses beyond it.

    Leave-one-out cannot see this; withholding the top of the range is what exposes it.
    """
    x = np.linspace(0, 100, 60)
    y = np.where(x < 70, 0.05 * x, 3.5 - 0.15 * (x - 70))   # rises, then falls
    skill, _ = holdout_high(x, y, 0.25)
    assert skill < 0, 'extrapolating a reversed relationship should score worse than the mean'


def test_loo_theilsen_excludes_each_point_from_its_own_fit():
    """Mutation testing killed the previous version of this test: with Theil-Sen's robustness, a
    single spiked point barely moves the fit, so "prediction is unaffected by the spike" passed even
    when the point WAS in its own training set. Compare against the all-points fit instead."""
    x = np.array([0.0, 1.0, 2.0, 3.0, 100.0])
    y = np.array([0.0, 1.0, 2.0, 3.0, 0.0])
    s, b0, _, _ = theilslopes(y, x)
    all_points_fit = b0 + s * x
    assert not np.allclose(loo_theilsen(x, y), all_points_fit), (
        'leave-one-out predictions are identical to the all-points fit, so nothing is being left out')


def test_loo_theilsen_is_robust_where_least_squares_is_not():
    """Pins the estimator itself. The previous version compared two OLS fits and survived swapping
    Theil-Sen for polyfit."""
    gen = np.random.default_rng(2)
    x = np.linspace(0, 100, 40)
    y = 0.5 * x + gen.normal(0, 0.5, 40)
    y[:3] = 500.0                                  # three wild points at the low end of x
    ts_slope = np.polyfit(x, loo_theilsen(x, y), 1)[0]
    ols_slope = np.polyfit(x, y, 1)[0]
    assert abs(ts_slope - 0.5) < 0.15, f'a robust fit should stay near 0.5, got {ts_slope:.3f}'
    assert abs(ols_slope - 0.5) > 0.5, 'the fixture must actually break least squares'


def test_cluster_holdout_scores_exactly_zero_when_the_predictor_is_the_baseline():
    """Both the fold structure and the baseline, pinned to an exact value.

    Uses a predictor that is NOT the mean, so the training-set mean and the global mean give
    different answers -- the previous version used the mean itself, where both the 'global baseline'
    and 'train on everything' mutations still produced 0 and passed.
    """
    labels = np.array([0, 0, 0, 1, 1, 1])
    values = np.array([1.0, 2.0, 3.0, 10.0, 11.0, 12.0])

    skill = cluster_holdout(lambda tr, hd: np.full(hd.sum(), values[tr].max()), values, labels)

    # By hand: fold 0 trains on group 1 -> predict 12.0, baseline mean 11.0
    #          fold 1 trains on group 0 -> predict 3.0,  baseline mean 2.0
    expected = compute_residual_skill_score(
        values - np.array([12.0, 12.0, 12.0, 3.0, 3.0, 3.0]),
        values - np.array([11.0, 11.0, 11.0, 2.0, 2.0, 2.0]))
    assert skill == pytest.approx(expected)


def test_cluster_holdout_never_trains_on_the_held_out_group():
    """A predictor that reports its own training-set size makes the fold membership observable."""
    labels = np.array([0, 0, 0, 0, 1, 1])
    values = np.arange(6, dtype=float)
    sizes = []

    def spy(tr, hd):
        sizes.append(int(tr.sum()))
        return np.full(hd.sum(), values[tr].mean())

    cluster_holdout(spy, values, labels)
    assert sorted(sizes) == [2, 4], f'training sets should exclude the held group, got {sorted(sizes)}'


def test_cluster_holdout_is_harsher_than_leave_one_out():
    """Leaving out a whole group asks whether the relationship transfers; leave-one-out does not."""
    gen = np.random.default_rng(3)
    labels = np.repeat(np.arange(5), 20)
    offsets = np.array([-3.0, -1.5, 0.0, 1.5, 3.0])[labels]
    values = offsets + gen.normal(0, 0.2, 100)

    loo_rmse = float(np.sqrt(np.mean((values - loo_cluster_pred(values, labels)) ** 2)))
    group_skill = cluster_holdout(lambda tr, hd: np.full(hd.sum(), values[tr].mean()), values, labels)

    assert loo_rmse < 1.0, 'leave-one-out should look excellent here'
    assert group_skill == pytest.approx(0.0), (
        'predicting a withheld group by the training mean IS the baseline, so skill is exactly zero '
        '-- the group structure buys nothing out-of-sample')


def test_holdout_high_baseline_is_the_training_mean_not_the_global_mean():
    """Pins which baseline is used, with a fixture where the two differ."""
    x = np.arange(50, dtype=float)
    y = np.where(x < 40, 0.0, 10.0)          # training mean 0.0, global mean 2.0
    skill, _ = holdout_high(x, y, 0.2)
    assert skill == pytest.approx(0.0), (
        'a flat training set predicts 0 and the training-mean baseline is also 0, so skill is '
        'exactly 0; using the global mean as baseline would give -0.25')
