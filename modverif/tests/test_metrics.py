"""
Tests for modverif.metrics module.
"""
import numpy as np
import pytest

from modverif.metrics import (
    ContingencyTable,
    compute_ane,
    compute_bias,
    compute_bias_domain,
    compute_ne,
    compute_ne_domain,
    compute_pearson_correlation,
    compute_rse,
)


class TestComputeNE:
    """Tests for compute_ne function."""
    def test_basic_calculation(self):
        source = np.array([100.0, 200.0, 50.0])
        test = np.array([110.0, 180.0, 75.0])
        result = compute_ne(source, test)
        expected = np.array([10, -10, 50], dtype=np.int16)
        np.testing.assert_array_equal(result, expected)

    def test_handles_zero_source(self):
        source = np.array([0.0, 1e-15, 100.0])
        test = np.array([10.0, 10.0, 110.0])
        result = compute_ne(source, test, epsilon=1e-10)
        assert result[0] == 0
        assert result[1] == 0
        assert result[2] == 10

class TestComputeANE:
    """Tests for compute_ane function."""
    def test_basic_calculation(self):
        source = np.array([100.0, 200.0, 50.0])
        test = np.array([110.0, 180.0, 75.0])
        result = compute_ane(source, test)
        expected = np.array([10, 10, 50], dtype=np.int16)
        np.testing.assert_array_equal(result, expected)

class TestComputeRSE:
    """Tests for compute_rse function."""
    def test_basic_calculation(self):
        source = np.array([100.0, 200.0, 50.0])
        test = np.array([110.0, 180.0, 75.0])
        result = compute_rse(source, test)
        expected = np.array([10.0, 20.0, 25.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)

class TestComputeBias:
    """Tests for compute_bias function."""
    def test_basic_calculation(self):
        source = np.array([100.0, 200.0, 50.0])
        test = np.array([110.0, 180.0, 75.0])
        result = compute_bias(source, test)
        expected = np.array([10.0, -20.0, 25.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)

class TestPearsonCorrelation:
    """Tests for compute_pearson_correlation function."""
    def test_perfect_correlation(self):
        x = np.array([1, 2, 3, 4, 5])
        y = x * 2 + 5
        assert compute_pearson_correlation(x, y) == pytest.approx(1.0)

    def test_negative_correlation(self):
        x = np.array([1, 2, 3, 4, 5])
        y = -x
        assert compute_pearson_correlation(x, y) == pytest.approx(-1.0)

class TestDomainMetrics:
    """Tests for domain-aggregated metrics."""
    def test_ne_domain(self):
        source = np.ones((1, 3, 3)) * 100
        test = np.ones((1, 3, 3)) * 110
        result = compute_ne_domain(source, test)
        assert result[0] == pytest.approx(10.0)

    def test_bias_domain(self):
        source = np.ones((1, 3, 3)) * 100
        test = np.ones((1, 3, 3)) * 110
        result = compute_bias_domain(source, test)
        assert result[0] == pytest.approx(10.0)

class TestContingencyTable:
    """Tests for ContingencyTable class and derived metrics."""
    @pytest.fixture
    def sample_table(self):
        # A=hits, B=false_alarms, C=misses, D=correct_negatives
        return ContingencyTable(hits=40, false_alarms=10, misses=20, correct_negatives=30)

    def test_from_data(self):
        source = np.array([0, 1, 2, 3])
        test = np.array([0, 2, 1, 3])
        threshold = 2
        # source_yes: [F, F, T, T]
        # test_yes:   [F, T, F, T]
        # Hits: source_yes & test_yes -> index 3 (1)
        # FA:   ~source_yes & test_yes -> index 1 (1)
        # Misses: source_yes & ~test_yes -> index 2 (1)
        # CN:   ~source_yes & ~test_yes -> index 0 (1)
        ct = ContingencyTable.from_data(source, test, threshold)
        assert ct.hits == 1
        assert ct.false_alarms == 1
        assert ct.misses == 1
        assert ct.correct_negatives == 1

    def test_pod(self, sample_table):
        # POD = 40 / (40 + 20) = 0.666...
        assert sample_table.pod() == pytest.approx(40/60)

    def test_far(self, sample_table):
        # FAR = 10 / (40 + 10) = 0.2
        assert sample_table.far() == pytest.approx(10/50)

    def test_csi(self, sample_table):
        # CSI = 40 / (40 + 10 + 20) = 40/70
        assert sample_table.csi() == pytest.approx(40/70)

    def test_gss(self, sample_table):
        # hits_random = (60 * 50) / 100 = 30
        # GSS = (40 - 30) / (40 + 10 + 20 - 30) = 10 / 40 = 0.25
        assert sample_table.gss() == pytest.approx(0.25)

    def test_bias(self, sample_table):
        # Bias = (40 + 10) / (40 + 20) = 50/60
        assert sample_table.bias() == pytest.approx(50/60)


# ================================================================ the two lag conventions, pinned
# These exist because the library now contains TWO cross-correlation functions that compute the same
# physical quantity with OPPOSITE sign conventions. That is deliberate -- published results depend on
# both -- but it is exactly the kind of thing a later "cleanup" unifies, silently inverting every
# timing conclusion drawn from one of them.
def _shifted_pair(shift, n=300):
    """An observation series and a model series in which the MODEL happens `shift` steps LATER."""
    base = np.sin(np.arange(n) / 9.0) + 0.4 * np.sin(np.arange(n) / 3.1)
    return base, np.roll(base, shift)


def test_the_two_lag_functions_disagree_on_sign_and_that_is_intentional():
    from modverif.metrics import compute_lagged_correlation, compute_xcorr_best_lag

    obs, model = _shifted_pair(+5)               # the model is 5 steps LATE

    best_lag, _, _, _ = compute_xcorr_best_lag(model, obs, 24, min_pairs=10)
    assert best_lag == +5, 'compute_xcorr_best_lag: positive must mean the MODEL IS LATER'

    lags, corrs = compute_lagged_correlation(model, obs, max_lag=24)
    assert int(lags[int(np.nanargmax(corrs))]) == -5, (
        'compute_lagged_correlation: positive means the model LEADS, so a late model is negative')


def test_the_lag_conventions_are_exact_mirrors_on_gap_free_data():
    from modverif.metrics import compute_lagged_correlation, compute_xcorr_best_lag

    for shift in (-7, -2, 0, 3, 8):
        obs, model = _shifted_pair(shift)
        best_lag, _, _, _ = compute_xcorr_best_lag(model, obs, 24, min_pairs=10)
        lags, corrs = compute_lagged_correlation(model, obs, max_lag=24)
        assert best_lag == -int(lags[int(np.nanargmax(corrs))]), f'mirror broken at shift {shift}'


def test_only_the_per_lag_masking_version_keeps_the_lag_axis_in_time():
    """The reason both exist. compute_lagged_correlation compacts gaps BEFORE lagging, so on a gappy
    series its lag is counted in surviving samples, not timesteps."""
    from modverif.metrics import compute_lagged_correlation, compute_xcorr_best_lag

    gen = np.random.default_rng(0)
    obs, model = _shifted_pair(+5)
    obs = obs.copy()
    obs[gen.choice(len(obs), 60, replace=False)] = np.nan   # a heavily gapped gauge

    best_lag, _, _, _ = compute_xcorr_best_lag(model, obs, 24, min_pairs=10)
    assert best_lag == +5, 'per-lag masking should still recover the true 5-step displacement'

    lags, corrs = compute_lagged_correlation(model, obs, max_lag=24)
    compacted = -int(lags[int(np.nanargmax(corrs))])
    assert compacted != 5, (
        'if the compacted version now agrees, the gap trap documented on it has been fixed and the '
        'warning in its docstring needs updating')


def test_compute_xcorr_best_lag_reports_the_zero_lag_correlation_for_comparison():
    from modverif.metrics import compute_xcorr_best_lag
    obs, model = _shifted_pair(+6)
    best_lag, r_best, r_zero, n = compute_xcorr_best_lag(model, obs, 24, min_pairs=10)
    assert best_lag == 6
    assert r_best > r_zero, 'aligning should beat not aligning, and both must be reported'
    assert n > 200


def test_compute_xcorr_best_lag_declines_when_no_lag_has_enough_pairs():
    from modverif.metrics import compute_xcorr_best_lag
    obs, model = _shifted_pair(+3, n=40)
    best_lag, r_best, _, n = compute_xcorr_best_lag(model, obs, 5, min_pairs=1000)
    assert np.isnan(best_lag) and np.isnan(r_best) and n == 0


def test_compute_xcorr_best_lag_skips_zero_variance_overlaps():
    """A constant slice has no correlation to measure; corrcoef returns NaN with a warning rather
    than an error, so the guard has to be explicit."""
    from modverif.metrics import compute_xcorr_best_lag
    # (both series are constant over the tested window)
    obs = np.concatenate([np.full(100, 7.0), np.sin(np.arange(100) / 5.0)])
    model = np.concatenate([np.full(100, 7.0), np.sin(np.arange(100) / 5.0)])
    best_lag, r_best, _, _ = compute_xcorr_best_lag(model[:100], obs[:100], 5, min_pairs=10)
    assert np.isnan(best_lag) and np.isnan(r_best), (
        'an entirely constant pair offers no lag information and must be declined, not scored')


# ============================================================ compute_residual_skill_score
# These exist because the final review found the function had NO direct tests: its only exercise
# computed the expected value BY CALLING IT, which is circular, and both of its documented design
# decisions survived mutation.
def test_residual_skill_score_worked_example():
    from modverif.metrics import compute_residual_skill_score
    # rmse([3,4]) = 3.5355..., rmse([6,8]) = 7.0710...  -> 1 - 0.5 = 0.5
    assert compute_residual_skill_score(np.array([3.0, 4.0]),
                                        np.array([6.0, 8.0])) == pytest.approx(0.5)
    assert compute_residual_skill_score(np.array([1.0, 1.0]),
                                        np.array([1.0, 1.0])) == pytest.approx(0.0)
    assert compute_residual_skill_score(np.array([2.0, 2.0]),
                                        np.array([1.0, 1.0])) == pytest.approx(-1.0)


def test_residual_skill_score_is_an_rmse_ratio_not_an_mse_ratio():
    """The two are routinely confused and differ substantially: an MSE ratio of 0.5 is an RMSE ratio
    of about 0.293."""
    from modverif.metrics import compute_residual_skill_score
    resid = np.full(4, 1.0)
    base = np.full(4, np.sqrt(2.0))          # MSE ratio exactly 0.5
    assert compute_residual_skill_score(resid, base) == pytest.approx(1 - 1 / np.sqrt(2.0))


def test_residual_skill_score_propagates_nan_rather_than_dropping_it():
    """Strict mean, not nanmean. A NaN residual means the caller's inputs are wrong and must surface,
    not be silently excluded from the denominator -- mutation testing showed this was undefended."""
    from modverif.metrics import compute_residual_skill_score
    assert np.isnan(compute_residual_skill_score(np.array([1.0, np.nan]), np.array([2.0, 2.0])))
    assert np.isnan(compute_residual_skill_score(np.array([1.0, 1.0]), np.array([2.0, np.nan])))


def test_residual_skill_score_declines_on_a_zero_baseline():
    """A baseline with no error cannot be improved on; the guard is `> 0`, not `>= 0`."""
    from modverif.metrics import compute_residual_skill_score
    assert np.isnan(compute_residual_skill_score(np.array([1.0, 1.0]), np.zeros(2)))
