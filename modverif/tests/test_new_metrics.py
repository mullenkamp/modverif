"""
Tests for new metric functions added in modverif.metrics.
"""
import numpy as np
import pytest

from modverif.metrics import (
    compute_ane_1d,
    compute_diurnal_stats,
    compute_fraction_field,
    compute_fss,
    compute_fss_multi_scale,
    compute_lagged_correlation,
    compute_mae,
    compute_mae_1d,
    compute_mean_bias,
    compute_ne_1d,
    compute_pearson_domain,
    compute_rmse_1d,
    compute_vector_rmse,
    compute_wind_direction_bias,
    compute_wind_speed_bias,
)


class TestComputeMAE:
    def test_basic(self):
        source = np.array([100.0, 200.0, 50.0])
        test = np.array([110.0, 180.0, 75.0])
        result = compute_mae(source, test)
        expected = np.array([10.0, 20.0, 25.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)

    def test_symmetric(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([3.0, 2.0, 1.0])
        np.testing.assert_array_almost_equal(compute_mae(a, b), compute_mae(b, a))


class TestStation1DMetrics:
    def test_mean_bias(self):
        model = np.array([11.0, 12.0, 13.0])
        obs = np.array([10.0, 10.0, 10.0])
        assert compute_mean_bias(model, obs) == pytest.approx(2.0)

    def test_mae_1d(self):
        model = np.array([12.0, 8.0, 10.0])
        obs = np.array([10.0, 10.0, 10.0])
        assert compute_mae_1d(model, obs) == pytest.approx(4 / 3)

    def test_rmse_1d(self):
        model = np.array([12.0, 8.0])
        obs = np.array([10.0, 10.0])
        # RMSE = sqrt(mean([4, 4])) = sqrt(4) = 2
        assert compute_rmse_1d(model, obs) == pytest.approx(2.0)

    def test_ne_1d(self):
        model = np.array([110.0, 110.0])
        obs = np.array([100.0, 100.0])
        assert compute_ne_1d(model, obs) == pytest.approx(10.0)

    def test_ne_1d_zero_obs(self):
        model = np.array([1.0, 2.0])
        obs = np.array([0.0, 0.0])
        assert compute_ne_1d(model, obs) == 0.0

    def test_ane_1d(self):
        model = np.array([90.0, 90.0])
        obs = np.array([100.0, 100.0])
        assert compute_ane_1d(model, obs) == pytest.approx(10.0)


class TestPearsonDomain:
    def test_perfect_correlation(self):
        source = np.arange(9, dtype=np.float64).reshape(1, 3, 3)
        test = source * 2 + 5
        result = compute_pearson_domain(source, test)
        assert result[0] == pytest.approx(1.0)

    def test_with_mask(self):
        source = np.random.rand(2, 5, 5)
        test = source * 3
        mask = np.ones((5, 5), dtype=bool)
        mask[0, 0] = False
        result = compute_pearson_domain(source, test, mask)
        np.testing.assert_allclose(result, 1.0, atol=1e-10)


class TestFSS:
    def test_perfect_forecast(self):
        field = np.zeros((20, 20))
        field[5:15, 5:15] = 10.0
        fss = compute_fss(field, field, threshold=5.0, neighborhood_size=1)
        assert fss == pytest.approx(1.0)

    def test_no_skill(self):
        source = np.zeros((20, 20))
        source[5:15, 5:15] = 10.0
        test = np.zeros((20, 20))
        # No overlap at all
        fss = compute_fss(source, test, threshold=5.0, neighborhood_size=1)
        assert fss < 0.5

    def test_fss_increases_with_scale(self):
        source = np.zeros((30, 30))
        source[10:20, 10:20] = 10.0
        test = np.zeros((30, 30))
        test[12:22, 12:22] = 10.0  # Shifted by 2 cells
        fss_results = compute_fss_multi_scale(
            source, test, threshold=5.0, neighborhood_sizes=[1, 5, 11, 21]
        )
        # FSS should generally increase with neighborhood size
        values = list(fss_results.values())
        assert values[-1] >= values[0]

    def test_fraction_field(self):
        field = np.zeros((5, 5), dtype=bool)
        field[2, 2] = True
        frac = compute_fraction_field(field, 3)
        # Center cell in 3x3 neighborhood: 1/9
        assert frac[2, 2] == pytest.approx(1.0 / 9, abs=1e-5)


class TestVectorWindMetrics:
    def test_vector_rmse_zero_error(self):
        u = np.array([1.0, 2.0, 3.0])
        v = np.array([1.0, 2.0, 3.0])
        assert compute_vector_rmse(u, v, u, v) == pytest.approx(0.0)

    def test_vector_rmse_known(self):
        su = np.array([1.0, 0.0])
        sv = np.array([0.0, 1.0])
        tu = np.array([2.0, 0.0])
        tv = np.array([0.0, 2.0])
        # du = [1, 0], dv = [0, 1], du^2 + dv^2 = [1, 1], mean=1, sqrt=1
        assert compute_vector_rmse(su, sv, tu, tv) == pytest.approx(1.0)

    def test_speed_bias_positive(self):
        su = np.array([1.0, 0.0])
        sv = np.array([0.0, 1.0])
        tu = np.array([2.0, 0.0])
        tv = np.array([0.0, 2.0])
        bias = compute_wind_speed_bias(su, sv, tu, tv)
        assert bias > 0  # Test is faster

    def test_direction_bias_zero(self):
        u = np.array([1.0, 0.0, -1.0])
        v = np.array([0.0, 1.0, 0.0])
        bias = compute_wind_direction_bias(u, v, u, v)
        assert bias == pytest.approx(0.0, abs=1e-10)


class TestDiurnalStats:
    def test_known_diurnal_bias(self):
        # Create data with known hourly pattern over 3 days
        n_hours = 72
        times = np.array([
            np.datetime64('2020-01-01T00:00') + np.timedelta64(h, 'h')
            for h in range(n_hours)
        ])
        obs = np.ones(n_hours) * 10.0
        model = np.ones(n_hours) * 10.0
        # Add +5 bias at hour 12
        for i in range(n_hours):
            if i % 24 == 12:
                model[i] = 15.0

        hours, values = compute_diurnal_stats(times, model, obs, metric='bias')
        assert hours.shape == (24,)
        assert values[12] == pytest.approx(5.0)
        assert values[0] == pytest.approx(0.0)

    def test_diurnal_rmse(self):
        n_hours = 48
        times = np.array([
            np.datetime64('2020-01-01T00:00') + np.timedelta64(h, 'h')
            for h in range(n_hours)
        ])
        obs = np.ones(n_hours) * 10.0
        model = np.ones(n_hours) * 12.0  # constant bias of 2

        hours, values = compute_diurnal_stats(times, model, obs, metric='rmse')
        # All hours should have RMSE = 2
        valid = ~np.isnan(values)
        np.testing.assert_allclose(values[valid], 2.0, atol=1e-10)


class TestLaggedCorrelation:
    def test_zero_lag_perfect(self):
        obs = np.sin(np.linspace(0, 4 * np.pi, 100))
        model = obs.copy()
        lags, corrs = compute_lagged_correlation(model, obs, max_lag=10)
        best = lags[np.argmax(corrs)]
        assert best == 0
        assert corrs[lags == 0][0] == pytest.approx(1.0, abs=1e-5)

    def test_positive_lag_model_leads(self):
        # Model signal arrives 3 steps before obs
        n = 100
        obs = np.zeros(n)
        obs[50:60] = 1.0
        model = np.zeros(n)
        model[47:57] = 1.0  # shifted 3 steps earlier

        lags, corrs = compute_lagged_correlation(model, obs, max_lag=10)
        best = lags[np.argmax(corrs)]
        assert best > 0  # model leads -> positive lag

    def test_negative_lag_model_lags(self):
        n = 100
        obs = np.zeros(n)
        obs[50:60] = 1.0
        model = np.zeros(n)
        model[53:63] = 1.0  # shifted 3 steps later

        lags, corrs = compute_lagged_correlation(model, obs, max_lag=10)
        best = lags[np.argmax(corrs)]
        assert best < 0  # model lags -> negative lag

    def test_short_series(self):
        model = np.array([1.0, 2.0])
        obs = np.array([1.0, 2.0])
        lags, corrs = compute_lagged_correlation(model, obs)
        assert len(lags) > 0

    def test_returns_correct_shape(self):
        model = np.random.rand(50)
        obs = np.random.rand(50)
        lags, corrs = compute_lagged_correlation(model, obs, max_lag=5)
        assert len(lags) == 11  # -5 to +5
        assert len(corrs) == 11
