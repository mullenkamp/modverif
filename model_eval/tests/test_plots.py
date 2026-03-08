"""
Tests for model_eval.plots module.

Smoke tests that verify figures are created with correct structure.
"""
import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for testing

from model_eval.plots import (
    plot_lagged_correlation,
    plot_scatter,
    plot_station_map,
    plot_timeseries,
    plot_performance_diagram,
    plot_taylor_diagram,
    plot_diurnal,
    plot_fss,
    plot_wind_rose_comparison,
)


class TestPlotScatter:
    def test_creates_figure(self):
        model = np.random.rand(100)
        obs = np.random.rand(100)
        fig, ax = plot_scatter(model, obs)
        assert fig is not None
        assert ax is not None
        assert ax.get_xlabel() != ''

    def test_with_density(self):
        model = np.random.rand(2000)
        obs = np.random.rand(2000)
        fig, ax = plot_scatter(model, obs, density=True)
        assert fig is not None

    def test_saves_file(self, tmp_path):
        model = np.random.rand(50)
        obs = np.random.rand(50)
        save_path = tmp_path / 'scatter.png'
        plot_scatter(model, obs, save_path=save_path)
        assert save_path.exists()

    def test_with_labels(self):
        model = np.random.rand(50)
        obs = np.random.rand(50)
        fig, ax = plot_scatter(model, obs, variable_name='Temperature', units='K')
        assert 'Temperature' in ax.get_xlabel()


class TestPlotStationMap:
    def test_creates_figure(self):
        lons = np.array([170.0, 171.0, 172.0])
        lats = np.array([-45.0, -44.0, -43.0])
        values = np.array([1.0, -0.5, 2.0])
        fig, ax = plot_station_map(lons, lats, values)
        assert fig is not None

    def test_symmetric_limits(self):
        lons = np.array([170.0, 171.0])
        lats = np.array([-45.0, -44.0])
        values = np.array([1.0, -2.0])
        fig, ax = plot_station_map(lons, lats, values, symmetric=True)
        assert fig is not None


class TestPlotTimeseries:
    def test_model_only(self):
        times = np.array([
            np.datetime64('2020-01-01') + np.timedelta64(i, 'D')
            for i in range(10)
        ])
        model = np.random.rand(10)
        fig, ax = plot_timeseries(times, model)
        assert fig is not None

    def test_model_vs_obs(self):
        times = np.array([
            np.datetime64('2020-01-01') + np.timedelta64(i, 'D')
            for i in range(10)
        ])
        model = np.random.rand(10)
        obs = np.random.rand(10)
        fig, ax = plot_timeseries(times, model, obs)
        assert len(ax.get_lines()) == 2


class TestPlotPerformanceDiagram:
    def test_single_point(self):
        fig, ax = plot_performance_diagram(0.8, 0.2)
        assert fig is not None
        assert ax.get_xlim() == (0, 1)

    def test_multiple_points(self):
        fig, ax = plot_performance_diagram(
            [0.8, 0.6, 0.9], [0.2, 0.3, 0.1],
            labels=['Model A', 'Model B', 'Model C'],
        )
        assert fig is not None


class TestPlotTaylorDiagram:
    def test_single_model(self):
        fig, ax = plot_taylor_diagram(1.0, 1.1, 0.95)
        assert fig is not None

    def test_multiple_models(self):
        fig, ax = plot_taylor_diagram(
            1.0, [0.9, 1.1, 1.2], [0.95, 0.85, 0.90],
            labels=['A', 'B', 'C'],
        )
        assert fig is not None


class TestPlotDiurnal:
    def test_model_only(self):
        hours = np.arange(24)
        model = np.sin(hours / 24 * 2 * np.pi) * 5 + 20
        fig, ax = plot_diurnal(hours, model)
        assert fig is not None

    def test_model_vs_obs(self):
        hours = np.arange(24)
        model = np.sin(hours / 24 * 2 * np.pi) * 5 + 20
        obs = np.sin(hours / 24 * 2 * np.pi) * 4 + 19
        fig, ax = plot_diurnal(hours, model, obs)
        assert fig is not None


class TestPlotLaggedCorrelation:
    def test_single_station(self):
        lags = np.arange(-10, 11)
        corrs = np.exp(-lags ** 2 / 10)
        fig, ax = plot_lagged_correlation(lags, corrs, variable_name='Temperature')
        assert fig is not None

    def test_multi_station(self):
        lags = np.arange(-5, 6)
        corrs = np.random.rand(11, 5)
        fig, ax = plot_lagged_correlation(lags, corrs, station_name='All')
        assert fig is not None


class TestPlotFSS:
    def test_basic(self):
        sizes = np.array([1, 3, 5, 9, 17])
        fss = np.array([0.2, 0.4, 0.6, 0.8, 0.9])
        fig, ax = plot_fss(sizes, fss, threshold=1.0)
        assert fig is not None

    def test_multi_times(self):
        sizes = np.array([1, 3, 5, 9])
        fig, ax = plot_fss(
            sizes, None,
            multi_times={'t=0': np.array([0.2, 0.4, 0.6, 0.8]), 't=1': np.array([0.3, 0.5, 0.7, 0.9])},
        )
        assert fig is not None


class TestPlotWindRose:
    def test_model_only(self):
        speed = np.random.exponential(5, 500)
        direction = np.random.uniform(0, 360, 500)
        fig, axes = plot_wind_rose_comparison(speed, direction)
        assert fig is not None

    def test_model_vs_obs(self):
        speed_m = np.random.exponential(5, 500)
        dir_m = np.random.uniform(0, 360, 500)
        speed_o = np.random.exponential(4, 500)
        dir_o = np.random.uniform(0, 360, 500)
        fig, axes = plot_wind_rose_comparison(speed_m, dir_m, speed_o, dir_o)
        assert len(axes) == 2
