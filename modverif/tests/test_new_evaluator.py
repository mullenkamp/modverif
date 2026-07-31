"""
Tests for new Evaluator methods: evaluate_fss, evaluate_wind, evaluate_diurnal, and pearson domain metric.
"""

import cfdb
import numpy as np
import pytest

from modverif.evaluate import evaluate_fss, evaluate_models_domain, evaluate_wind


def create_mock_cfdb(
    path, variables, n_times=4, n_y=10, n_x=10, data_func=None, start_time=None,
):
    if start_time is None:
        start_time = np.datetime64('2020-09-30T00:00')
    times = np.array([start_time + np.timedelta64(i, 'h') for i in range(n_times)])
    y_vals = np.arange(n_y, dtype='float32') * 1000.0
    x_vals = np.arange(n_x, dtype='float32') * 1000.0
    heights = np.array([0.0], dtype='float32')

    with cfdb.open_dataset(path, 'n', dataset_type='grid') as ds:
        ds.create.coord.time(data=times)
        ds.create.coord.y(data=y_vals)
        ds.create.coord.x(data=x_vals)
        ds.create.coord.height(data=heights)

        for var_name in variables:
            shape = (n_times, n_y, n_x)
            data = data_func(var_name, shape) if data_func else (np.random.rand(*shape).astype(np.float32) * 100 + 1)
            var = ds.create.data_var.generic(var_name, ('time', 'height', 'y', 'x'), dtype='float32')
            for t in range(n_times):
                var[(t, 0, slice(None), slice(None))] = data[t]

    return path


class TestDomainPearson:
    def test_pearson_in_domain_metrics(self, tmp_path):
        source_path = tmp_path / 'source.cfdb'
        test_path = tmp_path / 'test.cfdb'

        def source_data(var, shape):
            return np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + 1

        def test_data(var, shape):
            return (np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + 1) * 2

        create_mock_cfdb(source_path, ['air_temperature'], n_times=2, data_func=source_data)
        create_mock_cfdb(test_path, ['air_temperature'], n_times=2, data_func=test_data)

        output_path = tmp_path / 'output.cfdb'
        evaluate_models_domain(
            source_path, test_path, output_path,
            variables=['air_temperature'], metrics=['pearson'],
        )

        with cfdb.open_dataset(output_path) as ds:
            assert 'air_temperature' in ds.data_var_names
            data = ds['air_temperature'][(0, slice(None))].data
            # Perfect linear relationship -> correlation ~ 1.0
            np.testing.assert_allclose(data[0, 0], 1.0, atol=1e-5)


class TestEvaluateFSS:
    def test_basic_fss(self, tmp_path):
        source_path = tmp_path / 'source.cfdb'
        test_path = tmp_path / 'test.cfdb'
        n_y, n_x = 20, 20

        def source_data(var, shape):
            data = np.zeros(shape, dtype=np.float32)
            for t in range(shape[0]):
                data[t, 5:15, 5:15] = 10.0
            return data

        def test_data(var, shape):
            data = np.zeros(shape, dtype=np.float32)
            for t in range(shape[0]):
                data[t, 7:17, 7:17] = 10.0  # Shifted
            return data

        create_mock_cfdb(source_path, ['precipitation'], n_times=2, n_y=n_y, n_x=n_x, data_func=source_data)
        create_mock_cfdb(test_path, ['precipitation'], n_times=2, n_y=n_y, n_x=n_x, data_func=test_data)

        output_path = tmp_path / 'fss_output.cfdb'
        result = evaluate_fss(
            source_path, test_path, output_path,
            variables=['precipitation'], threshold=5.0,
            neighborhood_sizes=[1, 5, 11],
        )

        assert result.exists()
        with cfdb.open_dataset(output_path) as ds:
            assert 'precipitation' in ds.data_var_names
            assert 'scale' in ds.coord_names
            data = ds['precipitation'][(0, slice(None))].data
            # FSS should increase with scale
            assert data[0, 2] >= data[0, 0]  # larger neighborhood -> better FSS


class TestEvaluateWind:
    def test_basic_wind(self, tmp_path):
        source_path = tmp_path / 'source.cfdb'
        test_path = tmp_path / 'test.cfdb'

        def source_data(var, shape):
            if var == 'u_wind':
                return np.ones(shape, dtype=np.float32) * 5.0
            return np.ones(shape, dtype=np.float32) * 3.0

        def test_data(var, shape):
            if var == 'u_wind':
                return np.ones(shape, dtype=np.float32) * 6.0
            return np.ones(shape, dtype=np.float32) * 4.0

        create_mock_cfdb(source_path, ['u_wind', 'v_wind'], data_func=source_data)
        create_mock_cfdb(test_path, ['u_wind', 'v_wind'], data_func=test_data)

        output_path = tmp_path / 'wind_output.cfdb'
        result = evaluate_wind(
            source_path, test_path, output_path,
            metrics=['vector_rmse', 'speed_bias', 'direction_bias'],
        )

        assert result.exists()
        with cfdb.open_dataset(output_path) as ds:
            assert 'wind' in ds.data_var_names
            assert ds['wind'].shape[1] == 3

    def test_wind_invalid_metric(self, tmp_path):
        source_path = tmp_path / 'source.cfdb'
        test_path = tmp_path / 'test.cfdb'
        create_mock_cfdb(source_path, ['u_wind', 'v_wind'])
        create_mock_cfdb(test_path, ['u_wind', 'v_wind'])

        with pytest.raises(ValueError, match="Unknown wind metric"):
            evaluate_wind(
                source_path, test_path, tmp_path / 'output.cfdb',
                metrics='invalid_wind_metric',
            )


class TestEvaluateDiurnal:
    def test_basic_diurnal(self, tmp_path):
        source_path = tmp_path / 'source.cfdb'
        test_path = tmp_path / 'test.cfdb'

        # Create 48 hourly timesteps
        create_mock_cfdb(source_path, ['air_temperature'], n_times=48)
        create_mock_cfdb(test_path, ['air_temperature'], n_times=48)

        from modverif.evaluator import Evaluator
        evaluator = Evaluator(source_path, test_path)
        output_path = tmp_path / 'diurnal_output.cfdb'
        result = evaluator.evaluate_diurnal(output_path, variables=['air_temperature'], metrics=['bias', 'rmse'])

        assert result.exists()
        with cfdb.open_dataset(output_path) as ds:
            assert 'air_temperature' in ds.data_var_names
            assert 'hour' in ds.coord_names
            assert 'metric' in ds.coord_names
