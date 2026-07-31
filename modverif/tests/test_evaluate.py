"""
Tests for modverif evaluation pipeline using cfdb datasets.
"""

import cfdb
import numpy as np
import pytest

from modverif.evaluate import evaluate_cyclones, evaluate_models_cell, evaluate_models_domain
from modverif.metrics import AVAILABLE_DOMAIN_METRICS, AVAILABLE_METRICS


def create_mock_cfdb(
    path,
    variables,
    n_times=4,
    n_y=10,
    n_x=10,
    data_func=None,
    include_latlon=False,
    lat_range=None,
    lon_range=None,
    start_time=None,
):
    """
    Create a mock cfdb grid dataset for testing.

    Parameters
    ----------
    path : pathlib.Path
        Path for the cfdb dataset.
    variables : list[str]
        Variable names to create.
    n_times : int
        Number of timesteps (hourly from start_time).
    n_y, n_x : int
        Spatial dimensions.
    data_func : callable, optional
        Function(var_name, shape) -> np.ndarray. If None, random data.
    include_latlon : bool
        Add latitude/longitude as 1D coordinates.
    lat_range, lon_range : tuple, optional
        (min, max) for lat/lon coordinates.
    start_time : np.datetime64, optional
        Start time for time coordinate. Default '2020-09-30T00:00'.
    """
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

        if include_latlon:
            lr = lat_range or (-47.0, -42.0)
            lnr = lon_range or (165.0, 175.0)
            ds.create.coord.generic('latitude', data=np.linspace(lr[0], lr[1], n_y, dtype='float32'))
            ds.create.coord.generic('longitude', data=np.linspace(lnr[0], lnr[1], n_x, dtype='float32'))

        for var_name in variables:
            shape = (n_times, n_y, n_x)
            data = data_func(var_name, shape) if data_func else (np.random.rand(*shape).astype(np.float32) * 100 + 1)
            var = ds.create.data_var.generic(var_name, ('time', 'height', 'y', 'x'), dtype='float32')
            for t in range(n_times):
                var[(t, 0, slice(None), slice(None))] = data[t]

    return path


def create_mock_cyclone_cfdb(path, n_times=5, n_y=50, n_x=60, is_test=False):
    """Create a mock cfdb dataset with cyclone-like pressure patterns."""
    times = np.array([
        np.datetime64('2020-09-30T00:00') + np.timedelta64(i * 6, 'h')
        for i in range(n_times)
    ])
    lats = np.linspace(-48.0, -42.0, n_y, dtype='float32')
    lons = np.linspace(165.0, 175.0, n_x, dtype='float32')
    y_vals = np.arange(n_y, dtype='float32') * 10000.0
    x_vals = np.arange(n_x, dtype='float32') * 10000.0
    heights = np.array([0.0], dtype='float32')

    cyclone_lats = np.linspace(-46.0, -45.0, n_times)
    cyclone_lons = np.linspace(168.0, 170.0, n_times)
    lat_offset = 0.2 if is_test else 0.0
    lon_offset = 0.3 if is_test else 0.0

    lon_grid, lat_grid = np.meshgrid(lons, lats)

    with cfdb.open_dataset(path, 'n', dataset_type='grid') as ds:
        ds.create.coord.time(data=times)
        ds.create.coord.y(data=y_vals)
        ds.create.coord.x(data=x_vals)
        ds.create.coord.height(data=heights)
        ds.create.coord.generic('latitude', data=lats, dtype='float32')
        ds.create.coord.generic('longitude', data=lons, dtype='float32')

        mslp_var = ds.create.data_var.generic('mslp', ('time', 'height', 'y', 'x'), dtype='float32')
        precip_var = ds.create.data_var.generic('precipitation', ('time', 'height', 'y', 'x'), dtype='float32')

        for t in range(n_times):
            slp = np.full((n_y, n_x), 101325.0, dtype=np.float32)
            center_lat = cyclone_lats[t] + lat_offset
            center_lon = cyclone_lons[t] + lon_offset
            dlat = lat_grid - center_lat
            dlon = lon_grid - center_lon
            dist = np.sqrt(dlat**2 + (dlon * np.cos(np.radians(center_lat)))**2)
            slp -= 2000 * np.exp(-dist**2 / (2 * 2**2))
            mslp_var[(t, 0, slice(None), slice(None))] = slp

            precip = np.random.uniform(0, 10, (n_y, n_x)).astype(np.float32)
            precip_var[(t, 0, slice(None), slice(None))] = precip

    return path


class TestEvaluateModelsCell:
    """Tests for evaluate_models_cell using cfdb datasets."""

    def test_basic_ne_evaluation(self, tmp_path):
        """Should create output with NE values for each variable."""
        source_path = tmp_path / 'source.cfdb'
        test_path = tmp_path / 'test.cfdb'
        variables = ['air_temperature', 'u_wind']

        create_mock_cfdb(
            source_path, variables,
            data_func=lambda var, shape: np.ones(shape, dtype=np.float32) * 100,
        )
        create_mock_cfdb(
            test_path, variables,
            data_func=lambda var, shape: np.ones(shape, dtype=np.float32) * 110,
        )

        output_path = tmp_path / 'output.cfdb'
        result = evaluate_models_cell(source_path, test_path, output_path, variables=variables)

        assert result == output_path
        assert output_path.exists()

        with cfdb.open_dataset(output_path) as ds:
            assert 'air_temperature_ne' in ds.data_var_names
            assert 'u_wind_ne' in ds.data_var_names

            # NE should be 10% everywhere ((110-100)/100*100)
            data = ds['air_temperature_ne'][(0, slice(None), slice(None))].data[0]
            np.testing.assert_allclose(data, 10.0, atol=1)

    def test_single_metric_string(self, tmp_path):
        """Should accept a single metric as a string."""
        source_path = tmp_path / 'source.cfdb'
        test_path = tmp_path / 'test.cfdb'

        create_mock_cfdb(source_path, ['air_temperature'])
        create_mock_cfdb(test_path, ['air_temperature'])

        output_path = tmp_path / 'output.cfdb'
        evaluate_models_cell(
            source_path, test_path, output_path,
            variables=['air_temperature'], metrics='ane',
        )

        with cfdb.open_dataset(output_path) as ds:
            assert 'air_temperature_ane' in ds.data_var_names
            assert 'air_temperature_ne' not in ds.data_var_names

    def test_multiple_metrics(self, tmp_path):
        """Should compute multiple metrics when provided as a list."""
        source_path = tmp_path / 'source.cfdb'
        test_path = tmp_path / 'test.cfdb'

        create_mock_cfdb(
            source_path, ['air_temperature'],
            data_func=lambda var, shape: np.ones(shape, dtype=np.float32) * 100,
        )
        create_mock_cfdb(
            test_path, ['air_temperature'],
            data_func=lambda var, shape: np.ones(shape, dtype=np.float32) * 110,
        )

        output_path = tmp_path / 'output.cfdb'
        evaluate_models_cell(
            source_path, test_path, output_path,
            variables=['air_temperature'], metrics=['ne', 'ane', 'rse'],
        )

        with cfdb.open_dataset(output_path) as ds:
            assert 'air_temperature_ne' in ds.data_var_names
            assert 'air_temperature_ane' in ds.data_var_names
            assert 'air_temperature_rse' in ds.data_var_names

    def test_all_available_metrics(self, tmp_path):
        """Should be able to compute all available cell metrics."""
        source_path = tmp_path / 'source.cfdb'
        test_path = tmp_path / 'test.cfdb'

        create_mock_cfdb(source_path, ['air_temperature'])
        create_mock_cfdb(test_path, ['air_temperature'])

        output_path = tmp_path / 'output.cfdb'
        evaluate_models_cell(
            source_path, test_path, output_path,
            variables=['air_temperature'],
            metrics=list(AVAILABLE_METRICS),
            threshold=1.0,
        )

        with cfdb.open_dataset(output_path) as ds:
            for metric in AVAILABLE_METRICS:
                assert f'air_temperature_{metric}' in ds.data_var_names

    def test_raises_on_invalid_metric(self, tmp_path):
        """Should raise ValueError for unknown metric."""
        source_path = tmp_path / 'source.cfdb'
        test_path = tmp_path / 'test.cfdb'

        create_mock_cfdb(source_path, ['air_temperature'])
        create_mock_cfdb(test_path, ['air_temperature'])

        with pytest.raises(ValueError, match="Unknown metric"):
            evaluate_models_cell(
                source_path, test_path, tmp_path / 'output.cfdb',
                variables=['air_temperature'], metrics='invalid_metric',
            )

    def test_raises_on_missing_source(self, tmp_path):
        """Should raise FileNotFoundError if source dataset doesn't exist."""
        test_path = tmp_path / 'test.cfdb'
        create_mock_cfdb(test_path, ['air_temperature'])

        with pytest.raises(FileNotFoundError):
            evaluate_models_cell(
                tmp_path / 'nonexistent.cfdb', test_path,
                tmp_path / 'output.cfdb', variables=['air_temperature'],
            )

    def test_raises_on_missing_test(self, tmp_path):
        """Should raise FileNotFoundError if test dataset doesn't exist."""
        source_path = tmp_path / 'source.cfdb'
        create_mock_cfdb(source_path, ['air_temperature'])

        with pytest.raises(FileNotFoundError):
            evaluate_models_cell(
                source_path, tmp_path / 'nonexistent.cfdb',
                tmp_path / 'output.cfdb', variables=['air_temperature'],
            )

    def test_raises_on_missing_variable(self, tmp_path):
        """Should raise ValueError if variable not found in dataset."""
        source_path = tmp_path / 'source.cfdb'
        test_path = tmp_path / 'test.cfdb'

        create_mock_cfdb(source_path, ['air_temperature'])
        create_mock_cfdb(test_path, ['air_temperature'])

        with pytest.raises(ValueError, match="not found"):
            evaluate_models_cell(
                source_path, test_path, tmp_path / 'output.cfdb',
                variables=['nonexistent_var'],
            )

    def test_raises_on_spatial_shape_mismatch(self, tmp_path):
        """Should raise ValueError if source and test spatial shapes don't match."""
        source_path = tmp_path / 'source.cfdb'
        test_path = tmp_path / 'test.cfdb'

        create_mock_cfdb(source_path, ['air_temperature'], n_y=10, n_x=10)
        create_mock_cfdb(test_path, ['air_temperature'], n_y=8, n_x=8)

        with pytest.raises(ValueError, match="mismatch"):
            evaluate_models_cell(
                source_path, test_path, tmp_path / 'output.cfdb',
                variables=['air_temperature'],
            )

    def test_no_common_timesteps_raises(self, tmp_path):
        """Should raise ValueError if no common timesteps between datasets."""
        source_path = tmp_path / 'source.cfdb'
        test_path = tmp_path / 'test.cfdb'

        create_mock_cfdb(source_path, ['air_temperature'], n_times=2)
        create_mock_cfdb(
            test_path, ['air_temperature'], n_times=2,
            start_time=np.datetime64('2020-10-15T00:00'),
        )

        with pytest.raises(ValueError, match="No common timesteps"):
            evaluate_models_cell(
                source_path, test_path, tmp_path / 'output.cfdb',
                variables=['air_temperature'],
            )

    def test_creates_output_directory(self, tmp_path):
        """Should create output directory if it doesn't exist."""
        source_path = tmp_path / 'source.cfdb'
        test_path = tmp_path / 'test.cfdb'

        create_mock_cfdb(source_path, ['air_temperature'])
        create_mock_cfdb(test_path, ['air_temperature'])

        output_path = tmp_path / 'nested' / 'dir' / 'output.cfdb'
        evaluate_models_cell(
            source_path, test_path, output_path,
            variables=['air_temperature'],
        )

        assert output_path.exists()

    def test_time_filtering(self, tmp_path):
        """Should filter by start_time and end_time."""
        source_path = tmp_path / 'source.cfdb'
        test_path = tmp_path / 'test.cfdb'

        create_mock_cfdb(source_path, ['air_temperature'], n_times=10)
        create_mock_cfdb(test_path, ['air_temperature'], n_times=10)

        output_path = tmp_path / 'output.cfdb'
        evaluate_models_cell(
            source_path, test_path, output_path,
            variables=['air_temperature'],
            start_time='2020-09-30T02:00',
            end_time='2020-09-30T05:00',
        )

        with cfdb.open_dataset(output_path) as ds:
            times = ds['time'].data
            assert len(times) == 4  # hours 2, 3, 4, 5

    def test_handles_different_timestep_counts(self, tmp_path):
        """Should use only common timesteps when counts differ."""
        source_path = tmp_path / 'source.cfdb'
        test_path = tmp_path / 'test.cfdb'

        create_mock_cfdb(source_path, ['air_temperature'], n_times=6)
        create_mock_cfdb(test_path, ['air_temperature'], n_times=3)

        output_path = tmp_path / 'output.cfdb'
        evaluate_models_cell(
            source_path, test_path, output_path,
            variables=['air_temperature'],
        )

        with cfdb.open_dataset(output_path) as ds:
            times = ds['time'].data
            assert len(times) == 3  # common timesteps

    def test_spatial_mask(self, tmp_path):
        """Should apply 2D spatial mask when region is a boolean array."""
        source_path = tmp_path / 'source.cfdb'
        test_path = tmp_path / 'test.cfdb'
        n_y, n_x = 10, 10

        create_mock_cfdb(
            source_path, ['air_temperature'], n_y=n_y, n_x=n_x,
            data_func=lambda var, shape: np.ones(shape, dtype=np.float32) * 100,
        )
        create_mock_cfdb(
            test_path, ['air_temperature'], n_y=n_y, n_x=n_x,
            data_func=lambda var, shape: np.ones(shape, dtype=np.float32) * 110,
        )

        mask = np.zeros((n_y, n_x), dtype=bool)
        mask[3:7, 3:7] = True

        output_path = tmp_path / 'output.cfdb'
        evaluate_models_cell(
            source_path, test_path, output_path,
            variables=['air_temperature'], region=mask,
        )

        with cfdb.open_dataset(output_path) as ds:
            assert 'spatial_mask' in ds.data_var_names

            ne_data = ds['air_temperature_ne'][(0, slice(None), slice(None))].data[0]
            # Inside mask: NE = 10%
            np.testing.assert_allclose(ne_data[3:7, 3:7], 10.0, atol=1)
            # Outside mask: NaN
            assert np.isnan(ne_data[0, 0])

    def test_raises_on_mask_shape_mismatch(self, tmp_path):
        """Should raise ValueError when mask shape doesn't match domain."""
        source_path = tmp_path / 'source.cfdb'
        test_path = tmp_path / 'test.cfdb'

        create_mock_cfdb(source_path, ['air_temperature'], n_y=10, n_x=10)
        create_mock_cfdb(test_path, ['air_temperature'], n_y=10, n_x=10)

        wrong_mask = np.ones((5, 5), dtype=bool)

        with pytest.raises(ValueError, match="Mask shape .* does not match"):
            evaluate_models_cell(
                source_path, test_path, tmp_path / 'output.cfdb',
                variables=['air_temperature'], region=wrong_mask,
            )

    def test_raises_on_3d_mask(self, tmp_path):
        """Should raise ValueError for 3D mask array."""
        source_path = tmp_path / 'source.cfdb'
        test_path = tmp_path / 'test.cfdb'

        create_mock_cfdb(source_path, ['air_temperature'], n_y=5, n_x=5)
        create_mock_cfdb(test_path, ['air_temperature'], n_y=5, n_x=5)

        mask_3d = np.ones((2, 5, 5), dtype=bool)

        with pytest.raises(ValueError, match="Spatial mask must be 2D"):
            evaluate_models_cell(
                source_path, test_path, tmp_path / 'output.cfdb',
                variables=['air_temperature'], region=mask_3d,
            )

    def test_raises_on_invalid_region_type(self, tmp_path):
        """Should raise ValueError for invalid region type."""
        source_path = tmp_path / 'source.cfdb'
        test_path = tmp_path / 'test.cfdb'

        create_mock_cfdb(source_path, ['air_temperature'])
        create_mock_cfdb(test_path, ['air_temperature'])

        with pytest.raises(ValueError, match="region must be either"):
            evaluate_models_cell(
                source_path, test_path, tmp_path / 'output.cfdb',
                variables=['air_temperature'], region="invalid",
            )


class TestEvaluateModelsDomain:
    """Tests for evaluate_models_domain using cfdb datasets."""

    def test_basic_domain_evaluation(self, tmp_path):
        """Should create output with domain-aggregated metrics."""
        source_path = tmp_path / 'source.cfdb'
        test_path = tmp_path / 'test.cfdb'

        create_mock_cfdb(
            source_path, ['air_temperature'],
            data_func=lambda var, shape: np.ones(shape, dtype=np.float32) * 100,
        )
        create_mock_cfdb(
            test_path, ['air_temperature'],
            data_func=lambda var, shape: np.ones(shape, dtype=np.float32) * 110,
        )

        output_path = tmp_path / 'output.cfdb'
        result = evaluate_models_domain(
            source_path, test_path, output_path,
            variables=['air_temperature'], metrics=['ne', 'ane', 'rmse'],
        )

        assert result == output_path
        assert output_path.exists()

        with cfdb.open_dataset(output_path) as ds:
            assert 'air_temperature' in ds.data_var_names
            assert 'metric' in ds.coord_names

    def test_domain_ne_value(self, tmp_path):
        """Should compute correct domain-aggregated NE value."""
        source_path = tmp_path / 'source.cfdb'
        test_path = tmp_path / 'test.cfdb'

        create_mock_cfdb(
            source_path, ['air_temperature'], n_times=1,
            data_func=lambda var, shape: np.ones(shape, dtype=np.float32) * 100,
        )
        create_mock_cfdb(
            test_path, ['air_temperature'], n_times=1,
            data_func=lambda var, shape: np.ones(shape, dtype=np.float32) * 110,
        )

        output_path = tmp_path / 'output.cfdb'
        evaluate_models_domain(
            source_path, test_path, output_path,
            variables=['air_temperature'], metrics=['ne'],
        )

        with cfdb.open_dataset(output_path) as ds:
            data = ds['air_temperature'][(0, slice(None))].data
            # NE = (110 - 100) / 100 * 100 = 10%
            np.testing.assert_allclose(data[0, 0], 10.0, rtol=1e-3)

    def test_all_domain_metrics(self, tmp_path):
        """Should be able to compute all available domain metrics."""
        source_path = tmp_path / 'source.cfdb'
        test_path = tmp_path / 'test.cfdb'

        create_mock_cfdb(source_path, ['air_temperature'])
        create_mock_cfdb(test_path, ['air_temperature'])

        output_path = tmp_path / 'output.cfdb'
        evaluate_models_domain(
            source_path, test_path, output_path,
            variables=['air_temperature'],
            metrics=list(AVAILABLE_DOMAIN_METRICS),
            threshold=1.0,
        )

        with cfdb.open_dataset(output_path) as ds:
            assert 'air_temperature' in ds.data_var_names
            var = ds['air_temperature']
            assert var.shape[1] == len(AVAILABLE_DOMAIN_METRICS)

    def test_categorical_metrics(self, tmp_path):
        """Should compute categorical metrics when threshold is provided."""
        source_path = tmp_path / 'source.cfdb'
        test_path = tmp_path / 'test.cfdb'
        n_y, n_x = 10, 10

        # Source: 10 hits (row 0) + 5 misses (row 1, first 5) = 15 yes
        def source_data(var, shape):
            data = np.zeros(shape, dtype=np.float32)
            data[0, 0, 0:10] = 2.0  # hits
            data[0, 1, 0:5] = 2.0   # misses
            return data

        # Test: 10 hits (row 0) + 5 false alarms (row 2, first 5)
        def test_data(var, shape):
            data = np.zeros(shape, dtype=np.float32)
            data[0, 0, 0:10] = 2.0  # hits
            data[0, 2, 0:5] = 2.0   # false alarms
            return data

        create_mock_cfdb(
            source_path, ['precipitation'], n_times=1, n_y=n_y, n_x=n_x,
            data_func=source_data,
        )
        create_mock_cfdb(
            test_path, ['precipitation'], n_times=1, n_y=n_y, n_x=n_x,
            data_func=test_data,
        )

        output_path = tmp_path / 'output.cfdb'
        evaluate_models_domain(
            source_path, test_path, output_path,
            variables=['precipitation'], metrics=['pod', 'far'], threshold=1.0,
        )

        with cfdb.open_dataset(output_path) as ds:
            data = ds['precipitation'][(0, slice(None))].data
            # POD = Hits / (Hits + Misses) = 10 / 15
            np.testing.assert_allclose(data[0, 0], 10 / 15, rtol=1e-5)
            # FAR = FA / (Hits + FA) = 5 / 15
            np.testing.assert_allclose(data[0, 1], 5 / 15, rtol=1e-5)


class TestEvaluateCyclones:
    """Tests for evaluate_cyclones using cfdb datasets."""

    @pytest.fixture
    def mock_cyclone_files(self, tmp_path):
        """Create mock cfdb datasets with cyclone-like pressure patterns."""
        source_path = tmp_path / 'source_cyclone.cfdb'
        test_path = tmp_path / 'test_cyclone.cfdb'
        create_mock_cyclone_cfdb(source_path, is_test=False)
        create_mock_cyclone_cfdb(test_path, is_test=True)
        return source_path, test_path

    def test_basic_cyclone_evaluation(self, mock_cyclone_files, tmp_path):
        """Test basic cyclone evaluation with multiple metrics."""
        source_path, test_path = mock_cyclone_files
        output_path = tmp_path / 'cyclone_eval.cfdb'

        result = evaluate_cyclones(
            source_path, test_path, output_path,
            variables=['precipitation'],
            metrics=['ne', 'ane'],
            start_lat=-46.0,
            start_lon=168.0,
        )

        assert result.exists()

        with cfdb.open_dataset(output_path) as ds:
            var_names = set(ds.data_var_names)

            # Track variables
            assert 'source_latitude' in var_names
            assert 'source_longitude' in var_names
            assert 'source_pressure' in var_names
            assert 'source_radius' in var_names
            assert 'test_latitude' in var_names
            assert 'test_longitude' in var_names
            assert 'test_pressure' in var_names
            assert 'test_radius' in var_names

            # Comparison variables
            assert 'position_difference_km' in var_names
            assert 'pressure_difference' in var_names
            assert 'radius_difference' in var_names

            # Evaluation variable shape: (n_times=5, n_metrics=2)
            assert 'precipitation' in var_names
            precip_var = ds['precipitation']
            assert precip_var.shape == (5, 2)

            # Check metric coordinate
            assert 'metric' in ds.coord_names

    def test_cyclone_tracks_differ(self, mock_cyclone_files, tmp_path):
        """Source and test tracks should differ due to offset in mock data."""
        source_path, test_path = mock_cyclone_files
        output_path = tmp_path / 'cyclone_eval.cfdb'

        evaluate_cyclones(
            source_path, test_path, output_path,
            variables=['precipitation'],
            start_lat=-46.0,
            start_lon=168.0,
        )

        with cfdb.open_dataset(output_path) as ds:
            s_lats = np.array([ds['source_latitude'][(t,)].data[0] for t in range(5)])
            t_lats = np.array([ds['test_latitude'][(t,)].data[0] for t in range(5)])
            pos_diffs = np.array([ds['position_difference_km'][(t,)].data[0] for t in range(5)])

            # Tracks should differ
            assert not np.allclose(s_lats, t_lats, atol=0.1)
            # Position difference should be non-zero
            assert np.all(pos_diffs > 0)

    def test_all_domain_metrics_cyclone(self, mock_cyclone_files, tmp_path):
        """Test cyclone evaluation with all available domain metrics."""
        source_path, test_path = mock_cyclone_files
        output_path = tmp_path / 'cyclone_eval_all.cfdb'

        evaluate_cyclones(
            source_path, test_path, output_path,
            variables=['precipitation'],
            metrics=list(AVAILABLE_DOMAIN_METRICS),
            start_lat=-46.0,
            start_lon=168.0,
        )

        with cfdb.open_dataset(output_path) as ds:
            assert ds['precipitation'].shape[1] == len(AVAILABLE_DOMAIN_METRICS)

    def test_cyclone_with_smoothing(self, mock_cyclone_files, tmp_path):
        """Test cyclone evaluation with SLP smoothing."""
        source_path, test_path = mock_cyclone_files
        output_path = tmp_path / 'cyclone_eval_smooth.cfdb'

        result = evaluate_cyclones(
            source_path, test_path, output_path,
            variables=['precipitation'],
            start_lat=-46.0,
            start_lon=168.0,
            smoothing_sigma=2.0,
        )

        assert result.exists()

    def test_cyclone_invalid_metric(self, mock_cyclone_files, tmp_path):
        """Should raise ValueError for invalid metric."""
        source_path, test_path = mock_cyclone_files

        with pytest.raises(ValueError, match="Unknown metric"):
            evaluate_cyclones(
                source_path, test_path, tmp_path / 'output.cfdb',
                variables=['precipitation'],
                metrics=['invalid_metric'],
            )

    def test_cyclone_missing_variable(self, mock_cyclone_files, tmp_path):
        """Should raise ValueError for missing variable."""
        source_path, test_path = mock_cyclone_files

        with pytest.raises(ValueError, match="not found"):
            evaluate_cyclones(
                source_path, test_path, tmp_path / 'output.cfdb',
                variables=['nonexistent_var'],
            )


class TestEvaluateModelsCellIntegration:
    """Integration tests using real cfdb data.

    Skipped unless --source-dataset and --test-dataset are provided.

    Example usage:
        pytest --source-dataset=/path/to/source.cfdb --test-dataset=/path/to/test.cfdb \\
               --variables=air_temperature,u_wind \\
               --start-time=2020-09-30 --end-time=2020-10-15
    """

    def test_real_data_evaluation(
        self, real_datasets, variables, start_time, end_time, tmp_path
    ):
        """Test evaluation with real cfdb model data using all metrics."""
        source_dataset, test_dataset = real_datasets

        output_path = tmp_path / 'real_data_output.cfdb'
        result = evaluate_models_cell(
            source_dataset,
            test_dataset,
            output_path,
            variables=variables,
            metrics=list(AVAILABLE_METRICS),
            start_time=start_time,
            end_time=end_time,
            threshold=1.0,
        )

        assert result.exists()

        with cfdb.open_dataset(result) as ds:
            for var in variables:
                for metric in AVAILABLE_METRICS:
                    ds_name = f'{var}_{metric}'
                    assert ds_name in ds.data_var_names, f"Variable {ds_name} not found in output"

            assert 'time' in ds.coord_names


class TestEvaluateModelsDomainIntegration:
    """Integration tests for domain evaluation using real cfdb data.

    Skipped unless --source-dataset and --test-dataset are provided.
    """

    def test_real_data_domain_evaluation(
        self, real_datasets, variables, start_time, end_time, tmp_path
    ):
        """Test domain-aggregated evaluation with real model data."""
        source_dataset, test_dataset = real_datasets

        output_path = tmp_path / 'real_data_domain_output.cfdb'
        result = evaluate_models_domain(
            source_dataset,
            test_dataset,
            output_path,
            variables=variables,
            metrics=list(AVAILABLE_DOMAIN_METRICS),
            start_time=start_time,
            end_time=end_time,
            threshold=1.0,
        )

        assert result.exists()

        with cfdb.open_dataset(result) as ds:
            for var in variables:
                assert var in ds.data_var_names
                assert ds[var].shape[1] == len(AVAILABLE_DOMAIN_METRICS)

            assert 'time' in ds.coord_names
            assert 'metric' in ds.coord_names


class TestEvaluateCyclonesIntegration:
    """Integration tests for cyclone evaluation using real cfdb data.

    Skipped unless --source-dataset and --test-dataset are provided,
    along with --cyclone-start-lat and --cyclone-start-lon.
    """

    def test_real_data_cyclone_evaluation(
        self, real_datasets, variables, cyclone_start_lat, cyclone_start_lon, tmp_path
    ):
        """Test cyclone evaluation with real model data."""
        source_dataset, test_dataset = real_datasets

        if cyclone_start_lat is None or cyclone_start_lon is None:
            pytest.skip("Cyclone start position not provided. Use --cyclone-start-lat and --cyclone-start-lon.")

        output_path = tmp_path / 'real_cyclone_eval.cfdb'
        result = evaluate_cyclones(
            source_dataset,
            test_dataset,
            output_path,
            variables=variables,
            metrics=list(AVAILABLE_DOMAIN_METRICS),
            start_lat=cyclone_start_lat,
            start_lon=cyclone_start_lon,
            threshold=1.0,
        )

        assert result.exists()

        with cfdb.open_dataset(result) as ds:
            assert 'source_latitude' in ds.data_var_names
            assert 'test_latitude' in ds.data_var_names
            assert 'position_difference_km' in ds.data_var_names
