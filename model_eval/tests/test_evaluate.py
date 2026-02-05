"""
Tests for model_eval.evaluate module.
"""
import pathlib
from datetime import date

import h5py
import numpy as np
import pytest

from model_eval.evaluate import (
    AVAILABLE_DOMAIN_METRICS,
    AVAILABLE_METRICS,
    _find_latlon_bounds,
    compute_ane,
    compute_ane_domain,
    compute_ne,
    compute_ne_domain,
    compute_rmse_domain,
    compute_rse,
    evaluate_models_cell,
    evaluate_models_domain,
    find_wrfout_files,
)

# Southland region of South Island, New Zealand (approximate bounds)
SOUTHLAND_BOUNDS = (-46.5, -45.0, 166.5, 169.5)  # (min_lat, max_lat, min_lon, max_lon)


def make_wrfout_filename(domain: int, date_str: str, time_str: str = "00:00:00") -> str:
    """Generate a WRF output filename for testing."""
    return f"wrfout_d{domain:02d}_{date_str}_{time_str}"


class TestFindWrfoutFiles:
    """Tests for find_wrfout_files function."""

    def test_finds_matching_domain_files(self, tmp_path):
        """Should find wrfout files matching the specified domain."""
        # Create test files
        (tmp_path / make_wrfout_filename(4, "2020-09-30")).touch()
        (tmp_path / make_wrfout_filename(4, "2020-10-01")).touch()
        (tmp_path / make_wrfout_filename(3, "2020-09-30")).touch()  # Different domain

        result = find_wrfout_files(tmp_path, domain=4)

        assert len(result) == 2
        assert date(2020, 9, 30) in result
        assert date(2020, 10, 1) in result

    def test_domain_formatting(self, tmp_path):
        """Should correctly match domain with zero-padded format (d01, d04, d12)."""
        (tmp_path / make_wrfout_filename(1, "2020-09-30")).touch()
        (tmp_path / make_wrfout_filename(4, "2020-09-30")).touch()
        (tmp_path / make_wrfout_filename(12, "2020-09-30")).touch()

        assert len(find_wrfout_files(tmp_path, domain=1)) == 1
        assert len(find_wrfout_files(tmp_path, domain=4)) == 1
        assert len(find_wrfout_files(tmp_path, domain=12)) == 1
        assert len(find_wrfout_files(tmp_path, domain=2)) == 0

    def test_ignores_non_wrfout_files(self, tmp_path):
        """Should ignore files that don't start with wrfout_."""
        (tmp_path / make_wrfout_filename(4, "2020-09-30")).touch()
        (tmp_path / 'other_file.nc').touch()
        (tmp_path / 'wrfinput_d04').touch()

        result = find_wrfout_files(tmp_path, domain=4)

        assert len(result) == 1

    def test_ignores_directories(self, tmp_path):
        """Should ignore directories even if named like wrfout files."""
        (tmp_path / make_wrfout_filename(4, "2020-09-30")).touch()
        (tmp_path / make_wrfout_filename(4, "2020-10-01")).mkdir()

        result = find_wrfout_files(tmp_path, domain=4)

        assert len(result) == 1

    def test_empty_folder(self, tmp_path):
        """Should return empty dict for empty folder."""
        result = find_wrfout_files(tmp_path, domain=4)

        assert result == {}

    def test_no_matching_domain(self, tmp_path):
        """Should return empty dict when no files match domain."""
        (tmp_path / make_wrfout_filename(3, "2020-09-30")).touch()

        result = find_wrfout_files(tmp_path, domain=4)

        assert result == {}

    def test_start_date_filter(self, tmp_path):
        """Should filter files by start date (inclusive)."""
        (tmp_path / make_wrfout_filename(4, "2020-09-28")).touch()
        (tmp_path / make_wrfout_filename(4, "2020-09-29")).touch()
        (tmp_path / make_wrfout_filename(4, "2020-09-30")).touch()
        (tmp_path / make_wrfout_filename(4, "2020-10-01")).touch()

        result = find_wrfout_files(tmp_path, domain=4, start_date=date(2020, 9, 30))

        assert len(result) == 2
        assert date(2020, 9, 30) in result
        assert date(2020, 10, 1) in result

    def test_end_date_filter(self, tmp_path):
        """Should filter files by end date (inclusive)."""
        (tmp_path / make_wrfout_filename(4, "2020-09-28")).touch()
        (tmp_path / make_wrfout_filename(4, "2020-09-29")).touch()
        (tmp_path / make_wrfout_filename(4, "2020-09-30")).touch()
        (tmp_path / make_wrfout_filename(4, "2020-10-01")).touch()

        result = find_wrfout_files(tmp_path, domain=4, end_date=date(2020, 9, 29))

        assert len(result) == 2
        assert date(2020, 9, 28) in result
        assert date(2020, 9, 29) in result

    def test_date_range_filter(self, tmp_path):
        """Should filter files by both start and end date."""
        (tmp_path / make_wrfout_filename(4, "2020-09-28")).touch()
        (tmp_path / make_wrfout_filename(4, "2020-09-29")).touch()
        (tmp_path / make_wrfout_filename(4, "2020-09-30")).touch()
        (tmp_path / make_wrfout_filename(4, "2020-10-01")).touch()

        result = find_wrfout_files(
            tmp_path, domain=4, start_date=date(2020, 9, 29), end_date=date(2020, 9, 30)
        )

        assert len(result) == 2
        assert date(2020, 9, 29) in result
        assert date(2020, 9, 30) in result


class TestComputeNE:
    """Tests for compute_normalised_error function."""

    def test_basic_calculation(self):
        """Should compute NE = ((test - source) / source) * 100."""
        source = np.array([100.0, 200.0, 50.0])
        test = np.array([110.0, 180.0, 75.0])

        result = compute_ne(source, test)

        expected = np.array([10, -10, 50], dtype=np.int16)
        np.testing.assert_array_equal(result, expected)

    def test_handles_zero_source(self):
        """Should return 0 when source is near zero."""
        source = np.array([0.0, 1e-15, 100.0])
        test = np.array([10.0, 10.0, 110.0])

        result = compute_ne(source, test, epsilon=1e-10)

        assert result[0] == 0
        assert result[1] == 0
        assert result[2] == 10

    def test_clips_to_int16_range(self):
        """Should clip extreme values to int16 range."""
        source = np.array([1.0, 1.0])
        test = np.array([1000.0, -1000.0])  # 99900% and -100100%

        result = compute_ne(source, test)

        assert result[0] == 32767  # INT16_MAX
        assert result[1] == -32768  # INT16_MIN

    def test_handles_nan_and_inf(self):
        """Should handle NaN and Inf values gracefully."""
        source = np.array([1.0, 0.0, -0.0])
        test = np.array([np.nan, 1.0, 1.0])

        result = compute_ne(source, test)

        # All should be valid int16 values (no NaN/Inf)
        assert np.all(np.isfinite(result.astype(float)))

    def test_multidimensional_array(self):
        """Should work with multidimensional arrays."""
        source = np.ones((2, 3, 4)) * 100
        test = np.ones((2, 3, 4)) * 110

        result = compute_ne(source, test)

        assert result.shape == (2, 3, 4)
        assert np.all(result == 10)

    def test_returns_int16_dtype(self):
        """Should always return int16 dtype."""
        source = np.array([100.0], dtype=np.float64)
        test = np.array([110.0], dtype=np.float64)

        result = compute_ne(source, test)

        assert result.dtype == np.int16


class TestComputeANE:
    """Tests for compute_mean_absolute_normalised_error function."""

    def test_basic_calculation(self):
        """Should compute ANE = |((test - source) / source)| * 100."""
        source = np.array([100.0, 200.0, 50.0])
        test = np.array([110.0, 180.0, 75.0])  # +10%, -10%, +50%

        result = compute_ane(source, test)

        expected = np.array([10, 10, 50], dtype=np.int16)  # All positive
        np.testing.assert_array_equal(result, expected)

    def test_always_positive(self):
        """Should always return positive values."""
        source = np.array([100.0, 100.0])
        test = np.array([50.0, 150.0])  # -50% and +50%

        result = compute_ane(source, test)

        assert np.all(result >= 0)
        np.testing.assert_array_equal(result, [50, 50])

    def test_handles_zero_source(self):
        """Should return 0 when source is near zero."""
        source = np.array([0.0, 100.0])
        test = np.array([10.0, 110.0])

        result = compute_ane(source, test, epsilon=1e-10)

        assert result[0] == 0
        assert result[1] == 10

    def test_returns_int16_dtype(self):
        """Should always return int16 dtype."""
        source = np.array([100.0], dtype=np.float64)
        test = np.array([110.0], dtype=np.float64)

        result = compute_ane(source, test)

        assert result.dtype == np.int16


class TestComputeRSE:
    """Tests for compute_rse function."""

    def test_basic_calculation(self):
        """Should compute RSE = sqrt((test - source)^2)."""
        source = np.array([100.0, 200.0, 50.0])
        test = np.array([110.0, 180.0, 75.0])  # diff: 10, -20, 25

        result = compute_rse(source, test)

        expected = np.array([10.0, 20.0, 25.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)

    def test_always_positive(self):
        """Should always return positive values."""
        source = np.array([100.0, 100.0])
        test = np.array([50.0, 150.0])  # diff: -50 and +50

        result = compute_rse(source, test)

        assert np.all(result >= 0)
        np.testing.assert_array_almost_equal(result, [50.0, 50.0])

    def test_zero_when_identical(self):
        """Should return 0 when source and test are identical."""
        source = np.array([100.0, 200.0, 50.0])
        test = source.copy()

        result = compute_rse(source, test)

        np.testing.assert_array_equal(result, [0.0, 0.0, 0.0])

    def test_returns_float32_dtype(self):
        """Should always return float32 dtype."""
        source = np.array([100.0], dtype=np.float64)
        test = np.array([110.0], dtype=np.float64)

        result = compute_rse(source, test)

        assert result.dtype == np.float32


def create_mock_wrfout(path: pathlib.Path, variables: list[str], shape: tuple, data_func=None):
    """
    Create a mock WRF output file for testing.

    Parameters
    ----------
    path : pathlib.Path
        Output file path.
    variables : list[str]
        Variable names to create.
    shape : tuple
        Shape of each variable (time, y, x).
    data_func : callable, optional
        Function to generate data. If None, uses random data.
    """
    with h5py.File(path, 'w') as f:
        for var in variables:
            if data_func is not None:
                data = data_func(var, shape)
            else:
                data = np.random.rand(*shape).astype(np.float32) * 100 + 1
            f.create_dataset(var, data=data, chunks=(1, shape[1], shape[2]))


def create_mock_wrfout_with_latlon(
    path: pathlib.Path,
    variables: list[str],
    shape: tuple,
    lat_range: tuple[float, float],
    lon_range: tuple[float, float],
    data_func=None,
):
    """
    Create a mock WRF output file with XLAT and XLONG for testing region subsetting.

    Parameters
    ----------
    path : pathlib.Path
        Output file path.
    variables : list[str]
        Variable names to create.
    shape : tuple
        Shape of each variable (time, y, x).
    lat_range : tuple
        (min_lat, max_lat) for the domain.
    lon_range : tuple
        (min_lon, max_lon) for the domain.
    data_func : callable, optional
        Function to generate data. If None, uses random data.
    """
    n_time, n_y, n_x = shape

    # Create lat/lon grids
    lats = np.linspace(lat_range[0], lat_range[1], n_y)
    lons = np.linspace(lon_range[0], lon_range[1], n_x)
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    # WRF stores XLAT/XLONG with time dimension (time, y, x)
    xlat = np.broadcast_to(lat_grid, (n_time, n_y, n_x)).astype(np.float32)
    xlong = np.broadcast_to(lon_grid, (n_time, n_y, n_x)).astype(np.float32)

    with h5py.File(path, 'w') as f:
        f.create_dataset('XLAT', data=xlat)
        f.create_dataset('XLONG', data=xlong)

        for var in variables:
            if data_func is not None:
                data = data_func(var, shape)
            else:
                data = np.random.rand(*shape).astype(np.float32) * 100 + 1
            f.create_dataset(var, data=data, chunks=(1, n_y, n_x))


class TestEvaluateModelsCell:
    """Tests for evaluate_models function using mock data."""

    def test_basic_evaluation(self, tmp_path):
        """Should create output file with NE values for each variable."""
        source_dir = tmp_path / 'source'
        test_dir = tmp_path / 'test'
        source_dir.mkdir()
        test_dir.mkdir()

        variables = ['T2', 'Q2']
        shape = (4, 10, 10)

        # Create source with constant value 100
        def source_data(var, shape):
            return np.ones(shape, dtype=np.float32) * 100

        # Create test with constant value 110 (10% increase)
        def test_data(var, shape):
            return np.ones(shape, dtype=np.float32) * 110

        source_file = source_dir / make_wrfout_filename(4, "2020-09-30")
        test_file = test_dir / make_wrfout_filename(4, "2020-09-30")
        create_mock_wrfout(source_file, variables, shape, source_data)
        create_mock_wrfout(test_file, variables, shape, test_data)

        output_path = tmp_path / 'output.nc'
        result = evaluate_models_cell(source_dir, test_dir, output_path, domain=4, variables=variables)

        assert result == output_path
        assert output_path.exists()

        with h5py.File(output_path, 'r') as f:
            assert 'T2_ne' in f
            assert 'Q2_ne' in f
            # Check NE is 10% everywhere
            np.testing.assert_array_equal(f['T2_ne'][:], 10)
            np.testing.assert_array_equal(f['Q2_ne'][:], 10)

    def test_netcdf4_compliance(self, tmp_path):
        """Should create NetCDF4-compliant output file."""
        source_dir = tmp_path / 'source'
        test_dir = tmp_path / 'test'
        source_dir.mkdir()
        test_dir.mkdir()

        variables = ['T2']
        shape = (2, 5, 5)

        source_file = source_dir / make_wrfout_filename(4, "2020-09-30")
        test_file = test_dir / make_wrfout_filename(4, "2020-09-30")
        create_mock_wrfout(source_file, variables, shape)
        create_mock_wrfout(test_file, variables, shape)

        output_path = tmp_path / 'output.nc'
        evaluate_models_cell(source_dir, test_dir, output_path, domain=4, variables=variables)

        with h5py.File(output_path, 'r') as f:
            # Check global attributes
            assert f.attrs['Conventions'] == b'CF-1.8'
            assert 'history' in f.attrs
            assert f.attrs['domain'] == 4

            # Check dimension scales exist
            assert 'time' in f
            assert 'y' in f
            assert 'x' in f

            # Check dimension attributes
            assert f['time'].attrs['CLASS'] == b'DIMENSION_SCALE'
            assert f['time'].attrs['units'] == b'hours since 1970-01-01'

            # Check variable attributes
            assert f['T2_ne'].attrs['units'] == b'percent'
            assert 'DIMENSION_LIST' in f['T2_ne'].attrs

    def test_multiple_dates(self, tmp_path):
        """Should handle multiple dates and concatenate along time dimension."""
        source_dir = tmp_path / 'source'
        test_dir = tmp_path / 'test'
        source_dir.mkdir()
        test_dir.mkdir()

        variables = ['T2']
        shape = (3, 5, 5)  # 3 timesteps per file

        # Create files for two dates
        for date_str in ['2020-09-30', '2020-10-01']:
            source_file = source_dir / make_wrfout_filename(4, date_str)
            test_file = test_dir / make_wrfout_filename(4, date_str)
            create_mock_wrfout(source_file, variables, shape)
            create_mock_wrfout(test_file, variables, shape)

        output_path = tmp_path / 'output.nc'
        evaluate_models_cell(source_dir, test_dir, output_path, domain=4, variables=variables)

        with h5py.File(output_path, 'r') as f:
            # Should have 6 timesteps (3 per file * 2 files)
            assert f['T2_ne'].shape == (6, 5, 5)
            assert f['time'].shape == (6,)

    def test_date_range_filtering(self, tmp_path):
        """Should only process files within the specified date range."""
        source_dir = tmp_path / 'source'
        test_dir = tmp_path / 'test'
        source_dir.mkdir()
        test_dir.mkdir()

        variables = ['T2']
        shape = (2, 5, 5)

        # Create files for multiple dates
        for date_str in ['2020-09-28', '2020-09-29', '2020-09-30', '2020-10-01']:
            source_file = source_dir / make_wrfout_filename(4, date_str)
            test_file = test_dir / make_wrfout_filename(4, date_str)
            create_mock_wrfout(source_file, variables, shape)
            create_mock_wrfout(test_file, variables, shape)

        output_path = tmp_path / 'output.nc'
        evaluate_models_cell(
            source_dir,
            test_dir,
            output_path,
            domain=4,
            variables=variables,
            start_date='2020-09-29',
            end_date='2020-09-30',
        )

        with h5py.File(output_path, 'r') as f:
            # Should have 4 timesteps (2 per file * 2 files in range)
            assert f['T2_ne'].shape == (4, 5, 5)

    def test_date_range_with_date_objects(self, tmp_path):
        """Should accept date objects for start_date and end_date."""
        source_dir = tmp_path / 'source'
        test_dir = tmp_path / 'test'
        source_dir.mkdir()
        test_dir.mkdir()

        variables = ['T2']
        shape = (2, 5, 5)

        for date_str in ['2020-09-29', '2020-09-30', '2020-10-01']:
            source_file = source_dir / make_wrfout_filename(4, date_str)
            test_file = test_dir / make_wrfout_filename(4, date_str)
            create_mock_wrfout(source_file, variables, shape)
            create_mock_wrfout(test_file, variables, shape)

        output_path = tmp_path / 'output.nc'
        evaluate_models_cell(
            source_dir,
            test_dir,
            output_path,
            domain=4,
            variables=variables,
            start_date=date(2020, 9, 30),
            end_date=date(2020, 9, 30),
        )

        with h5py.File(output_path, 'r') as f:
            # Should have 2 timesteps (from single date)
            assert f['T2_ne'].shape == (2, 5, 5)

    def test_different_domains(self, tmp_path):
        """Should correctly select files by domain number."""
        source_dir = tmp_path / 'source'
        test_dir = tmp_path / 'test'
        source_dir.mkdir()
        test_dir.mkdir()

        variables = ['T2']
        shape = (2, 5, 5)

        # Create files for different domains
        for domain in [1, 2, 4]:
            source_file = source_dir / make_wrfout_filename(domain, "2020-09-30")
            test_file = test_dir / make_wrfout_filename(domain, "2020-09-30")
            create_mock_wrfout(source_file, variables, shape)
            create_mock_wrfout(test_file, variables, shape)

        # Test domain 2 specifically
        output_path = tmp_path / 'output.nc'
        evaluate_models_cell(source_dir, test_dir, output_path, domain=2, variables=variables)

        with h5py.File(output_path, 'r') as f:
            assert f.attrs['domain'] == 2

    def test_single_metric_string(self, tmp_path):
        """Should accept a single metric as a string."""
        source_dir = tmp_path / 'source'
        test_dir = tmp_path / 'test'
        source_dir.mkdir()
        test_dir.mkdir()

        variables = ['T2']
        shape = (2, 5, 5)

        source_file = source_dir / make_wrfout_filename(4, "2020-09-30")
        test_file = test_dir / make_wrfout_filename(4, "2020-09-30")
        create_mock_wrfout(source_file, variables, shape)
        create_mock_wrfout(test_file, variables, shape)

        output_path = tmp_path / 'output.nc'
        evaluate_models_cell(
            source_dir, test_dir, output_path, domain=4, variables=variables, metrics='ane'
        )

        with h5py.File(output_path, 'r') as f:
            assert 'T2_ane' in f
            assert 'T2_ne' not in f

    def test_multiple_metrics(self, tmp_path):
        """Should compute multiple metrics when provided as a list."""
        source_dir = tmp_path / 'source'
        test_dir = tmp_path / 'test'
        source_dir.mkdir()
        test_dir.mkdir()

        variables = ['T2']
        shape = (2, 5, 5)

        # Create source=100, test=110 (10% difference)
        def source_data(var, shape):
            return np.ones(shape, dtype=np.float32) * 100

        def test_data(var, shape):
            return np.ones(shape, dtype=np.float32) * 110

        source_file = source_dir / make_wrfout_filename(4, "2020-09-30")
        test_file = test_dir / make_wrfout_filename(4, "2020-09-30")
        create_mock_wrfout(source_file, variables, shape, source_data)
        create_mock_wrfout(test_file, variables, shape, test_data)

        output_path = tmp_path / 'output.nc'
        evaluate_models_cell(
            source_dir,
            test_dir,
            output_path,
            domain=4,
            variables=variables,
            metrics=['ne', 'ane', 'rse'],
        )

        with h5py.File(output_path, 'r') as f:
            # Check all metrics present
            assert 'T2_ne' in f
            assert 'T2_ane' in f
            assert 'T2_rse' in f

            # Check values
            np.testing.assert_array_equal(f['T2_ne'][:], 10)  # 10% NE
            np.testing.assert_array_equal(f['T2_ane'][:], 10)  # 10% ANE
            np.testing.assert_array_almost_equal(f['T2_rse'][:], 10.0)  # RSE = 10

            # Check dtypes
            assert f['T2_ne'].dtype == np.int16
            assert f['T2_ane'].dtype == np.int16
            assert f['T2_rse'].dtype == np.float32

    def test_all_available_metrics(self, tmp_path):
        """Should be able to compute all available metrics."""
        source_dir = tmp_path / 'source'
        test_dir = tmp_path / 'test'
        source_dir.mkdir()
        test_dir.mkdir()

        variables = ['T2']
        shape = (2, 5, 5)

        source_file = source_dir / make_wrfout_filename(4, "2020-09-30")
        test_file = test_dir / make_wrfout_filename(4, "2020-09-30")
        create_mock_wrfout(source_file, variables, shape)
        create_mock_wrfout(test_file, variables, shape)

        output_path = tmp_path / 'output.nc'
        evaluate_models_cell(
            source_dir,
            test_dir,
            output_path,
            domain=4,
            variables=variables,
            metrics=list(AVAILABLE_METRICS),
        )

        with h5py.File(output_path, 'r') as f:
            for metric in AVAILABLE_METRICS:
                assert f'T2_{metric}' in f

    def test_raises_on_invalid_metric(self, tmp_path):
        """Should raise ValueError for unknown metric."""
        source_dir = tmp_path / 'source'
        test_dir = tmp_path / 'test'
        source_dir.mkdir()
        test_dir.mkdir()

        shape = (2, 5, 5)
        source_file = source_dir / make_wrfout_filename(4, "2020-09-30")
        test_file = test_dir / make_wrfout_filename(4, "2020-09-30")
        create_mock_wrfout(source_file, ['T2'], shape)
        create_mock_wrfout(test_file, ['T2'], shape)

        with pytest.raises(ValueError, match="Unknown metric"):
            evaluate_models_cell(
                source_dir,
                test_dir,
                tmp_path / 'output.nc',
                domain=4,
                variables=['T2'],
                metrics='invalid_metric',
            )

    def test_raises_on_missing_source_folder(self, tmp_path):
        """Should raise FileNotFoundError if source folder doesn't exist."""
        test_dir = tmp_path / 'test'
        test_dir.mkdir()

        with pytest.raises(FileNotFoundError, match="Source folder not found"):
            evaluate_models_cell(
                tmp_path / 'nonexistent',
                test_dir,
                tmp_path / 'output.nc',
                domain=4,
                variables=['T2'],
            )

    def test_raises_on_missing_test_folder(self, tmp_path):
        """Should raise FileNotFoundError if test folder doesn't exist."""
        source_dir = tmp_path / 'source'
        source_dir.mkdir()

        with pytest.raises(FileNotFoundError, match="Test folder not found"):
            evaluate_models_cell(
                source_dir,
                tmp_path / 'nonexistent',
                tmp_path / 'output.nc',
                domain=4,
                variables=['T2'],
            )

    def test_raises_on_no_source_files(self, tmp_path):
        """Should raise ValueError if no wrfout files in source folder."""
        source_dir = tmp_path / 'source'
        test_dir = tmp_path / 'test'
        source_dir.mkdir()
        test_dir.mkdir()

        # Create test file but no source file
        test_file = test_dir / make_wrfout_filename(4, "2020-09-30")
        create_mock_wrfout(test_file, ['T2'], (2, 5, 5))

        with pytest.raises(ValueError, match="No wrfout files found for domain"):
            evaluate_models_cell(source_dir, test_dir, tmp_path / 'output.nc', domain=4, variables=['T2'])

    def test_raises_on_no_common_dates(self, tmp_path):
        """Should raise ValueError if no common dates between source and test."""
        source_dir = tmp_path / 'source'
        test_dir = tmp_path / 'test'
        source_dir.mkdir()
        test_dir.mkdir()

        shape = (2, 5, 5)
        source_file = source_dir / make_wrfout_filename(4, "2020-09-30")
        test_file = test_dir / make_wrfout_filename(4, "2020-10-01")  # Different date
        create_mock_wrfout(source_file, ['T2'], shape)
        create_mock_wrfout(test_file, ['T2'], shape)

        with pytest.raises(ValueError, match="No common dates found"):
            evaluate_models_cell(source_dir, test_dir, tmp_path / 'output.nc', domain=4, variables=['T2'])

    def test_raises_on_missing_variable(self, tmp_path):
        """Should raise ValueError if variable not found in file."""
        source_dir = tmp_path / 'source'
        test_dir = tmp_path / 'test'
        source_dir.mkdir()
        test_dir.mkdir()

        shape = (2, 5, 5)
        source_file = source_dir / make_wrfout_filename(4, "2020-09-30")
        test_file = test_dir / make_wrfout_filename(4, "2020-09-30")
        create_mock_wrfout(source_file, ['T2'], shape)
        create_mock_wrfout(test_file, ['T2'], shape)

        with pytest.raises(ValueError, match="Variable 'Q2' not found"):
            evaluate_models_cell(source_dir, test_dir, tmp_path / 'output.nc', domain=4, variables=['Q2'])

    def test_raises_on_spatial_shape_mismatch(self, tmp_path):
        """Should raise ValueError if source and test spatial shapes don't match."""
        source_dir = tmp_path / 'source'
        test_dir = tmp_path / 'test'
        source_dir.mkdir()
        test_dir.mkdir()

        source_file = source_dir / make_wrfout_filename(4, "2020-09-30")
        test_file = test_dir / make_wrfout_filename(4, "2020-09-30")
        create_mock_wrfout(source_file, ['T2'], (2, 5, 5))
        create_mock_wrfout(test_file, ['T2'], (2, 10, 10))  # Different spatial shape

        with pytest.raises(ValueError, match="Spatial shape mismatch"):
            evaluate_models_cell(source_dir, test_dir, tmp_path / 'output.nc', domain=4, variables=['T2'])

    def test_handles_different_timestep_counts(self, tmp_path):
        """Should use minimum timesteps when source and test have different counts."""
        source_dir = tmp_path / 'source'
        test_dir = tmp_path / 'test'
        source_dir.mkdir()
        test_dir.mkdir()

        variables = ['T2']

        # Source has 4 timesteps, test has 2
        source_file = source_dir / make_wrfout_filename(4, "2020-09-30")
        test_file = test_dir / make_wrfout_filename(4, "2020-09-30")
        create_mock_wrfout(source_file, variables, (4, 5, 5))
        create_mock_wrfout(test_file, variables, (2, 5, 5))

        output_path = tmp_path / 'output.nc'
        evaluate_models_cell(source_dir, test_dir, output_path, domain=4, variables=variables)

        with h5py.File(output_path, 'r') as f:
            # Should have 2 timesteps (minimum of 4 and 2)
            assert f['T2_ne'].shape == (2, 5, 5)

    def test_creates_output_directory(self, tmp_path):
        """Should create output directory if it doesn't exist."""
        source_dir = tmp_path / 'source'
        test_dir = tmp_path / 'test'
        source_dir.mkdir()
        test_dir.mkdir()

        shape = (2, 5, 5)
        source_file = source_dir / make_wrfout_filename(4, "2020-09-30")
        test_file = test_dir / make_wrfout_filename(4, "2020-09-30")
        create_mock_wrfout(source_file, ['T2'], shape)
        create_mock_wrfout(test_file, ['T2'], shape)

        output_path = tmp_path / 'nested' / 'dir' / 'output.nc'
        evaluate_models_cell(source_dir, test_dir, output_path, domain=4, variables=['T2'])

        assert output_path.exists()

    def test_output_dtype_is_int16(self, tmp_path):
        """Should store NE values as int16."""
        source_dir = tmp_path / 'source'
        test_dir = tmp_path / 'test'
        source_dir.mkdir()
        test_dir.mkdir()

        shape = (2, 5, 5)
        source_file = source_dir / make_wrfout_filename(4, "2020-09-30")
        test_file = test_dir / make_wrfout_filename(4, "2020-09-30")
        create_mock_wrfout(source_file, ['T2'], shape)
        create_mock_wrfout(test_file, ['T2'], shape)

        output_path = tmp_path / 'output.nc'
        evaluate_models_cell(source_dir, test_dir, output_path, domain=4, variables=['T2'])

        with h5py.File(output_path, 'r') as f:
            assert f['T2_ne'].dtype == np.int16


class TestFindLatlonBounds:
    """Tests for _find_latlon_bounds function."""

    def test_finds_bounds_within_domain(self, tmp_path):
        """Should find correct y/x slices for lat/lon bounds within domain."""
        # Create a file covering New Zealand South Island area
        # Domain: lat -47 to -42, lon 165 to 175
        file_path = tmp_path / 'test.nc'
        shape = (1, 50, 100)  # 50 y points, 100 x points
        create_mock_wrfout_with_latlon(
            file_path, ['T2'], shape, lat_range=(-47.0, -42.0), lon_range=(165.0, 175.0)
        )

        with h5py.File(file_path, 'r') as f:
            # Southland bounds should find a subset
            y_slice, x_slice = _find_latlon_bounds(f, SOUTHLAND_BOUNDS)

            # Verify slices are valid
            assert isinstance(y_slice, slice)
            assert isinstance(x_slice, slice)
            assert y_slice.start >= 0
            assert y_slice.stop <= 50
            assert x_slice.start >= 0
            assert x_slice.stop <= 100

            # Verify the selected region contains points within bounds
            xlat = f['XLAT'][0, y_slice, x_slice]
            xlong = f['XLONG'][0, y_slice, x_slice]
            min_lat, max_lat, min_lon, max_lon = SOUTHLAND_BOUNDS

            # At least some points should be within bounds
            assert np.any((xlat >= min_lat) & (xlat <= max_lat))
            assert np.any((xlong >= min_lon) & (xlong <= max_lon))

    def test_raises_when_no_cells_in_bounds(self, tmp_path):
        """Should raise ValueError when no grid cells fall within bounds."""
        # Create a file covering an area far from Southland
        file_path = tmp_path / 'test.nc'
        shape = (1, 10, 10)
        create_mock_wrfout_with_latlon(
            file_path, ['T2'], shape, lat_range=(40.0, 45.0), lon_range=(0.0, 10.0)  # Northern hemisphere
        )

        with h5py.File(file_path, 'r') as f:
            with pytest.raises(ValueError, match="No grid cells found within bounds"):
                _find_latlon_bounds(f, SOUTHLAND_BOUNDS)

    def test_raises_when_no_xlat_xlong(self, tmp_path):
        """Should raise ValueError when XLAT/XLONG variables are missing."""
        file_path = tmp_path / 'test.nc'
        create_mock_wrfout(file_path, ['T2'], (1, 10, 10))

        with h5py.File(file_path, 'r') as f:
            with pytest.raises(ValueError, match="XLAT and XLONG variables required"):
                _find_latlon_bounds(f, SOUTHLAND_BOUNDS)


class TestEvaluateModelsCellRegion:
    """Tests for evaluate_models region parameter."""

    def test_latlon_bounds_subsetting(self, tmp_path):
        """Should subset output to region defined by lat/lon bounds."""
        source_dir = tmp_path / 'source'
        test_dir = tmp_path / 'test'
        source_dir.mkdir()
        test_dir.mkdir()

        variables = ['T2']
        # Domain covering NZ South Island: lat -47 to -42, lon 165 to 175
        shape = (2, 50, 100)

        source_file = source_dir / make_wrfout_filename(4, "2020-09-30")
        test_file = test_dir / make_wrfout_filename(4, "2020-09-30")
        create_mock_wrfout_with_latlon(
            source_file, variables, shape, lat_range=(-47.0, -42.0), lon_range=(165.0, 175.0)
        )
        create_mock_wrfout_with_latlon(
            test_file, variables, shape, lat_range=(-47.0, -42.0), lon_range=(165.0, 175.0)
        )

        output_path = tmp_path / 'output.nc'
        evaluate_models_cell(
            source_dir,
            test_dir,
            output_path,
            domain=4,
            variables=variables,
            region=SOUTHLAND_BOUNDS,
        )

        with h5py.File(output_path, 'r') as f:
            # Output should be smaller than full domain
            out_shape = f['T2_ne'].shape
            assert out_shape[1] < 50  # y dimension reduced
            assert out_shape[2] < 100  # x dimension reduced

            # Check region attributes
            assert f.attrs['region_type'] == b'latlon_bounds'
            assert f.attrs['region_min_lat'] == SOUTHLAND_BOUNDS[0]
            assert f.attrs['region_max_lat'] == SOUTHLAND_BOUNDS[1]
            assert f.attrs['region_min_lon'] == SOUTHLAND_BOUNDS[2]
            assert f.attrs['region_max_lon'] == SOUTHLAND_BOUNDS[3]

    def test_spatial_mask(self, tmp_path):
        """Should apply 2D spatial mask to output."""
        source_dir = tmp_path / 'source'
        test_dir = tmp_path / 'test'
        source_dir.mkdir()
        test_dir.mkdir()

        variables = ['T2']
        shape = (2, 10, 10)

        # Create files with constant data for easy verification
        def source_data(var, shape):
            return np.ones(shape, dtype=np.float32) * 100

        def test_data(var, shape):
            return np.ones(shape, dtype=np.float32) * 110  # 10% error

        source_file = source_dir / make_wrfout_filename(4, "2020-09-30")
        test_file = test_dir / make_wrfout_filename(4, "2020-09-30")
        create_mock_wrfout(source_file, variables, shape, source_data)
        create_mock_wrfout(test_file, variables, shape, test_data)

        # Create a mask that includes only the center region
        mask = np.zeros((10, 10), dtype=bool)
        mask[3:7, 3:7] = True  # 4x4 center region

        output_path = tmp_path / 'output.nc'
        evaluate_models_cell(
            source_dir,
            test_dir,
            output_path,
            domain=4,
            variables=variables,
            region=mask,
        )

        with h5py.File(output_path, 'r') as f:
            # Output should have same spatial dimensions as input
            assert f['T2_ne'].shape == (2, 10, 10)

            # Check region attributes
            assert f.attrs['region_type'] == b'spatial_mask'

            # Check mask is stored
            assert 'spatial_mask' in f
            np.testing.assert_array_equal(f['spatial_mask'][:], mask.astype(np.int8))

            # Check masked values have fill value
            ne_data = f['T2_ne'][:]
            # Inside mask should be 10 (10% error)
            assert np.all(ne_data[:, 3:7, 3:7] == 10)
            # Outside mask should be fill value (INT16_MIN = -32768)
            assert np.all(ne_data[:, 0, 0] == -32768)

    def test_spatial_mask_with_rse(self, tmp_path):
        """Should use NaN as fill value for float32 metrics when masking."""
        source_dir = tmp_path / 'source'
        test_dir = tmp_path / 'test'
        source_dir.mkdir()
        test_dir.mkdir()

        variables = ['T2']
        shape = (2, 10, 10)

        def source_data(var, shape):
            return np.ones(shape, dtype=np.float32) * 100

        def test_data(var, shape):
            return np.ones(shape, dtype=np.float32) * 110

        source_file = source_dir / make_wrfout_filename(4, "2020-09-30")
        test_file = test_dir / make_wrfout_filename(4, "2020-09-30")
        create_mock_wrfout(source_file, variables, shape, source_data)
        create_mock_wrfout(test_file, variables, shape, test_data)

        mask = np.zeros((10, 10), dtype=bool)
        mask[3:7, 3:7] = True

        output_path = tmp_path / 'output.nc'
        evaluate_models_cell(
            source_dir,
            test_dir,
            output_path,
            domain=4,
            variables=variables,
            region=mask,
            metrics='rse',
        )

        with h5py.File(output_path, 'r') as f:
            rse_data = f['T2_rse'][:]
            # Inside mask should be 10 (RSE = sqrt((110-100)^2))
            np.testing.assert_array_almost_equal(rse_data[:, 3:7, 3:7], 10.0)
            # Outside mask should be NaN
            assert np.all(np.isnan(rse_data[:, 0, 0]))

    def test_raises_on_invalid_region_type(self, tmp_path):
        """Should raise ValueError for invalid region type."""
        source_dir = tmp_path / 'source'
        test_dir = tmp_path / 'test'
        source_dir.mkdir()
        test_dir.mkdir()

        shape = (2, 5, 5)
        source_file = source_dir / make_wrfout_filename(4, "2020-09-30")
        test_file = test_dir / make_wrfout_filename(4, "2020-09-30")
        create_mock_wrfout(source_file, ['T2'], shape)
        create_mock_wrfout(test_file, ['T2'], shape)

        with pytest.raises(ValueError, match="region must be either"):
            evaluate_models_cell(
                source_dir,
                test_dir,
                tmp_path / 'output.nc',
                domain=4,
                variables=['T2'],
                region="invalid",
            )

    def test_raises_on_wrong_bounds_length(self, tmp_path):
        """Should raise ValueError for bounds with wrong number of elements."""
        source_dir = tmp_path / 'source'
        test_dir = tmp_path / 'test'
        source_dir.mkdir()
        test_dir.mkdir()

        shape = (2, 5, 5)
        source_file = source_dir / make_wrfout_filename(4, "2020-09-30")
        test_file = test_dir / make_wrfout_filename(4, "2020-09-30")
        create_mock_wrfout(source_file, ['T2'], shape)
        create_mock_wrfout(test_file, ['T2'], shape)

        with pytest.raises(ValueError, match="region must be either"):
            evaluate_models_cell(
                source_dir,
                test_dir,
                tmp_path / 'output.nc',
                domain=4,
                variables=['T2'],
                region=(1.0, 2.0, 3.0),  # Only 3 elements
            )

    def test_raises_on_mask_shape_mismatch(self, tmp_path):
        """Should raise ValueError when mask shape doesn't match domain."""
        source_dir = tmp_path / 'source'
        test_dir = tmp_path / 'test'
        source_dir.mkdir()
        test_dir.mkdir()

        shape = (2, 10, 10)
        source_file = source_dir / make_wrfout_filename(4, "2020-09-30")
        test_file = test_dir / make_wrfout_filename(4, "2020-09-30")
        create_mock_wrfout(source_file, ['T2'], shape)
        create_mock_wrfout(test_file, ['T2'], shape)

        wrong_mask = np.ones((5, 5), dtype=bool)  # Wrong size

        with pytest.raises(ValueError, match="Mask shape .* does not match"):
            evaluate_models_cell(
                source_dir,
                test_dir,
                tmp_path / 'output.nc',
                domain=4,
                variables=['T2'],
                region=wrong_mask,
            )

    def test_raises_on_3d_mask(self, tmp_path):
        """Should raise ValueError for 3D mask array."""
        source_dir = tmp_path / 'source'
        test_dir = tmp_path / 'test'
        source_dir.mkdir()
        test_dir.mkdir()

        shape = (2, 5, 5)
        source_file = source_dir / make_wrfout_filename(4, "2020-09-30")
        test_file = test_dir / make_wrfout_filename(4, "2020-09-30")
        create_mock_wrfout(source_file, ['T2'], shape)
        create_mock_wrfout(test_file, ['T2'], shape)

        mask_3d = np.ones((2, 5, 5), dtype=bool)  # 3D mask

        with pytest.raises(ValueError, match="Spatial mask must be 2D"):
            evaluate_models_cell(
                source_dir,
                test_dir,
                tmp_path / 'output.nc',
                domain=4,
                variables=['T2'],
                region=mask_3d,
            )

    def test_no_region_uses_full_domain(self, tmp_path):
        """Should use full domain when region is None."""
        source_dir = tmp_path / 'source'
        test_dir = tmp_path / 'test'
        source_dir.mkdir()
        test_dir.mkdir()

        variables = ['T2']
        shape = (2, 10, 15)

        source_file = source_dir / make_wrfout_filename(4, "2020-09-30")
        test_file = test_dir / make_wrfout_filename(4, "2020-09-30")
        create_mock_wrfout(source_file, variables, shape)
        create_mock_wrfout(test_file, variables, shape)

        output_path = tmp_path / 'output.nc'
        evaluate_models_cell(
            source_dir, test_dir, output_path, domain=4, variables=variables, region=None
        )

        with h5py.File(output_path, 'r') as f:
            # Output should match full input spatial dimensions
            assert f['T2_ne'].shape == (2, 10, 15)
            # No region_type attribute
            assert 'region_type' not in f.attrs


class TestEvaluateModelsCellIntegration:
    """Integration tests using real WRF model data.

    These tests are skipped unless --source-folder and --test-folder are provided.

    Example usage:
        pytest --source-folder=/path/to/source --test-folder=/path/to/test \\
               --domain=4 --variables=T2,Q2 \\
               --start-date=2020-09-30 --end-date=2020-10-15
    """

    def test_real_data_evaluation(
        self, real_model_paths, domain, variables, start_date, end_date, tmp_path
    ):
        """Test evaluation with real WRF model data using all metrics."""
        source_folder, test_folder = real_model_paths

        output_path = tmp_path / 'real_data_output.nc'
        result = evaluate_models_cell(
            source_folder,
            test_folder,
            output_path,
            domain=domain,
            variables=variables,
            metrics=list(AVAILABLE_METRICS),
            start_date=start_date,
            end_date=end_date,
        )

        assert result.exists()

        with h5py.File(result, 'r') as f:
            # Check that all variables and metrics were processed
            for var in variables:
                for metric in AVAILABLE_METRICS:
                    ds_name = f'{var}_{metric}'
                    assert ds_name in f, f"Variable {ds_name} not found in output"
                    # Check shape is valid
                    assert len(f[ds_name].shape) == 3

            # Check dimensions exist
            assert 'time' in f
            assert 'y' in f
            assert 'x' in f

            print(f"\nOutput shape: {f[f'{variables[0]}_ne'].shape}")
            print(f"Metrics computed: {list(AVAILABLE_METRICS)}")

    def test_real_data_file_discovery(self, real_model_paths, domain, start_date, end_date):
        """Test that files are correctly discovered in real data folders."""
        source_folder, test_folder = real_model_paths

        source_files = find_wrfout_files(source_folder, domain, start_date, end_date)
        test_files = find_wrfout_files(test_folder, domain, start_date, end_date)

        assert len(source_files) > 0, f"No source files found for domain {domain}"
        assert len(test_files) > 0, f"No test files found for domain {domain}"

        # Check there are common dates
        common_dates = set(source_files.keys()) & set(test_files.keys())
        assert len(common_dates) > 0, "No common dates between source and test"

        print(f"\nFound {len(source_files)} source files")
        print(f"Found {len(test_files)} test files")
        print(f"Common dates: {sorted(common_dates)}")

    def test_real_data_variables_exist(self, real_model_paths, domain, variables):
        """Test that requested variables exist in real data files."""
        source_folder, _ = real_model_paths

        source_files = find_wrfout_files(source_folder, domain)
        assert len(source_files) > 0

        # Check first file for variables
        first_file = list(source_files.values())[0]
        with h5py.File(first_file, 'r') as f:
            for var in variables:
                assert var in f, f"Variable {var} not found in {first_file}"
                print(f"\n{var}: shape={f[var].shape}, dtype={f[var].dtype}")


##################################################
# Tests for domain-aggregated metrics
##################################################


class TestComputeNeDomain:
    """Tests for compute_ne_domain function."""

    def test_basic_calculation(self):
        """Should compute NE = ((sum(test) - sum(source)) / sum(source)) * 100."""
        # 2 timesteps, 3x3 spatial grid
        source = np.ones((2, 3, 3)) * 100  # sum = 900 per timestep
        test = np.ones((2, 3, 3)) * 110  # sum = 990 per timestep, 10% increase

        result = compute_ne_domain(source, test)

        assert result.shape == (2,)
        np.testing.assert_array_almost_equal(result, [10.0, 10.0])

    def test_with_mask(self):
        """Should only aggregate over masked cells."""
        source = np.ones((2, 4, 4)) * 100
        test = np.ones((2, 4, 4)) * 110

        # Mask that only includes center 2x2
        mask = np.zeros((4, 4), dtype=bool)
        mask[1:3, 1:3] = True  # 4 cells

        result = compute_ne_domain(source, test, mask=mask)

        # Still 10% error, just over fewer cells
        np.testing.assert_array_almost_equal(result, [10.0, 10.0])

    def test_handles_zero_source(self):
        """Should return 0 when source sum is near zero."""
        source = np.zeros((2, 3, 3))
        test = np.ones((2, 3, 3)) * 10

        result = compute_ne_domain(source, test, epsilon=1e-10)

        np.testing.assert_array_equal(result, [0.0, 0.0])

    def test_returns_float64(self):
        """Should return float64 dtype."""
        source = np.ones((2, 3, 3)) * 100
        test = np.ones((2, 3, 3)) * 110

        result = compute_ne_domain(source, test)

        assert result.dtype == np.float64


class TestComputeAneDomain:
    """Tests for compute_ane_domain function."""

    def test_returns_absolute_value(self):
        """Should return absolute value of NE."""
        source = np.ones((2, 3, 3)) * 100
        test = np.ones((2, 3, 3)) * 90  # -10% error

        result = compute_ane_domain(source, test)

        np.testing.assert_array_almost_equal(result, [10.0, 10.0])  # Absolute

    def test_always_positive(self):
        """Should always return positive values."""
        source = np.ones((2, 3, 3)) * 100
        test = np.ones((2, 3, 3)) * 50  # -50% error

        result = compute_ane_domain(source, test)

        assert np.all(result >= 0)


class TestComputeRmseDomain:
    """Tests for compute_rmse_domain function."""

    def test_basic_calculation(self):
        """Should compute RMSE = sqrt(mean((test - source)^2))."""
        source = np.ones((2, 3, 3)) * 100
        test = np.ones((2, 3, 3)) * 110  # diff = 10 at each cell

        result = compute_rmse_domain(source, test)

        # RMSE should be 10 (sqrt(mean(100)) = sqrt(100) = 10)
        np.testing.assert_array_almost_equal(result, [10.0, 10.0])

    def test_with_varying_errors(self):
        """Should correctly handle varying errors across domain."""
        source = np.array([[[100, 100], [100, 100]]])  # (1, 2, 2)
        test = np.array([[[110, 90], [120, 80]]])  # errors: 10, -10, 20, -20

        result = compute_rmse_domain(source, test)

        # squared errors: 100, 100, 400, 400; mean = 250; rmse = sqrt(250) ≈ 15.81
        expected = np.sqrt(250)
        np.testing.assert_array_almost_equal(result, [expected])

    def test_with_mask(self):
        """Should only include masked cells in RMSE calculation."""
        source = np.ones((1, 4, 4)) * 100
        test = np.ones((1, 4, 4)) * 100
        test[0, 0, 0] = 200  # Large error in one cell

        # Mask that excludes the error cell
        mask = np.ones((4, 4), dtype=bool)
        mask[0, 0] = False

        result = compute_rmse_domain(source, test, mask=mask)

        # Should be 0 since we excluded the error cell
        np.testing.assert_array_almost_equal(result, [0.0])

    def test_returns_float64(self):
        """Should return float64 dtype."""
        source = np.ones((2, 3, 3)) * 100
        test = np.ones((2, 3, 3)) * 110

        result = compute_rmse_domain(source, test)

        assert result.dtype == np.float64


class TestEvaluateModelsDomain:
    """Tests for evaluate_models_domain function."""

    def test_basic_evaluation(self, tmp_path):
        """Should create output file with domain-aggregated metrics."""
        source_dir = tmp_path / 'source'
        test_dir = tmp_path / 'test'
        source_dir.mkdir()
        test_dir.mkdir()

        variables = ['T2', 'Q2']
        shape = (4, 10, 10)

        def source_data(var, shape):
            return np.ones(shape, dtype=np.float32) * 100

        def test_data(var, shape):
            return np.ones(shape, dtype=np.float32) * 110

        source_file = source_dir / make_wrfout_filename(4, "2020-09-30")
        test_file = test_dir / make_wrfout_filename(4, "2020-09-30")
        create_mock_wrfout(source_file, variables, shape, source_data)
        create_mock_wrfout(test_file, variables, shape, test_data)

        output_path = tmp_path / 'output.nc'
        result = evaluate_models_domain(
            source_dir, test_dir, output_path, domain=4, variables=variables
        )

        assert result == output_path
        assert output_path.exists()

        with h5py.File(output_path, 'r') as f:
            # Check variables exist with shape (time, metric)
            assert 'T2' in f
            assert 'Q2' in f
            assert f['T2'].shape == (4, 1)  # 4 timesteps, 1 metric (default 'ne')

            # Check NE is 10% everywhere
            np.testing.assert_array_almost_equal(f['T2'][:, 0], 10.0)

    def test_multiple_metrics(self, tmp_path):
        """Should compute multiple metrics with metric dimension."""
        source_dir = tmp_path / 'source'
        test_dir = tmp_path / 'test'
        source_dir.mkdir()
        test_dir.mkdir()

        variables = ['T2']
        shape = (2, 5, 5)

        def source_data(var, shape):
            return np.ones(shape, dtype=np.float32) * 100

        def test_data(var, shape):
            return np.ones(shape, dtype=np.float32) * 110

        source_file = source_dir / make_wrfout_filename(4, "2020-09-30")
        test_file = test_dir / make_wrfout_filename(4, "2020-09-30")
        create_mock_wrfout(source_file, variables, shape, source_data)
        create_mock_wrfout(test_file, variables, shape, test_data)

        output_path = tmp_path / 'output.nc'
        evaluate_models_domain(
            source_dir,
            test_dir,
            output_path,
            domain=4,
            variables=variables,
            metrics=['ne', 'ane', 'rmse'],
        )

        with h5py.File(output_path, 'r') as f:
            # Check shape is (time, metric)
            assert f['T2'].shape == (2, 3)

            # Check metric coordinate
            assert 'metric' in f
            assert f['metric'].shape == (3,)
            assert f['metric'].attrs['flag_meanings'] == b'ne ane rmse'

            # Check values
            np.testing.assert_array_almost_equal(f['T2'][:, 0], 10.0)  # NE
            np.testing.assert_array_almost_equal(f['T2'][:, 1], 10.0)  # ANE
            np.testing.assert_array_almost_equal(f['T2'][:, 2], 10.0)  # RMSE

    def test_output_structure(self, tmp_path):
        """Should create proper NetCDF4 structure with dimensions."""
        source_dir = tmp_path / 'source'
        test_dir = tmp_path / 'test'
        source_dir.mkdir()
        test_dir.mkdir()

        shape = (3, 5, 5)
        source_file = source_dir / make_wrfout_filename(4, "2020-09-30")
        test_file = test_dir / make_wrfout_filename(4, "2020-09-30")
        create_mock_wrfout(source_file, ['T2'], shape)
        create_mock_wrfout(test_file, ['T2'], shape)

        output_path = tmp_path / 'output.nc'
        evaluate_models_domain(
            source_dir,
            test_dir,
            output_path,
            domain=4,
            variables=['T2'],
            metrics=['ne', 'rmse'],
        )

        with h5py.File(output_path, 'r') as f:
            # Check global attributes
            assert f.attrs['Conventions'] == b'CF-1.8'
            assert f.attrs['aggregation_type'] == b'domain'

            # Check dimensions exist
            assert 'time' in f
            assert 'metric' in f

            # Check T2 has correct dimensions
            assert f['T2'].shape == (3, 2)

    def test_raises_on_invalid_metric(self, tmp_path):
        """Should raise ValueError for unknown metric."""
        source_dir = tmp_path / 'source'
        test_dir = tmp_path / 'test'
        source_dir.mkdir()
        test_dir.mkdir()

        shape = (2, 5, 5)
        source_file = source_dir / make_wrfout_filename(4, "2020-09-30")
        test_file = test_dir / make_wrfout_filename(4, "2020-09-30")
        create_mock_wrfout(source_file, ['T2'], shape)
        create_mock_wrfout(test_file, ['T2'], shape)

        with pytest.raises(ValueError, match="Unknown metric"):
            evaluate_models_domain(
                source_dir,
                test_dir,
                tmp_path / 'output.nc',
                domain=4,
                variables=['T2'],
                metrics='invalid_metric',
            )

    def test_with_spatial_mask(self, tmp_path):
        """Should aggregate only over masked cells."""
        source_dir = tmp_path / 'source'
        test_dir = tmp_path / 'test'
        source_dir.mkdir()
        test_dir.mkdir()

        variables = ['T2']
        shape = (2, 10, 10)

        def source_data(var, shape):
            return np.ones(shape, dtype=np.float32) * 100

        def test_data(var, shape):
            data = np.ones(shape, dtype=np.float32) * 100
            # Only add error in center region
            data[:, 4:6, 4:6] = 110
            return data

        source_file = source_dir / make_wrfout_filename(4, "2020-09-30")
        test_file = test_dir / make_wrfout_filename(4, "2020-09-30")
        create_mock_wrfout(source_file, variables, shape, source_data)
        create_mock_wrfout(test_file, variables, shape, test_data)

        # Mask that only includes the center region with error
        mask = np.zeros((10, 10), dtype=bool)
        mask[4:6, 4:6] = True

        output_path = tmp_path / 'output.nc'
        evaluate_models_domain(
            source_dir,
            test_dir,
            output_path,
            domain=4,
            variables=variables,
            region=mask,
        )

        with h5py.File(output_path, 'r') as f:
            # NE should be 10% since we only look at the center
            np.testing.assert_array_almost_equal(f['T2'][:, 0], 10.0)

    def test_multiple_dates(self, tmp_path):
        """Should handle multiple dates and concatenate along time."""
        source_dir = tmp_path / 'source'
        test_dir = tmp_path / 'test'
        source_dir.mkdir()
        test_dir.mkdir()

        variables = ['T2']
        shape = (3, 5, 5)

        for date_str in ['2020-09-30', '2020-10-01']:
            source_file = source_dir / make_wrfout_filename(4, date_str)
            test_file = test_dir / make_wrfout_filename(4, date_str)
            create_mock_wrfout(source_file, variables, shape)
            create_mock_wrfout(test_file, variables, shape)

        output_path = tmp_path / 'output.nc'
        evaluate_models_domain(
            source_dir, test_dir, output_path, domain=4, variables=variables
        )

        with h5py.File(output_path, 'r') as f:
            # Should have 6 timesteps (3 per file * 2 files)
            assert f['T2'].shape == (6, 1)
            assert f['time'].shape == (6,)


class TestEvaluateModelsDomainIntegration:
    """Integration tests for evaluate_models_domain using real WRF model data.

    These tests are skipped unless --source-folder and --test-folder are provided.

    Example usage:
        pytest --source-folder=/path/to/source --test-folder=/path/to/test \\
               --domain=4 --variables=T2,Q2 \\
               --start-date=2020-09-30 --end-date=2020-10-15
    """

    def test_real_data_domain_evaluation(
        self, real_model_paths, domain, variables, start_date, end_date, tmp_path
    ):
        """Test domain-aggregated evaluation with real WRF model data."""
        source_folder, test_folder = real_model_paths

        output_path = tmp_path / 'real_data_domain_output.nc'
        result = evaluate_models_domain(
            source_folder,
            test_folder,
            output_path,
            domain=domain,
            variables=variables,
            metrics=list(AVAILABLE_DOMAIN_METRICS),
            start_date=start_date,
            end_date=end_date,
        )

        assert result.exists()

        with h5py.File(result, 'r') as f:
            # Check that all variables were processed
            for var in variables:
                assert var in f, f"Variable {var} not found in output"
                # Check shape is (time, n_metrics)
                assert len(f[var].shape) == 2
                assert f[var].shape[1] == len(AVAILABLE_DOMAIN_METRICS)

            # Check dimensions exist
            assert 'time' in f
            assert 'metric' in f

            # Check metric names are stored
            metric_names = f['metric'].attrs['flag_meanings'].decode().split()
            assert metric_names == list(AVAILABLE_DOMAIN_METRICS)

            print(f"\nOutput shape for {variables[0]}: {f[variables[0]].shape}")
            print(f"Time steps: {f['time'].shape[0]}")
            print(f"Metrics: {metric_names}")

            # Print sample values for first variable
            var = variables[0]
            for i, metric in enumerate(metric_names):
                values = f[var][:, i]
                print(f"{var} {metric}: min={values.min():.2f}, max={values.max():.2f}, mean={values.mean():.2f}")

    def test_real_data_domain_vs_cell_comparison(
        self, real_model_paths, domain, variables, start_date, end_date, tmp_path
    ):
        """Compare domain-aggregated metrics with cell-by-cell mean."""
        source_folder, test_folder = real_model_paths

        # Run domain evaluation
        domain_output = tmp_path / 'domain_output.nc'
        evaluate_models_domain(
            source_folder,
            test_folder,
            domain_output,
            domain=domain,
            variables=variables[:1],  # Just first variable
            metrics=['rmse'],
            start_date=start_date,
            end_date=end_date,
        )

        # Run cell evaluation
        cell_output = tmp_path / 'cell_output.nc'
        evaluate_models_cell(
            source_folder,
            test_folder,
            cell_output,
            domain=domain,
            variables=variables[:1],
            metrics=['rse'],  # Cell version uses RSE
            start_date=start_date,
            end_date=end_date,
        )

        with h5py.File(domain_output, 'r') as f_domain, h5py.File(cell_output, 'r') as f_cell:
            var = variables[0]

            # Get domain RMSE
            domain_rmse = f_domain[var][:, 0]

            # Calculate mean of cell RSE (which should approximate RMSE)
            cell_rse = f_cell[f'{var}_rse'][:]
            cell_mean_rse = np.sqrt(np.mean(cell_rse ** 2, axis=(1, 2)))

            # These should be similar (not identical due to different computation order)
            print(f"\nDomain RMSE mean: {domain_rmse.mean():.4f}")
            print(f"Cell RSE->RMSE mean: {cell_mean_rse.mean():.4f}")

            # Check they're in the same ballpark (within 50%)
            ratio = domain_rmse.mean() / cell_mean_rse.mean()
            assert 0.5 < ratio < 2.0, f"Domain and cell metrics differ too much: ratio={ratio}"

    def test_real_data_domain_with_region(
        self, real_model_paths, domain, variables, start_date, end_date, tmp_path
    ):
        """Test domain evaluation with Southland region mask."""
        source_folder, test_folder = real_model_paths

        output_path = tmp_path / 'real_data_southland_output.nc'

        try:
            result = evaluate_models_domain(
                source_folder,
                test_folder,
                output_path,
                domain=domain,
                variables=variables,
                metrics=['ne', 'rmse'],
                region=SOUTHLAND_BOUNDS,
                start_date=start_date,
                end_date=end_date,
            )

            assert result.exists()

            with h5py.File(result, 'r') as f:
                # Check region attributes
                assert f.attrs.get('region_type') == b'latlon_bounds'
                assert f.attrs.get('region_min_lat') == SOUTHLAND_BOUNDS[0]

                var = variables[0]
                print(f"\nSouthland region evaluation for {var}:")
                print(f"  Shape: {f[var].shape}")
                print(f"  NE range: {f[var][:, 0].min():.2f} to {f[var][:, 0].max():.2f}")

        except ValueError as e:
            if "No grid cells found within bounds" in str(e):
                pytest.skip("Test domain does not cover Southland region")
            raise
