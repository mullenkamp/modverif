"""
Tests for model_eval.evaluate module.
"""
import pathlib
from datetime import date

import h5py
import numpy as np
import pytest

from model_eval.metrics import (
    compute_ane,
    compute_ane_domain,
    compute_ne,
    compute_ne_domain,
    compute_rmse_domain,
    compute_rse,
)
from model_eval.evaluate import (
    AVAILABLE_DOMAIN_METRICS,
    AVAILABLE_METRICS,
    _find_latlon_bounds,
    evaluate_cyclones,
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
            threshold=1.0,
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


    def test_categorical_metrics(self, tmp_path):
        """Should compute categorical metrics when threshold is provided."""
        source_dir = tmp_path / 'source'
        test_dir = tmp_path / 'test'
        source_dir.mkdir()
        test_dir.mkdir()

        shape = (1, 10, 10)
        source_file = source_dir / make_wrfout_filename(4, "2020-09-30")
        test_file = test_dir / make_wrfout_filename(4, "2020-09-30")

        # 10x10 grid = 100 cells.
        # Hits: row 0, cols 0-9 (10 cells)
        # Misses: row 1, cols 0-4 (5 cells)
        # False Alarms: row 2, cols 0-4 (5 cells)
        
        # source: 10 hits (0, 0-9), 5 misses (1, 0-4) = 15 total yes
        source_data = np.zeros(shape, dtype=np.float32)
        source_data[0, 0, 0:10] = 2.0 
        source_data[0, 1, 0:5] = 2.0
        with h5py.File(source_file, 'w') as f:
            f.create_dataset('RAINNC', data=source_data)
        
        # test: 10 hits (0, 0-9), 5 false alarms (2, 0-4) = 15 total yes
        test_data = np.zeros(shape, dtype=np.float32)
        test_data[0, 0, 0:10] = 2.0
        test_data[0, 2, 0:5] = 2.0
        with h5py.File(test_file, 'w') as f:
            f.create_dataset('RAINNC', data=test_data)

        output_path = tmp_path / 'output.nc'
        evaluate_models_domain(
            source_dir, test_dir, output_path, domain=4, variables=['RAINNC'],
            metrics=['pod', 'far'], threshold=1.0
        )

        with h5py.File(output_path, 'r') as f:
            # POD = Hits / (Hits + Misses) = 10 / (10 + 5) = 10/15
            # FAR = FA / (Hits + FA) = 5 / (10 + 5) = 5/15
            np.testing.assert_allclose(f['RAINNC'][0, 0], 10/15, rtol=1e-5)
            np.testing.assert_allclose(f['RAINNC'][0, 1], 5/15, rtol=1e-5)


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


class TestEvaluateCyclones:
    """Unit tests for evaluate_cyclones function."""

    @pytest.fixture
    def mock_cyclone_files(self, tmp_path):
        """Create mock WRF files with cyclone-like pressure patterns."""
        n_times = 5
        n_y = 50
        n_x = 60

        # Create lat/lon grids centered around a test region
        lats = np.linspace(-48, -42, n_y)
        lons = np.linspace(165, 175, n_x)
        lon_grid, lat_grid = np.meshgrid(lons, lats)

        # Cyclone center moves over time
        cyclone_lats = [-46.0, -45.8, -45.5, -45.2, -45.0]
        cyclone_lons = [168.0, 168.5, 169.0, 169.5, 170.0]

        for name, lat_offset, lon_offset in [('source', 0, 0), ('test', 0.2, 0.3)]:
            filepath = tmp_path / f"wrfout_{name}.nc"
            with h5py.File(filepath, 'w') as f:
                # Global attributes
                f.attrs['DX'] = 10000.0
                f.attrs['DY'] = 10000.0

                # Coordinate grids
                xlat_data = np.broadcast_to(lat_grid, (n_times, n_y, n_x)).copy()
                xlong_data = np.broadcast_to(lon_grid, (n_times, n_y, n_x)).copy()
                f.create_dataset('XLAT', data=xlat_data.astype(np.float32))
                f.create_dataset('XLONG', data=xlong_data.astype(np.float32))

                # Terrain height (flat for simplicity)
                hgt_data = np.zeros((n_times, n_y, n_x), dtype=np.float32)
                f.create_dataset('HGT', data=hgt_data)

                # Create pressure field with cyclone minimum
                psfc_data = np.zeros((n_times, n_y, n_x), dtype=np.float32)
                t2_data = np.full((n_times, n_y, n_x), 285.0, dtype=np.float32)

                for t in range(n_times):
                    # Background pressure
                    psfc_data[t, :, :] = 101325.0

                    # Add cyclone depression (offset for test file)
                    center_lat = cyclone_lats[t] + lat_offset
                    center_lon = cyclone_lons[t] + lon_offset

                    # Distance from center
                    dlat = lat_grid - center_lat
                    dlon = lon_grid - center_lon
                    dist = np.sqrt(dlat**2 + (dlon * np.cos(np.radians(center_lat)))**2)

                    # Pressure depression (Gaussian-like)
                    pressure_drop = 2000 * np.exp(-dist**2 / (2 * 2**2))  # 20 hPa drop
                    psfc_data[t, :, :] -= pressure_drop

                f.create_dataset('PSFC', data=psfc_data)
                f.create_dataset('T2', data=t2_data)

                # Add a test variable (e.g., rainfall)
                rainnc = np.random.uniform(0, 10, (n_times, n_y, n_x)).astype(np.float32)
                f.create_dataset('RAINNC', data=rainnc)

        return tmp_path / "wrfout_source.nc", tmp_path / "wrfout_test.nc"

    def test_evaluate_cyclones_basic(self, mock_cyclone_files, tmp_path):
        """Test basic cyclone evaluation."""
        source_file, test_file = mock_cyclone_files
        output_path = tmp_path / 'cyclone_eval.nc'

        result = evaluate_cyclones(
            source_file,
            test_file,
            output_path,
            variables=['RAINNC'],
            metrics=['ne', 'ane'],
            start_lat=-46.0,
            start_lon=168.0,
        )

        assert result.exists()

        with h5py.File(result, 'r') as f:
            # Check track variables exist
            assert 'source_latitude' in f
            assert 'source_longitude' in f
            assert 'source_pressure' in f
            assert 'source_radius' in f
            assert 'test_latitude' in f
            assert 'test_longitude' in f
            assert 'test_pressure' in f
            assert 'test_radius' in f

            # Check comparison variables
            assert 'position_difference_km' in f
            assert 'pressure_difference' in f
            assert 'radius_difference' in f

            # Check evaluation variable
            assert 'RAINNC' in f
            assert f['RAINNC'].shape == (5, 2)  # (n_times, n_metrics)

            # Check dimensions
            assert 'time' in f
            assert 'metric' in f
            assert f['metric'].attrs['flag_meanings'] == b'ne ane'

    def test_evaluate_cyclones_all_metrics(self, mock_cyclone_files, tmp_path):
        """Test cyclone evaluation with all available metrics."""
        source_file, test_file = mock_cyclone_files
        output_path = tmp_path / 'cyclone_eval_all.nc'

        result = evaluate_cyclones(
            source_file,
            test_file,
            output_path,
            variables=['RAINNC'],
            metrics=list(AVAILABLE_DOMAIN_METRICS),
            start_lat=-46.0,
            start_lon=168.0,
        )

        with h5py.File(result, 'r') as f:
            assert f['RAINNC'].shape[1] == len(AVAILABLE_DOMAIN_METRICS)

    def test_evaluate_cyclones_tracks_independently(self, mock_cyclone_files, tmp_path):
        """Test that cyclones are tracked independently in source and test."""
        source_file, test_file = mock_cyclone_files
        output_path = tmp_path / 'cyclone_eval.nc'

        result = evaluate_cyclones(
            source_file,
            test_file,
            output_path,
            variables=['RAINNC'],
            start_lat=-46.0,
            start_lon=168.0,
        )

        with h5py.File(result, 'r') as f:
            source_lats = f['source_latitude'][:]
            test_lats = f['test_latitude'][:]
            source_lons = f['source_longitude'][:]
            test_lons = f['test_longitude'][:]

            # Tracks should be different (test has offset)
            assert not np.allclose(source_lats, test_lats, atol=0.1)
            assert not np.allclose(source_lons, test_lons, atol=0.1)

            # Position difference should be non-zero
            pos_diff = f['position_difference_km'][:]
            assert np.all(pos_diff > 0)

    def test_evaluate_cyclones_with_smoothing(self, mock_cyclone_files, tmp_path):
        """Test cyclone evaluation with SLP smoothing."""
        source_file, test_file = mock_cyclone_files
        output_path = tmp_path / 'cyclone_eval_smooth.nc'

        result = evaluate_cyclones(
            source_file,
            test_file,
            output_path,
            variables=['RAINNC'],
            start_lat=-46.0,
            start_lon=168.0,
            smoothing_sigma=2.0,
        )

        assert result.exists()

    def test_evaluate_cyclones_file_not_found(self, tmp_path):
        """Test that FileNotFoundError is raised for missing files."""
        with pytest.raises(FileNotFoundError):
            evaluate_cyclones(
                tmp_path / "nonexistent_source.nc",
                tmp_path / "nonexistent_test.nc",
                tmp_path / "output.nc",
                variables=['T2'],
            )

    def test_evaluate_cyclones_invalid_metric(self, mock_cyclone_files, tmp_path):
        """Test that ValueError is raised for invalid metric."""
        source_file, test_file = mock_cyclone_files

        with pytest.raises(ValueError, match="Unknown metric"):
            evaluate_cyclones(
                source_file,
                test_file,
                tmp_path / "output.nc",
                variables=['RAINNC'],
                metrics=['invalid_metric'],
            )

    def test_evaluate_cyclones_missing_variable(self, mock_cyclone_files, tmp_path):
        """Test that ValueError is raised for missing variable."""
        source_file, test_file = mock_cyclone_files

        with pytest.raises(ValueError, match="not found"):
            evaluate_cyclones(
                source_file,
                test_file,
                tmp_path / "output.nc",
                variables=['NONEXISTENT'],
            )


class TestEvaluateCyclonesIntegration:
    """Integration tests for evaluate_cyclones using real WRF model data.

    These tests are skipped unless --source-file and --test-file are provided.

    Example usage:
        pytest --source-file=/path/to/source.nc --test-file=/path/to/test.nc \\
               --cyclone-start-lat=-45.0 --cyclone-start-lon=170.0 \\
               --variables=RAINNC,T2
    """

    def test_real_data_cyclone_evaluation(
        self, real_cyclone_files, variables, cyclone_start_lat, cyclone_start_lon, tmp_path
    ):
        """Test cyclone evaluation with real WRF model data."""
        source_file, test_file = real_cyclone_files

        output_path = tmp_path / 'real_cyclone_eval.nc'
        result = evaluate_cyclones(
            source_file,
            test_file,
            output_path,
            variables=variables,
            metrics=list(AVAILABLE_DOMAIN_METRICS),
            start_lat=cyclone_start_lat,
            start_lon=cyclone_start_lon,
        )

        assert result.exists()

        with h5py.File(result, 'r') as f:
            # Check track variables
            assert 'source_latitude' in f
            assert 'test_latitude' in f
            assert 'position_difference_km' in f

            n_times = f['time'].shape[0]
            print(f"\nCyclone evaluation results:")
            print(f"  Number of timesteps: {n_times}")

            # Print track summary
            source_lats = f['source_latitude'][:]
            source_lons = f['source_longitude'][:]
            source_pressure = f['source_pressure'][:]
            test_lats = f['test_latitude'][:]
            test_lons = f['test_longitude'][:]
            pos_diff = f['position_difference_km'][:]

            print(f"\n  Source cyclone track:")
            print(f"    Lat range: {source_lats.min():.2f} to {source_lats.max():.2f}")
            print(f"    Lon range: {source_lons.min():.2f} to {source_lons.max():.2f}")
            print(f"    Pressure range: {source_pressure.min()/100:.1f} to {source_pressure.max()/100:.1f} hPa")

            print(f"\n  Test cyclone track:")
            print(f"    Lat range: {test_lats.min():.2f} to {test_lats.max():.2f}")
            print(f"    Lon range: {test_lons.min():.2f} to {test_lons.max():.2f}")

            print(f"\n  Position difference: {pos_diff.min():.1f} to {pos_diff.max():.1f} km")

            # Print evaluation metrics for each variable
            for var in variables:
                if var in f:
                    data = f[var][:]
                    print(f"\n  {var} metrics:")
                    for i, metric in enumerate(AVAILABLE_DOMAIN_METRICS):
                        print(f"    {metric}: {data[:, i].min():.2f} to {data[:, i].max():.2f}")

    def test_real_data_cyclone_tracking_consistency(
        self, real_cyclone_files, cyclone_start_lat, cyclone_start_lon, tmp_path
    ):
        """Test that cyclone tracking produces consistent results."""
        source_file, test_file = real_cyclone_files

        # Run evaluation twice
        output1 = tmp_path / 'cyclone_eval1.nc'
        output2 = tmp_path / 'cyclone_eval2.nc'

        evaluate_cyclones(
            source_file,
            test_file,
            output1,
            variables=['T2'],
            start_lat=cyclone_start_lat,
            start_lon=cyclone_start_lon,
        )

        evaluate_cyclones(
            source_file,
            test_file,
            output2,
            variables=['T2'],
            start_lat=cyclone_start_lat,
            start_lon=cyclone_start_lon,
        )

        # Results should be identical
        with h5py.File(output1, 'r') as f1, h5py.File(output2, 'r') as f2:
            np.testing.assert_array_equal(f1['source_latitude'][:], f2['source_latitude'][:])
            np.testing.assert_array_equal(f1['source_longitude'][:], f2['source_longitude'][:])
            np.testing.assert_array_equal(f1['T2'][:], f2['T2'][:])

    def test_real_data_cyclone_with_smoothing(
        self, real_cyclone_files, variables, cyclone_start_lat, cyclone_start_lon, tmp_path
    ):
        """Test cyclone evaluation with different smoothing levels."""
        source_file, test_file = real_cyclone_files

        results = {}
        for sigma in [None, 1.0, 3.0]:
            output_path = tmp_path / f'cyclone_sigma_{sigma}.nc'
            evaluate_cyclones(
                source_file,
                test_file,
                output_path,
                variables=variables[:1],  # Just first variable
                start_lat=cyclone_start_lat,
                start_lon=cyclone_start_lon,
                smoothing_sigma=sigma,
            )
            with h5py.File(output_path, 'r') as f:
                results[sigma] = {
                    'lats': f['source_latitude'][:].copy(),
                    'pressure': f['source_pressure'][:].copy(),
                }

        print("\nSmoothing comparison:")
        for sigma, data in results.items():
            print(f"  sigma={sigma}:")
            print(f"    Lat std: {np.std(data['lats']):.4f}")
            print(f"    Pressure std: {np.std(data['pressure'])/100:.2f} hPa")

    def test_real_data_required_variables_exist(self, real_cyclone_files):
        """Test that required variables for SLP calculation exist."""
        source_file, test_file = real_cyclone_files

        required_vars = ['PSFC', 'HGT', 'T2', 'XLAT', 'XLONG']

        for filepath in [source_file, test_file]:
            with h5py.File(filepath, 'r') as f:
                for var in required_vars:
                    assert var in f, f"Required variable {var} not found in {filepath}"
                print(f"\n{filepath.name}:")
                print(f"  PSFC shape: {f['PSFC'].shape}")
                print(f"  Has Q2: {'Q2' in f}")
