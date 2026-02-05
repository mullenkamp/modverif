"""
Tests for model_eval.evaluate module.
"""
import pathlib
from datetime import date

import h5py
import numpy as np
import pytest

from model_eval.evaluate import (
    AVAILABLE_METRICS,
    compute_ane,
    compute_ne,
    compute_rse,
    evaluate_models,
    find_wrfout_files,
)


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


class TestComputeNormalisedError:
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


class TestEvaluateModels:
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
        result = evaluate_models(source_dir, test_dir, output_path, domain=4, variables=variables)

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
        evaluate_models(source_dir, test_dir, output_path, domain=4, variables=variables)

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
        evaluate_models(source_dir, test_dir, output_path, domain=4, variables=variables)

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
        evaluate_models(
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
        evaluate_models(
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
        evaluate_models(source_dir, test_dir, output_path, domain=2, variables=variables)

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
        evaluate_models(
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
        evaluate_models(
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
        evaluate_models(
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
            evaluate_models(
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
            evaluate_models(
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
            evaluate_models(
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
            evaluate_models(source_dir, test_dir, tmp_path / 'output.nc', domain=4, variables=['T2'])

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
            evaluate_models(source_dir, test_dir, tmp_path / 'output.nc', domain=4, variables=['T2'])

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
            evaluate_models(source_dir, test_dir, tmp_path / 'output.nc', domain=4, variables=['Q2'])

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
            evaluate_models(source_dir, test_dir, tmp_path / 'output.nc', domain=4, variables=['T2'])

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
        evaluate_models(source_dir, test_dir, output_path, domain=4, variables=variables)

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
        evaluate_models(source_dir, test_dir, output_path, domain=4, variables=['T2'])

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
        evaluate_models(source_dir, test_dir, output_path, domain=4, variables=['T2'])

        with h5py.File(output_path, 'r') as f:
            assert f['T2_ne'].dtype == np.int16


class TestEvaluateModelsIntegration:
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
        result = evaluate_models(
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
