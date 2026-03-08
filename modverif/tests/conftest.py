"""
Pytest configuration and fixtures for modverif tests.
"""
import pathlib

import numpy as np
import pytest


def pytest_addoption(parser):
    """Add custom command-line options for test configuration."""
    parser.addoption(
        "--source-dataset",
        action="store",
        default=None,
        help="Path to source cfdb dataset for integration tests",
    )
    parser.addoption(
        "--test-dataset",
        action="store",
        default=None,
        help="Path to test cfdb dataset for integration tests",
    )
    parser.addoption(
        "--variables",
        action="store",
        default="air_temperature,u_wind",
        help="Comma-separated list of cfdb variable names to test (default: air_temperature,u_wind)",
    )
    parser.addoption(
        "--start-time",
        action="store",
        default=None,
        help="Start time for evaluation (ISO format, e.g., 2020-09-30)",
    )
    parser.addoption(
        "--end-time",
        action="store",
        default=None,
        help="End time for evaluation (ISO format, e.g., 2020-10-15)",
    )
    parser.addoption(
        "--cyclone-start-lat",
        action="store",
        default=None,
        help="Initial cyclone search latitude",
    )
    parser.addoption(
        "--cyclone-start-lon",
        action="store",
        default=None,
        help="Initial cyclone search longitude",
    )


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "integration: mark test as requiring real data")


@pytest.fixture
def source_dataset(request):
    """Fixture providing the source cfdb dataset path."""
    path = request.config.getoption("--source-dataset")
    if path is not None:
        return pathlib.Path(path)
    return None


@pytest.fixture
def test_dataset(request):
    """Fixture providing the test cfdb dataset path."""
    path = request.config.getoption("--test-dataset")
    if path is not None:
        return pathlib.Path(path)
    return None


@pytest.fixture
def variables(request):
    """Fixture providing the list of variables to test."""
    var_str = request.config.getoption("--variables")
    return [v.strip() for v in var_str.split(",")]


@pytest.fixture
def start_time(request):
    """Fixture providing the start time for evaluation."""
    val = request.config.getoption("--start-time")
    if val is not None:
        return np.datetime64(val)
    return None


@pytest.fixture
def end_time(request):
    """Fixture providing the end time for evaluation."""
    val = request.config.getoption("--end-time")
    if val is not None:
        return np.datetime64(val)
    return None


@pytest.fixture
def cyclone_start_lat(request):
    """Fixture providing initial cyclone search latitude."""
    val = request.config.getoption("--cyclone-start-lat")
    if val is not None:
        return float(val)
    return None


@pytest.fixture
def cyclone_start_lon(request):
    """Fixture providing initial cyclone search longitude."""
    val = request.config.getoption("--cyclone-start-lon")
    if val is not None:
        return float(val)
    return None


@pytest.fixture
def real_datasets(source_dataset, test_dataset):
    """Fixture that provides real dataset paths. Skips if not provided."""
    if source_dataset is None or test_dataset is None:
        pytest.skip("Real dataset paths not provided. Use --source-dataset and --test-dataset.")
    return source_dataset, test_dataset
