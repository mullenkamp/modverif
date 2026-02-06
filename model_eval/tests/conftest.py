"""
Pytest configuration and fixtures for model_eval tests.
"""
import pathlib
from datetime import date

import pytest


def pytest_addoption(parser):
    """Add custom command-line options for test configuration."""
    # Folder-based options (for evaluate_models_cell, evaluate_models_domain)
    parser.addoption(
        "--source-folder",
        action="store",
        default=None,
        help="Path to source WRF model output folder for integration tests",
    )
    parser.addoption(
        "--test-folder",
        action="store",
        default=None,
        help="Path to test WRF model output folder for integration tests",
    )
    parser.addoption(
        "--domain",
        action="store",
        default="4",
        help="WRF domain number to use for integration tests (default: 4)",
    )
    parser.addoption(
        "--variables",
        action="store",
        default="T2,Q2",
        help="Comma-separated list of variables to test (default: T2,Q2)",
    )
    parser.addoption(
        "--start-date",
        action="store",
        default=None,
        help="Start date for evaluation (ISO format, e.g., 2020-09-30)",
    )
    parser.addoption(
        "--end-date",
        action="store",
        default=None,
        help="End date for evaluation (ISO format, e.g., 2020-10-15)",
    )

    # File-based options (for evaluate_cyclones)
    parser.addoption(
        "--source-file",
        action="store",
        default=None,
        help="Path to source WRF file for cyclone integration tests",
    )
    parser.addoption(
        "--test-file",
        action="store",
        default=None,
        help="Path to test WRF file for cyclone integration tests",
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
    config.addinivalue_line(
        "markers", "cyclone: mark test as requiring cyclone data"
    )


# Folder-based fixtures

@pytest.fixture
def source_folder(request):
    """
    Fixture providing the source folder path.

    Returns None if not specified (for unit tests using mock data).
    """
    path = request.config.getoption("--source-folder")
    if path is not None:
        return pathlib.Path(path)
    return None


@pytest.fixture
def test_folder(request):
    """
    Fixture providing the test folder path.

    Returns None if not specified (for unit tests using mock data).
    """
    path = request.config.getoption("--test-folder")
    if path is not None:
        return pathlib.Path(path)
    return None


@pytest.fixture
def domain(request):
    """Fixture providing the WRF domain number."""
    return int(request.config.getoption("--domain"))


@pytest.fixture
def variables(request):
    """Fixture providing the list of variables to test."""
    var_str = request.config.getoption("--variables")
    return [v.strip() for v in var_str.split(",")]


@pytest.fixture
def start_date(request):
    """Fixture providing the start date for evaluation."""
    date_str = request.config.getoption("--start-date")
    if date_str is not None:
        return date.fromisoformat(date_str)
    return None


@pytest.fixture
def end_date(request):
    """Fixture providing the end date for evaluation."""
    date_str = request.config.getoption("--end-date")
    if date_str is not None:
        return date.fromisoformat(date_str)
    return None


@pytest.fixture
def real_model_paths(source_folder, test_folder):
    """
    Fixture that provides real model paths if configured.

    Skips test if paths are not provided.
    """
    if source_folder is None or test_folder is None:
        pytest.skip("Real model paths not provided. Use --source-folder and --test-folder.")
    return source_folder, test_folder


# File-based fixtures (for cyclone tests)

@pytest.fixture
def source_file(request):
    """Fixture providing the source file path for cyclone tests."""
    path = request.config.getoption("--source-file")
    if path is not None:
        return pathlib.Path(path)
    return None


@pytest.fixture
def test_file(request):
    """Fixture providing the test file path for cyclone tests."""
    path = request.config.getoption("--test-file")
    if path is not None:
        return pathlib.Path(path)
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
def real_cyclone_files(source_file, test_file):
    """
    Fixture that provides real cyclone file paths if configured.

    Skips test if paths are not provided.
    """
    if source_file is None or test_file is None:
        pytest.skip("Real cyclone files not provided. Use --source-file and --test-file.")
    return source_file, test_file
