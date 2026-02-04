"""
Pytest configuration and fixtures for model_eval tests.
"""
import pathlib
from datetime import date

import pytest


def pytest_addoption(parser):
    """Add custom command-line options for test configuration."""
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
