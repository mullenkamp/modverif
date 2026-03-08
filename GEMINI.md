# GEMINI.md

## Project Overview

`modverif` is a Python-based tool designed for the evaluation and comparison of multidimensional model outputs. Currently, it primarily focuses on the **WRF (Weather Research and Forecasting)** model, but it is architected to support other multidimensional data formats in the future.

The core purpose of the project is to provide a robust way to compare two or more model runs using various error metrics, supporting:
- **Cell-by-cell comparison:** High-resolution spatial error analysis.
- **Domain-aggregated metrics:** Bulk comparison of integrated quantities over a full domain or a sub-region.
- **Feature-based evaluation:** Specialized tracking and evaluation of features like cyclones, allowing comparison even when their spatial positions differ between runs.

### Key Technologies
- **Python:** >= 3.10 (Targeting 3.11)
- **UV:** Build system and environment management.
- **Data Handling:** `h5py` (HDF5/NetCDF4), `numpy`, `scipy`, `rechunkit` (for efficient large-scale data processing).
- **Visualization:** `matplotlib`, `cartopy`.
- **Quality Assurance:** `pytest`, `ruff`, `black`, `mypy`.
- **Documentation:** `mkdocs` with the Material theme and `mkdocstrings`.

---

## Building and Running

The project uses `uv` for task automation. Ensure you have `uv` installed.

### Environment Setup
Using UV (recommended):
```bash
# UV will automatically manage the environment when running commands
```

### Testing
Run the test suite using pytest:
```bash
uv run test
```
To run tests with coverage:
```bash
uv run cov
```

### Linting and Formatting
Check for style and typing issues:
```bash
uv run lint:all
```
Automatically fix formatting and linting issues:
```bash
uv run lint:fmt
```

### Documentation
Serve the documentation locally:
```bash
uv run docs-serve
```
Build the documentation:
```bash
uv run docs-build
```

### Building the Package
```bash
uv build
```

---

## Architecture and Development Conventions

### Package Structure
- `modverif/`: The main package directory.
    - `evaluate.py`: Contains the high-level API for model evaluation (`evaluate_models_cell`, `evaluate_models_domain`, `evaluate_cyclones`).
    - `wrfio.py`: Handles WRF-specific file I/O, providing abstractions for NetCDF4/HDF5 files.
    - `cyclone.py`: Specialized logic for cyclone tracking, sea level pressure (SLP) calculation, and radius estimation.
    - `tests/`: Comprehensive test suite using `pytest`.
- `docs/`: Source files for MkDocs documentation.
- `conda/`: Recipe for building Conda packages.

### Coding Style
- **Line Length:** 120 characters.
- **Formatting:** `black` with `skip-string-normalization = true`.
- **Linting:** `ruff` with a broad set of rules (see `pyproject.toml`).
- **Imports:** **Absolute imports only.** Relative imports are explicitly banned in the configuration.
- **Type Hinting:** Use type hints for all function signatures.

### Error Metrics
The project currently implements several standard metrics:
- **NE (Normalised Error):** `((test - source) / source) * 100`.
- **ANE (Absolute Normalised Error):** `|NE|`.
- **RSE (Root Squared Error):** `sqrt((test - source)^2)`.
- **RMSE (Root Mean Square Error):** Domain-aggregated version of RSE.

### Large Data Handling
The project utilizes `rechunkit` to handle large WRF output files that might not fit into memory. It processes data by rechunking it into a format suitable for timestep-by-timestep evaluation, ensuring efficiency and low memory overhead.
