# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`modverif` is a Python package for evaluating model outputs (e.g., comparing WRF weather model runs). It follows MET/METplus standards for meteorological verification. It uses UV for environment management, hatchling for the build system, and targets Python >= 3.10.

All inputs/outputs use the [cfdb](https://github.com/mullenkamp/cfdb) format. Station observation data uses the `ts_ortho` dataset type. Grid model data uses the `grid` dataset type.

## Code Style

- **Line length**: 120 characters (both black and ruff)
- **Formatter**: black with `skip-string-normalization`
- **Linter**: ruff targeting py311, with relative imports banned (`ban-relative-imports = "all"`)
- **Imports**: Use absolute imports (e.g., `from modverif.module import X`, not relative)
- **Indent**: 4 spaces, UTF-8, LF line endings

## Architecture

- `modverif/` — Main package. Version defined in `__init__.py`.
- `modverif/metrics.py` — All metric implementations (continuous, categorical, domain-aggregated, FSS, vector wind, diurnal).
- `modverif/evaluator.py` — `Evaluator` class for grid-to-grid comparison. Methods: `evaluate_cell`, `evaluate_domain`, `evaluate_fss`, `evaluate_wind`, `evaluate_diurnal`.
- `modverif/station.py` — `StationEvaluator` class for grid-to-point (model vs station) comparison. Methods: `evaluate`, `evaluate_aggregate`, `evaluate_wind`, `evaluate_diurnal`.
- `modverif/evaluate.py` — Convenience wrapper functions for all evaluator classes.
- `modverif/cyclone.py` — Cyclone tracking and cyclone-region evaluation.
- `modverif/plots.py` — Verification plots (scatter, station map, timeseries, performance diagram, Taylor diagram, diurnal cycle, FSS, wind rose).
- `modverif/wrfio.py` — Legacy WRF I/O using h5py.
- `modverif/tests/` — Test directory (pytest). Tests are excluded from sdist builds.
- `docs/` — MkDocs documentation with Material theme and mkdocstrings for API reference.
- `conda/meta.yaml` — Conda package recipe.
- `plans/` — Feature plans and TODO lists.

## Key Dependencies

- `cfdb` — CF-conventions database for data I/O
- `numpy` — Numerical computation
- `scipy` — FSS neighborhood filtering (`scipy.ndimage.uniform_filter`)
- `matplotlib` — Plotting
- `cartopy` — Geographic map projections (optional, graceful fallback)
- `pyproj` — CRS transformations

## Running Tests

```bash
uv run pytest modverif/tests/
```

Integration tests requiring real cfdb datasets are skipped by default. Use `--source-dataset` and `--test-dataset` flags to enable them.
