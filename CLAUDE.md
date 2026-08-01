# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`modverif` is a Python package for evaluating model outputs (e.g., comparing WRF weather model runs). It follows MET/METplus standards for meteorological verification. It uses UV for environment management, hatchling for the build system, and targets Python >= 3.10.

All inputs/outputs use the [cfdb](https://github.com/mullenkamp/cfdb) format. Station observation data uses the `ts_ortho` dataset type. Grid model data uses the `grid` dataset type.

## Code Style

- **Line length**: 120 characters (both black and ruff)
- **Formatter**: black with `skip-string-normalization`
- **Linter**: ruff (bounded `~=0.15.0`; target inferred from `requires-python`), relative imports banned (`ban-relative-imports = "all"`)
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
- `modverif/window.py` — Rolling-window accumulation maxima. Three functions, not one with a flag:
  the three missing-value conventions are not interchangeable, and tie-breaking is load-bearing.
- `modverif/match.py` — Point-to-grid neighbourhood matching plus the selection-bias null, vector
  coherence and field-shift objective. A neighbourhood search flatters even a skill-less model, so
  `null_improvement` is not optional decoration.
- `modverif/spatial.py` — Variogram (nugget-fixed fit + bootstrap), Moran's I with permutation
  inference, k-means regionalisation.
- `modverif/stats.py` — Permutation p-values, Holm correction, eta-squared. Separate from `spatial`
  because none of it is spatial.
- `modverif/crossval.py` — Out-of-sample assessment: leave-one-out, leave-one-group-out, and a
  hold-out-high extrapolation test.
- `modverif/wrfio.py` — Legacy WRF I/O using h5py.

⚠ **Two cross-correlation functions coexist in `metrics.py` with OPPOSITE sign conventions**
(`compute_lagged_correlation`: positive = model leads; `compute_xcorr_best_lag`: positive = model
later). Deliberate — they answer different questions and published results depend on both. Both take
`(model, obs)`; swapping the series returns the negated lag with an identical correlation.

⚠ **Random-generator contracts are load-bearing.** Some functions take a generator;
`bootstrap_variogram_params` takes a *seed* and builds its own (per-call reproducibility);
`null_improvement` consumes its generator in a rejection loop, so the draw count is data-dependent.
Reordering, batching or vectorising any of these changes published numbers. Read the docstrings
before "tidying".
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
