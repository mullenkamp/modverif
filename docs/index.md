# modverif

Model verification for multidimensional gridded and station data, following [MET/METplus](https://dtcenter.org/community-code/model-evaluation-tools-met) standards.

## Overview

modverif evaluates model output by comparing gridded datasets or comparing model grids against station observations. It computes standard verification metrics, produces publication-quality plots, and handles cyclone tracking and storm composite visualization. All data I/O uses the [cfdb](https://github.com/mullenkamp/cfdb) format with [CF conventions](https://cfconventions.org/).

## Evaluation Types

| Type | Class | Description |
|------|-------|-------------|
| Grid-to-Grid | `Evaluator` | Compare two gridded datasets cell-by-cell or domain-aggregated. Supports FSS, wind vectors, and diurnal analysis. |
| Grid-to-Point | `StationEvaluator` | Compare gridded model output to weather station observations with automatic interpolation and height matching. |
| Cyclone | `track_cyclone` + `compare_cyclone_tracks` | Track cyclones independently in two datasets and compare depth, timing and track separation. `evaluate_cyclones` additionally computes per-variable metrics inside the cyclone region. |

## Key Features

- **Cell-level and domain-aggregated metrics** -- NE, ANE, RMSE, Bias, MAE, Pearson correlation, POD, FAR, CSI, GSS, Frequency Bias
- **Fractions Skill Score (FSS)** -- multi-scale spatial verification for precipitation and threshold-based fields
- **Vector wind evaluation** -- Vector RMSE, speed bias, direction bias from U/V components
- **Diurnal cycle analysis** -- metrics grouped by hour-of-day with UTC offset support
- **Station evaluation** -- automatic grid-to-point interpolation, per-station and aggregate metrics, lagged correlation
- **Cyclone tracking** -- SLP-based tracking with radius estimation, an optional time window, and support for projected (`y`/`x` + CRS) model grids
- **Cyclone comparison** -- pressure bias, timing offset and track separation between two independently tracked storms, plus per-variable evaluation within the cyclone region
- **Storm composite plots** -- PWAT/MSLP/VIMF overlays with side-by-side model comparison and animated WebP output
- **Verification plots** -- scatter, station map, time series, performance diagram, Taylor diagram, FSS, wind rose
- **Spatial and temporal filtering** -- bounding box, boolean mask, start/end time
- **All I/O via cfdb** -- consistent CF-conventions database format for inputs and outputs
