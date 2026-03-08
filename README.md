# modverif

[![build](https://github.com/mullenkamp/modverif/workflows/Build/badge.svg)](https://github.com/mullenkamp/modverif/actions)
[![codecov](https://codecov.io/gh/mullenkamp/modverif/branch/master/graph/badge.svg)](https://codecov.io/gh/mullenkamp/modverif)
[![PyPI version](https://badge.fury.io/py/modverif.svg)](https://badge.fury.io/py/modverif)

---

**Documentation**: <a href="https://mullenkamp.github.io/modverif/" target="_blank">https://mullenkamp.github.io/modverif/</a>

**Source Code**: <a href="https://github.com/mullenkamp/modverif" target="_blank">https://github.com/mullenkamp/modverif</a>

---

A Python package for evaluating multidimensional model output, following [MET/METplus](https://dtcenter.org/community-code/model-evaluation-tools-met) standards for meteorological verification. All data I/O uses the [cfdb](https://github.com/mullenkamp/cfdb) format.

## Features

### Grid-to-Grid Evaluation (`Evaluator`)

Compare two gridded model runs (e.g., WRF outputs):

- **Cell-level metrics**: NE, ANE, RSE, Bias, MAE, POD, FAR, CSI, GSS, Frequency Bias
- **Domain-aggregated metrics**: NE, ANE, RMSE, Bias, Pearson correlation, POD, FAR, CSI, GSS, Frequency Bias
- **Fractions Skill Score (FSS)**: Multi-scale spatial verification for precipitation and other threshold-based fields
- **Vector wind metrics**: Vector RMSE, wind speed bias, wind direction bias from U/V components
- **Diurnal cycle analysis**: Metrics grouped by hour-of-day
- **Spatial subsetting**: Bounding box or 2D boolean mask
- **Time filtering**: Start/end time bounds

### Grid-to-Point Evaluation (`StationEvaluator`)

Compare gridded model output to weather station observations:

- Automatic grid-to-point interpolation via cfdb's `GridInterp.to_points()`
- Per-station, per-timestep metrics: Bias, MAE, NE, ANE
- Per-station aggregated metrics: RMSE, Pearson correlation
- Station-aggregated summary statistics
- Height level matching (single-level and multi-level observations)
- Vector wind evaluation at station locations
- Diurnal cycle analysis per station

### Cyclone Evaluation

Track cyclones independently in two datasets and compare:

- Cyclone tracking via SLP pressure minimum
- Track position, pressure, and radius differences
- Per-variable metrics within the cyclone region

### Verification Plots

Publication-quality plots following MET/METplus conventions:

- **Scatter plot**: Model vs observed with 1:1 line, statistics box, density option
- **Station map**: Geographic map of station metric values (cartopy optional)
- **Time series**: Model/observation comparison over time
- **Performance diagram**: POD vs Success Ratio with CSI contours and bias lines (Roebber 2009)
- **Taylor diagram**: Standard deviation, correlation, and centered RMSE (Taylor 2001)
- **Diurnal cycle**: Hour-of-day metric comparison
- **FSS scale plot**: Skill vs neighborhood size
- **Wind rose comparison**: Side-by-side model/observed wind roses

## Quick Start

```python
from modverif import Evaluator, StationEvaluator

# Grid-to-grid evaluation
evaluator = Evaluator('source.cfdb', 'test.cfdb')
evaluator.evaluate_domain('output.cfdb', variables=['air_temperature'], metrics=['bias', 'rmse', 'pearson'])

# Grid-to-point evaluation
station_eval = StationEvaluator(
    'model.cfdb', 'stations.cfdb',
    variable_heights={'air_temperature': 2.0, 'wind_speed': 10.0},
)
station_eval.evaluate('station_output.cfdb', variables=['air_temperature'], metrics=['bias', 'rmse'])

# FSS evaluation
evaluator.evaluate_fss('fss_output.cfdb', variables=['precipitation'], threshold=1.0)

# Vector wind evaluation
evaluator.evaluate_wind('wind_output.cfdb', metrics=['vector_rmse', 'speed_bias'])
```

Convenience functions are also available:

```python
from modverif.evaluate import (
    evaluate_models_cell,
    evaluate_models_domain,
    evaluate_stations,
    evaluate_fss,
    evaluate_wind,
)
```

### Plotting

```python
from modverif.plots import plot_scatter, plot_station_map, plot_performance_diagram

plot_scatter(model_values, obs_values, save_path='scatter.png', variable_name='Temperature', units='K')
plot_station_map(lons, lats, bias_values, save_path='map.png', metric_name='Bias')
plot_performance_diagram([0.85, 0.72], [0.15, 0.28], labels=['WRF-A', 'WRF-B'])
```

## Installation

```bash
pip install modverif
```

Or with UV:

```bash
uv add modverif
```

## Dependencies

- Python >= 3.10
- cfdb, numpy, scipy, matplotlib, pyproj
- cartopy (optional, for geographic map projections)

## License

This project is licensed under the terms of the Apache Software License 2.0.
