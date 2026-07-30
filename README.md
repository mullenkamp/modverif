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

## Cartopy Notes

modverif uses [cartopy](https://scitools.org.uk/cartopy/) for geographic map projections in composite and station map plots. Cartopy is optional -- plots fall back to plain matplotlib axes if cartopy is not installed -- but there are a few gotchas worth knowing about when working with projected WRF domains.

### Lambert Conformal `cutoff` clips Southern Hemisphere domains

Cartopy's `LambertConformal` projection has a `cutoff` parameter (default `-30`) that silently limits how far the map extends from the projection centre. For Northern Hemisphere projections this is fine, but for Southern Hemisphere domains the default boundary sits at 30 S. Any data north of 30 S is simply not displayed -- no error, no warning.

modverif sets `cutoff=30` for SH projections automatically (`pyproj_to_cartopy` in `composite.py`), but if you create your own cartopy axes for a SH Lambert Conformal domain, remember to pass an appropriate cutoff:

```python
import cartopy.crs as ccrs

proj = ccrs.LambertConformal(
    central_longitude=178, central_latitude=-34,
    standard_parallels=[-41, -41],
    cutoff=30,  # default -30 clips SH domains
)
```

### Antimeridian (180 longitude)

Domains that cross the 180 meridian (e.g., New Zealand) are a recurring source of issues:

- **NCAR ERA5 data** uses 0-360 longitude convention. When regridding to a WRF Lambert Conformal grid via pyproj, the transformed longitudes come back in -180/180 convention. A naive interpolation lookup will fail for points near the antimeridian because -175 is outside the ERA5 range of [144, 190]. The fix is to convert target longitudes to 0-360 (`lon % 360`) before interpolation.
- **cfdb's `GridInterp.to_grid()`** transforms target coordinates internally via pyproj, so it also returns -180/180 longitudes and cannot match ERA5's 0-360 range. For antimeridian-crossing domains, use `scipy.interpolate.RegularGridInterpolator` directly with explicit longitude convention handling instead of `to_grid()`.
- **Longitude values > 180** in WRF lat/lon auxiliary variables are masked out in modverif's composite plots (`_read_grid_from_ds`) to prevent cartopy from wrapping the map around the full globe. This only applies to datasets with `latitude`/`longitude` coordinates, not projected `y`/`x` grids.

### Comparison panel spacing

Side-by-side composite comparison plots auto-compute figure size and subplot spacing (`wspace`) from the domain's aspect ratio. Cartopy GeoAxes maintain the map's aspect ratio regardless of the subplot allocation, so different domain shapes need different spacing to avoid gaps or overlaps. The auto-computation handles this; if you need manual control, pass `figsize` and adjust `wspace` via the `GridSpec`.

## Dependencies

- Python >= 3.10
- cfdb, numpy, scipy, matplotlib, pyproj
- cartopy (optional, for geographic map projections)

## License

This project is licensed under the terms of the Apache Software License 2.0.
