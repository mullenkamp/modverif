# Verification Plots

modverif provides publication-quality plots following MET/METplus conventions. All plot functions return `(fig, ax)` and optionally save to file via a `save_path` parameter.

## Scatter Plot

Model vs observed with 1:1 reference line and statistics box showing R, RMSE, Bias, MAE, and N.

```python
from modverif.plots import plot_scatter

fig, ax = plot_scatter(
    model_values, obs_values,
    variable_name='Temperature',
    units='K',
    save_path='scatter.png',
)

# High-density scatter with hexbin
fig, ax = plot_scatter(
    model_values, obs_values,
    density=True,
)
```

## Station Map

Geographic map of station locations colored by a metric value. Uses cartopy for map projections if available.

```python
from modverif.plots import plot_station_map

fig, ax = plot_station_map(
    station_lons, station_lats, bias_values,
    metric_name='Bias',
    units='K',
    symmetric=True,     # symmetric color limits around zero
    save_path='map.png',
)
```

## Time Series

Model and observation values over time.

```python
from modverif.plots import plot_timeseries

fig, ax = plot_timeseries(
    times, model_values,
    obs_values=obs_values,
    variable_name='Temperature',
    units='K',
    model_label='WRF',
    obs_label='Station',
)
```

## Performance Diagram

POD vs Success Ratio with CSI contour lines and frequency bias diagonal lines (Roebber 2009). Accepts single values or lists for comparing multiple models.

```python
from modverif.plots import plot_performance_diagram

fig, ax = plot_performance_diagram(
    pod_values=[0.85, 0.72, 0.90],
    far_values=[0.15, 0.28, 0.10],
    labels=['WRF-A', 'WRF-B', 'ERA5'],
)
```

## Taylor Diagram

Standard deviation, correlation, and centered RMSE on a single polar plot (Taylor 2001). Useful for comparing multiple models against a common reference.

```python
from modverif.plots import plot_taylor_diagram

fig, ax = plot_taylor_diagram(
    std_obs=1.0,
    std_model=[0.9, 1.1, 1.2],
    correlations=[0.95, 0.85, 0.90],
    labels=['WRF-A', 'WRF-B', 'ERA5'],
)
```

## Diurnal Cycle

Hour-of-day comparison with optional standard deviation shading.

```python
from modverif.plots import plot_diurnal

fig, ax = plot_diurnal(
    hours, model_values,
    obs_values=obs_values,
    variable_name='Temperature',
    units='K',
    show_spread=True,
    model_std=model_std,
    obs_std=obs_std,
)
```

## Lagged Correlation

Cross-correlation at multiple time lags, with the optimal lag annotated.

```python
from modverif.plots import plot_lagged_correlation

fig, ax = plot_lagged_correlation(
    lags, correlations,
    variable_name='Temperature',
    station_name='Wellington',
)
```

## FSS Scale Plot

Fractions Skill Score vs neighborhood size. The dashed line at 0.5 marks the "useful skill" threshold.

```python
from modverif.plots import plot_fss

# Single time
fig, ax = plot_fss(
    neighborhood_sizes, fss_values,
    threshold=1.0,
)

# Multiple timesteps
fig, ax = plot_fss(
    neighborhood_sizes, None,
    multi_times={
        't=0h': fss_t0,
        't=6h': fss_t6,
        't=12h': fss_t12,
    },
)
```

## Wind Rose Comparison

Side-by-side polar plots of wind speed and direction distributions.

```python
from modverif.plots import plot_wind_rose_comparison

# Model vs observations
fig, axes = plot_wind_rose_comparison(
    model_speed, model_direction,
    obs_speed=obs_speed, obs_direction=obs_direction,
)

# Model only
fig, axes = plot_wind_rose_comparison(model_speed, model_direction)
```
