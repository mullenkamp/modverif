# Cyclone Evaluation

modverif can track cyclones through time using sea level pressure (SLP) minima, then evaluate model performance within the cyclone region.

## Cyclone Tracking

Track a cyclone through a dataset by following the SLP minimum:

```python
from modverif.cyclone import track_cyclone

positions = track_cyclone(
    'model.cfdb',
    start_lat=-40.0,           # initial search latitude
    start_lon=170.0,           # initial search longitude
    search_radius_km=500.0,    # search radius around previous position
    max_cyclone_radius_km=1000.0,
)
```

Each position in the returned list is a `CyclonePosition` containing:

- `time_index`, `y_index`, `x_index` -- array indices
- `latitude`, `longitude` -- geographic position
- `central_pressure` -- SLP at the cyclone center (Pa)
- `radius_km` -- estimated cyclone radius
- `time_str` -- timestamp string

### Multi-File Tracking

For datasets split across multiple files:

```python
from modverif.cyclone import track_cyclone_multi_file

positions = track_cyclone_multi_file(
    ['run_day1.cfdb', 'run_day2.cfdb', 'run_day3.cfdb'],
    start_lat=-40.0,
    start_lon=170.0,
)
```

### Tracking a Time Window

Pass `start_time` and/or `end_time` to track only part of a dataset -- useful when a cache spans
years but the storm of interest lasts days:

```python
positions = track_cyclone(
    'model.cfdb',
    start_time='2023-02-10',
    end_time='2023-02-15T23:59',
    smoothing_sigma=2.0,
)
```

Two things to know about the windowed form:

- `time_index` remains the **absolute** index into the dataset's time axis, not the position
  within the window, so a position can still be passed straight to `plot_cyclone_timestep`.
- With no `start_lat`/`start_lon`, the initial global search happens on the first timestep
  **inside the window**.

A window is deliberately not accepted by `track_cyclone_multi_file`; see its docstring.

### Projected Grids

Model output on a projected grid -- Lambert conformal, polar stereographic -- carries `y`/`x`
coordinates and a CRS rather than latitude and longitude. `read_latlon_2d` derives the geographic
coordinates that the tracking maths needs, so `track_cyclone` and `evaluate_cyclones` work
directly on raw WRF output:

```python
from modverif.cyclone import read_latlon_2d

with cfdb.open_dataset('wrfout.cfdb') as ds:
    xlat, xlong = read_latlon_2d(ds)
```

Note the longitude convention: the projected branch returns pyproj's `[-180, 180)`, while
datasets that store their own longitude coordinate commonly use `0-360`. Comparing positions
**across two datasets** therefore needs normalising -- `plot_cyclone_comparison` does this
internally. Distance calculations do not care, being periodic.

`plot_cyclone_timestep` raises `NotImplementedError` on projected grids: it plots through a
PlateCarree axis, which mangles a curvilinear domain and clips anything east of 180.

### SLP Computation

If the dataset does not contain an `mslp` variable, SLP is computed from surface pressure, terrain height, and 2m temperature using the hypsometric equation. A humidity correction is applied if `q2` (2m specific humidity) is available.

### Tracking Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `start_lat`, `start_lon` | None | Initial search location. If None, uses global minimum. |
| `search_radius_km` | 500.0 | Search radius around previous position for next timestep. |
| `pressure_threshold_pa` | 400.0 | Pressure difference threshold for radius estimation. |
| `max_cyclone_radius_km` | 1000.0 | Maximum allowed cyclone radius. |
| `smoothing_sigma` | None | Gaussian smoothing sigma for SLP field (grid cells). |
| `start_time`, `end_time` | None | Inclusive time window. Either may be given alone. |

## Comparing Two Tracks

Track the same storm independently in two datasets, then compare them at each track's **own
minimum** rather than at a fixed coordinate. That is what separates "the model under-deepens the
storm" from "the model has it slightly displaced, and a fixed-grid comparison smears it":

```python
from modverif.cyclone import compare_cyclone_tracks, plot_cyclone_comparison

model = track_cyclone('model.cfdb', start_time=start, end_time=end, smoothing_sigma=2.0)
# Seed the second track from the first, so both follow the same physical storm
reanalysis = track_cyclone(
    'era5.cfdb', start_lat=model[0].latitude, start_lon=model[0].longitude,
    start_time=start, end_time=end, smoothing_sigma=2.0,
)

pairs, metrics = compare_cyclone_tracks(model, reanalysis, tolerance_min=90)
plot_cyclone_comparison('compare.png', model, reanalysis, pairs, metrics,
                        label_a='WRF', label_b='ERA5')
```

`metrics` reports, with all signs as *a minus b*:

| Key | Meaning |
|-----|---------|
| `min_slp_bias_hpa` | `min(SLP_a) - min(SLP_b)`; positive means *a* is shallower |
| `a_min_hpa`, `b_min_hpa`, `a_min_time`, `b_min_time` | Each track's deepest point, and when |
| `timing_offset_h` | `t(a_min) - t(b_min)`; positive means *a* lags |
| `mean_track_sep_km`, `max_track_sep_km` | Centre separation over matched steps; `None` if none matched |
| `n_matched_timesteps`, `n_a_steps`, `n_b_steps` | Counts, so a thin match is visible |

Pairing is by **nearest timestamp within a tolerance**, so datasets with different output cadences
still compare -- a 3-hourly run and hourly reanalysis share no exact timestamps at all. Use
`evaluate_cyclones` below instead when you want per-variable metrics inside the cyclone region;
it intersects exact timestamps.

## Cyclone Visualization

Plot a single timestep with SLP field and cyclone position:

```python
from modverif.cyclone import plot_cyclone_timestep

plot_cyclone_timestep(
    'model.cfdb',
    positions[0],
    output_path='cyclone_t0.png',
)
```

Generate frames for all tracked positions:

```python
from modverif.cyclone import plot_cyclone_track

png_files = plot_cyclone_track(
    'model.cfdb',
    positions,
    output_dir='cyclone_frames/',
)
```

## Cyclone-Region Evaluation

Compare two datasets within the tracked cyclone region using the convenience function:

```python
from modverif.evaluate import evaluate_cyclones

evaluate_cyclones(
    'source.cfdb', 'test.cfdb', 'cyclone_eval.cfdb',
    variables=['air_temperature', 'precipitation'],
    metrics=['bias', 'rmse'],
    start_lat=-40.0,
    start_lon=170.0,
)
```

This tracks cyclones independently in both datasets and computes metrics within each dataset's cyclone region, along with track position, pressure, and radius differences.
