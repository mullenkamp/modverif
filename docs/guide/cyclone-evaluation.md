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
