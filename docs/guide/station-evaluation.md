# Station Evaluation

The `StationEvaluator` class compares gridded model output to weather station observations. It automatically interpolates model grid values to station locations.

## Setup

```python
from modverif import StationEvaluator

station_eval = StationEvaluator(
    'model.cfdb',           # grid dataset (model output)
    'stations.cfdb',        # ts_ortho dataset (station observations)
)
```

The model dataset must be a `grid` type cfdb, and the observations dataset must be `ts_ortho` with point geometries.

### Height Matching

Many variables are defined at specific heights above ground (e.g., 2m temperature, 10m wind). Specify heights to select the correct model level:

```python
# Per-variable heights
station_eval = StationEvaluator(
    'model.cfdb', 'stations.cfdb',
    variable_heights={'air_temperature': 2.0, 'wind_speed': 10.0},
)

# Single default height for all variables
station_eval = StationEvaluator(
    'model.cfdb', 'stations.cfdb',
    height=2.0,
)
```

### Interpolation

Control the spatial interpolation method:

```python
station_eval = StationEvaluator(
    'model.cfdb', 'stations.cfdb',
    interpolation_order=1,  # 0=nearest, 1=linear (default), 3=cubic
)
```

### Time Filtering

```python
station_eval = StationEvaluator(
    'model.cfdb', 'stations.cfdb',
    start_time='2023-02-12',
    end_time='2023-02-14',
)
```

## Per-Station Evaluation

Computes metrics at each station for each timestep. Output is a `ts_ortho` cfdb dataset with shape `(time, point)`.

```python
station_eval.evaluate(
    'station_output.cfdb',
    variables=['air_temperature'],
    metrics=['bias', 'mae', 'ne', 'ane'],
)
```

## Aggregate Evaluation

Computes summary statistics aggregated across stations. Output has shape `(time, metric)`.

```python
station_eval.evaluate_aggregate(
    'aggregate_output.cfdb',
    variables=['air_temperature'],
    metrics=['bias', 'rmse', 'pearson'],
)
```

## Wind Evaluation

Vector wind metrics at station locations:

```python
station_eval.evaluate_wind(
    'wind_output.cfdb',
    u_var='u_wind',
    v_var='v_wind',
    metrics=['vector_rmse', 'speed_bias', 'direction_bias'],
)
```

## Diurnal Cycle

Hour-of-day analysis per station with optional UTC offset for local time:

```python
station_eval.evaluate_diurnal(
    'diurnal_output.cfdb',
    variables=['air_temperature'],
    metrics=['bias', 'rmse'],
    utc_offset=12.0,  # NZST
)
```

## Lagged Correlation

Detect timing offsets between model and observations:

```python
station_eval.evaluate_lagged_correlation(
    'lag_output.cfdb',
    variables=['air_temperature'],
    max_lag=12,  # hours
)
```

The output contains per-station lag correlations, optimal lag (positive = model leads), and peak correlation values.
