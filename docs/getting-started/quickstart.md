# Quick Start

## Grid-to-Grid Evaluation

Compare two gridded model datasets:

```python
from modverif import Evaluator

evaluator = Evaluator('source.cfdb', 'test.cfdb')

# Domain-aggregated metrics (one value per timestep)
evaluator.evaluate_domain(
    'domain_output.cfdb',
    variables=['air_temperature'],
    metrics=['bias', 'rmse', 'pearson'],
)

# Cell-by-cell metrics (full spatial fields)
evaluator.evaluate_cell(
    'cell_output.cfdb',
    variables=['air_temperature'],
    metrics=['ne', 'bias'],
)

# Fractions Skill Score
evaluator.evaluate_fss(
    'fss_output.cfdb',
    variables=['precipitation'],
    threshold=1.0,
)

# Vector wind evaluation
evaluator.evaluate_wind(
    'wind_output.cfdb',
    metrics=['vector_rmse', 'speed_bias', 'direction_bias'],
)
```

## Station Evaluation

Compare gridded model output to weather station observations:

```python
from modverif import StationEvaluator

station_eval = StationEvaluator(
    'model.cfdb',
    'stations.cfdb',
    variable_heights={'air_temperature': 2.0, 'wind_speed': 10.0},
)

# Per-station, per-timestep metrics
station_eval.evaluate(
    'station_output.cfdb',
    variables=['air_temperature'],
    metrics=['bias', 'rmse'],
)

# Station-aggregated summary
station_eval.evaluate_aggregate(
    'aggregate_output.cfdb',
    variables=['air_temperature'],
    metrics=['bias', 'rmse', 'pearson'],
)
```

## Convenience Functions

Wrapper functions are available for one-call evaluation:

```python
from modverif.evaluate import (
    evaluate_models_cell,
    evaluate_models_domain,
    evaluate_stations,
    evaluate_fss,
    evaluate_wind,
)

evaluate_models_domain(
    'source.cfdb', 'test.cfdb', 'output.cfdb',
    variables=['air_temperature'],
    metrics=['bias', 'rmse', 'pearson'],
)
```

## Plotting

```python
from modverif.plots import plot_scatter, plot_station_map, plot_performance_diagram

# Scatter plot with statistics
plot_scatter(model_values, obs_values, save_path='scatter.png',
             variable_name='Temperature', units='K')

# Station map colored by metric
plot_station_map(lons, lats, bias_values, save_path='map.png',
                 metric_name='Bias')

# Performance diagram for multiple models
plot_performance_diagram(
    [0.85, 0.72], [0.15, 0.28],
    labels=['WRF-A', 'WRF-B'],
)
```
