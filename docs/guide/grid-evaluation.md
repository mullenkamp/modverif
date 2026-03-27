# Grid-to-Grid Evaluation

The `Evaluator` class compares two gridded cfdb datasets. The source dataset is the reference (e.g., a higher-resolution run or reanalysis), and the test dataset is the model being evaluated.

## Setup

```python
from modverif import Evaluator

evaluator = Evaluator('source.cfdb', 'test.cfdb')
```

Both datasets must share the same spatial grid and time coordinates.

### Spatial Subsetting

Restrict evaluation to a geographic region:

```python
# Bounding box (min_lon, min_lat, max_lon, max_lat) in WGS84
evaluator = Evaluator(
    'source.cfdb', 'test.cfdb',
    region=(170.0, -46.0, 178.0, -40.0),
)

# Boolean mask (shape must match domain)
import numpy as np
mask = np.zeros((ny, nx), dtype=bool)
mask[10:50, 20:60] = True
evaluator = Evaluator('source.cfdb', 'test.cfdb', region=mask)
```

### Time Filtering

```python
evaluator = Evaluator(
    'source.cfdb', 'test.cfdb',
    start_time='2023-02-12T06:00',
    end_time='2023-02-12T18:00',
)
```

## Cell-Level Evaluation

Computes metrics at every grid cell for every timestep. Output has shape `(time, y, x)`.

```python
evaluator.evaluate_cell(
    'cell_output.cfdb',
    variables=['air_temperature', 'precipitation'],
    metrics=['ne', 'bias', 'mae'],
)
```

Available cell-level metrics: `ne`, `ane`, `rse`, `bias`, `mae`

For categorical metrics, provide a threshold:

```python
evaluator.evaluate_cell(
    'categorical_output.cfdb',
    variables=['precipitation'],
    metrics=['pod', 'far', 'csi', 'fbias'],
    threshold=1.0,
)
```

## Domain-Aggregated Evaluation

Computes metrics aggregated over the spatial domain for each timestep. Output has shape `(time, metric)`.

```python
evaluator.evaluate_domain(
    'domain_output.cfdb',
    variables=['air_temperature'],
    metrics=['ne', 'ane', 'rmse', 'bias', 'pearson'],
)
```

Available domain metrics: `ne`, `ane`, `rmse`, `bias`, `pearson`, `pod`, `far`, `csi`, `gss`, `fbias`

## Fractions Skill Score (FSS)

Multi-scale spatial verification for precipitation and other threshold-based fields. Computes FSS at multiple neighborhood sizes.

```python
evaluator.evaluate_fss(
    'fss_output.cfdb',
    variables=['precipitation'],
    threshold=1.0,
    neighborhood_sizes=[1, 3, 5, 9, 17, 33, 65],  # default
)
```

FSS ranges from 0 (no skill) to 1 (perfect). A value of 0.5 is commonly used as the "useful skill" threshold. Output has shape `(time, scale)`.

## Vector Wind Evaluation

Evaluates U/V wind component pairs together.

```python
evaluator.evaluate_wind(
    'wind_output.cfdb',
    u_var='u_wind',
    v_var='v_wind',
    metrics=['vector_rmse', 'speed_bias', 'direction_bias'],
)
```

## Diurnal Cycle Analysis

Groups metrics by hour-of-day (0--23 UTC).

```python
evaluator.evaluate_diurnal(
    'diurnal_output.cfdb',
    variables=['air_temperature'],
    metrics=['bias', 'rmse'],
)
```

## Output Format

All evaluation methods write results to cfdb datasets. Variable naming in the output follows the pattern `{variable_name}` with metric values stored along a metric coordinate dimension (for domain/diurnal) or directly as spatial fields (for cell-level).
