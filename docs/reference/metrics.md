# Metrics Reference

## Continuous Cell-Level Metrics

These metrics are computed at every grid cell for every timestep.

| Key | Name | Formula | Units |
|-----|------|---------|-------|
| `ne` | Normalised Error | `(test - source) / source * 100` | percent |
| `ane` | Absolute Normalised Error | `abs((test - source) / source) * 100` | percent |
| `rse` | Root Squared Error | `sqrt((test - source)^2)` | same as variable |
| `bias` | Mean Error (Bias) | `test - source` | same as variable |
| `mae` | Mean Absolute Error | `abs(test - source)` | same as variable |

## Categorical Cell-Level Metrics

Require a `threshold` parameter. Events are defined as values exceeding the threshold.

| Key | Name | Formula | Range |
|-----|------|---------|-------|
| `pod` | Probability of Detection | `Hits / (Hits + Misses)` | 0--1 |
| `far` | False Alarm Ratio | `False Alarms / (Hits + False Alarms)` | 0--1 |
| `csi` | Critical Success Index | `Hits / (Hits + False Alarms + Misses)` | 0--1 |
| `fbias` | Frequency Bias | `(Hits + False Alarms) / (Hits + Misses)` | 0--inf |

## Domain-Aggregated Metrics

Computed over the spatial domain for each timestep.

| Key | Name | Description |
|-----|------|-------------|
| `ne` | Normalised Error | NE computed from domain-summed values |
| `ane` | Absolute Normalised Error | ANE computed from domain-summed values |
| `rmse` | Root Mean Square Error | Spatially aggregated RMSE |
| `bias` | Mean Error | Spatially aggregated bias |
| `pearson` | Pearson Correlation | Spatial correlation coefficient (-1 to 1) |
| `pod` | Probability of Detection | From domain contingency table |
| `far` | False Alarm Ratio | From domain contingency table |
| `csi` | Critical Success Index | From domain contingency table |
| `gss` | Gilbert Skill Score | Equitable threat score from domain contingency table |
| `fbias` | Frequency Bias | From domain contingency table |

## Station Metrics

Per-station time series metrics.

| Key | Name | Description |
|-----|------|-------------|
| `bias` | Mean Bias | Mean difference (model - obs) |
| `mae` | Mean Absolute Error | Mean absolute difference |
| `rmse` | Root Mean Square Error | RMS of differences |
| `ne` | Normalised Error | Normalised by observed values |
| `ane` | Absolute Normalised Error | Absolute normalised difference |
| `pearson` | Pearson Correlation | Temporal correlation per station |

## Wind Metrics

Vector wind evaluation from U/V components.

| Key | Name | Units | Description |
|-----|------|-------|-------------|
| `vector_rmse` | Vector RMSE | m/s | `sqrt(mean(du^2 + dv^2))` |
| `speed_bias` | Wind Speed Bias | m/s | `mean(test_speed - source_speed)` |
| `direction_bias` | Wind Direction Bias | degrees | Mean directional difference (-180 to 180) |

## Fractions Skill Score (FSS)

Spatial verification at multiple neighborhood scales (default: 1, 3, 5, 9, 17, 33, 65 grid cells).

| Value | Interpretation |
|-------|---------------|
| 0.0 | No skill |
| 0.5 | Useful skill threshold |
| 1.0 | Perfect |

Requires a `threshold` parameter to binarize the fields before computing fractions.

## Diurnal Metrics

Any of the following metrics grouped by hour-of-day (0--23):

`bias`, `rmse`, `mae`, `pearson`

Supports a `utc_offset` parameter for local time conversion.


## API

Full signatures for every function in `modverif.metrics`, including
`compute_residual_skill_score` (residuals rather than model/obs pairs, and an **RMSE**-ratio
rather than the more common MSE-ratio skill score).

::: modverif.metrics
    options:
      show_root_heading: false
      show_source: false
