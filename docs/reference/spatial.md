# Spatial Structure

Does a model's point-wise bias have exploitable **spatial** structure — does the error at one station
tell you anything about the error at the next, and over what distance?

The empirical variogram answers this directly. Its two headline numbers are the ones that decide
whether a bias field can be interpolated at all:

| quantity | reads as |
|---|---|
| **range** | the distance beyond which errors are effectively unrelated |
| **nugget** | the irreducible point-scale component — variance that no amount of interpolation removes |
| **relative nugget** (`nugget / sill`) | how much of the field is noise. High means heavy smoothing and gauges that will not be reproduced exactly |

## Use the log-ratio, not the raw difference

Fit on `log(model / obs)` rather than `model - obs`. A multiplicative bias field is usually the
stationary one; the raw difference typically keeps climbing with lag instead of reaching a clean sill,
which means the stationarity assumption underpinning any interpolation does not hold.

[`fit_bias_variogram`](#modverif.spatial.fit_bias_variogram) reports `ch_n` and `mat_n` — the
Cressie–Hawkins and Matheron estimates near the origin — precisely so this is checkable rather than
assumed. **A ratio near 1 means the field is well behaved at short range**; far from 1 means heavy
tails that the classical estimator will exaggerate.

## The nugget is fixed, not fitted

[`fit_exponential_variogram`](#modverif.spatial.fit_exponential_variogram) holds the nugget at a
data-derived near-origin estimate and fits only `(psill, range)`.

This is deliberate and worth understanding before changing it. On a typical station network there are
too few sub-10 km pairs to anchor the intercept, and a free three-parameter fit collapses the nugget
toward zero — which then reads as a smoothly interpolable bias field when the data say no such thing.

!!! warning "Quote a range with its resampling band, or not at all"

    With few stations the range is poorly identified — worst when the field is nugget-dominated,
    where many `(range, sill)` pairs fit a flattish cloud about equally well.
    [`bootstrap_variogram_params`](#modverif.spatial.bootstrap_variogram_params) gives the p5/p50/p95
    band. A range reported without it is a number with no error bar, and the band is often wide
    enough to change the conclusion.

    That function takes a **seed** and builds its own generator rather than accepting one. This is
    load-bearing: callers typically invoke it once per field, and each call is meant to start from
    the same stream so results are independently reproducible. Threading a shared generator through
    instead silently changes every result after the first.

## Example

```python
import numpy as np
from pyproj import Transformer
from modverif.spatial import fit_bias_variogram, bootstrap_variogram_params

# Coordinates must be in a PROJECTED, metric CRS -- the lags are distances in km.
T = Transformer.from_crs(4326, 2193, always_xy=True)   # e.g. NZTM
gx, gy = T.transform(lons, lats)
x_km, y_km = np.asarray(gx) / 1000.0, np.asarray(gy) / 1000.0

fit = fit_bias_variogram(x_km, y_km, np.log(model / obs))
band = bootstrap_variogram_params(x_km, y_km, np.log(model / obs))

if fit['fit_ok']:
    print(f"range {fit['range_km']:.0f} km  (p5-p95 {band['range_km'][0]:.0f}-{band['range_km'][2]:.0f})")
    print(f"nugget effect {100 * fit['rel_nugget']:.0f}%")
```

## API

::: modverif.spatial
    options:
      show_root_heading: false
      show_source: false
