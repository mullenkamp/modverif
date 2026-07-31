# Point-to-Grid Matching

Comparing a model against point observations at the model's own nearest cell punishes a forecast that
is *displaced* just as hard as one that is *wrong*. Relaxing the comparison to "the best-matching cell
within R" separates those two failures — the model got the amount right but put it in the wrong place,
versus the model got the amount wrong.

!!! danger "A neighborhood search is a biased estimator, always"

    In a 3 km box on a 1 km grid there are roughly 50 candidate cells. A near-matching value exists
    almost regardless of whether the model has any skill at that location, so **best-matching always
    improves the apparent fit**, including for a model with no skill at all.

    An improvement figure quoted on its own is therefore not a result. It has to be compared against
    [`null_improvement`](#modverif.match.null_improvement) — the same search re-run with the whole
    point set displaced to deliberately wrong locations. If the real improvement sits inside that
    distribution, what you measured was the search's optimism, not the model.

## Which function answers which question

| function | question |
|---|---|
| [`neighborhood_match`](#modverif.match.neighborhood_match) | Could the model have been right nearby? How intense does it get nearby at all? |
| [`best_match_locate`](#modverif.match.best_match_locate) | **Where** is the match, and **when** did it fall there? |
| [`grid_best_match`](#modverif.match.grid_best_match) | Values only, for a whole point set — the null's workhorse |
| [`logvar_improvement`](#modverif.match.logvar_improvement) | How much scatter did the search remove? |
| [`null_improvement`](#modverif.match.null_improvement) | How much of that was luck? |

`best_match_locate` is the one that turns matching into a *measurement*: knowing a matching value sits
4 km away is weaker than knowing the match sits consistently northwest of every gauge. The first is a
scatter; the second is a displacement vector.

## Why the null uses one common offset

`null_improvement` displaces **every** point by the *same* random offset each trial, rather than
jittering each point independently. That preserves the network's geometry and its relationship to the
field's spatial structure, and destroys only the correspondence — which is what makes it a null for
*displacement* specifically.

!!! warning "Do not restructure the null's sampling loop"

    The generator is consumed inside a rejection loop, so **the number of draws depends on the data**.
    Each attempt takes exactly a bearing then a distance, *including attempts later discarded* for
    landing off-footprint. Batching the draws with a `size=` argument, reordering them, or
    precomputing the offsets changes every downstream number while looking like a speed-up.

## Example

```python
import numpy as np
from modverif.match import grid_best_match, logvar_improvement, null_improvement

radius_m = 3000.0
point_vals, best_vals = grid_best_match(field, gx, gy, sx, sy, obs, radius_m)
real, n_used = logvar_improvement(point_vals, best_vals, obs)

null = null_improvement(field, gx, gy, sx, sy, obs, radius_m,
                        n_trials=999, rng=np.random.default_rng(0))

if len(null):                       # may be empty -- percentile of an empty array raises
    p = (np.sum(null >= real) + 1) / (len(null) + 1)
    print(f'improvement {real:.3f} vs null median {np.median(null):.3f}, p = {p:.3f}')
```

!!! note "Coordinates must be projected"

    All coordinates and radii are in a **projected, metric CRS** — the boxes are built by
    `searchsorted` on each axis in the axis's own units. Passing degrees produces a box whose
    east–west extent varies with latitude.

    Boxes are square (Chebyshev), not circular. That is deliberate: it makes each search a pair of
    cheap axis slices rather than a distance computation over every candidate cell.

## API

::: modverif.match
    options:
      show_root_heading: false
      show_source: false
