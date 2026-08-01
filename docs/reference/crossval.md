# Out-of-Sample Assessment

Does a candidate predictor of model bias **still work where you did not measure?**

That is a different question from "does it fit?", and it is the only one that matters when the point
of a bias correction is to reach the ungauged terrain *between* the gauges. A predictor fitted and
scored on the same points will nearly always look useful.

!!! warning "Predicting a point from a group that contains it is not out-of-sample"

    [`loo_cluster_pred`](#modverif.crossval.loo_cluster_pred) recomputes each group's mean **without**
    the point being predicted. Including it is the standard way to make a regionalisation look
    skilful when it is merely descriptive — the point is partly predicting itself, and the effect is
    largest for exactly the small groups a clustering tends to produce.

    A singleton group falls back to the leave-one-out **global** mean, because a group of one carries
    no information about itself.

## The baseline matters as much as the method

[`loo_global_mean`](#modverif.crossval.loo_global_mean) is the aspatial baseline — "correct
everything by the same amount". Any regionalisation, covariate or clustering has to beat *this* to
have earned its complexity, and a surprising amount of apparent skill disappears when it is measured
against a leave-one-out global mean rather than against no correction at all.

Turn two sets of residuals into one number with
[`compute_residual_skill_score`](metrics.md) — positive means the method beat the baseline, negative
means the baseline won, which is the outcome worth reporting rather than hiding.

## Nothing here is spatial

`loo_cluster_pred` takes values and labels. The labels may come from a spatial clustering
([`modverif.spatial.best_kmeans`](spatial.md)) or from anything else — land-use class, elevation
band, forecast regime. That is why these live apart from the structure diagnostics.

## API

::: modverif.crossval
    options:
      show_root_heading: false
      show_source: false
