"""
Out-of-sample assessment of candidate predictors for a scattered error field.

The question here is not "does this predictor fit?" but **"does it still work where you did not
measure?"** -- which is the only version that matters when the point of a bias correction is to reach
the ungauged terrain between the gauges.

Nothing in this module is spatial. `loo_cluster_pred` takes values and labels; the labels may come
from a spatial clustering (`modverif.spatial.best_kmeans`) or from anything else. Keeping these
apart from the structure diagnostics is what makes them findable by the next consumer, who may have
no variogram in sight.

Pair the residuals these produce with `modverif.metrics.compute_residual_skill_score`, which is the
common way to turn two sets of residuals into a single "did it beat the baseline?" number.
"""
import numpy as np
from scipy.stats import theilslopes

from modverif.metrics import compute_residual_skill_score


def loo_cluster_pred(z: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    Leave-one-out prediction from cluster means.

    Each point is predicted from its own cluster's mean computed **without it**. Including the point
    in its own predictor is the standard way to make a regionalisation look skilful when it is
    merely descriptive.

    Parameters
    ----------
    z : np.ndarray
        Field values.
    labels : np.ndarray
        Cluster label per point.

    Returns
    -------
    np.ndarray
        Prediction per point. A singleton cluster falls back to the leave-one-out **global** mean,
        since a cluster of one carries no information about itself.
    """
    n = len(z)
    pred = np.empty(n)
    for i in range(n):
        same = (labels == labels[i])
        same[i] = False
        pred[i] = z[same].mean() if same.any() else z[np.arange(n) != i].mean()
    return pred


def loo_global_mean(z: np.ndarray) -> np.ndarray:
    """
    Leave-one-out global mean -- the aspatial baseline any regionalisation must beat.

    Parameters
    ----------
    z : np.ndarray
        Field values.

    Returns
    -------
    np.ndarray
        Mean of all other points, per point.
    """
    n = len(z)
    tot = z.sum()
    return (tot - z) / (n - 1)


def loo_theilsen(covariate: np.ndarray, values: np.ndarray) -> np.ndarray:
    """
    Leave-one-out Theil--Sen prediction of ``values`` from a scalar ``covariate``.

    Refits the regression n times, each time omitting the point being predicted. Theil--Sen rather
    than least squares because a bias field's relationship to a covariate is usually driven by the
    bulk of the points, and a handful of extreme stations should not set the slope.

    Parameters
    ----------
    covariate : np.ndarray
        Predictor per point -- elevation, distance to coast, whatever is being assessed.
    values : np.ndarray
        Field being predicted, typically ``log(model / obs)``.

    Returns
    -------
    np.ndarray
        Out-of-sample prediction per point.

    Notes
    -----
    Leave-one-out still shares *most* of the training set between folds, so it is the mildest
    honest test available. It says nothing about whether the relationship **extrapolates** beyond
    the range the points cover -- for that, use `holdout_high`.
    """
    n = len(values)
    pred = np.empty(n)
    for i in range(n):
        k = np.arange(n) != i
        s, b0, _, _ = theilslopes(values[k], covariate[k])
        pred[i] = b0 + s * covariate[i]
    return pred


def holdout_high(covariate: np.ndarray, values: np.ndarray, q: float):
    """
    Train on the low end of a covariate and predict the high end -- an **extrapolation** test.

    This is the decisive check when the operational claim is "correct the bias where we have no
    measurements", and the ungauged places sit at the *extreme* of the covariate. Rain gauges cluster
    in valleys, so a correction applied to ridges extrapolates the elevation slope beyond any support
    in the data. Leave-one-out cannot detect that; withholding the top of the range can.

    A negative skill here is the informative result: it says the relationship, however significant
    in-sample, does not reach the places the correction was wanted for.

    Parameters
    ----------
    covariate : np.ndarray
        Predictor per point. The **highest** ``q`` fraction is withheld.
    values : np.ndarray
        Field being predicted.
    q : float
        Fraction to withhold; at least 2 points are always held out.

    Returns
    -------
    skill : float
        `modverif.metrics.compute_residual_skill_score` of the fitted prediction against the
        **training-set mean** -- the honest baseline, since a model that cannot beat "assume the
        training average" has bought nothing.
    diagnostics : tuple
        ``(slope, intercept, slope_lo, slope_hi, max_train_covariate, held_indices)``. The fourth
        and fifth elements are what make the extrapolation visible: the slope's confidence interval,
        and the covariate value beyond which the prediction has no support at all.
    """
    n = len(values)
    order = np.argsort(covariate)
    nh = max(int(np.ceil(q * n)), 2)
    held = order[-nh:]
    tr = order[:-nh]
    s, b0, lo, hi = theilslopes(values[tr], covariate[tr])
    pm = b0 + s * covariate[held]
    pb = values[tr].mean()
    return (compute_residual_skill_score(values[held] - pm, values[held] - pb),
            (float(s), float(b0), float(lo), float(hi), covariate[tr].max(), held))


def cluster_holdout(pred_fn, values: np.ndarray, labels: np.ndarray) -> float:
    """
    Leave-one-**group**-out: predict each held-out group from all the others.

    Much harsher than leave-one-out, and much closer to the real question. Leave-one-out leaves a
    point surrounded by its own neighbours; leaving out a whole region asks whether the relationship
    transfers to somewhere it was never fitted. A regionalisation that scores well under
    leave-one-out and badly here is describing the gauges, not the field.

    Parameters
    ----------
    pred_fn : callable
        ``pred_fn(train_mask, held_mask) -> predictions for the held points``. Taking a callback
        rather than a fixed model is what lets the same protocol score a covariate regression, a
        group mean, or anything else, against the identical baseline.
    values : np.ndarray
        Field being predicted.
    labels : np.ndarray
        Group label per point.

    Returns
    -------
    float
        Skill against the **training-set mean** of each fold.
    """
    n = len(values)
    pm = np.empty(n)
    pb = np.empty(n)
    for c in np.unique(labels):
        held = labels == c
        tr = ~held
        pm[held] = pred_fn(tr, held)
        pb[held] = values[tr].mean()
    return compute_residual_skill_score(values - pm, values - pb)
