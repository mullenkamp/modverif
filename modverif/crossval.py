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
