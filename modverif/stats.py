"""
Resampling inference and multiple-comparison corrections.

Deliberately **not** in `modverif.spatial`: nothing here is spatial. A permutation p-value, a
family-wise error correction and a variance-explained ratio are general statistical tools, and the
next code to need them is as likely to be a teleconnection analysis as a variogram. Burying them in a
module named for one application is how a second consumer ends up writing its own copy.

**WARNING:**
   **Permutation tests are one-sided here, and the side is the caller's to choose.** The default
   (``'greater'``) asks "is the observed statistic unusually *high*?", which suits Moran's I and
   variance-explained. A test against a *negative* prior -- "is this correlation unusually low?" --
   needs ``side='less'``, and calling the default on it silently reports very nearly its complement.
"""
import numpy as np


def holm_adjust(pvals) -> np.ndarray:
    """
    Holm--Bonferroni step-down adjusted p-values, returned in the input order.

    Controls the family-wise error rate: the probability of *any* false positive across the whole
    family. Uniformly more powerful than plain Bonferroni at the same guarantee, so there is no
    reason to prefer Bonferroni.

    Use this when a single false positive would change the conclusion -- for example when scanning
    distance bands for spatial structure, where one spurious band is enough to claim structure that
    is not there. When the question is instead "what fraction of my many discoveries are false?",
    a false-discovery-rate procedure is the better tool and this is needlessly conservative.

    Parameters
    ----------
    pvals : array-like
        Unadjusted p-values.

    Returns
    -------
    np.ndarray
        Adjusted p-values, aligned to the input order, each capped at 1.0 and non-decreasing in the
        sorted order (the step-down monotonicity that makes the family interpretable).
    """
    p = np.asarray(pvals, float)
    if not np.isfinite(p).all():
        # Python's max() propagates NaN unpredictably through the step-down running maximum, so a
        # NaN input silently emerges as a finite -- and often significant -- adjusted value. A
        # p-value that could not be computed must not be laundered into one that can be reported.
        raise ValueError('holm_adjust received non-finite p-values; a p-value that could not be '
                         'computed cannot be corrected. Drop or handle them before calling.')
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for rank, idx in enumerate(np.argsort(p)):
        run = max(run, (m - rank) * p[idx])
        adj[idx] = min(run, 1.0)
    return adj


def permutation_pvalue(observed: float, null, side: str = 'greater') -> float:
    """
    One-sided permutation p-value with the add-one correction.

    The ``+1`` in both numerator and denominator counts the observed statistic as one of its own
    null draws. Without it a p-value of exactly 0 is reportable, which is never true -- it just means
    the null sample was too small to resolve the tail.

    Parameters
    ----------
    observed : float
        The statistic computed on the real data.
    null : array-like
        Statistics from the permuted replicates. Non-finite entries are **dropped**, and the
        denominator counts only what survived -- a replicate whose statistic failed to compute is
        not evidence either way, and leaving it in the denominator deflates the p-value.
    side : {'greater', 'less'}
        ``'greater'`` (default) tests whether ``observed`` is unusually **high**; ``'less'`` whether
        it is unusually **low**. There is no two-sided option: the tests this serves all have a
        directional prior, and picking a side after seeing the data is what the correction above
        cannot save you from.

    Returns
    -------
    float
        p-value in ``(0, 1]``.

    Raises
    ------
    ValueError
        If ``observed`` is not finite, or no finite null replicates remain.

    Notes
    -----
    **A non-finite ``observed`` is refused rather than scored.** Every comparison against NaN is
    False, so a NaN statistic would score zero extremes and return the *smallest reportable
    p-value* -- on **both** sides simultaneously. That is the most dangerous possible failure for
    this function: a statistic that could not be computed reported as maximally significant.
    Callers whose statistic can legitimately fail (too few samples, say) must check before calling.
    """
    null = np.asarray(null, float)
    if not np.isfinite(observed):
        raise ValueError(
            f'permutation_pvalue received a non-finite observed statistic ({observed!r}). NaN '
            f'compares False against everything, so scoring it would report maximum significance '
            f'on both sides. Check your statistic is computable before testing it.')
    finite = np.isfinite(null)
    if not finite.any():
        raise ValueError('permutation_pvalue received no finite null replicates')
    null = null[finite]
    if side == 'greater':
        extreme = np.sum(null >= observed)
    elif side == 'less':
        extreme = np.sum(null <= observed)
    else:
        raise ValueError(f"side must be 'greater' or 'less', got {side!r}")
    return (extreme + 1) / (len(null) + 1)


def eta_squared(z, labels) -> float:
    """
    Fraction of a field's variance explained by group means (between-SS / total-SS).

    The natural effect size for "do these groups capture anything?" -- but on its own it is not
    evidence, because **eta-squared rises mechanically with the number of groups**. Enough clusters
    will explain any field perfectly. Pair it with a permutation test (shuffle the values across
    fixed group labels) or with out-of-sample skill before concluding the grouping is real.

    Parameters
    ----------
    z : array-like
        Field values.
    labels : array-like
        Group label per value; any hashable labels, not necessarily contiguous integers.

    Returns
    -------
    float
        Ratio in ``[0, 1]``, or NaN if the field has no variance to explain.
    """
    z = np.asarray(z, float)
    labels = np.asarray(labels)
    gmean = z.mean()
    tot = float(np.sum((z - gmean) ** 2))
    if tot <= 0:
        return np.nan
    between = 0.0
    for c in np.unique(labels):
        m = labels == c
        between += m.sum() * (z[m].mean() - gmean) ** 2
    return float(between / tot)
