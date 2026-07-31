"""
Spatial-structure diagnostics for scattered model-minus-observation error.

The question this answers is whether a model's point-wise bias has exploitable *spatial* structure:
does the error at one station tell you anything about the error at the next one, and over what
distance? The empirical variogram is the standard tool, and its fitted range and nugget say directly
how far a bias field can be interpolated and how much of it is irreducible point-scale noise.

Intended on the log-ratio ``log(model / obs)`` rather than the raw difference: a multiplicative bias
field is usually the stationary one, and comparing the Cressie--Hawkins and Matheron estimators near
the origin is a cheap test of which behaves better (see `fit_bias_variogram`'s ``ch_n`` /
``mat_n``).

The fitted length scale is called ``range_km`` throughout -- parameter and result key alike. Nothing
here named ``rng`` is anything but a ``numpy.random.Generator``.
"""
import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.optimize import curve_fit
from scipy.spatial.distance import pdist

# Bins with fewer pairs than this are dropped: a semivariance estimated from a handful of pairs is
# noise, and a noisy near-origin bin distorts the whole fit.
MIN_PAIRS_PER_BIN = 30
# Cap lags at this percentile of the pairwise distance distribution. More robust than the common
# "half the maximum distance" rule, which is driven by a single far-flung pair.
MAX_LAG_PCT = 60
# Window for the data-derived nugget estimate.
NEAR_ORIGIN_KM = 10.0


def ch_gamma(abs_diffs: np.ndarray) -> float:
    """
    Cressie--Hawkins robust semivariance from a set of ``|z_i - z_j|`` pair values.

    Built on the mean square-root difference rather than the mean squared difference, so a single
    outlying pair cannot dominate the estimate. Prefer this to `matheron_gamma` on
    heavy-tailed fields; their ratio near the origin is itself a diagnostic.

    Parameters
    ----------
    abs_diffs : np.ndarray
        Absolute pairwise differences of the field.

    Returns
    -------
    float
        Semivariance, or NaN if no pairs were supplied.
    """
    n = len(abs_diffs)
    if n == 0:
        return np.nan
    m = np.mean(np.sqrt(abs_diffs))
    return 0.5 * m ** 4 / (0.457 + 0.494 / n + 0.045 / n ** 2)


def matheron_gamma(abs_diffs: np.ndarray) -> float:
    """
    Classical Matheron semivariance from a set of ``|z_i - z_j|`` pair values.

    Parameters
    ----------
    abs_diffs : np.ndarray
        Absolute pairwise differences of the field.

    Returns
    -------
    float
        Semivariance, or NaN if no pairs were supplied.
    """
    return 0.5 * np.mean(abs_diffs ** 2) if len(abs_diffs) else np.nan


def empirical_variogram(
    distances: np.ndarray,
    abs_diffs: np.ndarray,
    estimator: str = 'ch',
    min_pairs_per_bin: int = MIN_PAIRS_PER_BIN,
    max_lag_pct: float = MAX_LAG_PCT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Bin scattered pairs by separation distance and estimate semivariance in each bin.

    The bin count is derived from the number of pairs rather than fixed, so the same call works for
    a few dozen stations and for a few thousand without retuning.

    Parameters
    ----------
    distances : np.ndarray
        Condensed pairwise distances (as from ``scipy.spatial.distance.pdist``), in km.
    abs_diffs : np.ndarray
        Absolute pairwise field differences, in the **same condensed order** as ``distances``.
    estimator : {'ch', 'matheron'}
        ``'ch'`` selects `ch_gamma`, anything else `matheron_gamma`.
    min_pairs_per_bin : int
        Bins with fewer pairs are dropped entirely rather than reported noisily.
    max_lag_pct : float
        Percentile of ``distances`` at which to cap the lag axis.

    Returns
    -------
    centers : np.ndarray
        Bin centre distances, km.
    gammas : np.ndarray
        Semivariance per retained bin.
    counts : np.ndarray
        Pair count per retained bin -- the natural fit weight.
    max_lag : float
        The lag cap actually used, km.
    """
    max_lag = float(np.percentile(distances, max_lag_pct))
    n_bins = int(np.clip(round(np.sqrt(len(distances)) / 4), 10, 20))
    edges = np.linspace(0.0, max_lag, n_bins + 1)
    est = ch_gamma if estimator == 'ch' else matheron_gamma
    centers, gammas, counts = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        sel = (distances >= lo) & (distances < hi)
        c = int(sel.sum())
        if c < min_pairs_per_bin:
            continue
        centers.append(0.5 * (lo + hi))
        gammas.append(est(abs_diffs[sel]))
        counts.append(c)
    return np.array(centers), np.array(gammas), np.array(counts), max_lag


def exponential_variogram(h: np.ndarray, nugget: float, psill: float, range_km: float) -> np.ndarray:
    """
    Isotropic exponential semivariogram model.

    Parameters
    ----------
    h : np.ndarray
        Lag distances, km.
    nugget : float
        Semivariance at zero separation -- the irreducible point-scale component.
    psill : float
        Partial sill; the model asymptotes to ``nugget + psill``.
    range_km : float
        e-folding length scale, km. Reported under the same name in the fit result dicts.

    Returns
    -------
    np.ndarray
        Modelled semivariance at each lag.
    """
    return nugget + psill * (1.0 - np.exp(-h / range_km))


def fit_exponential_variogram(
    centers: np.ndarray,
    gammas: np.ndarray,
    counts: np.ndarray,
    var_z: float,
    max_lag: float,
    d_min: float,
    nugget: float,
) -> dict | None:
    """
    Fit an isotropic exponential variogram with the nugget **fixed**, not free.

    Fixing it is deliberate. With a typical station network there are too few sub-10 km pairs to
    anchor the intercept, and a free three-parameter fit collapses the nugget toward zero -- which
    then reads as a smoothly interpolable bias field when the data say no such thing. Only
    ``(psill, range)`` are fitted, weighted by pair count.

    Parameters
    ----------
    centers, gammas, counts : np.ndarray
        Empirical variogram points from `empirical_variogram`.
    var_z : float
        Variance of the field; bounds the sill.
    max_lag : float
        Upper bound for the fitted range, km.
    d_min : float
        Smallest pairwise distance; lower bound for the fitted range.
    nugget : float
        The data-derived near-origin estimate to hold fixed. Clipped into ``[0, var_z]``.

    Returns
    -------
    dict or None
        ``nugget``, ``psill``, ``sill``, ``range_km``, ``rel_nugget``; or None if the fit
        did not converge or there were too few points to attempt one.
    """
    if len(centers) < 2 or var_z <= 0:
        return None
    nugget = float(min(max(nugget, 0.0), var_z))

    def model(h, psill, range_km):
        return nugget + psill * (1.0 - np.exp(-h / range_km))

    lo = [0.0, max(d_min, 1e-6)]
    hi = [2.0 * var_z, max_lag]
    p0 = [min(max(var_z - nugget, 1e-9), hi[0]), max_lag / 3.0]
    try:
        popt, _ = curve_fit(model, centers, gammas, p0=p0, bounds=(lo, hi),
                            sigma=1.0 / np.sqrt(counts), maxfev=10000)
    except Exception as e:
        # Printed rather than raised or warned, preserved verbatim from the pre-graduation code
        # because callers' console output is part of their provenance record. A library writing to
        # stdout is not good practice; changing it is a separate, deliberate decision.
        print(f'  [fit] exponential fit did not converge ({e.__class__.__name__}); showing cloud only')
        return None
    psill, range_km = float(popt[0]), float(popt[1])
    sill = nugget + psill
    return {'nugget': nugget, 'psill': psill, 'sill': sill, 'range_km': range_km,
            'rel_nugget': (nugget / sill) if sill > 0 else np.nan}


def fit_bias_variogram(
    x_km: np.ndarray,
    y_km: np.ndarray,
    z: np.ndarray,
    estimator: str = 'ch',
    near_origin_km: float = NEAR_ORIGIN_KM,
) -> dict:
    """
    Full variogram pipeline for a scattered field: empirical estimate plus a nugget-fixed fit.

    Single entry point so a diagnostic and any downstream interpolation cannot disagree about which
    variogram they are using.

    Parameters
    ----------
    x_km, y_km : np.ndarray
        Point coordinates in a **projected, metric** CRS, in km.
    z : np.ndarray
        Field value per point -- typically ``log(model / obs)``.
    estimator : {'ch', 'matheron'}
        Semivariance estimator, and which near-origin estimate becomes the fixed nugget.
    near_origin_km : float
        Separation below which pairs count toward the nugget estimate.

    Returns
    -------
    dict
        The fit parameters (``nugget``, ``psill``, ``sill``, ``range_km``, ``rel_nugget``), the empirical
        points (``centers``, ``gammas``, ``counts``, ``max_lag``), and diagnostics (``var_z``,
        ``ch_n``, ``mat_n``, ``n_near``, ``fit_ok``). Fit parameters are NaN when ``fit_ok`` is False.

        ``ch_n / mat_n`` is a stationarity read: a ratio near 1 means the field is well behaved at
        short range, and a ratio far from 1 means heavy tails that a Matheron estimate will
        exaggerate.
    """
    xy = np.column_stack([np.asarray(x_km, float), np.asarray(y_km, float)])
    z = np.asarray(z, float)
    d = pdist(xy)                      # condensed pairwise distance (km)
    abs_diffs = pdist(z[:, None])      # |z_i - z_j| in the same condensed order
    near = d < near_origin_km
    ch_n, mat_n = ch_gamma(abs_diffs[near]), matheron_gamma(abs_diffs[near])
    nugget_fix = ch_n if estimator == 'ch' else mat_n
    centers, gammas, counts, max_lag = empirical_variogram(d, abs_diffs, estimator)
    var_z = float(np.var(z))
    fit = fit_exponential_variogram(centers, gammas, counts, var_z, max_lag, float(d.min()), nugget_fix)
    out = {'centers': centers, 'gammas': gammas, 'counts': counts, 'max_lag': max_lag,
           'var_z': var_z, 'ch_n': float(ch_n), 'mat_n': float(mat_n),
           'n_near': int(near.sum()), 'fit_ok': fit is not None}
    out.update(fit if fit else {'nugget': np.nan, 'psill': np.nan, 'sill': np.nan,
                                'range_km': np.nan, 'rel_nugget': np.nan})
    return out


def bootstrap_variogram_params(
    x_km: np.ndarray,
    y_km: np.ndarray,
    z: np.ndarray,
    estimator: str = 'ch',
    n_boot: int = 400,
    drop_frac: float = 0.1,
    seed: int = 0,
) -> dict | None:
    """
    Resampling sensitivity of the fitted range and nugget.

    Refits on ``n_boot`` random subsamples, each dropping ``drop_frac`` of the points, and reports
    p5/p50/p95 bands. With few points the **range** is poorly identified -- worst when the field is
    nugget-dominated, where many ``(range, sill)`` pairs fit a flattish cloud about equally well --
    so its band is usually wide. The **nugget**, anchored to near-origin data, is much tighter. A
    range quoted without this band is a number with no error bar.

    Subsampling without replacement, deliberately: resampling with replacement would create
    zero-distance duplicate pairs and corrupt the near-origin estimate.

    **WARNING:**
       **This function takes a ``seed`` and constructs its own generator; it does not accept one.**
       That is intentional and load-bearing. Callers commonly invoke it more than once per run (say,
       for a log-ratio field and a raw-difference field), and each call is meant to start from the
       same stream so the two are independently reproducible. Threading one shared generator through
       instead -- the usual tidy-up -- silently changes every result after the first call.

    Parameters
    ----------
    x_km, y_km : np.ndarray
        Point coordinates in a projected, metric CRS, in km.
    z : np.ndarray
        Field value per point.
    estimator : {'ch', 'matheron'}
        Passed through to `fit_bias_variogram`.
    n_boot : int
        Number of subsample refits to attempt.
    drop_frac : float
        Fraction of points dropped per subsample; at least 6 points are always kept.
    seed : int
        Seed for this call's generator.

    Returns
    -------
    dict or None
        ``range_km`` and ``nugget``, each a ``(p5, p50, p95)`` tuple, plus ``drop_frac`` and ``n_ok``
        (how many refits converged). None if fewer than half the refits converged, since a band from
        a minority of fits would misrepresent the uncertainty rather than describe it.
    """
    x_km, y_km, z = np.asarray(x_km, float), np.asarray(y_km, float), np.asarray(z, float)
    n = len(z)
    keep = max(n - int(round(drop_frac * n)), 6)
    gen = np.random.default_rng(seed)
    ranges, nuggets = [], []
    for _ in range(n_boot):
        idx = gen.permutation(n)[:keep]
        f = fit_bias_variogram(x_km[idx], y_km[idx], z[idx], estimator)
        if f['fit_ok'] and np.isfinite(f['range_km']):
            ranges.append(f['range_km'])
            nuggets.append(f['nugget'])
    if len(ranges) < n_boot // 2:
        return None

    def band(a):
        return tuple(float(v) for v in np.percentile(a, [5, 50, 95]))

    return {'range_km': band(ranges), 'nugget': band(nuggets), 'drop_frac': drop_frac,
            'n_ok': len(ranges)}


# ---------------------------------------------------------------------------- spatial autocorrelation
def morans_i(zc: np.ndarray, weights: np.ndarray, s0: float, n: int, denom: float) -> float:
    """
    Moran's I for a pre-centred field under a given weight matrix.

    Takes ``s0``, ``n`` and ``denom`` as arguments rather than deriving them because callers evaluate
    many weight matrices, or many permutations, against the same field -- recomputing the invariants
    each time is the dominant cost.

    Parameters
    ----------
    zc : np.ndarray
        Field with its mean already removed.
    weights : np.ndarray
        ``(n, n)`` spatial weight matrix, zero on the diagonal.
    s0 : float
        Sum of ``weights``.
    n : int
        Number of locations.
    denom : float
        ``zc @ zc``, the field's total squared deviation.

    Returns
    -------
    float
        Moran's I. Its null expectation is ``-1 / (n - 1)``, **not zero** -- a small positive value
        can still be below chance.
    """
    return (n / s0) * float(zc @ (weights @ zc)) / denom


def morans_i_permutation(
    zc: np.ndarray,
    weight_mats,
    n: int,
    denom: float,
    rng: np.random.Generator,
    n_perm: int = 999,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Permutation null for Moran's I across one or more weight matrices.

    **WARNING:**
       **One shuffle is evaluated against every weight matrix, by design.** Testing each matrix with
       its own independent permutations would draw ``n_perm x len(weight_mats)`` times instead of
       ``n_perm``, and -- more importantly -- would destroy the correlation *between* the bands'
       nulls. A correlogram's bands are not independent tests of independent quantities; they are one
       field viewed at several scales, and a multiple-comparison correction applied across
       independently-generated nulls is answering a different question from the one asked.

       This is why the function takes a *list* of weight matrices rather than being called once per
       matrix. Refactoring it into a per-matrix helper changes both the draw count and the result.

    Parameters
    ----------
    zc : np.ndarray
        Field with its mean already removed.
    weight_mats : sequence of (np.ndarray, float)
        ``(weights, s0)`` pairs -- e.g. one per distance band of a correlogram, or a single entry for
        a global statistic.
    n : int
        Number of locations.
    denom : float
        ``zc @ zc``.
    rng : np.random.Generator
        Caller's generator. Consumed once per permutation, in order.
    n_perm : int
        Number of permutations.

    Returns
    -------
    observed : np.ndarray
        Moran's I per weight matrix on the real data.
    null : np.ndarray
        ``(n_perm, len(weight_mats))`` null statistics.
    """
    observed = np.array([morans_i(zc, w, s0, n, denom) for w, s0 in weight_mats])
    null = np.empty((n_perm, len(weight_mats)))
    for p in range(n_perm):
        zp = zc[rng.permutation(n)]
        for j, (w, s0) in enumerate(weight_mats):
            null[p, j] = morans_i(zp, w, s0, n, denom)
    return observed, null


# ------------------------------------------------------------------------------------ regionalisation
# (the out-of-sample PREDICTORS that consume these groupings live in modverif.crossval --
#  neither of them uses coordinates, so neither belongs in a spatial-structure module)
def best_kmeans(xy: np.ndarray, k: int, rng: np.random.Generator, n_restart: int = 20):
    """
    k-means on point coordinates, keeping the lowest-inertia labelling over several restarts.

    k-means converges to a local optimum that depends on initialisation, so a single run is a coin
    toss dressed as an answer. Restarts make the labelling reproducible in practice rather than only
    in principle.

    **NOTE:**
       The restart seeds are drawn in **one vectorised call**. Replacing that with per-restart scalar
       draws consumes the caller's generator differently and changes every downstream result.

    Parameters
    ----------
    xy : np.ndarray
        ``(n, 2)`` coordinates, in a projected metric CRS.
    k : int
        Number of clusters.
    rng : np.random.Generator
        Caller's generator, used only to seed the restarts.
    n_restart : int
        Restarts to attempt.

    Returns
    -------
    np.ndarray or None
        Cluster label per point, or **None** if every restart collapsed to fewer than ``k`` non-empty
        clusters -- which is the honest answer for a k the point set cannot support, and callers must
        handle it rather than assume an array.
    """
    best_lab, best_inertia = None, np.inf
    for s in rng.integers(0, 2**31 - 1, size=n_restart):
        try:
            cen, lab = kmeans2(xy, k, minit='++', seed=int(s), missing='raise')
        except Exception:  # noqa: S112 -- a failed restart is expected; the next seed may succeed
            continue
        if len(np.unique(lab)) < k:
            # Belt-and-braces, and measured as such: with missing='raise' scipy raises on an empty
            # cluster rather than returning a short labelling, so this branch did not fire once in
            # 800 attempts across degenerate inputs (identical points, duplicate locations, k > the
            # number of distinct sites). Kept because it is the guard that would matter if that
            # kwarg ever changed -- but do not spend effort trying to cover it with a test.
            continue
        inertia = float(np.sum((xy - cen[lab]) ** 2))
        if inertia < best_inertia:
            best_lab, best_inertia = lab, inertia
    return best_lab
