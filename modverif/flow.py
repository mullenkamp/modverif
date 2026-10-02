"""
Water-balance checks for river-flow records against model forcing: water years, coverage, annual
volumes, the implied precipitation multiplier, step detection and a pre-registered verdict.

Built for thalweg's step-0 water-balance gate (``thalweg/docs/evaluation.md`` §1), but nothing here
knows about thalweg, WRF or a particular gauge: inputs are hourly series on a period-ending UTC axis
and plain arrays.

**Time convention.** Every hourly series is labelled by the **end** of its hour, in UTC: the value
at ``t`` covers ``(t - 1 h, t]``. An hour belongs to the water year that contains its *start*, so
the hour labelled ``YYYY-07-01T00`` is the last hour of the previous July-start water year. Water
year ``Y`` starts at 00 UTC on the first day of ``start_month`` of calendar year ``Y``.

**WARNING:**
   **Coverage is measured against the calendar, never against the timestamps supplied.** An hour
   that is absent from the input is as missing as one that is NaN. Counting only the hours present
   would let a record with whole months absent report full coverage.
"""
from dataclasses import dataclass

import numpy as np
from scipy.stats import rankdata

from modverif.stats import permutation_pvalue

HOUR = np.timedelta64(1, 'h')

# The pre-registered water-balance classes (thalweg docs/evaluation.md §1). Inclusive and exclusive
# edges are part of the rule: green is closed at both ends, amber is open towards green.
GREEN = (0.8, 1.25)
AMBER = (0.67, 1.5)


def _hour_axis(times) -> np.ndarray:
    t = np.asarray(times).astype('datetime64[h]')
    if t.ndim != 1:
        raise ValueError('times must be one-dimensional')
    if len(t) > 1 and not (np.diff(t) > np.timedelta64(0, 'h')).all():
        raise ValueError('times must be strictly increasing (duplicate or unsorted hour labels)')
    return t


def water_year(times, start_month: int) -> np.ndarray:
    """
    Water-year label of each period-ending hourly timestamp.

    Parameters
    ----------
    times : array-like of datetime64
        Period-ending UTC hour labels.
    start_month : int
        First month of the water year (1-12).

    Returns
    -------
    np.ndarray of int
        The calendar year in which each hour's water year starts.
    """
    if not 1 <= start_month <= 12:
        raise ValueError(f'start_month must be 1-12, got {start_month}')
    start = np.asarray(times).astype('datetime64[h]') - HOUR
    year = start.astype('datetime64[Y]').astype(int) + 1970
    month = start.astype('datetime64[M]').astype(int) % 12 + 1
    return np.where(month >= start_month, year, year - 1)


def water_year_hours(year: int, start_month: int) -> int:
    """Number of hours in water year ``year``."""
    a = np.datetime64(f'{year:04d}-{start_month:02d}-01T00', 'h')
    b = np.datetime64(f'{year + 1:04d}-{start_month:02d}-01T00', 'h')
    return int((b - a) / HOUR)


def coverage(times, valid, start_month: int) -> dict:
    """
    Fraction of each water year's hours that hold a valid value.

    Parameters
    ----------
    times : array-like of datetime64
        Period-ending UTC hour labels, strictly increasing. Need not be contiguous.
    valid : array-like of bool
        Whether the value at each timestamp is usable.
    start_month : int
        First month of the water year.

    Returns
    -------
    dict
        ``{water_year: fraction}`` for every water year touched by ``times``. The denominator is the
        calendar length of the water year, so a partial year at either end of the record, or any
        absent hour, lowers the fraction.
    """
    t = _hour_axis(times)
    valid = np.asarray(valid, bool)
    if valid.shape != t.shape:
        raise ValueError('valid must have the same length as times')
    wy = water_year(t, start_month)
    out = {}
    for y in np.unique(wy):
        out[int(y)] = float(valid[wy == y].sum()) / water_year_hours(int(y), start_month)
    return out


def counted_years(cov: dict, threshold: float) -> list:
    """Water years whose coverage is at least ``threshold`` (e.g. 0.95), in order."""
    if not 0 < threshold <= 1:
        raise ValueError(f'threshold must be in (0, 1], got {threshold}')
    return sorted(y for y, c in cov.items() if c >= threshold)


def annual_sum(times, values, start_month: int, years) -> np.ndarray:
    """
    Sum of a complete hourly series over each requested water year.

    For forcing (precipitation, ET) in mm per hour. A forcing record must be complete, so a water year
    with any missing or non-finite hour **raises** instead of summing what is there.
    """
    t = _hour_axis(times)
    v = np.asarray(values, float)
    if v.shape != t.shape:
        raise ValueError('values must have the same length as times')
    wy = water_year(t, start_month)
    out = []
    for y in years:
        sel = wy == y
        n = water_year_hours(int(y), start_month)
        if sel.sum() != n or not np.isfinite(v[sel]).all():
            raise ValueError(f'water year {y} is incomplete ({int(sel.sum())} of {n} hours present, '
                             f'{int((~np.isfinite(v[sel])).sum())} non-finite); a forcing sum needs every hour')
        out.append(v[sel].sum())
    return np.array(out)


def annual_flow_mm(times, flow_m3s, area_m2: float, start_month: int, years) -> tuple:
    """
    Annual flow volume as a depth over the catchment, from hourly mean flow.

    Missing hours are filled by the mean of the year's valid hours (the volume is the valid mean
    times the year's length). This is unbiased only when gaps are not tied to flow; outages cluster
    in floods, which is why callers restrict this to years with high coverage and report the
    missing share returned here.

    Parameters
    ----------
    times : array-like of datetime64
        Period-ending UTC hour labels.
    flow_m3s : array-like
        Hourly mean flow in m³/s; NaN marks a missing hour.
    area_m2 : float
        Catchment area in m².
    start_month : int
        First month of the water year.
    years : iterable of int
        Water years to compute.

    Returns
    -------
    (np.ndarray, np.ndarray)
        Volume in mm per water year, and the fraction of each year's hours that was filled.
    """
    if not area_m2 > 0:
        raise ValueError(f'area_m2 must be positive, got {area_m2}')
    t = _hour_axis(times)
    q = np.asarray(flow_m3s, float)
    if q.shape != t.shape:
        raise ValueError('flow_m3s must have the same length as times')
    if (q < 0).any():
        raise ValueError(f'{int((q < 0).sum())} negative flow values (a sentinel?); clean them before summing')
    wy = water_year(t, start_month)
    vol, filled = [], []
    for y in years:
        n = water_year_hours(int(y), start_month)
        qy = q[wy == y]
        ok = np.isfinite(qy)
        if not ok.any():
            raise ValueError(f'water year {y} has no valid flow')
        vol.append(qy[ok].mean() * n * 3600.0 / area_m2 * 1000.0)
        filled.append(1.0 - ok.sum() / n)
    return np.array(vol), np.array(filled)


def implied_p_multiplier(flow_mm: dict, et_mm: dict, precip_mm: dict) -> float:
    """
    m = (sum of Q + sum of ET) / sum of P over the counted water years.

    The precipitation multiplier that would close the long-term balance if ET were right. It is the
    ratio of sums, not the mean of annual ratios (which weights dry years up).

    Parameters
    ----------
    flow_mm, et_mm, precip_mm : dict
        ``{water_year: total in mm}``. Keyed by year so a flow set for one span cannot be paired with
        forcing for another: the three key sets must be identical.

    Raises
    ------
    ValueError
        On mismatched years, an empty set, non-finite totals or a non-positive precipitation total,
        rather than returning a number that would be classified.
    """
    if not all(isinstance(d, dict) for d in (flow_mm, et_mm, precip_mm)):
        raise TypeError('flow_mm, et_mm and precip_mm must be dicts keyed by water year')
    years = set(flow_mm)
    if not years or years != set(et_mm) or years != set(precip_mm):
        raise ValueError('flow_mm, et_mm and precip_mm must cover the same, non-empty set of water years')
    ys = sorted(years)
    q, e, p = (np.array([d[y] for y in ys], float) for d in (flow_mm, et_mm, precip_mm))
    if not (np.isfinite(q).all() and np.isfinite(e).all() and np.isfinite(p).all()):
        raise ValueError('implied_p_multiplier received non-finite annual totals')
    if not p.sum() > 0:
        raise ValueError('precipitation total must be positive')
    return float((q.sum() + e.sum()) / p.sum())


@dataclass(frozen=True)
class PettittResult:
    """Pettitt change-point test on an ordered series."""

    K: float
    """The statistic, max |U_t|."""
    change: int
    """Label of the first element **after** the step (e.g. the first water year of the new regime)."""
    p: float
    """Permutation p-value (add-one corrected)."""
    shift: float
    """Mean after the step minus mean before it, in the series' own units."""
    n: int


def _pettitt_u(ranks: np.ndarray) -> np.ndarray:
    # U_t = 2 * sum_{i<=t} r_i - t (n + 1), t = 1 .. n-1, along the last axis.
    n = ranks.shape[-1]
    t = np.arange(1, n)
    return 2.0 * np.cumsum(ranks, axis=-1)[..., :-1] - t * (n + 1)


def pettitt(x, labels, n_perm: int = 9999, rng=None) -> PettittResult:
    """
    Pettitt test for a single change in location, with a permutation p-value.

    Ties take average ranks. The p-value comes from ``n_perm`` random permutations of the ranks, not
    from the closed-form approximation, whose true size at n of about 34 is near 2.5 % at a nominal
    5 %.

    Parameters
    ----------
    x : array-like
        The ordered series (e.g. annual Q/P for the counted water years, oldest first). No NaN.
    labels : array-like of int
        One label per element, e.g. its water year. The change point is reported as a label, never
        as a position, because a series with dropped years has positions that differ between gauges.
    n_perm : int
        Number of permutations.
    rng : numpy.random.Generator, optional
        Source of the permutations. Pass one for reproducible p-values.

    Returns
    -------
    PettittResult
    """
    x = np.asarray(x, float)
    labels = np.asarray(labels)
    if x.ndim != 1 or labels.shape != x.shape:
        raise ValueError('x and labels must be one-dimensional and the same length')
    if not np.isfinite(x).all():
        raise ValueError('pettitt received non-finite values')
    if len(labels) > 1 and not (np.diff(labels) > 0).all():
        raise ValueError('labels must be strictly increasing: the test assumes the series is in time order')
    n = len(x)
    if n < 4:
        raise ValueError(f'pettitt needs at least 4 values, got {n}')
    if rng is None:
        rng = np.random.default_rng()
    r = rankdata(x)
    u = np.abs(_pettitt_u(r))
    k = float(u.max())
    t = int(u.argmax()) + 1  # length of the first segment
    perms = rng.permuted(np.broadcast_to(r, (n_perm, n)), axis=1)
    null = np.abs(_pettitt_u(perms)).max(axis=1)
    p = permutation_pvalue(k, null, side='greater')
    return PettittResult(K=k, change=labels[t].item(), p=p, shift=float(x[t:].mean() - x[:t].mean()), n=n)


def regional_series(series: dict, min_gauges: int = 2) -> tuple:
    """
    Mean of the gauges' standardised series, by label.

    Each gauge's series is standardised over its own counted years (mean 0, sd 1), then averaged
    across the gauges present in each year. A step common to all gauges adds up in the mean while
    gauge-specific noise partly cancels.

    Parameters
    ----------
    series : dict
        ``{gauge: (labels, values)}``, e.g. water years and log(Q/P).
    min_gauges : int
        A label enters the regional series only if at least this many gauges have it.

    Returns
    -------
    (np.ndarray, np.ndarray, np.ndarray)
        Labels, regional means and the number of gauges contributing to each.
    """
    z = {}
    for g, (lab, val) in series.items():
        lab = np.asarray(lab)
        val = np.asarray(val, float)
        if lab.shape != val.shape or not np.isfinite(val).all():
            raise ValueError(f'gauge {g}: labels and values must match and be finite')
        if len(np.unique(lab)) != len(lab):
            raise ValueError(f'gauge {g}: duplicate labels')
        sd = val.std(ddof=1)
        if not sd > 0:
            raise ValueError(f'gauge {g}: zero spread, cannot standardise')
        z[g] = dict(zip(lab.tolist(), ((val - val.mean()) / sd).tolist(), strict=True))
    labels = sorted(set().union(*(d.keys() for d in z.values())))
    out_l, out_v, out_n = [], [], []
    for lab in labels:
        vals = [d[lab] for d in z.values() if lab in d]
        if len(vals) >= min_gauges:
            out_l.append(lab)
            out_v.append(np.mean(vals))
            out_n.append(len(vals))
    return np.array(out_l), np.array(out_v), np.array(out_n)


def step_signs(series: dict, change) -> dict:
    """
    Sign of each gauge's own shift across a given change label: +1, -1 or 0. A diagnostic.

    The shift is the mean of the gauge's values at labels >= ``change`` minus the mean before it.
    A gauge with no values on one side gets 0.

    Not used to decide a common step: at the other gauges the sign of a shift that is not there is a
    coin flip, so sign agreement lets one gauge's step pass as common (17-42 % of the time in
    thalweg's self-tests). :func:`common_step` uses leave-one-out instead.
    """
    out = {}
    for g, (lab, val) in series.items():
        lab = np.asarray(lab)
        val = np.asarray(val, float)
        after = lab >= change
        if after.all() or not after.any():
            out[g] = 0
        else:
            out[g] = int(np.sign(val[after].mean() - val[~after].mean()))
    return out


@dataclass(frozen=True)
class CommonStep:
    """Outcome of :func:`common_step`."""

    common: bool
    """True when the regional step is significant, stays significant with any one gauge removed, and
    every leave-one-out change year lies within ``agree_years`` of the regional one."""
    regional: PettittResult
    """Pettitt on the regional series of all gauges."""
    leave_one_out: dict
    """``{gauge: PettittResult}`` on the regional series without that gauge; empty if the full test failed."""
    reason: str
    """Why ``common`` is what it is."""


def common_step(series: dict, alpha: float = 0.05, agree_years: int = 2, n_perm: int = 9999, rng=None,
                min_gauges: int = 2) -> CommonStep:
    """
    Test for a step shared by the gauges, as opposed to steps in individual gauges' records.

    Three conditions, all required:

    1. the regional series (:func:`regional_series`) of all gauges has a significant step
       (:func:`pettitt`, p < ``alpha``);
    2. it stays significant with each gauge left out in turn, so no single gauge can carry it (a
       rating change or site move disappears when that gauge is removed);
    3. every leave-one-out change point lies within ``agree_years`` water years of the regional one,
       so steps at different gauges in different years (independent rating changes) do not add up to
       a "common" step.

    Parameters
    ----------
    series : dict
        ``{gauge: (labels, values)}`` with at least three gauges; labels are water years (ints).
    alpha : float
        Significance level for the full and every leave-one-out test.
    agree_years : int
        Largest allowed distance, in label units, between a leave-one-out change point and the
        regional one. Pettitt places change points to about ±2-3 years at n of about 34.
    n_perm : int
        Permutations per Pettitt test.
    rng : numpy.random.Generator, optional
        Source of the permutations. Pass a seeded one for a reproducible (recorded) gate run.
    min_gauges : int
        Passed to :func:`regional_series`.

    Returns
    -------
    CommonStep
    """
    if len(series) < 3:
        raise ValueError(f'common_step needs at least three gauges, got {len(series)}')
    if rng is None:
        rng = np.random.default_rng()
    lab, val, _ = regional_series(series, min_gauges=min_gauges)
    regional = pettitt(val, lab, n_perm=n_perm, rng=rng)
    if not regional.p < alpha:
        return CommonStep(common=False, regional=regional, leave_one_out={},
                          reason=f'regional step not significant (p={regional.p:.3f})')
    loo = {}
    for g in series:
        sub = {k: v for k, v in series.items() if k != g}
        lab_g, val_g, _ = regional_series(sub, min_gauges=min_gauges)
        loo[g] = pettitt(val_g, lab_g, n_perm=n_perm, rng=rng)
    weak = [g for g, r in loo.items() if not r.p < alpha]
    if weak:
        return CommonStep(common=False, regional=regional, leave_one_out=loo,
                          reason=f'not significant without {weak}: carried by those gauges')
    apart = [g for g, r in loo.items() if abs(r.change - regional.change) > agree_years]
    if apart:
        return CommonStep(common=False, regional=regional, leave_one_out=loo,
                          reason=f'change years disagree without {apart} (more than {agree_years} y from '
                                 f'{regional.change}): separate steps, not a common one')
    return CommonStep(common=True, regional=regional, leave_one_out=loo,
                      reason=f'common step at {regional.change} (p={regional.p:.3f})')


def multiplier_class(m: float) -> str:
    """'green', 'amber' or 'red' for an implied precipitation multiplier. NaN raises."""
    if not np.isfinite(m):
        raise ValueError(f'multiplier is not finite ({m!r}); it cannot be classified')
    if GREEN[0] <= m <= GREEN[1]:
        return 'green'
    if AMBER[0] <= m <= AMBER[1]:
        return 'amber'
    return 'red'


_ORDER = {'green': 0, 'amber': 1, 'red': 2}


def wb_verdict(m: float, gauge_step: PettittResult, common: bool, alpha: float = 0.05) -> tuple:
    """
    The water-balance verdict for one gauge, with its reasons.

    - The multiplier sets green, amber or red by the pre-registered ranges.
    - A common step (``common`` is True, from :func:`common_step`) makes every gauge red: it points to
      the forcing.
    - Otherwise a significant step in this gauge's own series (p < ``alpha``) makes it at least
      amber: a step that is not common points to the gauge's record.

    The worst of these applies. Returns ``(verdict, reasons)``.

    ``common`` must be a bool: passing the :class:`CommonStep` object itself (always truthy) is
    refused rather than read as red.
    """
    if not isinstance(common, (bool, np.bool_)):
        raise TypeError(f'common must be a bool (e.g. CommonStep.common), got {type(common).__name__}')
    reasons = []
    verdict = multiplier_class(m)
    if verdict != 'green':
        reasons.append(f'multiplier {m:.3f} is {verdict}')
    if not np.isfinite(gauge_step.p):
        raise ValueError('gauge step p-value is not finite')
    if common:
        verdict = 'red'
        reasons.append('common step across gauges (forcing)')
    elif gauge_step.p < alpha:
        if _ORDER[verdict] < _ORDER['amber']:
            verdict = 'amber'
        reasons.append(f'single-gauge step at {gauge_step.change} (p={gauge_step.p:.3f})')
    return verdict, reasons


def double_mass(x, y) -> tuple:
    """Cumulative sums of two aligned series (e.g. annual P and Q), for a double-mass curve."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if x.shape != y.shape or not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError('double_mass needs two finite series of the same length')
    return np.cumsum(x), np.cumsum(y)
