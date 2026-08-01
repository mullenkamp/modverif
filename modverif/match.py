"""
Point-to-grid neighborhood matching, and the selection-bias null that keeps it honest.

When a model is compared against point observations at the model's own nearest cell, a spatially
displaced but otherwise correct forecast scores badly. Relaxing the comparison to "the best-matching
cell within R" separates an *amplitude* error from a *placement* error -- but only if you also price
what that search buys by luck, which is what `null_improvement` is for.

**NOTE:**
   **This is the point-based neighborhood family. The grid-based one lives in**
   `modverif.metrics` -- ``compute_fss``, ``compute_fraction_field``,
   ``compute_fss_multi_scale``. The fractions skill score compares two *fields* by the fraction of
   each neighborhood exceeding a threshold; the functions here match *scattered points* against a
   field by value. Related ideas, different inputs, different answers.

**WARNING:**
   **Searching a neighborhood for the closest value is a biased estimator, always.** In a 3 km box
   on a 1 km grid there are ~50 candidate cells, so a near-matching value exists almost regardless of
   whether the model has any real skill at that location. Any improvement from best-matching must be
   compared against `null_improvement` -- the same search re-run at deliberately wrong
   locations -- or it means nothing. Reporting the improvement alone overstates the model.

Coordinates are in a **projected, metric CRS**; radii are in the same units (metres). Boxes are
square (Chebyshev), not circular, and that is deliberate: it makes the search a cheap
``searchsorted`` slice on each axis.
"""
import numpy as np

# Distance band for the common-offset null, km. Far enough that a displaced gauge set lands on
# genuinely unrelated cells, close enough to stay on the model footprint.
NULL_DIST_KM = (30.0, 80.0)

# Minimum number of usable gauges before a log-variance ratio means anything.
MIN_GAUGES_FOR_IMPROVEMENT = 8


def nearest_indices(coords: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    Nearest index per point on an ascending, **regularly spaced** axis.

    Computed arithmetically from the first two coordinates rather than by search, so it is O(1) per
    point -- and therefore assumes even spacing. On an irregular axis use ``np.searchsorted``.

    Parameters
    ----------
    coords : np.ndarray
        1-D ascending, regularly spaced axis.
    points : np.ndarray
        Positions to locate, same units.

    Returns
    -------
    np.ndarray
        Integer indices, clipped to the axis bounds -- a point outside the axis snaps to the nearest
        end rather than raising.
    """
    return np.clip(np.round((points - coords[0]) / (coords[1] - coords[0])).astype(int),
                   0, len(coords) - 1)


def neighborhood_match(
    field: np.ndarray,
    gx: np.ndarray,
    gy: np.ndarray,
    sx: np.ndarray,
    sy: np.ndarray,
    obs: np.ndarray,
    radii_m,
    within_frac: float = 0.2,
) -> dict:
    """
    Per point and per search radius: the best-matching cell value, the neighborhood maximum, and
    whether any cell falls within a tolerance of the observation.

    The three answer different questions. ``bm`` asks "could the model have been right nearby?",
    ``nmx`` asks "how intense does the model get nearby at all?", and ``near`` reduces the first to
    a pass/fail that survives tabulation.

    Parameters
    ----------
    field : np.ndarray
        Gridded model values, shape ``(len(gy), len(gx))``. NaN marks off-footprint cells.
    gx, gy : np.ndarray
        Ascending grid axis coordinates in a projected, metric CRS.
    sx, sy : np.ndarray
        Point coordinates in the same CRS.
    obs : np.ndarray
        Observed value per point.
    radii_m : iterable of float
        Half-widths of the square search boxes, in the CRS's units.
    within_frac : float
        Relative tolerance for the pass/fail flag.

    Returns
    -------
    dict
        ``{radius: {'bm': ndarray, 'nmx': ndarray, 'near': ndarray}}``, each array aligned to the
        input points. Entries are NaN (or False) where the box contained no finite cell.
    """
    sxa, sya, ob = np.asarray(sx, float), np.asarray(sy, float), np.asarray(obs, float)
    out = {}
    for r in radii_m:
        bm = np.full(len(ob), np.nan)
        nmx = np.full(len(ob), np.nan)
        near = np.zeros(len(ob), bool)
        for i in range(len(ob)):
            ix0 = int(np.searchsorted(gx, sxa[i] - r))
            ix1 = int(np.searchsorted(gx, sxa[i] + r))
            iy0 = int(np.searchsorted(gy, sya[i] - r))
            iy1 = int(np.searchsorted(gy, sya[i] + r))
            blk = field[iy0:iy1, ix0:ix1]
            blk = blk[np.isfinite(blk)]
            if blk.size:
                nmx[i] = float(blk.max())
                bm[i] = float(blk[np.argmin(np.abs(blk - ob[i]))])
                if ob[i] > 0 and (np.abs(blk - ob[i]) <= within_frac * ob[i]).any():
                    near[i] = True
        out[r] = {'bm': bm, 'nmx': nmx, 'near': near}
    return out


def best_match_locate(
    field: np.ndarray,
    start_hours: np.ndarray,
    gx: np.ndarray,
    gy: np.ndarray,
    px: float,
    py: float,
    obs: float,
    radius_m: float,
) -> tuple[float, int, int, float]:
    """
    Locate the best-matching cell for a single point: its value, **where it is**, and its timing.

    The extra return values are the point. Knowing that a matching value exists 4 km away is a weaker
    statement than knowing the match sits consistently to the northwest of every gauge, which is what
    turns a scatter of best-matches into a displacement measurement.

    Parameters
    ----------
    field : np.ndarray
        Gridded model values, shape ``(len(gy), len(gx))``.
    start_hours : np.ndarray
        Per-cell accumulation-window start, same shape as ``field`` -- so the matched cell's timing
        comes for free rather than needing a second lookup.
    gx, gy : np.ndarray
        Ascending grid axis coordinates, projected metric CRS.
    px, py : float
        Point coordinates in the same CRS.
    obs : float
        Observed value to match.
    radius_m : float
        Half-width of the square search box.

    Returns
    -------
    value : float
        Best-matching cell value, NaN if the box held no finite cell.
    iy, ix : int
        Indices of that cell in ``field``, or -1 if none.
    start : float
        That cell's window start, NaN if none.
    """
    ix0 = int(np.searchsorted(gx, px - radius_m))
    ix1 = int(np.searchsorted(gx, px + radius_m))
    iy0 = int(np.searchsorted(gy, py - radius_m))
    iy1 = int(np.searchsorted(gy, py + radius_m))
    blk = field[iy0:iy1, ix0:ix1]
    fin = np.isfinite(blk)
    if not fin.any():
        return np.nan, -1, -1, np.nan
    masked = np.where(fin, np.abs(blk - obs), np.inf)
    jy, jx = np.unravel_index(int(np.argmin(masked)), masked.shape)
    return float(blk[jy, jx]), iy0 + int(jy), ix0 + int(jx), float(start_hours[iy0 + jy, ix0 + jx])


def grid_best_match(
    field: np.ndarray,
    gx: np.ndarray,
    gy: np.ndarray,
    sx: np.ndarray,
    sy: np.ndarray,
    obs: np.ndarray,
    radius_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Point values and best-match values for a whole point set at one radius.

    The values-only workhorse behind `null_improvement`, which calls it once per null trial.

    Parameters
    ----------
    field : np.ndarray
        Gridded model values, shape ``(len(gy), len(gx))``.
    gx, gy : np.ndarray
        Ascending, **regularly spaced** grid axes (see `nearest_indices`).
    sx, sy : np.ndarray
        Point coordinates in the same CRS.
    obs : np.ndarray
        Observed value per point.
    radius_m : float
        Half-width of the square search box.

    Returns
    -------
    point_vals : np.ndarray
        Value at each point's own nearest cell, NaN where that cell is not finite.
    best_vals : np.ndarray
        Closest value to ``obs`` within the box, NaN where the box held no finite cell.
    """
    n = len(sx)
    pt = np.full(n, np.nan)
    bm = np.full(n, np.nan)
    jx = nearest_indices(gx, sx)
    jy = nearest_indices(gy, sy)
    for i in range(n):
        v = field[jy[i], jx[i]]
        pt[i] = v if np.isfinite(v) else np.nan
        ix0 = int(np.searchsorted(gx, sx[i] - radius_m))
        ix1 = int(np.searchsorted(gx, sx[i] + radius_m))
        iy0 = int(np.searchsorted(gy, sy[i] - radius_m))
        iy1 = int(np.searchsorted(gy, sy[i] + radius_m))
        blk = field[iy0:iy1, ix0:ix1]
        blk = blk[np.isfinite(blk)]
        if blk.size:
            bm[i] = float(blk[np.argmin(np.abs(blk - obs[i]))])
    return pt, bm


def logvar_improvement(point_vals: np.ndarray, best_vals: np.ndarray,
                       obs: np.ndarray) -> tuple[float, int]:
    """
    Fractional reduction in log-ratio variance from allowing the neighborhood search.

    ``1 - var(log(best / obs)) / var(log(point / obs))``.

    **Fractional, not absolute, and that is what makes the null comparison fair.** Random locations
    start from a much worse point fit, so an absolute difference would flatter them mechanically;
    a ratio asks the same question of the real and displaced gauge sets.

    Parameters
    ----------
    point_vals, best_vals : np.ndarray
        Nearest-cell and best-match values, as from `grid_best_match`.
    obs : np.ndarray
        Observed value per point.

    Returns
    -------
    improvement : float
        Fraction of log-variance removed, or NaN if fewer than
        ``MIN_GAUGES_FOR_IMPROVEMENT`` points are usable or the point fit had no variance.
    n_used : int
        How many points were finite and strictly positive in all three inputs -- the log needs
        positives, so zeros and gaps drop out.
    """
    m = (np.isfinite(point_vals) & np.isfinite(best_vals)
         & (point_vals > 0) & (best_vals > 0) & (obs > 0))
    if m.sum() < MIN_GAUGES_FOR_IMPROVEMENT:
        return np.nan, int(m.sum())
    lv_pt = float(np.var(np.log(point_vals[m] / obs[m])))
    lv_bm = float(np.var(np.log(best_vals[m] / obs[m])))
    return (1.0 - lv_bm / lv_pt) if lv_pt > 0 else np.nan, int(m.sum())


def null_improvement(
    field: np.ndarray,
    gx: np.ndarray,
    gy: np.ndarray,
    sx: np.ndarray,
    sy: np.ndarray,
    obs: np.ndarray,
    radius_m: float,
    n_trials: int,
    rng: np.random.Generator,
    min_frac: float = 0.8,
    null_dist_km: tuple[float, float] = NULL_DIST_KM,
) -> np.ndarray:
    """
    Price what the neighborhood search buys by luck alone.

    Per trial, displace **every** point by one *common* random offset and recompute the
    point-to-best-match improvement there. A common offset rather than independent per-point jitter
    is what makes this a null for *displacement*: it preserves the gauge network's geometry and its
    relationship to the field's spatial structure, and destroys only the correspondence.

    Compare the real improvement against this distribution. If it sits inside, the apparent skill of
    best-matching is search luck.

    **WARNING:**
       **The generator is consumed inside a rejection loop, so the number of draws depends on the
       data.** Each attempt takes exactly two values (bearing, then distance) *including attempts
       later discarded for landing off-footprint*. Batching the draws, reordering them, or
       precomputing offsets changes every downstream number while looking like a speed-up. Pass a
       generator in; do not restructure this loop.

    Parameters
    ----------
    field : np.ndarray
        Gridded model values.
    gx, gy : np.ndarray
        Ascending, regularly spaced grid axes.
    sx, sy : np.ndarray
        True point coordinates -- displaced internally, not modified.
    obs : np.ndarray
        Observed value per point.
    radius_m : float
        Half-width of the square search box; must match the radius being tested.
    n_trials : int
        Target number of usable trials. The loop gives up after ``10 * n_trials`` attempts, so a
        short return means the footprint was hard to land on.
    rng : np.random.Generator
        Caller's generator. See the warning above.
    min_frac : float
        Minimum fraction of displaced points that must land on finite cells for a trial to count.
    null_dist_km : tuple of float
        ``(min, max)`` offset magnitude, km, sampled uniformly.

    Returns
    -------
    np.ndarray
        Improvement per usable trial. May be shorter than ``n_trials``, and **may be empty** -- take
        percentiles only after checking, since ``np.percentile`` of an empty array raises.
    """
    out: list[float] = []
    attempts = 0
    while len(out) < n_trials and attempts < n_trials * 10:
        attempts += 1
        ang = rng.uniform(0, 2 * np.pi)
        dist = rng.uniform(*null_dist_km) * 1000.0
        ox, oy = dist * np.sin(ang), dist * np.cos(ang)
        pt, bm = grid_best_match(field, gx, gy, sx + ox, sy + oy, obs, radius_m)
        if np.isfinite(pt).mean() < min_frac:
            continue
        imp, _ = logvar_improvement(pt, bm, obs)
        if np.isfinite(imp):
            out.append(imp)
    return np.array(out)


def vector_coherence(dx: np.ndarray, dy: np.ndarray) -> tuple[float, float, float, float]:
    """
    Do the offset vectors point the same way? The second selection-bias guard.

    A real displacement makes neighbouring points' offsets align; search luck scatters them
    uniformly. This reduces that to one number -- the mean resultant length of the bearings, which is
    1 for perfect alignment and near 0 for random directions -- plus a Rayleigh test of uniformity.

    Use it alongside `null_improvement`. That one prices the *magnitude* of the apparent
    improvement; this one asks whether the offsets have a coherent *direction*, which a value-search
    on an unrelated field has no reason to produce.

    Parameters
    ----------
    dx, dy : np.ndarray
        Offset components per point, in a projected metric CRS. Pass only points that actually moved
        -- zero-length vectors have no bearing, and including them deflates the resultant.

    Returns
    -------
    rbar : float
        Mean resultant length in ``[0, 1]``.
    p : float
        Rayleigh p-value against the uniform null, ``exp(-n * rbar**2)``.
    mean_dx, mean_dy : float
        Mean offset vector, same units as the inputs.

    Notes
    -----
    Bearings are computed in **compass convention** -- ``arctan2(dx, dy)``, so 0 is north and angles
    increase clockwise. Note that ``rbar`` and ``p`` are **invariant to that choice**: rotating every
    bearing by a constant leaves the resultant length unchanged, so no convention is recoverable from
    them. The convention is documented because it governs how to interpret the returned mean offset
    vector, and because this is deliberately *not* the meteorological wind convention used by
    `modverif.metrics.compute_wind_direction_bias` -- the two answer different questions.
    """
    th = np.arctan2(dx, dy)   # compass-style: 0 = north
    n = len(th)
    if n == 0:
        return np.nan, np.nan, np.nan, np.nan
    c, s = np.cos(th).sum(), np.sin(th).sum()
    rbar = float(np.hypot(c, s) / n)
    p = float(np.exp(-n * rbar ** 2))
    return rbar, p, float(np.mean(dx)), float(np.mean(dy))


def field_shift_objective(
    field: np.ndarray,
    gx: np.ndarray,
    gy: np.ndarray,
    sx: np.ndarray,
    sy: np.ndarray,
    obs: np.ndarray,
    max_shift_km: float,
    step_km: float,
):
    """
    One whole-field displacement vector, with the objective surface that produced it.

    Where `neighborhood_match` lets every point find its own best cell -- and can therefore
    cherry-pick -- this asks a single question of the entire field: which rigid translation makes the
    model best match *all* the points at once? One number, no per-point freedom, and the surface
    around the optimum shows whether it is well determined or a broad flat basin.

    The scored point set is **fixed across every candidate shift**: only points whose full
    ``±max_shift`` box is finite and strictly positive take part. Otherwise shifts that happen to
    move points onto the footprint would score against a different sample from those that do not,
    and the comparison would be meaningless.

    Parameters
    ----------
    field : np.ndarray
        Gridded model values, shape ``(len(gy), len(gx))``.
    gx, gy : np.ndarray
        Ascending, regularly spaced grid axes in a projected metric CRS.
    sx, sy : np.ndarray
        Point coordinates in the same CRS.
    obs : np.ndarray
        Observed value per point; only strictly positive values participate (the objective is a
        log-ratio).
    max_shift_km : float
        Half-width of the shift search, km.
    step_km : float
        Shift increment, km; rounded to at least one grid cell.

    Returns
    -------
    offsets_km : np.ndarray or None
        Shift values along each axis, km. None if fewer than
        `MIN_GAUGES_FOR_IMPROVEMENT` points survived the fixed-sample rule.
    objective : np.ndarray or None
        ``J[a, b]`` = mean squared log-ratio at shift ``(dx=offsets[b], dy=offsets[a])`` -- **rows are
        northing, columns are easting**. None when ``offsets_km`` is None.
    used : np.ndarray
        Boolean mask of the points that were scored, returned even on the None path so a caller can
        report *why* it declined.

    Notes
    -----
    **Sign convention.** A positive optimum ``(u, v)`` means the model values that match the
    observations sit at ``point + (u, v)`` -- i.e. the field is displaced **by** ``+(u, v)``.
    Correcting it means shifting the field by ``-(u, v)``. Getting this backwards inverts every
    conclusion drawn from it, so it is stated here and in the returned array's axis order.
    """
    res = float(gx[1] - gx[0])
    if res <= 0 or gy[1] - gy[0] <= 0:
        # A descending axis is silently catastrophic here rather than merely wrong: nearest_indices
        # still works (the negative spacing cancels), so the function runs to completion and returns
        # a plausible surface with the north/south sign INVERTED -- exactly the conclusion-flipping
        # failure the Notes below warn about. Refuse instead.
        raise ValueError('gx and gy must be ascending; a descending axis silently inverts the '
                         'displacement sign')
    s = int(round(max_shift_km * 1000.0 / res))
    step = max(1, int(round(step_km * 1000.0 / res)))
    jx = nearest_indices(gx, sx)
    jy = nearest_indices(gy, sy)
    used = (obs > 0) & (jx >= s) & (jx < len(gx) - s) & (jy >= s) & (jy < len(gy) - s)
    for i in np.where(used)[0]:
        box = field[jy[i] - s:jy[i] + s + 1, jx[i] - s:jx[i] + s + 1]
        if not (np.isfinite(box).all() and (box > 0).all()):
            used[i] = False
    if used.sum() < MIN_GAUGES_FOR_IMPROVEMENT:
        return None, None, used
    # Built outward from 0 so the no-displacement hypothesis is ALWAYS scored. A plain
    # arange(-s, s+1, step) omits zero whenever step does not divide s -- the one shift a
    # displacement study must never fail to evaluate.
    half = np.arange(step, s + 1, step)
    offs = np.concatenate([-half[::-1], [0], half])
    objective = np.full((len(offs), len(offs)), np.nan)
    lo = np.log(obs[used])
    for a, v in enumerate(offs):       # rows: dy (north)
        for b, u in enumerate(offs):   # cols: dx (east)
            vals = field[jy[used] + v, jx[used] + u]
            objective[a, b] = float(np.mean((np.log(vals) - lo) ** 2))
    return offs * res / 1000.0, objective, used
