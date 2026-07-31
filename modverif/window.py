"""
Rolling-window accumulation maxima over time series and gridded fields.

The event-scale question these answer is "what is the largest N-step accumulation, and when did it
start?" -- the basis of an n-hour-maximum precipitation verification (24 h maxima being the common
case).

**Three missing-value conventions live here, deliberately, and they are not interchangeable.**
Unifying them behind a single ``nan_policy`` would be a behaviour change wearing the clothes of a
cleanup, so each is a separate function with its convention stated in its own docstring:

============================  =========================  ==================================
function                      input                      a window containing a gap...
============================  =========================  ==================================
:func:`rolling_window_max`    gridded ``(nt, ny, nx)``    ...counts, with gaps read as zero
:func:`rolling_max_valid`     1-D series                 ...is disqualified outright
:func:`rolling_window_sums`   1-D series                 ...is summed AND flagged, caller decides
============================  =========================  ==================================

The grid convention suits gap-free model output, where a NaN means "no precipitation recorded here"
rather than "unknown". The series convention suits observations, where a gauge window with a missing
hour must not compete against complete ones on an artificially low total. The third exists so that a
model series and an observation series can be reduced by *one* implementation -- which is what stops
the two sides' window start times from drifting apart numerically.

.. warning::
   **Tie-breaking is load-bearing and ties are common.** Zero-padded windows sum to bitwise-identical
   totals surprisingly often -- for a 6-hour burst inside a 72-hour record, 19 of 49 24-hour windows
   tie exactly. :func:`rolling_window_max` resolves a tie via ``argmax`` and :func:`max_window` via a
   ``>= vmax - _TIE_TOL`` first-candidate scan; **they agree only because both scan first-to-last.**
   Reversing either scan direction shifts reported window start times by hours. Do not "harmonise"
   the two rules, and do not reorder the scans.
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

# Absolute slack when deciding which windows count as maximal. Absolute rather than relative because
# it exists to absorb floating-point summation noise between windows over the same units, not to
# express a physical tolerance.
_TIE_TOL = 1e-9


def rolling_window_max(field: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-cell maximum rolling ``window``-step sum over the time axis of a gridded field.

    Missing values are read as zero, so every window competes. This suits gap-free model output,
    where a NaN means "nothing here" rather than "unknown"; for observations, where a gap must
    disqualify its window, use :func:`rolling_max_valid`.

    Parameters
    ----------
    field : np.ndarray
        Per-step increments, shape ``(nt, ny, nx)``. NaNs are treated as 0.
    window : int
        Number of consecutive time steps per accumulation window.

    Returns
    -------
    max_grid : np.ndarray
        Maximum window sum per cell, shape ``(ny, nx)``, float32.
    start_idx : np.ndarray
        Time index of the first step of each cell's maximising window, shape ``(ny, nx)``.
        Ties resolve to the **earliest** window (``argmax`` semantics) -- see the module warning.
    """
    field = np.nan_to_num(field, nan=0.0).astype('float64')
    windows = sliding_window_view(field, window_shape=window, axis=0)  # (nt-w+1, ny, nx, w)
    roll = windows.sum(axis=-1)              # (nt-w+1, ny, nx)
    start_idx = roll.argmax(axis=0)          # (ny, nx)
    max_grid = roll.max(axis=0).astype('float32')
    return max_grid, start_idx


def rolling_max_valid(series: np.ndarray, window: int) -> tuple[float, int]:
    """
    Maximum rolling ``window``-step sum of a 1-D series, ignoring windows that contain a gap.

    A window with any missing step is disqualified rather than summed over what is present -- an
    incomplete gauge window would otherwise compete on an artificially low total and could win.

    Parameters
    ----------
    series : np.ndarray
        1-D per-step increments. Non-finite entries mark gaps.
    window : int
        Number of consecutive steps per accumulation window.

    Returns
    -------
    max_sum : float
        Largest sum over fully-observed windows, or NaN if the series is shorter than one window or
        no window is complete.
    n_valid : int
        Count of finite steps in the whole series -- a coverage diagnostic, reported even when no
        complete window exists.
    """
    n_valid = int(np.isfinite(series).sum())
    if len(series) < window:
        return np.nan, n_valid
    w = sliding_window_view(series, window)
    ok = np.isfinite(w).all(axis=1)
    if not ok.any():
        return np.nan, n_valid
    return float(w[ok].sum(axis=1).max()), n_valid


def rolling_window_sums(series: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Every rolling ``window``-step sum of a 1-D series, plus which windows are fully observed.

    Reports both conventions instead of choosing: sums treat gaps as zero (the
    :func:`rolling_window_max` grid rule), while ``valid`` marks the all-finite windows (the
    :func:`rolling_max_valid` observation rule). One implementation can therefore reduce a model
    series and an observation series, which is what keeps the two sides' window start times from
    drifting apart numerically. Pair with :func:`max_window` to select.

    Parameters
    ----------
    series : np.ndarray
        1-D per-step increments. Non-finite entries mark gaps.
    window : int
        Number of consecutive steps per accumulation window.

    Returns
    -------
    sums : np.ndarray
        Rolling sums, shape ``(len(series) - window + 1,)``, gaps summed as zero.
    valid : np.ndarray
        Boolean, same shape, True where the window contains no gap.
    """
    w = sliding_window_view(np.asarray(series, 'float64'), window)
    sums = np.nansum(w, axis=1)
    valid = np.isfinite(w).all(axis=1)
    return sums, valid


def max_window(sums: np.ndarray, valid: np.ndarray, tol: float) -> tuple[float, int, float, bool]:
    """
    Select the maximal window from :func:`rolling_window_sums` output, with an ambiguity diagnostic.

    The diagnostic is the point: over a long event the maximum is often nearly flat, so the reported
    start time can move by hours at almost no cost in depth. ``spread`` quantifies that directly, and
    ``clamped`` flags a maximum pinned against the edge of the record, where the true maximum may lie
    outside the observed period.

    Parameters
    ----------
    sums : np.ndarray
        Rolling window sums. **Assumed non-negative** -- accumulations of a non-negative quantity,
        which is what :func:`rolling_window_sums` produces. With a negative maximum and ``tol > 0``,
        ``(1 - tol) * vmax`` exceeds ``vmax``, no window qualifies as near-maximal, and ``spread``
        raises on the empty selection. Unreachable for precipitation; stated because this is public
        API and the assumption is otherwise invisible.
    valid : np.ndarray
        Boolean mask of windows eligible to win.
    tol : float
        Relative depth tolerance defining "near-maximal" for ``spread``; e.g. ``0.02`` counts every
        window within 2 % of the maximum.

    Returns
    -------
    max_sum : float
        Largest sum over valid windows, or NaN if none are valid.
    start_idx : int
        Index of the maximising window, or -1 if none are valid. Ties resolve to the **earliest**
        window -- see the module warning; this must not be reordered.
    spread : float
        Index range spanned by valid windows within ``tol`` of the maximum: how far the reported
        start could move at less than ``tol`` cost in depth. NaN if no window is valid.
    clamped : bool
        True if the maximising window sits at the first or last available position, i.e. the record
        may be cutting the event off.
    """
    if not valid.any():
        return np.nan, -1, np.nan, False
    vmax = float(sums[valid].max())
    cand = np.where(valid & (sums >= vmax - _TIE_TOL))[0]
    start = int(cand[0])
    near = np.where(valid & (sums >= (1.0 - tol) * vmax))[0]
    spread = int(near.max() - near.min())
    clamped = start == 0 or start == len(sums) - 1
    return vmax, start, spread, clamped
