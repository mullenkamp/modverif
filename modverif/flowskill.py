"""
Skill scores for simulated against observed river flow, and the benchmarks they are compared with.

Built for thalweg's stage-comparison gate and calibration objective (``thalweg/docs/evaluation.md`` §2.2), but
nothing here knows about thalweg or a model: inputs are plain arrays of the hours to be scored.

**Inputs are the scored hours only.** Each function takes 1-D arrays of equal length with no NaN; selecting
which hours count (valid flow codes, coverage, warm-up) is the caller's job and is never done silently here.
A NaN, a negative flow, a length mismatch or a constant observed series raises.

**The scores:**

- **KGE′** (Kling, Fuchs and Paulin 2012): 1 − √((r − 1)² + (β − 1)² + (γ − 1)²), with β = μ_sim/μ_obs and
  γ the ratio of coefficients of variation. Calibrated on √Q, it is thalweg's objective. On raw Q it is the
  gate's high-flow end.
- **The low-flow score:** 1 − √((r − 1)² + (α − 1)² + (β_low − 1)²). Here r and α = σ_sim/σ_obs are computed
  on log(Q + ε), and β_low on raw flows. α, not the CV ratio, because the CV of log flows depends on the flow
  unit (Santos et al. 2018). ε is one fixed value per gauge and split direction, the observed mean over the
  calibration half's scored hours divided by 100 (``low_flow_epsilon``). It is fixed so that per-year
  bootstrap statistics stay exact.
- **β_low:** the sum of the lowest k = max(1, ⌊0.3·n⌋) simulated flows over the sum of the lowest k observed
  flows, each series sorted on its own (the flow-duration low segment, Yilmaz et al. 2008).

**Degenerate cases:** a constant series is detected exactly (max equals min), never by a small standard
deviation. A constant simulation has r := 0 (as airGR's ``ErrorCrit_KGE2``). A constant observed series, a
negative flow, or a simulated mean of zero raises. Flows are non-negative, so a zero observed mean is a constant
series and is refused with it.
"""
from dataclasses import dataclass

import numpy as np

LOW_FLOW_FRACTION = 0.3
CLIMATOLOGY_HALF_WINDOW = 15  # days either side: a 31-day centred window (thalweg D10)


@dataclass(frozen=True)
class KGEPrime:
    score: float
    r: float
    beta: float
    gamma: float


@dataclass(frozen=True)
class LowFlow:
    score: float
    r: float
    alpha: float
    beta_low: float


def _pair(sim, obs):
    sim = np.asarray(sim, dtype=np.float64)
    obs = np.asarray(obs, dtype=np.float64)
    if sim.ndim != 1 or sim.shape != obs.shape:
        raise ValueError(f'sim and obs must be 1-D and the same length, got {sim.shape} and {obs.shape}')
    if len(obs) < 2:
        raise ValueError('at least two scored hours are needed')
    if not (np.isfinite(sim).all() and np.isfinite(obs).all()):
        raise ValueError('sim and obs must be finite: select the scored hours before scoring')
    if sim.min() < 0.0 or obs.min() < 0.0:
        raise ValueError('flows must be non-negative')
    if obs.max() == obs.min():
        raise ValueError('the observed series is constant: no score is defined')
    return sim, obs


def _r(a, b):
    """Pearson r; 0 for a constant ``a`` (``b`` is never constant here)."""
    if a.max() == a.min():
        return 0.0
    da, db = a - a.mean(), b - b.mean()
    return float((da * db).sum() / np.sqrt((da * da).sum() * (db * db).sum()))


def kge_prime(sim, obs):
    """KGE′ with its components (``KGEPrime``)."""
    sim, obs = _pair(sim, obs)
    mu_s, mu_o = sim.mean(), obs.mean()
    if mu_s == 0.0:
        raise ValueError('the simulated mean is zero: the CV ratio is undefined')
    r = _r(sim, obs)
    beta = mu_s / mu_o
    gamma = 0.0 if sim.max() == sim.min() else (sim.std() / mu_s) / (obs.std() / mu_o)
    score = 1.0 - np.sqrt((r - 1) ** 2 + (beta - 1) ** 2 + (gamma - 1) ** 2)
    return KGEPrime(float(score), r, float(beta), float(gamma))


def beta(sim, obs):
    """The volume ratio μ_sim/μ_obs on raw flows."""
    sim, obs = _pair(sim, obs)
    return float(sim.mean() / obs.mean())


def beta_low(sim, obs, fraction=LOW_FLOW_FRACTION):
    """Sum of the lowest k simulated over the lowest k observed flows, k = max(1, ⌊fraction·n⌋), each sorted on
    its own (so timing does not enter)."""
    sim, obs = _pair(sim, obs)
    k = max(1, int(np.floor(fraction * len(obs))))
    low_obs = np.partition(obs, k - 1)[:k].sum()
    if low_obs <= 0.0:
        raise ValueError('the lowest observed flows sum to zero or less: β_low is undefined')
    return float(np.partition(sim, k - 1)[:k].sum() / low_obs)


def low_flow_epsilon(obs_calibration):
    """ε for the low-flow score: the observed mean over the calibration half's scored hours, divided by 100."""
    obs = np.asarray(obs_calibration, dtype=np.float64)
    if obs.ndim != 1 or not len(obs) or not np.isfinite(obs).all():
        raise ValueError('ε needs the finite observed flows of the calibration half')
    return float(obs.mean() / 100.0)


def low_flow_score(sim, obs, eps, fraction=LOW_FLOW_FRACTION):
    """The low-flow score with its components (``LowFlow``); ``eps`` from ``low_flow_epsilon``."""
    sim, obs = _pair(sim, obs)
    if not (np.isfinite(eps) and eps > 0.0):
        raise ValueError(f'ε must be positive and finite, got {eps}')
    ls, lo = np.log(sim + eps), np.log(obs + eps)
    r = _r(ls, lo)
    alpha = float(ls.std() / lo.std())
    bl = beta_low(sim, obs, fraction)
    return LowFlow(float(1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (bl - 1) ** 2)), r, alpha, bl)


def _day_of_year(times):
    """Day of the (non-leap) year, 1–365, of each period-ending hour's start; 29 February is the 28th."""
    start = np.asarray(times).astype('datetime64[h]') - np.timedelta64(1, 'h')
    day = start.astype('datetime64[D]')
    year = day.astype('datetime64[Y]')
    doy = (day - year).astype(int) + 1  # 1-366
    leap = ((year.astype(int) + 1970) % 4 == 0) & (((year.astype(int) + 1970) % 100 != 0) |
                                                   ((year.astype(int) + 1970) % 400 == 0))
    return np.where(leap & (doy >= 60), doy - 1, doy)  # 29 Feb (60 in a leap year) joins 28 Feb (59)


def doy_climatology(cal_times, cal_obs, target_times, half_window=CLIMATOLOGY_HALF_WINDOW):
    """
    The day-of-year climatology benchmark: for each target hour, the mean of the calibration hours' observed
    flows whose day of year lies within ±``half_window`` days, wrapping at the year end (thalweg D10: a 31-day
    window). ``cal_obs`` must be the calibration half's scored hours only. Raises if any day of the year has no
    calibration hour within its window.
    """
    cal_obs = np.asarray(cal_obs, dtype=np.float64)
    cal_doy = _day_of_year(cal_times)
    if cal_doy.shape != cal_obs.shape or not np.isfinite(cal_obs).all():
        raise ValueError('cal_times and cal_obs must be the same length and finite')
    sums = np.bincount(cal_doy - 1, weights=cal_obs, minlength=365)
    counts = np.bincount(cal_doy - 1, minlength=365).astype(float)
    offsets = np.arange(-half_window, half_window + 1)
    win_sums = sum(np.roll(sums, -o) for o in offsets)
    win_counts = sum(np.roll(counts, -o) for o in offsets)
    if (win_counts == 0).any():
        raise ValueError('a day of the year has no calibration hours within its window')
    clim = win_sums / win_counts
    return clim[_day_of_year(target_times) - 1]
