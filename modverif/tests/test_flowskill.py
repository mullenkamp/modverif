"""Flow skill scores and benchmarks (thalweg docs/evaluation.md §2.2). Failure cases first."""
import math

import numpy as np
import pytest

from modverif import flowskill
from modverif.flowskill import beta_low, doy_climatology, kge_prime, low_flow_epsilon, low_flow_score


def flows(n=2000, seed=3):
    """A positive, skewed, autocorrelated hourly series: recessions punctuated by events."""
    rng = np.random.default_rng(seed)
    q = np.empty(n)
    q[0] = 5.0
    for t in range(1, n):
        q[t] = 0.98 * q[t - 1] + 0.1 + (rng.gamma(0.3, 20.0) if rng.random() < 0.02 else 0.0)
    return q


Q = flows()
EPS = low_flow_epsilon(Q)


# --- refusals ---------------------------------------------------------------------------------------


@pytest.mark.parametrize('func', [lambda s, o: kge_prime(s, o), lambda s, o: low_flow_score(s, o, EPS),
                                  lambda s, o: beta_low(s, o), lambda s, o: flowskill.beta(s, o)])
@pytest.mark.parametrize('sim, obs', [(Q[:-1], Q), (np.r_[Q[:-1], np.nan], Q), (Q, np.r_[Q[:-1], np.inf]),
                                      (Q, np.full(len(Q), 3.0)), (Q[:1], Q[:1]), (np.r_[Q[:-1], -1e-9], Q),
                                      (Q, np.r_[Q[:-1], -1e-9]), (Q, np.full(len(Q), 0.1)),
                                      (Q.reshape(40, 50), Q.reshape(40, 50))])
def test_refuses_what_cannot_be_scored(func, sim, obs):
    with pytest.raises(ValueError):
        func(sim, obs)


def test_kge_refuses_a_zero_simulated_mean():
    with pytest.raises(ValueError, match='zero'):
        kge_prime(np.zeros(len(Q)), Q)


@pytest.mark.parametrize('eps', [0.0, -1.0, math.nan])
def test_low_flow_score_refuses_a_bad_epsilon(eps):
    with pytest.raises(ValueError):
        low_flow_score(Q, Q, eps)


# --- known values -----------------------------------------------------------------------------------


def test_perfect_simulation_scores_one():
    k, lf = kge_prime(Q, Q), low_flow_score(Q, Q, EPS)
    assert abs(k.score - 1.0) < 1e-12 and abs(lf.score - 1.0) < 1e-12
    assert lf.beta_low == 1.0


def test_double_flow_scores_zero_on_raw_and_known_value_on_sqrt():
    """sim = 2·obs: β = 2, γ = 1, r = 1, so KGE′ = 0 on raw Q; on √Q, β = √2, so KGE′ = 1 − (√2 − 1)."""
    assert abs(kge_prime(2 * Q, Q).score) < 1e-12
    assert abs(kge_prime(np.sqrt(2 * Q), np.sqrt(Q)).score - (1 - (math.sqrt(2) - 1))) < 1e-12


def test_kge_uses_the_cv_ratio_not_the_sd_ratio():
    """Kling 2012: with the σ ratio (KGE 2009), sim = 2·obs would give γ = 2 and a score of 1 − √2."""
    assert kge_prime(2 * Q, Q).gamma == pytest.approx(1.0, abs=1e-12)


def test_in_sample_mean_flow_benchmark_is_one_minus_root_two():
    """Knoben et al. 2019: a constant at the observed mean has r := 0, β = 1, γ = 0."""
    k = kge_prime(np.full(len(Q), Q.mean()), Q)
    assert (k.r, k.gamma) == (0.0, 0.0)
    assert abs(k.score - (1 - math.sqrt(2))) < 1e-12


def test_constant_simulation_is_detected_exactly():
    """np.full(n, c).std() is not exactly 0 for most c; the constant test must not depend on it."""
    c = next(v for v in np.linspace(0.1, 0.2, 1001) if np.full(len(Q), v).std() != 0.0)  # premise: one exists
    for v in (c, 1.0):  # 1.0: the mean is exact, so an unguarded r would be 0/0
        k = kge_prime(np.full(len(Q), v), Q)
        assert k.r == 0.0 and k.gamma == 0.0
    lf = low_flow_score(np.full(len(Q), 1.0), Q, EPS)
    assert lf.r == 0.0 and math.isfinite(lf.score)


# --- invariances and what each component can see -----------------------------------------------------


def test_scores_do_not_depend_on_the_flow_unit():
    k1, k2 = kge_prime(0.7 * Q, Q), kge_prime(700 * Q, 1000 * Q)
    assert abs(k1.score - k2.score) < 1e-12
    l1 = low_flow_score(0.7 * Q, Q, EPS)
    l2 = low_flow_score(700 * Q, 1000 * Q, 1000 * EPS)
    assert abs(l1.score - l2.score) < 1e-12


def test_proportional_low_flow_bias_moves_beta_low():
    """r and α on log flows barely see a uniform proportional bias; β_low does (review thalweg-design-1 B5)."""
    lf = low_flow_score(1.5 * Q, Q, EPS)
    assert lf.beta_low == pytest.approx(1.5, abs=1e-12)
    assert abs(lf.r - 1.0) < 1e-3 and abs(lf.alpha - 1.0) < 1e-2


def test_beta_low_sorts_each_series_on_its_own():
    """A time shift does not change the flow-duration curve, so β_low stays 1; pairing by time would not."""
    assert beta_low(np.roll(Q, 500), Q) == pytest.approx(1.0, abs=1e-12)


def test_beta_low_counts_the_lowest_thirty_percent():
    obs = np.arange(1.0, 11.0)  # k = 3: 1 + 2 + 3 = 6
    sim = obs.copy()
    sim[np.argsort(obs)[3]] = 0.5  # the 4th lowest drops below: now in the lowest 3 of sim
    assert beta_low(sim, obs) == pytest.approx((0.5 + 1 + 2) / 6)
    assert beta_low(obs, obs, fraction=0.05) == 1.0  # k = max(1, 0)


def test_epsilon_is_the_calibration_mean_over_one_hundred():
    assert low_flow_epsilon(Q) == pytest.approx(Q.mean() / 100.0)


# --- the climatology benchmark ---------------------------------------------------------------------


def hours(start, end):
    return np.arange(np.datetime64(start, 'h'), np.datetime64(end, 'h'))


def test_climatology_reproduces_a_seasonal_cycle():
    t = hours('2001-01-01T01', '2005-01-01T01')
    doy = flowskill._day_of_year(t)
    q = 10 + 5 * np.sin(2 * np.pi * doy / 365.0)
    clim = doy_climatology(t, q, t)
    assert np.abs(clim - q).max() < 0.15  # a 31-day mean of a sine loses about 1.2 % of its amplitude


def test_climatology_window_wraps_at_the_year_end():
    t = hours('2001-01-01T01', '2002-01-01T01')
    q = np.where(flowskill._day_of_year(t) <= 5, 100.0, 0.0)  # flow only in the first five days
    clim = doy_climatology(t, q, np.array([np.datetime64('2001-12-30T12', 'h')]))
    assert clim[0] > 0.0  # 30 December's window reaches into January


def test_climatology_maps_29_february_to_the_28th():
    t = np.array([np.datetime64('2004-02-29T12', 'h'), np.datetime64('2004-02-28T12', 'h'),
                  np.datetime64('2004-03-01T12', 'h'), np.datetime64('2005-03-01T12', 'h')])
    doy = flowskill._day_of_year(t)
    assert doy.tolist() == [59, 59, 60, 60]


def test_climatology_uses_each_hours_start_for_its_day():
    """Period-ending labels: the hour labelled 00:00 on 2 January belongs to 1 January."""
    assert flowskill._day_of_year(np.array([np.datetime64('2001-01-02T00', 'h')])).tolist() == [1]


def test_climatology_refuses_a_gap_wider_than_its_window():
    t = hours('2001-01-01T01', '2001-03-01T01')  # only January and February
    with pytest.raises(ValueError, match='no calibration hours'):
        doy_climatology(t, np.ones(len(t)), t)


def test_constant_observed_series_is_refused_even_when_its_std_is_not_zero():
    """0.1 repeated has a computed standard deviation of about 1e-17: max == min still sees it as constant."""
    obs = np.full(len(Q), 0.1)
    assert obs.std() != 0.0  # premise
    with pytest.raises(ValueError, match='constant'):
        kge_prime(Q, obs)


def test_beta_low_refuses_a_low_segment_of_zeros():
    obs = np.r_[np.zeros(40), np.arange(1.0, 61.0)]  # the lowest 30 of 100 are all zero
    with pytest.raises(ValueError, match='lowest observed'):
        beta_low(obs + 0.5, obs)


def test_beta_low_takes_the_floor_of_thirty_percent():
    """n = 9: 0.3·9 = 2.7, so k = 2 (round would give 3, ceil 3)."""
    obs = np.arange(1.0, 10.0)
    sim = obs.copy()
    sim[2] = 100.0  # the third-lowest: outside k = 2, so β_low stays 1
    assert beta_low(sim, obs) == 1.0


def test_climatology_window_is_exactly_thirty_one_days():
    """An impulse on one day reaches exactly the 15 days either side (D10)."""
    t = np.arange(np.datetime64('2001-01-01T01', 'h'), np.datetime64('2002-01-01T01', 'h'))
    doy = flowskill._day_of_year(t)
    q = np.where(doy == 100, 1.0, 0.0)
    clim = doy_climatology(t, q, t)
    reached = np.unique(doy[clim > 0])
    assert reached.tolist() == list(range(85, 116))
    assert flowskill.CLIMATOLOGY_HALF_WINDOW == 15
