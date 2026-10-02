"""
Tests for modverif.flow, written failure-case first: each test names the defect it exists to catch.
"""
import numpy as np
import pytest

from modverif import flow


def hours(start, end):
    """Period-ending hour labels from ``start`` (exclusive) to ``end`` (inclusive)."""
    a = np.datetime64(start, 'h')
    b = np.datetime64(end, 'h')
    return np.arange(a + flow.HOUR, b + flow.HOUR, flow.HOUR)


# --- water years --------------------------------------------------------------------------------


def test_water_year_assigns_hour_by_its_start():
    # The hour labelled 07-01T00 covers 06-30T23..07-01T00, so it closes the previous water year.
    t = np.array(['2020-07-01T00', '2020-07-01T01'], dtype='datetime64[h]')
    assert flow.water_year(t, 7).tolist() == [2019, 2020]


def test_water_year_january_start_is_calendar_year():
    t = np.array(['2020-01-01T00', '2020-01-01T01', '2020-12-31T23'], dtype='datetime64[h]')
    assert flow.water_year(t, 1).tolist() == [2019, 2020, 2020]


def test_water_year_hours_counts_leap_day():
    assert flow.water_year_hours(2019, 7) == 8784  # Jul 2019 - Jun 2020 holds 29 Feb 2020
    assert flow.water_year_hours(2020, 7) == 8760


def test_water_year_rejects_bad_month():
    with pytest.raises(ValueError):
        flow.water_year(np.array(['2020-01-01T00'], dtype='datetime64[h]'), 13)


# --- coverage -----------------------------------------------------------------------------------


def test_coverage_counts_absent_hours_as_missing():
    # Defect caught: a denominator of "hours present" would report a year with a month absent as full.
    t = hours('2020-07-01T00', '2021-07-01T00')
    keep = ~((t > np.datetime64('2020-09-01T00')) & (t <= np.datetime64('2020-10-01T00')))
    cov = flow.coverage(t[keep], np.ones(keep.sum(), bool), 7)
    assert cov[2020] == pytest.approx(1 - 30 * 24 / 8760)


def test_coverage_counts_invalid_values():
    t = hours('2020-07-01T00', '2021-07-01T00')
    valid = np.ones(len(t), bool)
    valid[:876] = False
    assert flow.coverage(t, valid, 7)[2020] == pytest.approx(0.9)


def test_coverage_refuses_duplicate_labels():
    t = np.array(['2020-07-01T01', '2020-07-01T01'], dtype='datetime64[h]')
    with pytest.raises(ValueError, match='strictly increasing'):
        flow.coverage(t, [True, True], 7)


def test_counted_years_threshold_is_inclusive_and_excluded_set_is_below_it():
    cov = {2000: 0.95, 2001: 0.9499999, 2002: 1.0, 2003: 0.5}
    kept = flow.counted_years(cov, 0.95)
    assert kept == [2000, 2002]
    # Assert on the excluded set too: every dropped year is below the threshold.
    assert all(cov[y] < 0.95 for y in set(cov) - set(kept))


# --- annual totals ------------------------------------------------------------------------------


def test_annual_sum_refuses_an_incomplete_forcing_year():
    t = hours('2020-07-01T00', '2021-07-01T00')
    v = np.ones(len(t))
    v[100] = np.nan
    with pytest.raises(ValueError, match='incomplete'):
        flow.annual_sum(t, v, 7, [2020])
    with pytest.raises(ValueError, match='incomplete'):
        flow.annual_sum(t[1:], np.ones(len(t) - 1), 7, [2020])


def test_annual_sum_value():
    t = hours('2020-07-01T00', '2021-07-01T00')
    assert flow.annual_sum(t, np.full(len(t), 0.5), 7, [2020]).tolist() == [4380.0]


def test_annual_flow_mm_depth_and_fill():
    # 1 m3/s for a year over 1 km2 = 8760 * 3600 m3 / 1e6 m2 = 31.536 m = 31536 mm.
    t = hours('2020-07-01T00', '2021-07-01T00')
    q = np.ones(len(t))
    vol, filled = flow.annual_flow_mm(t, q, 1e6, 7, [2020])
    assert vol[0] == pytest.approx(31536.0)
    assert filled[0] == 0.0
    q[:876] = np.nan
    vol, filled = flow.annual_flow_mm(t, q, 1e6, 7, [2020])
    assert vol[0] == pytest.approx(31536.0)  # filled with the valid mean
    assert filled[0] == pytest.approx(0.1)


def test_annual_flow_mm_refuses_negative_flow():
    t = hours('2020-07-01T00', '2021-07-01T00')
    q = np.ones(len(t))
    q[5] = -9999.0
    with pytest.raises(ValueError, match='negative'):
        flow.annual_flow_mm(t, q, 1e6, 7, [2020])


def test_annual_flow_mm_counts_absent_hours_as_filled():
    t = hours('2020-07-01T00', '2021-07-01T00')[:-876]
    _, filled = flow.annual_flow_mm(t, np.ones(len(t)), 1e6, 7, [2020])
    assert filled[0] == pytest.approx(0.1)


def test_implied_p_multiplier_is_a_ratio_of_sums_not_a_mean_of_ratios():
    # Asymmetric years: ratio of sums (300 + 2100) / (500 + 2500) = 0.8; mean of ratios 0.72.
    q = {2000: 100.0, 2001: 1500.0}
    e = {2000: 200.0, 2001: 600.0}
    p = {2000: 500.0, 2001: 2500.0}
    assert flow.implied_p_multiplier(q, e, p) == pytest.approx(0.8)


def test_implied_p_multiplier_refusals():
    q = {2000: 400.0, 2001: 600.0}
    e = {2000: 500.0, 2001: 500.0}
    p = {2000: 1000.0, 2001: 1000.0}
    assert flow.implied_p_multiplier(q, e, p) == pytest.approx(1.0)
    with pytest.raises(ValueError, match='same'):
        flow.implied_p_multiplier(q, e, {2001: 1000.0, 2002: 1000.0})  # forcing for a different span
    with pytest.raises(ValueError):
        flow.implied_p_multiplier({**q, 2001: np.nan}, e, p)
    with pytest.raises(ValueError):
        flow.implied_p_multiplier(q, e, {2000: 0.0, 2001: 0.0})
    with pytest.raises(ValueError):
        flow.implied_p_multiplier({}, {}, {})
    with pytest.raises(TypeError):
        flow.implied_p_multiplier([400, 600], [500, 500], [1000, 1000])


# --- multiplier classes -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ('m', 'cls'),
    [
        (0.8, 'green'),
        (1.25, 'green'),
        (1.0, 'green'),
        (0.7999999, 'amber'),
        (1.2500001, 'amber'),
        (0.67, 'amber'),
        (1.5, 'amber'),
        (0.6699999, 'red'),
        (1.5000001, 'red'),
    ],
)
def test_multiplier_class_edges(m, cls):
    assert flow.multiplier_class(m) == cls


def test_multiplier_class_refuses_nan():
    # NaN compares False everywhere; it must never fall through to a class.
    with pytest.raises(ValueError):
        flow.multiplier_class(float('nan'))


# --- Pettitt ------------------------------------------------------------------------------------


def test_pettitt_finds_a_planted_step_and_reports_the_label():
    # The step is mid-series: Pettitt's |U_t| is t(n - t) under clean separation, so its location
    # estimate is pulled towards the middle and is exact only there (an off-centre step at index 20
    # is reported at 19 by both this code and pyhomogeneity).
    rng = np.random.default_rng(0)
    x = rng.normal(0, 0.1, 34)
    x[17:] += 0.5
    labels = np.arange(1990, 2024)
    r = flow.pettitt(x, labels, n_perm=1999, rng=np.random.default_rng(1))
    assert r.change == 2007
    assert r.p < 0.01
    assert r.shift > 0.4


def test_pettitt_reports_labels_not_positions_when_years_are_dropped():
    x = np.r_[np.zeros(10), np.ones(10)] + np.random.default_rng(2).normal(0, 0.05, 20)
    labels = np.r_[np.arange(1990, 2000), np.arange(2005, 2015)]  # five years dropped
    r = flow.pettitt(x, labels, n_perm=999, rng=np.random.default_rng(3))
    assert r.change == 2005


def test_pettitt_refuses_labels_out_of_order():
    with pytest.raises(ValueError, match='increasing'):
        flow.pettitt([1.0, 2.0, 3.0, 4.0, 5.0], [2001, 2000, 2002, 2003, 2004])


def test_pettitt_refuses_nonfinite():
    with pytest.raises(ValueError):
        flow.pettitt([1.0, 2.0, np.nan, 3.0, 4.0], [1, 2, 3, 4, 5])


def test_pettitt_permutation_size_is_near_nominal():
    # Under no step, p < 0.05 should occur about 5 % of the time (the closed form gives ~2.5 %).
    rng = np.random.default_rng(4)
    hits = 0
    n_rep = 400
    for _ in range(n_rep):
        x = rng.normal(size=34)
        hits += flow.pettitt(x, np.arange(34), n_perm=499, rng=rng).p < 0.05
    assert 0.02 <= hits / n_rep <= 0.09


def test_pettitt_matches_pyhomogeneity():
    pyh = pytest.importorskip('pyhomogeneity')
    rng = np.random.default_rng(5)
    for _ in range(50):
        x = rng.normal(size=34)
        x[rng.integers(5, 29):] += rng.normal(0, 1)
        ours = flow.pettitt(x, np.arange(34), n_perm=99, rng=rng)
        theirs = pyh.pettitt_test(x)
        assert ours.K == pytest.approx(theirs.U)
        assert ours.change == theirs.cp  # pyhomogeneity's cp is the first index after the change


# --- regional series and common step ------------------------------------------------------------


def test_regional_series_standardises_each_gauge_and_applies_min_gauges():
    s = {'a': ([2000, 2001, 2002], [1.0, 2.0, 3.0]), 'b': ([2001, 2002], [10.0, 30.0])}
    lab, val, n = flow.regional_series(s, min_gauges=2)
    assert lab.tolist() == [2001, 2002]
    za = (np.array([1.0, 2.0, 3.0]) - 2.0) / 1.0
    zb = (np.array([10.0, 30.0]) - 20.0) / np.std([10.0, 30.0], ddof=1)
    assert val.tolist() == pytest.approx([(za[1] + zb[0]) / 2, (za[2] + zb[1]) / 2])
    assert n.tolist() == [2, 2]


def test_step_signs():
    s = {
        'a': ([1, 2, 3, 4], [0, 0, 1, 1]),
        'b': ([1, 2, 3, 4], [1, 1, 0, 0]),
    }
    assert flow.step_signs(s, 3) == {'a': 1, 'b': -1}


def test_common_step_needs_three_gauges():
    s = {'a': ([1, 2, 3, 4], [0.0, 0, 1, 1]), 'b': ([1, 2, 3, 4], [0.0, 0, 1, 1])}
    with pytest.raises(ValueError, match='three'):
        flow.common_step(s)


def _years():
    return np.arange(1990, 2024)


def test_common_step_true_for_a_shared_step_false_for_one_gauge():
    rng = np.random.default_rng(9)
    years = _years()
    noise = rng.normal(0, 0.05, (4, 34))
    shared = {f'g{i}': (years, noise[i] + np.r_[np.zeros(17), np.full(17, 0.3)]) for i in range(4)}
    r = flow.common_step(shared, n_perm=999, rng=rng)
    assert r.common
    assert set(r.leave_one_out) == set(shared)
    one = {f'g{i}': (years, noise[i] + (i == 1) * np.r_[np.zeros(17), np.full(17, 0.3)]) for i in range(4)}
    r1 = flow.common_step(one, n_perm=999, rng=rng)
    # The full regional series may well be significant (one gauge carries it); leaving that gauge out
    # must break it.
    assert not r1.common


def test_common_step_false_for_unrelated_steps_in_different_years():
    # Defect caught: three gauges stepping in different years (independent rating changes) add up to
    # a significant regional series that survives leave-one-out, but they are not one common step.
    rng = np.random.default_rng(12)
    years = _years()
    noise = rng.normal(0, 0.02, (4, 34))
    steps = {0: 8, 1: 17, 2: 26}
    s = {f'g{i}': (years, noise[i] + (np.arange(34) >= steps[i]) * 0.4 if i in steps else noise[i]) for i in range(4)}
    r = flow.common_step(s, n_perm=999, rng=rng)
    assert not r.common
    assert 'disagree' in r.reason or 'not significant' in r.reason


def test_common_step_agreement_window_edge():
    # Four noise-free gauges: three step at index 16, the fourth at 17. The regional series of all four
    # changes at index 17 (1990 + 17 = 2007); leaving the fourth gauge out, at index 16 (2006). So one
    # leave-one-out change point lies exactly 1 year from the regional one: agree_years=1 passes,
    # agree_years=0 fails. (Measured with these exact inputs; see the assertions on the change years.)
    years = _years()
    base = (np.arange(34) >= 16).astype(float)
    late = (np.arange(34) >= 17).astype(float)
    s = {'a': (years, base), 'b': (years, base * 1.1), 'c': (years, base * 0.9), 'd': (years, late)}
    r1 = flow.common_step(s, agree_years=1, n_perm=999, rng=np.random.default_rng(16))
    assert r1.regional.change == 2007
    assert r1.leave_one_out['d'].change == 2006
    assert r1.common
    r0 = flow.common_step(s, agree_years=0, n_perm=999, rng=np.random.default_rng(16))
    assert not r0.common and 'disagree' in r0.reason


def test_common_step_alpha_is_applied_to_leave_one_out():
    # With alpha tiny no step is significant anywhere, so nothing is common.
    rng = np.random.default_rng(14)
    years = _years()
    noise = rng.normal(0, 0.05, (4, 34))
    s = {f'g{i}': (years, noise[i] + np.r_[np.zeros(17), np.full(17, 0.3)]) for i in range(4)}
    assert not flow.common_step(s, alpha=1e-6, n_perm=999, rng=rng).common


def test_common_step_min_gauges_drops_thin_years():
    # Gauges a, b cover 2000-2013 and c 2004-2017: three gauges in 2004-2013 (10 years), at least two in
    # 2000-2013 (14 years).
    rng = np.random.default_rng(15)
    yab = np.arange(2000, 2014)
    yc = np.arange(2004, 2018)
    s = {'a': (yab, rng.normal(size=14)), 'b': (yab, rng.normal(size=14)), 'c': (yc, rng.normal(size=14))}
    assert flow.common_step(s, n_perm=99, rng=rng, min_gauges=3).regional.n == 10
    assert flow.common_step(s, n_perm=99, rng=rng, min_gauges=2).regional.n == 14  # 2014-17 have c only


# --- verdict ------------------------------------------------------------------------------------

NO_STEP = flow.PettittResult(K=1.0, change=2000, p=0.6, shift=0.0, n=34)
STEP = flow.PettittResult(K=9.0, change=2010, p=0.01, shift=0.2, n=34)


def test_verdict_green_without_steps():
    assert flow.wb_verdict(1.0, NO_STEP, common=False)[0] == 'green'


def test_verdict_common_step_is_red_even_with_a_green_multiplier():
    assert flow.wb_verdict(1.0, NO_STEP, common=True)[0] == 'red'


def test_verdict_single_gauge_step_is_amber_and_never_red():
    assert flow.wb_verdict(1.0, STEP, common=False)[0] == 'amber'


def test_verdict_keeps_the_worst_class():
    assert flow.wb_verdict(2.0, STEP, common=False)[0] == 'red'
    assert flow.wb_verdict(0.7, NO_STEP, common=False)[0] == 'amber'


def test_verdict_alpha_is_005():
    # The pre-registered level: p just below 0.05 is a step, just above is not.
    below = flow.PettittResult(K=9.0, change=2010, p=0.049, shift=0.1, n=34)
    above = flow.PettittResult(K=9.0, change=2010, p=0.051, shift=0.1, n=34)
    assert flow.wb_verdict(1.0, below, common=False)[0] == 'amber'
    assert flow.wb_verdict(1.0, above, common=False)[0] == 'green'


def test_verdict_refuses_a_non_bool_common():
    cs = flow.CommonStep(common=False, regional=NO_STEP, leave_one_out={}, reason='')
    with pytest.raises(TypeError):
        flow.wb_verdict(1.0, NO_STEP, cs)


def test_double_mass():
    a, b = flow.double_mass([1, 2, 3], [2, 2, 2])
    assert a.tolist() == [1, 3, 6] and b.tolist() == [2, 4, 6]
    with pytest.raises(ValueError):
        flow.double_mass([1, 2], [1, np.nan])


def test_verdict_refuses_nan():
    with pytest.raises(ValueError):
        flow.wb_verdict(float('nan'), NO_STEP, common=False)
    with pytest.raises(ValueError):
        flow.wb_verdict(1.0, flow.PettittResult(K=1.0, change=2000, p=float('nan'), shift=0.0, n=34), common=False)


# --- behaviour of the whole rule on synthetic gauges --------------------------------------------


def _gauges(rng, cvs, step_at=None, step=0.0, which=None, n=34, rho=0.5):
    """Annual Q/P for len(cvs) gauges: lognormal noise with between-gauge correlation, optional step."""
    g = len(cvs)
    cov = np.full((g, g), rho) + (1 - rho) * np.eye(g)
    z = rng.multivariate_normal(np.zeros(g), cov, size=n).T
    sig = np.sqrt(np.log(1 + np.asarray(cvs) ** 2))[:, None]
    x = np.exp(sig * z)
    if step_at is not None:
        rows = range(g) if which is None else which
        for i in rows:
            x[i, step_at:] *= 1 + step
    years = np.arange(1990, 1990 + n)
    return {f'g{i}': (years, np.log(x[i])) for i in range(g)}


def _classify(series, rng):
    common = flow.common_step(series, n_perm=499, rng=rng).common
    out = []
    for lab_g, val_g in series.values():
        step = flow.pettitt(val_g, lab_g, n_perm=499, rng=rng)
        out.append(flow.wb_verdict(1.0, step, common)[0])
    return out


def test_rule_turns_a_large_common_step_red():
    rng = np.random.default_rng(6)
    reds = sum('red' in _classify(_gauges(rng, [0.13, 0.06, 0.18, 0.15], 17, 0.3), rng) for _ in range(40))
    assert reds >= 34  # >= 85 %; the design-level power is measured separately


def test_rule_keeps_a_single_gauge_step_out_of_red():
    rng = np.random.default_rng(7)
    reds = 0
    for _ in range(40):
        v = _classify(_gauges(rng, [0.13, 0.06, 0.18, 0.15], 17, 0.4, which=[1]), rng)
        reds += 'red' in v
    assert reds <= 4


def test_rule_rarely_turns_red_without_a_step():
    rng = np.random.default_rng(8)
    reds = sum('red' in _classify(_gauges(rng, [0.13, 0.06, 0.18, 0.15]), rng) for _ in range(80))
    assert reds <= 8
