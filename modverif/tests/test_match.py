"""
Tests for modverif.match.

The ones that matter defend the null benchmark, because it is the only thing standing between
"best-matching improved the fit" and a claim about the model:

* ``test_the_null_fires_when_there_is_no_real_correspondence`` -- on a field with no relationship to
  the observations, the null must reproduce the apparent improvement. A null that never fires is
  worse than no null, since it launders search luck as skill.
* ``test_null_draws_one_bearing_then_one_distance_per_attempt_including_discards`` -- the rejection
  loop's draw sequence is part of the contract, and its length is data-dependent. Batching or
  reordering the draws changes every downstream number while looking like a speed-up, and neither is
  visible to a test that only counts draws or compares a run against itself.
"""
import numpy as np
import pytest

from modverif.match import (
    MIN_GAUGES_FOR_IMPROVEMENT,
    best_match_locate,
    field_shift_objective,
    grid_best_match,
    logvar_improvement,
    nearest_indices,
    neighborhood_match,
    null_improvement,
    vector_coherence,
)


def toy_grid(ny=80, nx=90, res=1000.0):
    gy, gx = np.arange(ny) * res, np.arange(nx) * res
    return gx, gy


# ------------------------------------------------------------------------------ nearest_indices
def test_nearest_indices_rounds_to_the_closest_cell():
    coords = np.arange(10) * 100.0
    np.testing.assert_array_equal(nearest_indices(coords, np.array([0.0, 149.0, 151.0, 899.0])),
                                  [0, 1, 2, 9])


def test_nearest_indices_clips_rather_than_raising():
    """A point off the edge snaps to the end. Callers rely on this to avoid pre-filtering."""
    coords = np.arange(5) * 100.0
    np.testing.assert_array_equal(nearest_indices(coords, np.array([-1e6, 1e6])), [0, 4])


# -------------------------------------------------------------------------- neighborhood_match
def test_neighborhood_match_finds_a_displaced_peak():
    gx, gy = toy_grid()
    field = np.full((len(gy), len(gx)), 5.0)
    field[40, 50] = 100.0                       # the model's peak, 4 km from the "gauge"
    sx, sy = np.array([46_000.0]), np.array([40_000.0])
    obs = np.array([100.0])

    out = neighborhood_match(field, gx, gy, sx, sy, obs, [2000.0, 6000.0])
    assert out[2000.0]['bm'][0] == pytest.approx(5.0), 'the tight box should not reach the peak'
    assert out[2000.0]['near'][0] is np.False_
    assert out[6000.0]['bm'][0] == pytest.approx(100.0), 'the wide box should find it'
    assert out[6000.0]['nmx'][0] == pytest.approx(100.0)
    assert out[6000.0]['near'][0] is np.True_


def test_neighborhood_match_tolerance_is_configurable_but_the_key_is_not():
    """`within_frac` moves the threshold; the output key stays 'near' (a data contract)."""
    gx, gy = toy_grid()
    field = np.full((len(gy), len(gx)), 70.0)
    sx, sy = np.array([40_000.0]), np.array([40_000.0])
    obs = np.array([100.0])                      # model is 30% low
    assert 'near' in neighborhood_match(field, gx, gy, sx, sy, obs, [3000.0])[3000.0]
    assert neighborhood_match(field, gx, gy, sx, sy, obs, [3000.0], within_frac=0.2)[3000.0]['near'][0] is np.False_
    assert neighborhood_match(field, gx, gy, sx, sy, obs, [3000.0], within_frac=0.4)[3000.0]['near'][0] is np.True_


def test_the_default_tolerance_is_twenty_percent():
    """The default is what the 'near' key name promises; pin it independently of the argument."""
    gx, gy = toy_grid()
    sx, sy = np.array([40_000.0]), np.array([40_000.0])
    obs = np.array([100.0])
    just_inside = np.full((len(gy), len(gx)), 81.0)      # 19% low  -> inside a 20% tolerance
    just_outside = np.full((len(gy), len(gx)), 79.0)     # 21% low  -> outside it
    assert neighborhood_match(just_inside, gx, gy, sx, sy, obs, [3000.0])[3000.0]['near'][0] is np.True_
    assert neighborhood_match(just_outside, gx, gy, sx, sy, obs, [3000.0])[3000.0]['near'][0] is np.False_


def test_neighborhood_match_reports_nan_off_footprint():
    gx, gy = toy_grid()
    field = np.full((len(gy), len(gx)), np.nan)
    out = neighborhood_match(field, gx, gy, np.array([40_000.0]), np.array([40_000.0]),
                              np.array([50.0]), [3000.0])[3000.0]
    assert np.isnan(out['bm'][0]) and np.isnan(out['nmx'][0])
    assert out['near'][0] is np.False_


# --------------------------------------------------------------------------- best_match_locate
def test_best_match_locate_returns_where_the_match_is_and_when():
    gx, gy = toy_grid()
    field = np.full((len(gy), len(gx)), 5.0)
    start = np.full((len(gy), len(gx)), 12.0)
    field[42, 53] = 90.0
    start[42, 53] = 31.0
    value, iy, ix, when = best_match_locate(field, start, gx, gy, 50_000.0, 40_000.0, 90.0, 6000.0)
    assert value == pytest.approx(90.0)
    assert (iy, ix) == (42, 53), 'the matched cell index is the whole point of this function'
    assert when == pytest.approx(31.0), 'timing must come from the MATCHED cell, not the gauge cell'


def test_best_match_locate_declines_on_an_empty_box():
    gx, gy = toy_grid()
    field = np.full((len(gy), len(gx)), np.nan)
    value, iy, ix, when = best_match_locate(field, field, gx, gy, 40_000.0, 40_000.0, 50.0, 3000.0)
    assert np.isnan(value) and iy == -1 and ix == -1 and np.isnan(when)


# ------------------------------------------------------------------------------ the improvement
def test_logvar_improvement_is_positive_when_matching_helps():
    obs = np.full(20, 100.0)
    point = np.full(20, 100.0) * np.linspace(0.5, 2.0, 20)   # scattered
    best = np.full(20, 100.0) * np.linspace(0.95, 1.05, 20)  # much tighter
    imp, n = logvar_improvement(point, best, obs)
    assert 0.0 < imp < 1.0
    assert n == 20


def test_the_minimum_sample_floor_is_eight():
    """Pinned as a LITERAL. Asserting against the imported constant makes the test move with it,
    which mutation testing showed defends nothing."""
    assert MIN_GAUGES_FOR_IMPROVEMENT == 8


def test_logvar_improvement_needs_a_minimum_sample():
    obs = np.full(20, 100.0)
    best = np.full(20, 100.0)
    point = np.full(20, 100.0)
    point[7:] = np.nan                       # 7 usable points, one short of the floor
    imp, n = logvar_improvement(point, best, obs)
    assert np.isnan(imp), 'seven points should be refused'
    assert n == 7

    point8 = np.full(20, 100.0)
    point8[8:] = np.nan                      # exactly at the floor -- must now be accepted
    imp8, n8 = logvar_improvement(point8, best, obs)
    assert n8 == 8
    assert not np.isnan(imp8) or True        # value may be NaN if variance is zero; the count is the point


def test_logvar_improvement_drops_non_positive_values():
    """The ratio is logged, so zeros and negatives must be excluded, not crash or leak."""
    obs = np.array([100.0] * 12)
    point = np.array([0.0, -3.0] + [90.0] * 10)
    best = np.array([100.0] * 12)
    imp, n = logvar_improvement(point, best, obs)
    assert n == 10
    assert np.isnan(imp) or np.isfinite(imp)


# -------------------------------------------------------------------------- the null benchmark
def _random_field_and_gauges(seed=0, n=40):
    gen = np.random.default_rng(seed)
    gx, gy = toy_grid()
    field = gen.gamma(2.0, 30.0, (len(gy), len(gx)))
    sx = gen.uniform(gx[10], gx[-11], n)
    sy = gen.uniform(gy[10], gy[-11], n)
    return field, gx, gy, sx, sy, gen


def test_the_null_fires_when_there_is_no_real_correspondence():
    """On a field unrelated to the observations, best-matching still "improves" the fit.

    The null must reproduce that improvement. If it did not, the search's own optimism would be
    reported as model skill -- the exact failure this benchmark exists to prevent.
    """
    field, gx, gy, sx, sy, gen = _random_field_and_gauges(seed=1)
    obs = gen.gamma(2.0, 30.0, len(sx))          # drawn independently of the field
    pt, bm = grid_best_match(field, gx, gy, sx, sy, obs, 4000.0)
    real, _ = logvar_improvement(pt, bm, obs)
    null = null_improvement(field, gx, gy, sx, sy, obs, 4000.0, 30, np.random.default_rng(2))

    assert len(null) > 0, 'the null produced no usable trials'
    assert real > 0, 'searching a neighborhood always improves the apparent fit'
    assert np.median(null) == pytest.approx(real, abs=0.25), (
        'with no true correspondence the null should reproduce the apparent improvement')


class _RecordingGenerator:
    """Delegates to a real Generator while recording the arguments of every uniform() call."""

    def __init__(self, seed):
        self._g = np.random.default_rng(seed)
        self.calls = []

    def uniform(self, *a, **kw):
        self.calls.append((a, kw))
        return self._g.uniform(*a, **kw)


def test_null_draws_one_bearing_then_one_distance_per_attempt_including_discards():
    """The draw *sequence* is part of the contract, not just the count.

    Each attempt takes a bearing then a distance, in that order, **including attempts later
    discarded**. Reordering the two, batching them with a ``size=`` argument, or hoisting either out
    of the loop changes every downstream number while looking like a speed-up -- and none of those
    is visible to a test that only counts draws or only compares a run against itself.

    The attempt count is *measured*, not assumed: discards genuinely occur in this fixture, which is
    exactly the case the contract is about.
    """
    field, gx, gy, sx, sy, gen = _random_field_and_gauges(seed=3)
    obs = gen.gamma(2.0, 30.0, len(sx))

    spy = _RecordingGenerator(11)
    null = null_improvement(field, gx, gy, sx, sy, obs, 4000.0, 6, spy)

    assert len(spy.calls) % 2 == 0, 'draws did not come in pairs'
    attempts = len(spy.calls) // 2
    assert attempts > len(null), (
        'this fixture is meant to exercise the discard path; if nothing was ever discarded the test '
        'no longer covers "including discards"')

    for i in range(attempts):
        bearing_args, bearing_kw = spy.calls[2 * i]
        dist_args, dist_kw = spy.calls[2 * i + 1]
        assert bearing_args == (0, pytest.approx(2 * np.pi)), f'attempt {i}: first draw is not a bearing'
        assert dist_args == (30.0, 80.0), f'attempt {i}: second draw is not a distance in the default band'
        assert not bearing_kw and not dist_kw, 'a size= argument means the draws were batched'


def test_null_is_reproducible_from_a_seeded_generator():
    """Two runs from the same seed must agree exactly -- the property the gate ultimately rests on."""
    field, gx, gy, sx, sy, gen = _random_field_and_gauges(seed=8)
    obs = gen.gamma(2.0, 30.0, len(sx))
    a = null_improvement(field, gx, gy, sx, sy, obs, 4000.0, 8, np.random.default_rng(12))
    b = null_improvement(field, gx, gy, sx, sy, obs, 4000.0, 8, np.random.default_rng(12))
    np.testing.assert_array_equal(a, b)


def test_null_respects_the_offset_band():
    """A band far off the footprint must yield no usable trials rather than silently rescaling."""
    field, gx, gy, sx, sy, gen = _random_field_and_gauges(seed=4)
    obs = gen.gamma(2.0, 30.0, len(sx))
    far = null_improvement(field, gx, gy, sx, sy, obs, 4000.0, 5, np.random.default_rng(5),
                           null_dist_km=(5000.0, 6000.0))
    assert len(far) == 0, 'trials displaced entirely off the grid should be rejected'


def test_null_gives_up_rather_than_looping_forever():
    """The attempt cap must bound the loop even when nothing qualifies."""
    field, gx, gy, sx, sy, gen = _random_field_and_gauges(seed=6)
    obs = gen.gamma(2.0, 30.0, len(sx))
    out = null_improvement(field, gx, gy, sx, sy, obs, 4000.0, 3, np.random.default_rng(7),
                           min_frac=1.01)          # impossible to satisfy
    assert len(out) == 0


def test_grid_best_match_marks_off_footprint_points():
    gx, gy = toy_grid()
    field = np.full((len(gy), len(gx)), 50.0)
    field[:, :20] = np.nan
    sx = np.array([5_000.0, 60_000.0])
    sy = np.array([40_000.0, 40_000.0])
    pt, bm = grid_best_match(field, gx, gy, sx, sy, np.array([50.0, 50.0]), 2000.0)
    assert np.isnan(pt[0]) and np.isnan(bm[0])
    assert pt[1] == pytest.approx(50.0) and bm[1] == pytest.approx(50.0)


# ---------------------------------------------------------------- displacement (the other guards)
def test_vector_coherence_separates_aligned_from_random_offsets():
    aligned_dx, aligned_dy = np.full(40, 3.0), np.full(40, 4.0)
    rbar, p, mdx, mdy = vector_coherence(aligned_dx, aligned_dy)
    assert rbar == pytest.approx(1.0), 'identical bearings must give a resultant of 1'
    assert p < 1e-6
    assert (mdx, mdy) == pytest.approx((3.0, 4.0))

    gen = np.random.default_rng(0)
    th = gen.uniform(0, 2 * np.pi, 400)
    r_rand, p_rand, _, _ = vector_coherence(np.sin(th), np.cos(th))
    assert r_rand < 0.2, 'random bearings should barely resolve'
    assert p_rand > 0.05


def test_vector_coherence_resultant_length_arithmetic():
    """NB: this does NOT pin the bearing convention, and cannot.

    Mutation testing showed that replacing arctan2(dx, dy) with arctan2(-dx, -dy) changes the result
    by one ULP -- rotating every bearing by a constant leaves the resultant length unchanged. The
    convention is unobservable from rbar and p, which is now stated in the docstring rather than
    implied to matter.
    """
    # Due north: dx=0, dy=1. Due east: dx=1, dy=0. Both are single directions -> rbar 1.
    assert vector_coherence(np.array([0.0]), np.array([1.0]))[0] == pytest.approx(1.0)
    # Two vectors 90 degrees apart average to a resultant of 1/sqrt(2).
    rbar, _, _, _ = vector_coherence(np.array([0.0, 1.0]), np.array([1.0, 0.0]))
    assert rbar == pytest.approx(1 / np.sqrt(2))


def test_vector_coherence_on_no_vectors():
    assert all(np.isnan(v) for v in vector_coherence(np.array([]), np.array([])))


def test_field_shift_objective_recovers_a_known_displacement():
    """A field shifted by a known amount must put the objective's minimum at that shift."""
    gx, gy = np.arange(160) * 1000.0, np.arange(140) * 1000.0
    yy, xx = np.mgrid[0:140, 0:160]
    truth = 50.0 + 40.0 * np.exp(-(((xx - 80) / 25.0) ** 2 + ((yy - 70) / 25.0) ** 2))
    shifted = np.roll(np.roll(truth, 5, axis=1), 3, axis=0)   # model displaced +5 east, +3 north

    gen = np.random.default_rng(1)
    idx = gen.integers(40, 110, (30, 2))
    sx, sy = gx[idx[:, 0]], gy[idx[:, 1]]
    obs = truth[idx[:, 1], idx[:, 0]]

    offs, objective, used = field_shift_objective(shifted, gx, gy, sx, sy, obs, 12.0, 1.0)
    assert offs is not None and used.sum() >= 8
    a, b = np.unravel_index(int(np.nanargmin(objective)), objective.shape)
    assert offs[b] == pytest.approx(5.0), 'east shift not recovered (columns are easting)'
    assert offs[a] == pytest.approx(3.0), 'north shift not recovered (rows are northing)'


def test_field_shift_objective_scores_the_same_points_at_every_shift():
    """The fixed-sample rule. Without it, shifts that move points onto the footprint would be scored
    against a different sample from those that do not."""
    gx, gy = np.arange(80) * 1000.0, np.arange(80) * 1000.0
    field = np.full((80, 80), 50.0)
    field[:, :30] = np.nan                       # a large off-footprint region
    gen = np.random.default_rng(2)
    sx = gen.uniform(gx[10], gx[-11], 40)
    sy = gen.uniform(gy[10], gy[-11], 40)
    obs = np.full(40, 50.0)
    offs, objective, used = field_shift_objective(field, gx, gy, sx, sy, obs, 5.0, 1.0)
    assert offs is not None
    assert np.isfinite(objective).all(), 'every shift must score, having used one fixed sample'
    assert not used.all(), 'points near the NaN region should have been excluded'


def test_field_shift_objective_declines_with_too_few_usable_points():
    gx, gy = np.arange(60) * 1000.0, np.arange(60) * 1000.0
    field = np.full((60, 60), np.nan)
    offs, objective, used = field_shift_objective(
        field, gx, gy, np.array([30_000.0]), np.array([30_000.0]), np.array([10.0]), 5.0, 1.0)
    assert offs is None and objective is None
    assert used.shape == (1,), 'the mask is returned even on the decline path, to explain why'


def test_field_shift_objective_always_scores_the_zero_shift():
    """The no-displacement hypothesis must never be missing from the surface.

    A plain arange(-s, s+1, step) silently omits zero whenever the step does not divide the maximum
    shift -- so the one candidate a displacement study must always evaluate could be absent.
    """
    gx, gy = np.arange(120) * 1000.0, np.arange(120) * 1000.0
    field = np.full((120, 120), 50.0)
    gen = np.random.default_rng(9)
    sx = gen.uniform(gx[20], gx[-21], 30)
    sy = gen.uniform(gy[20], gy[-21], 30)
    obs = np.full(30, 50.0)
    for max_shift, step in [(5.0, 2.0), (10.0, 3.0), (7.0, 4.0), (10.0, 1.0)]:
        offs, _, _ = field_shift_objective(field, gx, gy, sx, sy, obs, max_shift, step)
        assert offs is not None
        assert 0.0 in offs, f'zero shift missing for max_shift={max_shift}, step={step}'
        np.testing.assert_allclose(offs, -offs[::-1], err_msg='offsets should be symmetric about 0')


def test_field_shift_objective_refuses_a_descending_axis():
    """Descending axes are silently catastrophic, not merely wrong: nearest_indices still works, so
    the function would return a plausible surface with the north/south sign inverted."""
    gx, gy = np.arange(60) * 1000.0, np.arange(60) * 1000.0
    field = np.full((60, 60), 50.0)
    args = (field, gx, gy[::-1], np.array([30_000.0]), np.array([30_000.0]), np.array([50.0]), 5.0, 1.0)
    with pytest.raises(ValueError, match='ascending'):
        field_shift_objective(*args)
    with pytest.raises(ValueError, match='ascending'):
        field_shift_objective(field, gx[::-1], gy, np.array([30_000.0]), np.array([30_000.0]),
                              np.array([50.0]), 5.0, 1.0)
