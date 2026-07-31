"""
Tests for modverif.window.

Two of these exist because a code review found the properties load-bearing and undefended:

* ``test_ties_are_common_and_resolve_to_the_earliest_window`` -- zero-padded accumulation windows tie
  bitwise far more often than intuition suggests, so the tie-break rule is not a detail. It is pinned
  here in both directions: that ties occur, and that both selection paths resolve them the same way.
* ``test_the_two_series_paths_agree_bitwise_on_complete_data`` -- ``rolling_max_valid`` and
  ``rolling_window_sums`` + ``max_window`` are separate implementations on purpose, but on
  all-finite input they must produce the identical float. If a future simplification builds one on
  the other, or changes a summation order, this says so before the science does.
"""
import numpy as np
import pytest

from modverif.window import max_window, rolling_max_valid, rolling_window_max, rolling_window_sums


def burst_series(n: int = 72, start: int = 30, length: int = 6, amount: float = 10.0) -> np.ndarray:
    """A short burst inside an otherwise dry record -- the case that produces tied windows."""
    s = np.zeros(n)
    s[start:start + length] = amount
    return s


# ---------------------------------------------------------------------------------- rolling_window_max
def test_rolling_window_max_finds_the_burst():
    field = np.zeros((48, 2, 3))
    field[10:14, 1, 2] = 5.0          # 20 mm over 4 steps at one cell
    max_grid, start_idx = rolling_window_max(field, 6)
    assert max_grid[1, 2] == pytest.approx(20.0)
    assert max_grid[0, 0] == 0.0
    # The earliest window fully containing the burst starts at step 8 (covers 8..13).
    assert start_idx[1, 2] == 8


def test_rolling_window_max_reads_gaps_as_zero():
    """The grid convention: a NaN contributes 0 and its window still competes."""
    field = np.zeros((10, 1, 1))
    field[2:5, 0, 0] = 4.0
    field[3, 0, 0] = np.nan           # 12.0 -> 8.0, but the window is NOT disqualified
    max_grid, _ = rolling_window_max(field, 3)
    assert np.isfinite(max_grid[0, 0])
    assert max_grid[0, 0] == pytest.approx(8.0)


def test_rolling_window_max_returns_float32_and_int_start():
    max_grid, start_idx = rolling_window_max(np.ones((12, 2, 2)), 4)
    assert max_grid.dtype == np.float32
    assert np.issubdtype(start_idx.dtype, np.integer)


# ---------------------------------------------------------------------------------- rolling_max_valid
def test_rolling_max_valid_disqualifies_windows_containing_a_gap():
    """The observation convention -- the opposite of rolling_window_max, and deliberately so."""
    s = np.array([1.0, 1.0, 1.0, np.nan, 9.0, 9.0, 9.0])
    # The 9-heavy window (4..6) is complete and wins; any window touching index 3 is out.
    max_sum, n_valid = rolling_max_valid(s, 3)
    assert max_sum == pytest.approx(27.0)
    assert n_valid == 6


def test_rolling_max_valid_reports_coverage_even_with_no_complete_window():
    s = np.array([1.0, np.nan, 2.0, np.nan, 3.0])
    max_sum, n_valid = rolling_max_valid(s, 3)
    assert np.isnan(max_sum)
    assert n_valid == 3           # the diagnostic survives the failure to select


def test_rolling_max_valid_handles_a_series_shorter_than_the_window():
    max_sum, n_valid = rolling_max_valid(np.array([1.0, 2.0]), 5)
    assert np.isnan(max_sum)
    assert n_valid == 2


# ---------------------------------------------------------------------------------- rolling_window_sums
def test_rolling_window_sums_reports_both_conventions():
    s = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    sums, valid = rolling_window_sums(s, 2)
    assert sums.shape == (4,) and valid.shape == (4,)
    np.testing.assert_allclose(sums, [3.0, 2.0, 4.0, 9.0])   # gaps summed as zero
    np.testing.assert_array_equal(valid, [True, False, False, True])


def test_rolling_window_sums_accepts_integer_input():
    sums, valid = rolling_window_sums(np.array([1, 2, 3, 4]), 2)
    np.testing.assert_allclose(sums, [3.0, 5.0, 7.0])
    assert valid.all()


# ---------------------------------------------------------------------------------- max_window
def test_max_window_reports_start_spread_and_clamp():
    s = burst_series()
    sums, valid = rolling_window_sums(s, 24)
    vmax, start, spread, clamped = max_window(sums, valid, 0.02)
    assert vmax == pytest.approx(60.0)
    assert start == 12                       # earliest window fully containing steps 30..35
    assert spread > 0                        # a flat maximum, by construction
    assert clamped is False


def test_max_window_flags_a_maximum_pinned_to_the_start_of_the_record():
    s = np.zeros(30)
    s[:4] = 10.0                             # the event is cut off at the start of the record
    sums, valid = rolling_window_sums(s, 24)
    _, start, _, clamped = max_window(sums, valid, 0.02)
    assert start == 0
    assert clamped is True


def test_max_window_flags_a_maximum_pinned_to_the_end_of_the_record():
    """The mirror of the test above. Without it, deleting the ``start == len(sums) - 1`` half of the
    clamp check passes the whole suite, and an event running past the end of the record stops being
    flagged as truncated."""
    s = np.zeros(30)
    s[-4:] = 10.0                            # the event is cut off at the END of the record
    sums, valid = rolling_window_sums(s, 24)
    _, start, _, clamped = max_window(sums, valid, 0.02)
    assert start == len(sums) - 1
    assert clamped is True


# ---------------------------------------------------------------------- the tie tolerance itself
# These pin _TIE_TOL from BOTH sides. Without them the constant is undefended: mutation testing
# showed that setting it to 0.0 or to 1e-3 leaves every other test passing, so nothing would stop a
# later "cleanup" from changing the window start times the whole displacement analysis rests on.
def test_a_near_tie_within_the_tolerance_resolves_to_the_earlier_window():
    """Pins _TIE_TOL from below: at 0.0, the later exact maximum would win instead."""
    sums = np.array([1.0 - 5e-10, 0.1, 0.2, 0.3, 0.4, 1.0])
    valid = np.ones(6, bool)
    _, start, _, _ = max_window(sums, valid, 0.02)
    assert start == 0, 'a sum within _TIE_TOL of the maximum must count as tied, and earliest wins'


def test_a_near_miss_outside_the_tolerance_does_not_steal_the_maximum():
    """Pins _TIE_TOL from above: at 1e-3, the earlier near-miss would wrongly win."""
    sums = np.array([1.0 - 1e-4, 0.1, 0.2, 0.3, 0.4, 1.0])
    valid = np.ones(6, bool)
    _, start, _, _ = max_window(sums, valid, 0.02)
    assert start == 5, '_TIE_TOL is a float-noise allowance, not a physical tolerance'


def test_max_window_with_nothing_valid():
    sums = np.array([1.0, 2.0, 3.0])
    vmax, start, spread, clamped = max_window(sums, np.zeros(3, bool), 0.02)
    assert np.isnan(vmax) and start == -1 and np.isnan(spread) and clamped is False


def test_max_window_ignores_invalid_windows_even_when_they_are_larger():
    sums = np.array([100.0, 5.0, 7.0])
    vmax, start, _, _ = max_window(sums, np.array([False, True, True]), 0.02)
    assert vmax == pytest.approx(7.0)
    assert start == 2


# ------------------------------------------------------------------ the two load-bearing properties
def test_ties_are_common_and_resolve_to_the_earliest_window():
    """Ties are not a rare edge case, and both selection paths must break them the same way.

    A 6-step burst inside a 72-step record leaves 19 of the 49 24-step windows summing to *bitwise*
    identical totals -- every window that fully contains the burst and nothing else. Both the grid
    path (argmax) and the series path (>= vmax - tol, first candidate) scan first-to-last, so both
    pick the earliest. Reversing either scan would move reported start times by many steps.
    """
    s = burst_series()
    sums, valid = rolling_window_sums(s, 24)
    tied = np.flatnonzero(np.isclose(sums, sums.max(), atol=0, rtol=0))
    assert len(tied) == 19, 'the tie structure this test exists to pin has changed'
    assert len(sums) == 49

    _, start_series, _, _ = max_window(sums, valid, 0.02)
    _, start_grid = rolling_window_max(s[:, None, None], 24)

    assert start_series == int(start_grid[0, 0]), 'the two tie-break rules have diverged'
    assert start_series == int(tied[0]), 'a tie no longer resolves to the earliest window'
    assert int(tied[-1]) != int(tied[0]), 'the choice is real -- the last tied window differs'


def test_the_two_series_paths_agree_bitwise_on_complete_data():
    """rolling_max_valid and rolling_window_sums+max_window are separate code on purpose.

    They must still agree exactly on all-finite input, where the two missing-value conventions
    coincide. This catches a changed summation order, a dtype change, or a future "simplification"
    that rebuilds one on the other without checking.
    """
    rng = np.random.default_rng(0)
    for _ in range(50):
        s = rng.gamma(shape=1.5, scale=3.0, size=96)
        direct, _ = rolling_max_valid(s, 24)
        sums, valid = rolling_window_sums(s, 24)
        via_sums, _, _, _ = max_window(sums, valid, 0.02)
        assert direct == via_sums, 'the two series paths no longer produce the identical float'


def test_the_two_conventions_disagree_when_data_is_incomplete():
    """The complement of the test above: with a gap present, the conventions MUST differ.

    Without this, a regression that quietly unified them would still pass the agreement test.
    """
    s = np.zeros(48)
    s[10:20] = 8.0
    # Two gaps placed so that EVERY 24-step window (there are 25) contains at least one: windows
    # 0..5 span index 5, windows 2..24 span index 25.
    s[5] = np.nan
    s[25] = np.nan
    sums, valid = rolling_window_sums(s, 24)
    assert not valid.any(), 'the gap placement this test relies on no longer disqualifies every window'

    lenient, _, _, _ = max_window(sums, np.ones_like(valid), 0.02)   # gaps-as-zero convention
    strict, _ = rolling_max_valid(s, 24)                             # gaps-disqualify convention
    assert np.isfinite(lenient), 'the lenient rule should still report a maximum'
    assert np.isnan(strict), 'every window spans a gap, so the strict rule must decline'
