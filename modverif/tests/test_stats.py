"""
Tests for modverif.stats.

``test_permutation_pvalue_is_never_zero`` and the ``side`` tests matter most: a p-value of 0 is never
true, and calling the default on a test with a negative prior reports very nearly its complement --
a wrong answer that looks entirely reasonable.
"""
import numpy as np
import pytest

from modverif.stats import eta_squared, holm_adjust, permutation_pvalue


# ----------------------------------------------------------------------------------- holm_adjust
def test_holm_adjust_matches_a_worked_example():
    # m=4. Sorted: .01(x4=.04) .02(x3=.06) .03(x2=.06) .04(x1=.04 -> held at .06 by monotonicity)
    np.testing.assert_allclose(holm_adjust([0.01, 0.02, 0.03, 0.04]), [0.04, 0.06, 0.06, 0.06])


def test_holm_adjust_preserves_input_order():
    scrambled = holm_adjust([0.04, 0.01, 0.03, 0.02])
    np.testing.assert_allclose(scrambled, [0.06, 0.04, 0.06, 0.06])


def test_holm_adjust_is_monotone_and_capped():
    p = np.array([0.001, 0.2, 0.5, 0.9, 0.95])
    adj = holm_adjust(p)
    assert (adj <= 1.0).all(), 'an adjusted p-value above 1 is meaningless'
    assert (adj >= p).all(), 'adjustment must never make a p-value more significant'
    assert (np.diff(adj[np.argsort(p)]) >= 0).all(), 'step-down monotonicity violated'


def test_holm_adjust_is_less_conservative_than_bonferroni():
    """The reason to prefer it -- same family-wise guarantee, uniformly more power."""
    p = np.array([0.01, 0.02, 0.03, 0.04])
    assert (holm_adjust(p) <= len(p) * p + 1e-12).all()
    assert (holm_adjust(p) < len(p) * p).any(), 'Holm should be strictly better somewhere'


def test_holm_adjust_on_a_single_value_is_the_identity():
    np.testing.assert_allclose(holm_adjust([0.031]), [0.031])


# ---------------------------------------------------------------------------- permutation_pvalue
def test_permutation_pvalue_is_never_zero():
    """The add-one correction. A p of exactly 0 would claim more resolution than the null sample has."""
    null = np.zeros(99)
    p = permutation_pvalue(1e6, null)
    assert p > 0
    assert p == pytest.approx(1 / 100)


def test_permutation_pvalue_counts_ties_as_extreme():
    null = np.array([1.0, 2.0, 3.0, 3.0])
    assert permutation_pvalue(3.0, null) == pytest.approx(3 / 5)


def test_permutation_pvalue_sides_are_complementary_in_the_right_direction():
    null = np.linspace(-3, 3, 199)
    high, low = 2.5, -2.5
    assert permutation_pvalue(high, null, side='greater') < 0.1
    assert permutation_pvalue(high, null, side='less') > 0.9
    assert permutation_pvalue(low, null, side='less') < 0.1
    assert permutation_pvalue(low, null, side='greater') > 0.9


def test_the_default_side_is_greater():
    """Pinned separately: a flipped default would silently invert every existing caller."""
    null = np.linspace(-3, 3, 199)
    assert permutation_pvalue(2.5, null) == permutation_pvalue(2.5, null, side='greater')
    assert permutation_pvalue(2.5, null) != permutation_pvalue(2.5, null, side='less')


def test_permutation_pvalue_rejects_an_unknown_side():
    with pytest.raises(ValueError, match='greater'):
        permutation_pvalue(1.0, np.zeros(10), side='two-sided')


# --------------------------------------------------------------------------------- eta_squared
def test_eta_squared_is_one_for_perfectly_separated_groups():
    z = np.array([1.0, 1.0, 1.0, 5.0, 5.0, 5.0])
    labels = np.array([0, 0, 0, 1, 1, 1])
    assert eta_squared(z, labels) == pytest.approx(1.0)


def test_eta_squared_is_near_zero_when_groups_carry_nothing():
    gen = np.random.default_rng(0)
    z = gen.normal(size=400)
    labels = gen.integers(0, 4, 400)
    assert eta_squared(z, labels) < 0.05


def test_eta_squared_is_nan_when_there_is_no_variance_to_explain():
    assert np.isnan(eta_squared(np.full(10, 3.0), np.arange(10)))


def test_eta_squared_rises_mechanically_with_more_groups():
    """The caveat the docstring warns about, pinned so nobody treats the raw value as evidence.

    With one group per point, eta-squared is 1 regardless of whether the grouping means anything.
    """
    gen = np.random.default_rng(1)
    z = gen.normal(size=60)
    few = eta_squared(z, gen.integers(0, 2, 60))
    every_point_its_own = eta_squared(z, np.arange(60))
    assert every_point_its_own == pytest.approx(1.0)
    assert few < every_point_its_own


def test_eta_squared_accepts_non_contiguous_labels():
    z = np.array([1.0, 1.0, 5.0, 5.0])
    assert eta_squared(z, np.array([7, 7, 99, 99])) == pytest.approx(1.0)
    assert eta_squared(z, np.array(['a', 'a', 'b', 'b'])) == pytest.approx(1.0)
