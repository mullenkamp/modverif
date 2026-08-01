"""
Tests for modverif.spatial.

Three of these defend design decisions rather than arithmetic, because those are what a later
"cleanup" silently reverses:

* ``test_the_nugget_is_fixed_not_fitted`` -- the fit deliberately holds the nugget at a data-derived
  value. A free three-parameter fit collapses it toward zero, which reads as a smoothly interpolable
  bias field when the data say otherwise.
* ``test_bootstrap_starts_a_fresh_stream_on_every_call`` -- the function takes a *seed* and builds its
  own generator, and callers rely on repeated calls being independently reproducible. Threading one
  shared generator through instead is the obvious tidy-up and it silently changes every result after
  the first.
* ``test_fit_bias_variogram_returns_the_full_key_contract`` -- downstream code indexes these keys by
  name, including the awkward ``'range_km'`` (which means *range*).
"""
import numpy as np
import pytest

from modverif.spatial import (
    MAX_LAG_PCT,
    MIN_PAIRS_PER_BIN,
    NEAR_ORIGIN_KM,
    best_kmeans,
    bootstrap_variogram_params,
    ch_gamma,
    empirical_variogram,
    exponential_variogram,
    fit_bias_variogram,
    fit_exponential_variogram,
    matheron_gamma,
    morans_i,
    morans_i_at_points,
    morans_i_permutation,
)
from modverif.stats import permutation_pvalue


def correlated_field(n=180, rng_km=60.0, psill=1.0, nugget=0.05, seed=0):
    """Scattered points carrying a known exponential covariance structure."""
    gen = np.random.default_rng(seed)
    x, y = gen.uniform(0, 400, n), gen.uniform(0, 400, n)
    d = np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :])
    cov = psill * np.exp(-d / rng_km) + nugget * np.eye(n)
    z = np.linalg.cholesky(cov + 1e-9 * np.eye(n)) @ gen.normal(size=n)
    return x, y, z


# ------------------------------------------------------------------------------ the estimators
def test_matheron_gamma_is_half_the_mean_squared_difference():
    d = np.array([1.0, 2.0, 3.0])
    assert matheron_gamma(d) == pytest.approx(0.5 * np.mean(d ** 2))


def test_estimators_return_nan_on_no_pairs():
    empty = np.array([])
    assert np.isnan(ch_gamma(empty))
    assert np.isnan(matheron_gamma(empty))


def test_ch_gamma_is_more_robust_to_an_outlying_pair():
    """The reason the Cressie-Hawkins estimator is the default."""
    base = np.full(200, 1.0)
    spiked = base.copy()
    spiked[0] = 50.0
    ch_shift = ch_gamma(spiked) / ch_gamma(base)
    mat_shift = matheron_gamma(spiked) / matheron_gamma(base)
    assert ch_shift < mat_shift, 'CH should absorb an outlier better than Matheron'


# ------------------------------------------------------------------------- empirical variogram
def test_the_tuning_constants_have_their_documented_values():
    """Pinned as LITERALS on purpose.

    Every other test here must also avoid asserting against the imported constant, or it silently
    re-derives its own expectation when the constant changes and defends nothing. Mutation testing
    caught exactly that: three constants could be changed with the whole suite still green.
    """
    assert MIN_PAIRS_PER_BIN == 30
    assert MAX_LAG_PCT == 60
    assert NEAR_ORIGIN_KM == 10.0


def test_empirical_variogram_drops_sparse_bins():
    gen = np.random.default_rng(1)
    d = gen.uniform(0, 100, 5000)
    adz = gen.gamma(2.0, 1.0, 5000)
    centers, gammas, counts, max_lag = empirical_variogram(d, adz)
    assert len(centers) == len(gammas) == len(counts)
    assert (counts >= 30).all(), 'a bin below the 30-pair floor survived'
    assert max_lag == pytest.approx(float(np.percentile(d, 60))), 'lag cap is not the 60th percentile'
    assert (centers <= max_lag).all()


def test_empirical_variogram_uses_the_estimator_it_was_asked_for():
    """Asserts WHICH estimator ran, not merely that the two differ.

    A 'they differ' assertion stays true if the selection is inverted -- which mutation testing
    confirmed, so this checks the returned value against the estimator applied by hand.
    """
    gen = np.random.default_rng(2)
    d = gen.uniform(0, 100, 4000)
    adz = gen.gamma(0.5, 3.0, 4000)          # heavy-tailed, so the estimators genuinely disagree
    centers, g_ch, _, max_lag = empirical_variogram(d, adz, 'ch')
    _, g_mat, _, _ = empirical_variogram(d, adz, 'matheron')
    assert not np.allclose(g_ch, g_mat), 'the estimator argument had no effect at all'

    # Rebuild the first bin's membership exactly as the function does, then check the reported
    # semivariance against each estimator applied by hand.
    width = centers[1] - centers[0]
    lo, hi = centers[0] - width / 2, centers[0] + width / 2
    sel = (d >= lo) & (d < hi)
    assert g_ch[0] == pytest.approx(ch_gamma(adz[sel])), "'ch' did not run the Cressie-Hawkins estimator"
    assert g_mat[0] == pytest.approx(matheron_gamma(adz[sel])), "'matheron' did not run Matheron"


# -------------------------------------------------------------------------------- the model
def test_exponential_variogram_has_the_expected_shape():
    nugget, psill, range_km = 0.2, 1.0, 50.0
    assert exponential_variogram(np.array([0.0]), nugget, psill, range_km)[0] == pytest.approx(nugget)
    far = exponential_variogram(np.array([1e6]), nugget, psill, range_km)[0]
    assert far == pytest.approx(nugget + psill)
    at_range = exponential_variogram(np.array([range_km]), nugget, psill, range_km)[0]
    assert at_range == pytest.approx(nugget + psill * (1 - np.exp(-1)))


# ---------------------------------------------------------------------------------- the fit
def test_fit_recovers_a_known_range():
    x, y, z = correlated_field(rng_km=60.0, seed=3)
    out = fit_bias_variogram(x, y, z)
    assert out['fit_ok']
    assert 20.0 < out['range_km'] < 180.0, f'range {out["range_km"]:.1f} km is not in the plausible band'
    assert out['sill'] == pytest.approx(out['nugget'] + out['psill'])


def test_the_nugget_is_fixed_not_fitted():
    """The fit must return the nugget it was handed, not one it chose.

    A free fit collapses the nugget toward zero on sparse networks, which overstates how
    interpolable the field is. Holding it is the whole point of the function's design.
    """
    x, y, z = correlated_field(seed=4)
    centers, gammas, counts, max_lag = empirical_variogram(
        *_pairs(x, y, z), 'ch')
    var_z = float(np.var(z))
    for asked in (0.05, 0.30):
        fit = fit_exponential_variogram(centers, gammas, counts, var_z, max_lag, 1.0, asked)
        assert fit is not None
        assert fit['nugget'] == pytest.approx(asked), 'the nugget was refitted rather than held'


def test_the_fixed_nugget_is_clipped_into_range():
    x, y, z = correlated_field(seed=5)
    centers, gammas, counts, max_lag = empirical_variogram(*_pairs(x, y, z), 'ch')
    var_z = float(np.var(z))
    assert fit_exponential_variogram(centers, gammas, counts, var_z, max_lag, 1.0, -5.0)['nugget'] == 0.0
    assert fit_exponential_variogram(
        centers, gammas, counts, var_z, max_lag, 1.0, 10 * var_z)['nugget'] == pytest.approx(var_z)


def test_fit_declines_rather_than_guesses_on_degenerate_input():
    assert fit_exponential_variogram(np.array([1.0]), np.array([1.0]), np.array([50]),
                                     1.0, 10.0, 1.0, 0.1) is None
    assert fit_exponential_variogram(np.array([1.0, 2.0]), np.array([1.0, 1.0]), np.array([50, 50]),
                                     0.0, 10.0, 1.0, 0.1) is None


def test_fit_bias_variogram_returns_the_full_key_contract():
    """Downstream code indexes these by name; dropping or renaming one is a breaking change."""
    x, y, z = correlated_field(seed=6)
    out = fit_bias_variogram(x, y, z)
    assert set(out) == {'centers', 'gammas', 'counts', 'max_lag', 'var_z', 'ch_n', 'mat_n',
                        'n_near', 'fit_ok', 'nugget', 'psill', 'sill', 'range_km', 'rel_nugget'}
    assert out['n_near'] > 0, 'no near-origin pairs, so the nugget estimate is unanchored'


def test_fit_bias_variogram_reports_nan_parameters_when_the_fit_fails():
    """fit_ok must not be the only honest signal -- the parameters have to be NaN too."""
    gen = np.random.default_rng(7)
    x, y = gen.uniform(0, 5, 40), gen.uniform(0, 5, 40)
    out = fit_bias_variogram(x, y, np.zeros(40))     # zero variance -> no fit possible
    assert out['fit_ok'] is False
    for k in ('nugget', 'psill', 'sill', 'range_km', 'rel_nugget'):
        assert np.isnan(out[k]), f'{k} should be NaN when the fit failed'


def test_near_origin_window_controls_the_nugget_estimate():
    """Literal windows, not multiples of the imported constant.

    Passing ``NEAR_ORIGIN_KM`` and ``4 * NEAR_ORIGIN_KM`` scales both sides together, so the test
    survives any change to the constant -- mutation testing confirmed it did.
    """
    x, y, z = correlated_field(seed=8)
    tight = fit_bias_variogram(x, y, z, near_origin_km=10.0)
    wide = fit_bias_variogram(x, y, z, near_origin_km=40.0)
    assert tight['n_near'] < wide['n_near']
    assert tight['ch_n'] != wide['ch_n'], 'the near-origin window had no effect on the nugget'


def test_the_default_near_origin_window_is_the_module_constant():
    """Separately pins that the DEFAULT is 10 km, which the test above deliberately no longer does."""
    x, y, z = correlated_field(seed=8)
    assert fit_bias_variogram(x, y, z)['n_near'] == fit_bias_variogram(x, y, z, near_origin_km=10.0)['n_near']
    assert fit_bias_variogram(x, y, z)['n_near'] != fit_bias_variogram(x, y, z, near_origin_km=40.0)['n_near']


# ------------------------------------------------------------------------------- the bootstrap
def test_bootstrap_starts_a_fresh_stream_on_every_call():
    """Two identical calls must give identical answers.

    The function takes a seed and builds its own generator precisely so that a caller running it
    once per field gets reproducible, independent bands. If it were refactored to accept a shared
    generator, the second call would consume a different part of the stream and this fails -- which
    is the entire point of the test.
    """
    x, y, z = correlated_field(seed=9)
    first = bootstrap_variogram_params(x, y, z, n_boot=40)
    second = bootstrap_variogram_params(x, y, z, n_boot=40)
    assert first is not None
    assert first == second, 'repeated calls diverged, so the generator is no longer per-call'


def test_bootstrap_seed_actually_changes_the_band():
    x, y, z = correlated_field(seed=10)
    a = bootstrap_variogram_params(x, y, z, n_boot=40, seed=0)
    b = bootstrap_variogram_params(x, y, z, n_boot=40, seed=1)
    assert a is not None and b is not None
    assert a != b, 'the seed argument had no effect'


def test_bootstrap_declines_when_too_few_refits_converge():
    gen = np.random.default_rng(11)
    x, y = gen.uniform(0, 3, 30), gen.uniform(0, 3, 30)
    assert bootstrap_variogram_params(x, y, np.zeros(30), n_boot=20) is None


def test_bootstrap_band_is_ordered_and_reports_its_sample_size():
    x, y, z = correlated_field(seed=12)
    out = bootstrap_variogram_params(x, y, z, n_boot=60)
    assert out is not None
    for key in ('range_km', 'nugget'):
        lo, mid, hi = out[key]
        assert lo <= mid <= hi, f'{key} band is not ordered'
    assert out['n_ok'] >= 30
    assert out['drop_frac'] == pytest.approx(0.1)


def _pairs(x, y, z):
    """Condensed pairwise distances and absolute differences, as fit_bias_variogram builds them."""
    from scipy.spatial.distance import pdist
    return pdist(np.column_stack([x, y])), pdist(np.asarray(z, float)[:, None])


# =============================================================== spatial autocorrelation + regions
class _PermutationCounter:
    """Delegates to a real Generator while counting permutation() calls."""

    def __init__(self, seed):
        self._g = np.random.default_rng(seed)
        self.n_permutation = 0
        self.n_integers = 0

    def permutation(self, *a, **kw):
        self.n_permutation += 1
        return self._g.permutation(*a, **kw)

    def integers(self, *a, **kw):
        self.n_integers += 1
        return self._g.integers(*a, **kw)


def _weights_from_coords(xy):
    d = np.hypot(xy[:, 0][:, None] - xy[:, 0][None, :], xy[:, 1][:, None] - xy[:, 1][None, :])
    w = np.where(d > 0, 1.0 / np.where(d > 0, d, 1.0), 0.0)
    np.fill_diagonal(w, 0.0)
    return w, float(w.sum())


def test_morans_i_is_positive_for_a_clustered_field_and_negative_for_a_checkerboard():
    side = 8
    yy, xx = np.mgrid[0:side, 0:side]
    xy = np.column_stack([xx.ravel().astype(float), yy.ravel().astype(float)])
    w, s0 = _weights_from_coords(xy)
    n = len(xy)

    checker = ((xx + yy) % 2).ravel().astype(float)
    zc = checker - checker.mean()
    assert morans_i(zc, w, s0, n, float(zc @ zc)) < 0, 'a checkerboard is anti-correlated'

    clustered = (xx < side / 2).ravel().astype(float)
    zc2 = clustered - clustered.mean()
    assert morans_i(zc2, w, s0, n, float(zc2 @ zc2)) > 0, 'a split field is positively correlated'


def test_morans_i_permutation_draws_once_per_permutation_not_once_per_weight_matrix():
    """The load-bearing property: ONE shuffle is evaluated against EVERY weight matrix.

    Testing each matrix with its own permutations would draw ``n_perm x n_matrices`` times and
    produce independent nulls -- a different statistical question, and a different answer from any
    multiple-comparison correction applied on top.
    """
    gen = np.random.default_rng(3)
    xy = gen.uniform(0, 100, (40, 2))
    z = gen.normal(size=40)
    zc = z - z.mean()
    n, denom = 40, float(zc @ zc)
    w, s0 = _weights_from_coords(xy)
    mats = [(w, s0), (np.ones((n, n)) - np.eye(n), float(n * (n - 1))), (w, s0)]

    spy = _PermutationCounter(5)
    observed, null = morans_i_permutation(zc, mats, n, denom, spy, n_perm=50)

    assert spy.n_permutation == 50, (
        f'expected one shuffle per permutation, got {spy.n_permutation} for {len(mats)} matrices')
    assert null.shape == (50, 3)
    np.testing.assert_array_equal(null[:, 0], null[:, 2],
                                  err_msg='identical weight matrices must see the SAME shuffle')
    assert observed[0] == observed[2]


def test_shared_shuffle_couples_the_nulls_where_independent_shuffles_would_not():
    """The statistical consequence of sharing the shuffle, stated as a comparison rather than a
    threshold.

    Bands of a correlogram are one field viewed at several scales, not independent tests. Sharing the
    shuffle preserves that coupling in the null; drawing per-band permutations destroys it, and a
    family-wise correction applied to independently-generated nulls answers a different question.

    Asserting "shared is more coupled than independent" is the actual claim. An absolute correlation
    threshold would just be a number tuned until the test passed.
    """
    gen = np.random.default_rng(4)
    xy = gen.uniform(0, 100, (50, 2))
    z = gen.normal(size=50)
    zc = z - z.mean()
    n, denom, n_perm = 50, float(zc @ zc), 400
    w, s0 = _weights_from_coords(xy)
    near = (np.hypot(xy[:, 0][:, None] - xy[:, 0][None, :],
                     xy[:, 1][:, None] - xy[:, 1][None, :]) < 30).astype(float)
    np.fill_diagonal(near, 0.0)
    mats = [(w, s0), (near, float(near.sum()))]

    _, shared = morans_i_permutation(zc, mats, n, denom, np.random.default_rng(6), n_perm=n_perm)
    r_shared = abs(np.corrcoef(shared[:, 0], shared[:, 1])[0, 1])

    # The same two matrices, but each with its own independent permutations.
    g = np.random.default_rng(6)
    indep = np.column_stack([morans_i_permutation(zc, [m], n, denom, g, n_perm=n_perm)[1][:, 0]
                             for m in mats])
    r_indep = abs(np.corrcoef(indep[:, 0], indep[:, 1])[0, 1])

    assert r_shared > r_indep, (
        f'sharing the shuffle should couple the nulls: shared r={r_shared:.3f} vs '
        f'independent r={r_indep:.3f}')


def test_best_kmeans_declines_when_k_cannot_be_supported():
    """Returns None rather than an array -- callers must handle it, so it is pinned."""
    xy = np.zeros((5, 2))                     # every point identical
    assert best_kmeans(xy, 4, np.random.default_rng(0)) is None


def test_best_kmeans_recovers_obvious_clusters():
    gen = np.random.default_rng(7)
    xy = np.vstack([gen.normal((0, 0), 0.4, (40, 2)), gen.normal((30, 30), 0.4, (40, 2))])
    lab = best_kmeans(xy, 2, np.random.default_rng(1))
    assert lab is not None
    assert len(np.unique(lab)) == 2
    assert len(np.unique(lab[:40])) == 1 and len(np.unique(lab[40:])) == 1


def test_best_kmeans_draws_its_restart_seeds_in_one_vectorised_call():
    """Per-restart scalar draws would consume the caller's generator differently."""
    gen = np.random.default_rng(8)
    xy = gen.uniform(0, 50, (40, 2))
    spy = _PermutationCounter(2)
    best_kmeans(xy, 3, spy, n_restart=20)
    assert spy.n_integers == 1, f'expected a single vectorised integers() call, got {spy.n_integers}'


def test_morans_i_at_points_agrees_with_the_low_level_pair():
    """The convenience wrapper must be a wrapper, not a second implementation.

    It exists because the raw-ingredients API was easier to re-implement than to call -- so the one
    thing it must never become is a third copy that drifts.
    """
    gen = np.random.default_rng(20)
    x, y = gen.uniform(0, 200, 45), gen.uniform(0, 200, 45)
    v = gen.normal(size=45)

    got_i, got_p = morans_i_at_points(v, x, y, np.random.default_rng(3), n_perm=199)

    d = np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :])
    with np.errstate(divide='ignore'):
        w = np.where(d > 0, 1.0 / d, 0.0)
    s0, zc = float(w.sum()), v - v.mean()
    (exp_i,), null = morans_i_permutation(zc, [(w, s0)], 45, float(zc @ zc),
                                          np.random.default_rng(3), 199)
    assert got_i == exp_i
    assert got_p == permutation_pvalue(exp_i, null[:, 0])


def test_morans_i_at_points_declines_on_too_few_points():
    gen = np.random.default_rng(21)
    assert all(np.isnan(v) for v in
               morans_i_at_points(gen.normal(size=5), gen.uniform(0, 10, 5),
                                  gen.uniform(0, 10, 5), np.random.default_rng(0)))


def test_morans_i_at_points_detects_a_clustered_field():
    gen = np.random.default_rng(22)
    x = np.concatenate([gen.uniform(0, 30, 30), gen.uniform(170, 200, 30)])
    y = gen.uniform(0, 200, 60)
    v = np.concatenate([np.full(30, 1.0), np.full(30, -1.0)]) + gen.normal(0, 0.1, 60)
    i, p = morans_i_at_points(v, x, y, np.random.default_rng(4), n_perm=499)
    assert i > 0 and p < 0.05, 'two separated blocks of opposite sign should read as clustered'


def test_morans_i_at_points_declines_on_a_constant_field():
    """Zero variance means Moran's I is 0/0. The guard must return NaN rather than divide."""
    gen = np.random.default_rng(23)
    x, y = gen.uniform(0, 100, 30), gen.uniform(0, 100, 30)
    assert all(np.isnan(v) for v in
               morans_i_at_points(np.full(30, 4.0), x, y, np.random.default_rng(0), n_perm=99))
