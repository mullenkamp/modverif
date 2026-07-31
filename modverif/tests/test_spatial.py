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
  name, including the awkward ``'rng'`` (which means *range*).
"""
import numpy as np
import pytest

from modverif.spatial import (
    MAX_LAG_PCT,
    MIN_PAIRS_PER_BIN,
    NEAR_ORIGIN_KM,
    bootstrap_variogram_params,
    ch_gamma,
    empirical_variogram,
    exponential_variogram,
    fit_bias_variogram,
    fit_exponential_variogram,
    matheron_gamma,
)


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
    assert 20.0 < out['rng'] < 180.0, f'range {out["rng"]:.1f} km is not in the plausible band'
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
                        'n_near', 'fit_ok', 'nugget', 'psill', 'sill', 'rng', 'rel_nugget'}
    assert out['n_near'] > 0, 'no near-origin pairs, so the nugget estimate is unanchored'


def test_fit_bias_variogram_reports_nan_parameters_when_the_fit_fails():
    """fit_ok must not be the only honest signal -- the parameters have to be NaN too."""
    gen = np.random.default_rng(7)
    x, y = gen.uniform(0, 5, 40), gen.uniform(0, 5, 40)
    out = fit_bias_variogram(x, y, np.zeros(40))     # zero variance -> no fit possible
    assert out['fit_ok'] is False
    for k in ('nugget', 'psill', 'sill', 'rng', 'rel_nugget'):
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
    for key in ('rng', 'nugget'):
        lo, mid, hi = out[key]
        assert lo <= mid <= hi, f'{key} band is not ordered'
    assert out['n_ok'] >= 30
    assert out['drop_frac'] == pytest.approx(0.1)


def _pairs(x, y, z):
    """Condensed pairwise distances and absolute differences, as fit_bias_variogram builds them."""
    from scipy.spatial.distance import pdist
    return pdist(np.column_stack([x, y])), pdist(np.asarray(z, float)[:, None])
