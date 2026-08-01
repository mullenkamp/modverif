# Changelog

Notable changes to `modverif`, newest first. Versions follow [semantic versioning](https://semver.org):
a **breaking** change to anything importable bumps the minor while below 1.0.

Releases are published manually, so an entry stays marked `unreleased` until the version is actually
on PyPI — a changelog that claims a release which is not installable is worse than none.

Entries record *why* a change was made where the reason is not obvious from the name. Anything that
can change a number a user has already published is called out explicitly.

## [0.4.0] — 2026-08-01

Five new modules of spatial-verification methods, graduated from a private analysis codebase where
they had been in production use on real storm assessments. **Purely additive**: nothing that shipped
in 0.3.0 changed behaviour, so no existing code needs updating.

### Added

- **`modverif.window`** — rolling-window accumulation maxima over series and gridded fields, the
  basis of *n*-hour-maximum precipitation verification. Three functions rather than one with a
  `nan_policy` flag, because the three missing-value conventions are genuinely not interchangeable:
  gaps-as-zero for gap-free model output, gaps-disqualify-the-window for observations, and a
  both-conventions primitive so a model series and an observation series can be reduced by the *same*
  code.
- **`modverif.match`** — point-to-grid neighbourhood matching, and the selection-bias null that keeps
  it honest. A neighbourhood search improves the apparent fit *even for a model with no skill*, so
  `null_improvement` re-runs the same search at deliberately wrong locations to price that. Also
  `vector_coherence` and `field_shift_objective`, the other two guards against the same bias.
- **`modverif.spatial`** — empirical variogram with a nugget-fixed exponential fit, bootstrap
  uncertainty on the fitted range, Moran's I with permutation inference, and *k*-means regionalisation.
- **`modverif.stats`** — permutation p-values (one-sided, with the add-one correction), Holm–Bonferroni
  correction, and eta-squared. Deliberately separate from `spatial`: none of it is spatial, and the
  next code to need a family-wise correction is as likely to be a teleconnection analysis.
- **`modverif.crossval`** — out-of-sample assessment of candidate predictors: leave-one-out,
  leave-one-group-out, and a hold-out-high **extrapolation** test for the case where the correction is
  wanted beyond the range the observations cover.
- **`metrics.compute_residual_skill_score`** — skill from two sets of *residuals* rather than
  model/obs pairs, for baselines with no per-point prediction to subtract. An **RMSE** ratio, not the
  more common MSE-ratio skill score.
- **`metrics.compute_xcorr_best_lag`** — best-correlating lag with the finite mask recomputed **at each
  lag**. See the note below.
- Reference documentation for each new module.

### Changed

- `metrics.compute_lagged_correlation` — **documentation only, no behaviour change.** Its docstring now
  warns that non-finite pairs are dropped *before* lagging, which compacts the series: on a gappy
  record the returned lag is counted in surviving samples rather than in time.

### Notes for users

**Two cross-correlation functions now coexist, with opposite sign conventions.** This is deliberate —
they were written for different questions and unifying them would change results that have already
been published.

| | positive lag means | gaps |
|---|---|---|
| `compute_lagged_correlation` | the **model leads** | dropped before lagging (lag is in surviving samples) |
| `compute_xcorr_best_lag` | the **model is later** | re-masked per lag (lag stays in real time) |

Use `compute_xcorr_best_lag` for displaced-timing work on gappy observations. Both take
`(model, obs)`, so habit gives the right answer — **swapping the two series returns the negated lag
with a bitwise-identical correlation, and no output reveals the mistake.**

**Coordinates must be projected.** Everything in `match`, `spatial` and `crossval` works in a
projected, metric CRS: radii and lags are in the axis's own units, and search boxes are built by
`searchsorted` per axis. Passing degrees produces boxes whose east–west extent varies with latitude.

## [0.3.0] — 2026-07-31

### Added

- `cyclone.read_latlon_2d` — latitude/longitude for three dataset layouts, including **projected
  grids** (`y`/`x` plus a CRS), which previously could not be tracked at all.
- `cyclone.match_cyclone_positions`, `cyclone.compare_cyclone_tracks`, `cyclone.plot_cyclone_comparison`
  — a track-comparison layer with neutral A/B metric names, so it is not tied to one model pairing.
- `cyclone.haversine_distance` and `composite.pyproj_to_cartopy` promoted from private helpers to
  public API; downstream code had been importing the private forms.
- `track_cyclone` accepts `start_time` / `end_time`, so a track can be confined to an event window.
  Deliberately **not** plumbed through `track_cyclone_multi_file`, whose index bookkeeping assumes one
  position per timestep.

### Fixed

- `plot_cyclone_timestep` now raises `NotImplementedError` on projected grids rather than emitting a
  wrong map: the longitude wrap produced contour artefacts and `set_extent` clipped the eastern edge
  of a dateline-crossing domain.
- Two `zip()` calls in `plots.py` silently truncated when given unequal-length inputs.

### Changed

- CI gained a lint gate alongside the 3.10/3.11/3.12 test matrix.

---

Releases before 0.3.0 predate this changelog; see the git history.
