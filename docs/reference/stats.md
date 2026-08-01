# Resampling Inference

Permutation tests and multiple-comparison corrections. Deliberately **not** in
[`modverif.spatial`](spatial.md) — nothing here is spatial, and the next code to need a
family-wise correction is as likely to be a teleconnection analysis as a variogram.

!!! warning "Permutation tests here are one-sided, and the side is yours to choose"

    The default, `side='greater'`, asks *"is the observed statistic unusually **high**?"* — which
    suits Moran's I and variance-explained. A test against a **negative** prior — *"is this
    correlation unusually low?"* — needs `side='less'`.

    Calling the default on a lower-sided test reports very nearly its complement: a wrong answer that
    looks entirely reasonable, and one no amount of correction downstream will catch.

    There is no two-sided option. Every test this serves has a directional prior, and choosing a side
    after seeing the data is precisely what a p-value cannot survive.

## Why the add-one correction

[`permutation_pvalue`](#modverif.stats.permutation_pvalue) counts the observed statistic as one of
its own null draws — `(extreme + 1) / (n_null + 1)`. Without it, a p-value of exactly zero is
reportable, and that is never true: it only means the null sample was too small to resolve the tail.
With 999 permutations the smallest honest p-value is 0.001.

## Holm, and when not to use it

[`holm_adjust`](#modverif.stats.holm_adjust) controls the **family-wise error rate** — the chance of
*any* false positive across the whole family. It is uniformly more powerful than plain Bonferroni at
the same guarantee, so there is no reason to prefer Bonferroni.

Use it when a single false positive would change the conclusion — scanning distance bands for spatial
structure, say, where one spurious band is enough to claim structure that is not there. When the
question is instead *"what fraction of my many discoveries are false?"*, a false-discovery-rate
procedure is the right tool and Holm is needlessly conservative.

!!! note "Eta-squared is an effect size, not evidence"

    [`eta_squared`](#modverif.stats.eta_squared) rises **mechanically** with the number of groups —
    one group per point explains any field perfectly. Always pair it with a permutation test
    (shuffle values across fixed labels) or with out-of-sample skill before concluding a grouping is
    real.

## API

::: modverif.stats
    options:
      show_root_heading: false
      show_source: false
