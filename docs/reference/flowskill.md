# Flow Skill Scores

Skill scores for simulated against observed river flow, and the benchmarks they are compared with. Built for
thalweg's stage-comparison gate and calibration objective, but model-agnostic: the inputs are plain arrays of
the hours to be scored.

!!! warning "Select the scored hours first"

    Every function takes 1-D arrays of the hours that count, with no NaN. Choosing those hours (valid flow
    codes, coverage, warm-up) is the caller's job and is never done silently here. NaN, a negative flow, a
    length mismatch or a constant observed series raises.

## Which score for which job

- **`kge_prime`** on raw flow weights floods. On √Q it balances the range better, which is why thalweg
  calibrates on it.
- **`low_flow_score`** combines r and α on log(Q + ε) with **β_low**. r and α are nearly blind to a uniform
  proportional bias in low flows, and β_low sees it.
- **ε** comes from `low_flow_epsilon` on the *calibration* half and is then held fixed. Recomputing it on
  each resample or half changes the transform under the score.
- **`doy_climatology`** is the benchmark a model should beat: the typical flow for the date, smoothed over a
  31-day window.

## API

::: modverif.flowskill
    options:
      show_root_heading: false
      show_source: false
