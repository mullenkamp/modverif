# Rolling Windows

Rolling-window accumulation maxima over time series and gridded fields — the basis of an
n-hour-maximum verification (24 h maxima being the common case).

!!! warning "Three missing-value conventions, deliberately not unified"

    These functions differ in how a window containing a gap is treated, and the differences are
    load-bearing rather than incidental. Picking the wrong one silently changes which window wins.

    | function | input | a window containing a gap... |
    |---|---|---|
    | [`rolling_window_max`](#modverif.window.rolling_window_max) | gridded `(nt, ny, nx)` | ...counts, with gaps read as zero |
    | [`rolling_max_valid`](#modverif.window.rolling_max_valid) | 1-D series | ...is disqualified outright |
    | [`rolling_window_sums`](#modverif.window.rolling_window_sums) | 1-D series | ...is summed **and** flagged; the caller decides |

    Use the grid convention for gap-free model output, where a NaN means "nothing here" rather than
    "unknown". Use the series convention for observations, where a gauge window with a missing step
    must not compete against complete ones on an artificially low total. Use the third when a model
    series and an observation series must be reduced by the *same* code, so their window start times
    cannot drift apart numerically.

!!! danger "Tie-breaking is load-bearing"

    Zero-padded windows sum to bitwise-identical totals surprisingly often — for a 6-step burst
    inside a 72-step record, 19 of 49 24-step windows tie exactly. `rolling_window_max` resolves a
    tie via `argmax`, `max_window` via a first-candidate scan within a small absolute tolerance;
    **they agree only because both scan first-to-last.** Reversing either scan direction shifts
    reported window start times by many steps.

## Example

```python
import numpy as np
from modverif.window import rolling_window_sums, max_window, rolling_max_valid

hourly = np.array([...])          # one gauge's hourly totals, NaN where unobserved

# Lenient: gaps count as zero, every window competes.
sums, valid = rolling_window_sums(hourly, 24)
depth, start, spread, clamped = max_window(sums, valid, tol=0.02)

# Strict: any window containing a gap is disqualified.
depth_strict, n_observed = rolling_max_valid(hourly, 24)
```

`spread` is the ambiguity diagnostic: over a long event the maximum is often nearly flat, so the
reported start can move by many steps at almost no cost in depth. `clamped` flags a maximum pinned
against the edge of the record, where the true maximum may lie outside the observed period.

## API

::: modverif.window
    options:
      show_root_heading: false
      show_source: false
