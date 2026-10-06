# Water-Balance Checks

Water years, coverage, annual volumes and the step tests behind thalweg's step-0 water-balance gate.

!!! warning "Coverage is measured against the calendar"

    An hour absent from the input is as missing as one that is NaN. Counting only the hours present would let
    a record with whole months absent report full coverage.

## API

::: modverif.flow
    options:
      show_root_heading: false
      show_source: false
