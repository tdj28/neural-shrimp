# Broader-data tiny-LM design review

Coordinator review, 8 October 2026. Reviewed design commit
`2a3f074a31bcdebe234d0bf5a09364773fed1621` and
`docs/plans/explore-tiny-lm-broader-data.md` before implementation or target
outcomes. **Pass for implementation with the clarifications below.** This
review does not authorize a target run before source checks and the committed
profile card; it makes no scientific-result claim.

The fixed nested 8/256-window comparison changes training coverage while
holding initialization, exact step size, architecture, optimizer, precision
and update count fixed. It does not match runtime or token cost; the separately
specified 25-versus-800-step comparison correctly matches 204,800 training
targets. Initialization replication uses two seeds on one fixed text pool,
so neither uncertainty nor conclusions may imply replicated data populations.

Independently checked the stated selected-window endpoints and disjointness
against previous starts, assessment/validation windows and final-ten-percent
boundary. The 3,204 float32 states contain 202,185,216 raw bytes. Source
identity, ASCII offset equivalence and integer token mappings still require
executable checks. The profile plus collection/analysis ceilings including
cleanup fit the declared aggregate limit. The throughput extrapolation is a
prospective conservative feasibility rule, not a measured runtime yet.

Continuous 16-update progress is an explicitly disclosed change of endpoint
after the sparse negative calibration. The fitting/tuning/late intervals have
the required label embargo, train-only transformations and per-arm scaling.
Ordinary-comparator and penalty choices use tuning only, while the separate
probe cannot select a setting or checkpoint. Overlapping horizons do not
supply independent replicates. B3 remains a nonlinear summary of information
already supplied to B2; any benefit must be assessed against both B2 and the
selected stronger ordinary comparator, with all comparators reported.

Required source/card clarifications sent to the producer before coding:

- State explicitly that the text pool is fixed and the two seeds replicate
  initialization only.
- Prespecify rank-deficient AR fitting and failed/nonfinite recursive forecasts.
  Retain failures; do not silently omit a comparator or eligible row. Any such
  baseline failure blocks the positive forecasting interpretation.

Preserve the failed prior unit and its gate. No old assessment, protected text,
controller, rate search, inference experiment or paid compute is part of this
design. Source review should inspect these boundaries and the interruption
retention path before the engineering profile.
