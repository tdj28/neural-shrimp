# EXP007 independent runtime audit

Date: 2026-09-11. Status: **no unresolved blocker for the prospective freeze**.
This is a local source and synthetic-test audit, not a result, a target run, a
paid review, or a guarantee about numerical outcomes. The reviewer did not
change the runner or prepare code and did not integrate a neural trajectory.
Line references below refer to the reviewed prefreeze versions.

## Scope and resolved findings

Reviewed `tools/run_hopfield_holdout.py`,
`tools/prepare_hopfield_holdout.py`, the candidate/scoring helper, and the
inherited trace, sensitivity, waveform-metric and baseline APIs. Anchor
selection has a separate independent design audit.

One failure-schema issue was reported and fixed before freeze: a nominally
complete preview with a missing/non-real score could make `np.isfinite` raise
instead of invalidating the required policy subset. The runner now uses
`valid_score` (runner lines 69–80), requiring a finite real value in [0,2] and
excluding booleans. Eight malformed-score cases pass. The physical bound
follows from the componentwise tanh activation range; this is validation, not
a change to the endpoint or a replacement rule.

Two other corrections were present in the final reviewed source: the zero-dose
floor is undefined if any required zero trial fails, and the preview census
explicitly accounts for anchors whose baselines are not evaluable. Neither
report silently computes a success-only aggregate.

## Numerical and selection checks

- **Census:** runner lines 31–53 fix 12 anchors, four phase starts, four nonzero
  doses, seven arms, and two integration steps. The complete target census is
  `12*4*(4*7+1)*2 = 2784`, including 96 zero-dose trials. The preview census is
  `12*4*4*21 = 4032`; ineligible anchors retain their planned status count.
- **Candidate universe and costs:** runner lines 64–87 use the tested helper's
  16 uniform native-unit directions, local signs at 16/17, tangent signs at
  18/19 and atlas direction at 20. Exact coordinate axes occur at 0/4/8/12.
  Duplicate actions remain separate entries. Policy subsets have declared
  preview-call counts 2/2/4/16/21; atlas requires zero previews. Exact ties use
  the lowest index. A failed required candidate invalidates that subset;
  failures outside a subset do not invalidate it. A failed atlas preview does
  not invalidate the fixed atlas policy, which does not use the preview.
- **Phase epoch and state:** runner lines 199–209 use endpoint-excluded 2700
  sample orbits, indices 337/1012/1687/2362, and `index*T/2700`. These are the
  actual sampled phases, not rounded exact eighths. All previews and both
  target steps restart from the same copied original orbit state. They do not
  continue from the end of a preview. The native wrapper additionally copies
  its initial-state input (`hopfield_drift_probe.py:74`).
- **Endpoint separation:** preview scoring uses only elapsed 0 through 75,
  with the original phase epoch passed to the common activation metric
  (runner lines 132–145). Primary targets use the strict mask `times>75`,
  yielding 2250 samples from 75.1 through 300; a separate metric scores all
  3001 samples from 0 through 300 (lines 148–164). Each score fits one global
  phase offset to its own scored interval, using the frozen 128-point phase
  search. There is no frequency adjustment or time warp. This separates
  samples; it does not create statistically independent future dynamics.
- **Local derivative:** the sensitivity starts with zero parameter derivative
  at the fixed initial state and propagates through 0–300. Its Gram matrix uses
  only samples with `times>75` (runner lines 206–210), projecting out one global
  phase direction. This matches the local approximation to the scored tail.
  It assumes knowledge of the simulator and a nominal future sensitivity; its
  cost cannot be treated as free. This derivative is not a perturbed target
  outcome or an adaptively updated intervention.
- **Coarse/fine comparison:** all decisions use the single dt=.005 preview
  pool and are then reused for dt=.01 and dt=.005 targets (runner lines
  214–226). The coarse result is a numerical comparison, not an independently
  selected policy or an independent statistical replicate. Each condition's
  pool is consumed before loop parameters change, avoiding a late-bound
  thread callback selecting a different radius or phase.
- **Hybrid interpretation:** local is used at three doses and atlas at the
  largest dose. Thus the pooled local-minus-hybrid difference is exactly one
  quarter of its largest-dose difference under complete equal weighting.
  These are not four independent demonstrations. Uniform is a separate
  geometry-free direction set; the augmented search includes local and atlas
  candidates and has their information requirements.

## Source, failures, and replay

Prepare lines 11–23 use only the old EXP004 atlas and the EXP006 development
manifest to construct and hash the source packet. The claim boundary (prepare
line 43) correctly identifies new starting points on a previously seen atlas,
not a new network or an unseen parameter map. The development-informed hybrid
rule is explicit. No paid tool or resource launch is present.

Runner lines 166–180 require clean, remotely contained source, verify the
declared old input hashes, create an exclusive attempt directory, record the
source and manifest hash, and enforce the frozen hard cap. Native failures
write their partial state arrays before being caught. A fatal sensitivity
failure propagates its alpha/gamma, initial state, requested/returned times,
partial augmented states and solver costs to
`unhandled-numerical-failure.npz`; the outer failure marker is written and the
attempt cannot be restarted in the same directory (lines 237–240).

Complete records retain failures and undefined policies; summary lines 90–129
require the entire target ID census and invalidate affected equal-weight
means/contrasts rather than delete failed trials. Baseline failures are
retained without replacement. An abruptly interrupted condition can have
saved raw traces before its condition decision JSON is written; such an
attempt is incomplete and must not be presented as a complete preview census.
The failure/timeout gate prevents accepting it as a completed experiment.

Replay (runner lines 243–320) checks receipt bytes/hashes, rejects a failure or
timeout marker, reconstructs actions and policy choices, checks saved initial
states, weights, steps and sample grids, replays the tail Gram and substantive
waveform objectives, checks preview accounting, and rederives summaries.
Saved certificate gates are checked; this saved-data verification does not
reintegrate shooting certificates or independently prove solver accuracy.
Coarse/fine evidence and existing independent certificate tests address those
separate numerical concerns. Floating-point arithmetic replay uses the
inherited portable tolerances; raw artifacts remain bound by exact hashes.

## Independent validation performed

`tests/test_hopfield_holdout_audit.py` adds two injected, non-neural tests:

1. A 75-unit preview and both 300-unit target steps receive identical original
   states and parameter actions, while the preview receives the exact phase
   epoch and its own full sample interval.
2. An injected fatal sensitivity failure preserves the complete payload, all
   12 baseline records and the failure marker, runs no target, emits no complete
   receipt, and rejects a second run into the same attempt directory.

The independent audit tests, runner tests, and selector helper tests pass
together: **32 passed in 1.00 s**. All solvers in the two added orchestration
tests are injected; no EXP007 target output was consulted or generated.
