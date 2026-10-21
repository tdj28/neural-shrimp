# Activation organization as a local regime diagnostic

ARP1 is an exploratory pilot requested on 21 October 2026. It asks whether
16–128 consecutive activation observations distinguish numerically periodic
and finite-time chaotic trajectories at one parameter setting, without a new
parameter-plane scan. It also tests whether the proposed activation measures
improve on direct recurrence of the 896-dimensional feedback state.

All experiment-specific code, plans and compact results live in this directory.
Raw arrays, logs and figures live outside Git under
`/data2/neural-shrimp-data/audits/activation-regime-probe/`. Existing research,
renderers and media are inputs only and are not modified. The shared research
log receives run-card pointers. No rented compute or paid review is used.

## Design before target outcomes

The visual observation motivated the hypothesis; it is not held-out evidence.
The pilot acquires fresh float64 forward passes of the pinned Qwen2.5-0.5B
soft-token loop, using the established fixed prompt and one ordinary start per
parameter setting. Ten selected settings are listed in `plan.json`: two for
calibration and eight held out by parameter setting. These are a small,
purposefully selected challenge set, not a random sample of the phase plane.
Prior diagnostics guide selection but never supply a new trajectory's label.

After 400 settling steps, save the first 128 steps' MLP activations at every
layer and the full feedback state. Then use a separate 512-step continuation
for reference labels. Two nearby trajectories, at absolute offsets 1e-6 and
1e-8, are renormalized each step; their first 128 growth values are discarded
for tangent alignment. The batch layout stays fixed throughout acquisition.
Activations are stored as float32 for compactness; full states, growth values,
and all model computations are float64. No precision below the activation
storage floor is inferred from activation recurrence.

Reference labels are deliberately three-way:

- **Numerically periodic:** the smallest lag up to 64 closes across the last
  384 reference states at relative tolerance 1e-9, all smaller lags fail, and
  both finite-time growth estimates are negative. This is a numerical
  recurrence and contraction diagnostic, not a new interval certificate.
- **Finite-time chaotic:** no such recurrence, both exponent means exceed
  0.01, all four 128-step block means at both offsets are positive, and the
  two means differ by at most max(0.03, 20% of the larger absolute mean).
- **Unresolved:** every other outcome, including inconsistent diagnostics.

The long-window label is a finite-horizon reference, not proof of infinite-time
chaos. Drift or late capture can invalidate an early-window prediction; retain
and report such cases. A fixed point counts as period 1. No visual island is
classified as a shrimp by this test.

## Measures and baseline

For each layer and each observation budget (16, 32, 64, 128):

1. Temporal effective dimensionality: covariance participation ratio, both raw
   and after standardizing varying neurons. Constant trajectories score zero.
2. Visual roughness: mean squared neighbor differences on a 64 x 76 neuron
   grid, divided by twice the across-neuron variance. Positions are learned
   solely from the two calibration trajectories, using their standardized
   activation profiles and the first two principal components. Lower scores
   indicate smoother spatial patches. This is a display-dependent descriptor.
3. Baseline: smallest full-state return error over candidate periods allowing
   at least three repeats within the observation window. Declare recurrence
   only below relative error 1e-8; otherwise output "unresolved", not "chaos".

Aggregate activation scores by the median over all 24 layers. For each budget
and score, the binary exploratory classifier uses the midpoint of the two
calibration scores; its orientation is also learned on calibration only.
Record whether that orientation agrees with the visual hypothesis. Calibration
requires one reference-periodic and one reference-chaotic trajectory; otherwise
classification fails loudly. Do not select the best layer, window or metric
after viewing the test labels. Report all rows, all budgets and all metrics.

Binary accuracy is conditional on reference-resolved cases. Report the full
denominator and unresolved cases separately. For a fair descriptive comparison,
also score the recurrence baseline's closed-set guess (no detected recurrence
=> chaotic candidate), explicitly distinguishing it from its conservative
three-way output. An activation-based chaotic prediction is likewise only a
candidate unless independently checked. The principal question is whether
activation scores improve on recurrence at the same observation budget.

## Resource and integrity limits

Use local CPU only, four numerical threads, one model instance and a fixed
30-row batch (10 trajectories plus two twins each). A short benchmark measures
throughput and the actual batch before the full acquisition. Limit the run to
one hour and record partial progress and failed attempts. Check patched norms
and rotary embeddings, and compare central differences with an exact float64
JVP at three halved step sizes before trusting growth estimates.

Commit source before acquisition, append a run card, preserve exact source and
plan hashes, and hash every raw output. Tests use synthetic periodic, drifting
and chaotic series to check metric invariance, recurrence and refusal gates.
No screenshot is used as numerical input.

## Interpretation limits

This pilot can establish feasibility or expose a failure on these selected
settings. It cannot establish universal detection, generalization to other
prompts/models, a benefit specific to shrimp geometry, or better task accuracy.
Low dimensionality is partly expected from a short periodic cycle: p distinct
states have centered covariance rank at most p-1. A more orderly appearance
therefore need not add information beyond recurrence. Longer periodic cycles,
quasiperiodicity, slow transients and low-dimensional chaos are important
confounds; a detector must be allowed to abstain.

Commands and actual outcomes are recorded in `research-log.md`; an outcome
report and compact receipt will be added only after every row is checked.
