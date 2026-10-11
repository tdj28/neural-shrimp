# EXP-009: numerical refinement of known EXP-008 target discrepancies

Status: draft prospective execution protocol, 14 September 2026. The case
selection below uses already observed EXP-008 outcomes and is permanently a
post-run numerical sensitivity study. No EXP-009 target integration has been
performed in preparing this protocol. The complete executable implementation,
machine plan, exact selected-case census and local audit must be committed and
pushed before this new attempt starts. This is not a new utility confirmation,
a replacement EXP-008 attempt, or authorization for paid resources.

## Question and scope

Does reducing the native integration step stabilize the recorded later
waveform errors for the known difficult EXP-008 cases, and do the original
frozen policy comparisons change when their same selected actions receive
finer numerical evaluation? A small fixed independent-solver panel checks
whether apparent RK4 agreement survives a different integration method.

This packet keeps the model, initial conditions, reference waveform,
destinations, sample times and task endpoint unchanged. It does not regenerate
the reference orbit, local sensitivity, candidate directions, previews,
features, models or policy choices. Numerical fitting portability and strict
EXP-008 replay are separate questions. Neither a stable subset nor agreement
of policy means makes the unresolved original replay gate pass.

## Complete prior-case census and deterministic selection

Read both `development-targets.jsonl` and `test-targets.jsonl` from the
receipt-bound EXP-008 attempt. Its receipt SHA-256 is
`ddb22c78a6b5f11a023088fd90f3403d25a2797e00624f01c316171ce35706fb`.
The canonical case key is `(role, anchor, phase, radius, candidate)` with the
numeric radius value, not its display string. Each of the 3,504 cases has
exactly one complete target at each old step, 0.01 and 0.005. Abort preparation
on duplicate, missing, nonfinite or invalid records instead of silently
changing the census. Bind the original receipt, journals, decisions,
baselines, every selected old raw trace, and referenced source files by hash.

For each case define `old_delta = abs(old_rmse_0.01 - old_rmse_0.005)` using
the saved full-precision JSON numbers. Form the union of the following sets:

1. **Sensitive cases:** every case with `old_delta > 0.001`, strictly. This
   diagnostic inclusion threshold was chosen after the old discrepancies
   were known; it is not an acceptable task-quality loss or inferential test.
2. **Frozen selected actions:** every unique case selected by any of baseline,
   geometry, shuffled geometry or minimum preview in all 96 saved test
   decisions. Preserve all 384 policy assignments even when their trajectories
   share one unique case. Do not recompute an argmin or refit any model.
3. **Matched stable numerical controls:** for each origin/phase/radius context
   containing a sensitive case, consider candidate indices `[0,4,8,12,16,17]`,
   the signed coordinate axes and the two local directions. Among those with
   `old_delta <= 0.001`, select minimum `(old_delta, candidate_index)`. This is
   intentionally a prior-selected stable comparison, not a random control or
   an unbiased prevalence estimate. Keep any overlap with the other sets as
   a membership label, and integrate the unique case once per method.
4. **Zero-dose controls:** all 48 original origin/phase shams in both roles.

Preparation from the saved evidence gives the following exact counts:

| Set | Development | Test | Total |
|---|---:|---:|---:|
| Sensitive cases | 622 | 390 | 1,012 |
| Frozen selected unique actions | 0 | 220 | 220 |
| Matched stable controls | 62 | 46 | 108 |
| Zero-dose controls | 24 | 24 | 48 |
| Union, counted once | 708 | 661 | 1,369 |

There are 109 affected contexts across all 24 origins. The development context
`development-05`, phase 675, radius 0.016 has no qualifying stable axis/local
control. Retain this unmatched context; do not substitute a new origin or
broaden its control rule. Nineteen matched controls overlap selected actions;
no selected action crosses the sensitive-case threshold. The largest selected
old discrepancy is 0.0009998078186881088 for shuffled geometry, which is why
all selected actions are included independently of that threshold.

Sensitive counts at radii 0.00025, 0.001, 0.004 and 0.016 are respectively
16, 57, 198 and 351 in development, and 0, 4, 100 and 286 in test. The largest
old discrepancy is 0.47685138414318257 at `development-09`, phase 675,
radius 0.004, candidate 7. These are prior observations used for selection,
not new EXP-009 outcomes.

Sort unique cases lexicographically by the canonical tuple, serialize the
resulting list of lists with Python `json.dumps(..., sort_keys=True,
separators=(',', ':'), allow_nan=False)` and UTF-8 without a trailing newline.
The prepared union's SHA-256 is
`3d222c46b0c6c446e1b1c55b3442911a34aa3055c72b22bcec8bc153f28ecdad`.
The machine plan must retain all 3,504 prior pairs with inclusion/exclusion
reasons, not merely the selected list.

## Fixed integrations and independent panel

Run every selected case using the unchanged native float64 RK4 equations at
both `dt=0.0025` and `dt=0.00125`: 2,738 mandatory new traces. Both restart at
time zero from the exact saved EXP-008 initial state and run to 300, retaining
all 3,001 samples at cadence 0.1. The native step counts are 120,000 and
240,000 with strides 40 and 80. No new settling, continuation, state
perturbation, phase choice, resampling or task-dependent stopping is allowed.

Construct an independent panel entirely from old deltas before execution:
take the largest-delta sensitive case and largest-delta matched stable control
in every nonempty `(role, radius)` stratum; the largest-delta zero-dose case
in each role; and the largest-delta selected case separately for each of the
four policies. Ties use the lexicographically smallest canonical key. Keep
empty strata explicit. Deduplication gives 19 cases with 20 selection labels.

For all 19 panel cases, independently integrate the Python ODE with SciPy
DOP853 twice: `rtol=1e-11, atol=1e-13, max_step=0.05`, and
`rtol=1e-12, atol=1e-14, max_step=0.025`. Both use the same initial state,
destination and output times as RK4; both are mandatory regardless of earlier
results. Save solver status, evaluations, returned samples and failures.
This yields 38 independent traces, for **2,776 mandatory traces overall**.
Independent float64 solvers do not provide rigorous error bounds. If the
finest methods disagree, retain an unresolved result; no automatic further
halving, extra sentinel, threshold relaxation or favorable method selection.

## Unchanged waveform definition

Load destination alpha/gamma, delta and initial state directly from each
selected old fine-step raw trace. Validate them against its journal and old
coarse trace; do not reconstruct a direction with a new eigensolver. Load the
original certificate's 2,700 endpoint-excluded activation samples, period and
phase epoch. The epoch remains `phase * period / 2700` and the initial state
must equal the original certificate state at the saved phase index.

The target remains all-neuron activation RMSE over `75 < t <= 300`, using
the same reference and `waveform_errors(..., phase_grid=128)`: one constant
phase shift fitted on that whole target interval for each new trace, no
frequency fit, time warp, per-window refit or altered time cutoff. Preserve
the synchronous RMSE, component scores, fitted phase, per-window scores and
cadence diagnostics too. Fitting the same permitted constant phase for a new
trace is part of evaluating the unchanged endpoint; it does not mean choosing
a new starting state or reusing a favorable old trajectory phase.

Also score each new trace at the saved old-fine phase as a diagnostic using
the same reference interpolator. Label this fixed-phase sensitivity separately
from the original task endpoint. Store raw states and compare directly at
the common recorded times so phase fitting cannot hide waveform disagreement.

## Analysis and missingness

Report all prior and new RMSE values and signed/absolute differences per case.
For native refinement, report `d1=abs(rmse_0.005-rmse_0.0025)` and
`d2=abs(rmse_0.0025-rmse_0.00125)`, including their ratio when `d1>0`.
Report the finest RK4 versus each DOP853 result and the DOP853 pair difference
on every independent-panel case. Do not assume an RK4 order-of-convergence
ratio holds along sensitive trajectories.

A descriptive **RMSE-stable at this resolution** flag requires `d2<=1e-4`
and `d2<=d1`; show these two gates individually. The threshold defines this
numerical diagnostic only. It is not task noninferiority, a corrected
EXP-008 tolerance, or convergence of the full state. For the independent
panel, additionally report whether all three pairwise RMSE differences among
finest RK4 and both DOP853 results are at most 1e-4.

Report direct common-time state and activation RMS/max discrepancies,
over the whole trace and the target tail, separately from phase-aligned task
scores. A secondary **activation-waveform agreement** flag requires tail
all-component RMS difference <=1e-3 and maximum absolute component difference
<=1e-2 for the finest native pair. Apply the same diagnostic to the three
independent-panel comparisons. None of these flags certifies long-time chaos,
orbit stability, basin structure or every unselected EXP-008 candidate.

For each selection set and role/radius stratum, retain denominators,
overlaps, all failures, and median/95th percentile/maximum differences.
Selected controls cannot estimate a population convergence rate. Re-evaluate
the four saved test policies' fixed-action means and original baseline-minus-
geometry error contrast at both new native steps, including all 12 block
values and all 96 contexts. Call these post-run numerical sensitivity of the
old selections, never new holdout success. Do not recompute a finite-action
oracle, all-candidate prediction MSE, model fit or regret from this incomplete
candidate subset.

Every planned trace retains a status. Failed integrations preserve raw partial
arrays, attempted final state and progress where available. A required missing
or failed score makes its comparison undefined; a missing selected-action
score makes the corresponding whole-policy summary undefined. Show completed
descriptive values only with their exact denominator and never replace the
all-required result by a complete-case result. A timeout or interrupted attempt
is retained and not restarted in place.

## Workload, execution and evidence

Use the local CPU reference path with at most four workers and one thread per
BLAS/OpenMP library. Check shared-server load and free memory before launch;
do not interfere with other agents. The native work is 492,840,000 RK4 steps.
The same selected cases' old 0.005 target calls sum to 65.07420594 call-wall
seconds. Linear integration-step scaling gives 390.44523564 summed call-wall
seconds for the two new native levels, before new platform effects and
independent solves. This is a planning proxy from historical calls, which
include scoring and overlap; it is not measured zeta throughput, elapsed time,
or an observed speedup. Original paired raw traces total 210,604,526 bytes;
the two new native levels should have similar sampled-output volume because
cadence is unchanged. Retain a 2 GiB output reserve.

Freeze a 3,600-second target-execution ceiling and 60-second per-trace
limit, including native and independent solves. Final receipt sealing and
saved-array verification are timed separately after workers have stopped.
A timeout must terminate only this task's worker and retain its
failure record. No paid compute is planned. Preparation, compilation and
synthetic fixtures may run before the freeze; target-case timing is itself
target execution and must wait for the pushed source and machine plan. If a
measurement requires changing the budget or solver, prepare a dated successor
before any additional target outcomes, preserving the original attempt.

Write new raw evidence to
`/data2/neural-shrimp-data/raw/EXP-009/attempt-001/`, outside the Git checkout,
using exclusive attempt creation and an append-only progress journal. Record
the exact source commit and source/plan hashes, bound original inputs,
platform/compiler/library/arithmetic details, worker PIDs, start/end times,
per-call timing, status, peak memory and every raw artifact's bytes/SHA-256.
Keep old EXP-008 evidence immutable. Store compact plan, census, receipt,
derived summaries and audit reports in the repository with stable references
to the external raw inventory. Audit hashes and the complete planned census
before interpreting outcomes; reproduce scoring from retained arrays without
new integration, and retain independent-audit disagreements verbatim.
