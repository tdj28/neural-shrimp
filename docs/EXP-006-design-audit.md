# EXP-006 prospective design audit

Date: 2026-09-11. Scope: the deterministic atlas selector, paired analysis,
compiled plan and drift orchestration, before EXP-006 target trajectories.
This is a local audit under `EXPERIMENT_INTEGRITY_SKILLS.md` sections 4 and
4A, including its matched-contrast and predictor/outcome circularity checks.
It is not a paid review or approval for paid resources. Numerical continuation
and the native probe have separate numerical audits and tests.

## The question is behavior, with recurrence as a secondary response

The primary experiment asks whether a direction selected exclusively from the
old EXP-004 atlas preserves an original neural activation waveform better than
specified local and sham policies under equal finite changes of connection
weights. The response is the three-neuron activation RMSE over 300 physical
time units, permitting one constant phase shift. It does not fit a new period,
warp time, or align each cycle separately. The reference remains the original
certified orbit. Same-time error, phase offset and zero-dose error are retained.

This avoids treating the association between an old period-nine map and a new
period-nine label as task utility. Long-horizon recurrence remains useful as a
separate mechanism measurement. A positive result is conditional utility for
this explicitly defined small-network waveform task. A null or reversal
against the local policy argues against paying for this atlas policy on the
tested cases. Neither outcome establishes a training or reasoning benefit.

## Matched interventions and a strong ordinary comparator

Eleven nonzero arms at a fixed anchor, orbit phase, radius and integration step
start at the exact same certified state. The radius is the raw Frobenius norm
of the weight perturbation, equivalent to Euclidean norm in `(alpha,gamma)`.
Rectangle-width scaling only spreads anchor attempts; it never sets the
intervention norm. Four doses are fixed: 0.00025, 0.001, 0.004 and 0.016.

The primary geometry policy maximizes the original atlas's nearest-pixel
distance to an observed non-nine pixel among 512 fixed angular directions.
The inward normal remains a separately displayed arm; the policy does not
assume that an arbitrarily large inward step stays in the window. No target
outcome selects an angle, a sign, a dose, or an anchor.

The local comparator minimizes the finite-horizon activation sensitivity
quadratic form after projecting out one global phase direction. Its state
sensitivity obeys `dS/dt = J(x)S + B(x)`, with nonzero parameter forcing entries
`B[0,0]=tanh(x3)` and `B[2,1]=tanh(x1)`. This comparator knows the baseline
neural dynamics but sees no atlas. It addresses the possibility that geometry
merely recovers ordinary alpha-versus-gamma sensitivity. Fixed-weight Floquet
multipliers alone would have been an insufficient comparator because they
measure state perturbations rather than weight-direction sensitivity.

The two signs of the local and random policies are averaged, representing a
symmetric sign choice. Their better observed sign is never selected after
outcomes. The shuffled comparator uses the preceding selected anchor's inward
normal. This is a deterministic derangement of donor identities; similar or
identical donor directions remain possible and must not be called independent
random geometry. Every signed arm remains visible in the complete data.

## Anchor selection and atlas limitations

Twelve anchor attempts are fixed: four in each rank third of the raw atlas
margin among eligible cells of the core's eight-connected period-nine
component. A candidate's nearest observed non-nine pixel must be closer than
its nearest crop edge. The selector never pads the rectangle with presumed
exterior cells. Within each third, a seeded first point and maximin spacing
in rectangle-width coordinates determine four attempts. Ties and the random
generator seed are fixed. Failed long-run baseline certificates do not cause
replacement or reselection.

These are lower, middle and upper atlas-margin strata, not demonstrated
anatomical classifications of the shrimp's body and limbs. Pixel distances
are not continued bifurcation boundaries. The old atlas can contain finite-time
misclassification, and a destination assigned to a nine-peak pixel need not
share that pixel's attractor. New trajectories test this transfer instead of
assuming it.

Inspection of the compiled old-data choices found **48** anchor-radius
geometry directions, **zero** all-zero-score fallbacks, and **12** selected
scores flagged as crop-censored. These are design facts, not EXP-006 outcomes.
The latter scores can overestimate unseen margin beyond the crop. Their flags
and all 512 candidate scores per choice are retained; the audit did not change
the rule after inspecting these choices. Directions outside the crop or whose
nearest pixel belongs outside the core component receive zero score. A future
all-zero case uses the explicit inward fallback, without searching new data.

## Census, failures and descriptive uncertainty

The full prospective census is
`12 anchors × 4 phases × (4 doses × 11 arms + 1 zero control) × 2 steps`,
or **4,320 trial-step records**. Both steps independently initialize each
baseline from `(0.1,0.1,0.1)` and use the fixed 10,000-unit burn and 2,000-unit
measurement. A baseline must retain nine peaks at both steps and receive an
independent attracting-cycle certificate before its paired drift assay.
Unqualified anchors remain in the 12-attempt denominator and receive all their
planned `baseline_not_evaluable` records. Conditional paired estimates report
their eligible-anchor coverage explicitly.

The primary numerical step is 0.005. Step 0.01 is a separate sensitivity
analysis, not another independent replicate and not an outcome-selected step.
Within each anchor the analysis averages phases and the four fixed doses;
eligible anchors receive equal weight. Benefit is comparator RMSE minus
atlas-policy RMSE, so positive means lower error for the geometry policy.

The analysis rejects duplicate and unplanned trials. Missing, failed,
nonfinite or out-of-range errors make the affected radius-step contrast
undefined, including failures in a required diagnostic arm. A failed or missing
zero control invalidates every radius contrast at that step. Any such missing
radius also makes the across-radius estimate undefined. There is no
success-only deletion. All rows and failure payloads are preserved.

Bootstrap resampling acts on anchors, with 10,000 seeded draws and common
resampling indices across contrasts. The 2.5th and 97.5th percentiles describe
variation under this fixed-anchor resampling scheme. Purposively selected
anchors and a maximum of twelve eligible cases do not justify general
population confidence coverage or formal confirmatory p-values.

## Checks completed without target integration

`tests/test_hopfield_drift_design.py`: **9 tests passed**. Synthetic fixtures
check deterministic rank-stratum selection, raw norms, connected-component
membership, observed-boundary and crop rules, lexicographic nearest-neighbor
ties, direction-score selection and explicit fallback, source-byte and row
census binding, paired sign/phase/radius/anchor aggregation, missing and failed
trials, nonfinite errors, duplicates, invalid phases and zero eligible anchors.

The audit independently called `compile_plan()` using only retained EXP-004/005
data and obtained an object exactly equal to the saved EXP-006 plan. No target
integration was used to choose the design. A runner-verifier improvement was
reported before freeze: reconstruct each arm direction from the frozen plan
and retained local sensitivity, and check every raw initial state against its
declared certified orbit phase. Norm equality alone cannot detect swapped
directions or wrongly initialized arms. The strengthened
`verify_complete_trial` now performs both checks, including the first saved
trajectory state, alongside independent waveform-score reconstruction.

A subsequent protocol-to-code pass corrected wording without changing the
design: local directions are computed from new baseline orbits under the frozen
algorithm; 48 accepted continuation successors permit 49 stored nodes including
the seed; the branch figure plots the dominant multiplier while retaining both;
and the illustrative trace explicitly uses step 0.005. Its first eligible
upper-margin selection agrees with the plotter, with an unavailable panel when
none qualifies and no fallback to another stratum. Receipt wall time is captured
before final verification and is now described accordingly. The 1,800-second
watchdog still spans verification. No EXP-006 target integration was used in
this protocol-to-code review.

## Claim boundary and cost

The design can compare fixed policies at equal disturbance norm and under a
shared behavior metric. It cannot prove exact shrimp boundary topology, prove
all unresolved trajectories chaotic, establish an asymptotic robustness radius,
or show a speedup by omitting atlas construction and diagnostic costs. The
atlas's recorded 140.54-second integration cost and local-sensitivity costs
must remain visible. A complete controller, an amortization argument, and
transfer to another neural task require additional experiments.

No design-level blocker was found for the stated exploratory claim. The
complete plan, code and local numerical checks must be committed and pushed
before target execution. Continuation remains separate from drift selection
and cannot retrospectively redefine any policy or endpoint.
