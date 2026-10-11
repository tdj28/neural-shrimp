# EXP-013 feature, model-interface and evaluation review

17 September 2026. Pre-outcome local review of the draft implementation;
**not an executable freeze or permission to interpret fresh results**. No new
trajectory, historical production fit or fresh label was generated or opened
for this review. The complete protocol, runner, plan, tests and private remote
freeze remain the caller's responsibility.

## Scope and findings

The feature review reads `fresh_preview_features.py` and its tests independently
of their author, and checks their interface against `preview_final_model.py`
and the draft runner. The evaluator's author checks its implementation and
synthetic outputs; a separate reviewer checks its policy/statistics contract.
These are different checks, not two independent implementations of every test.

The fresh adapter preserves the original 37 shared and six geometry fields,
exact phase indices, four radii, 18 charged candidate identities and complete
24-origin/192-context/3,456-row census. It has no file loader, fit or integration
entry point. Selection reads only supplied old atlas/manifest metadata, retains
the 36 physical exclusions and aliases, and keeps the fixed leaf order.
Out-of-crop candidates remain present with the inherited coverage indicators.

Reference inputs retain the 2,701-point endpoint-inclusive certificate and
all eight certificate gates; the shared-feature primitive removes the duplicate
endpoint as before. Sensitivity inputs retain 3,001 samples over 0..300, the
exact reference-phase initial state, zero initial parameter sensitivity and
original DOP853 settings. Algebraic checks bind activation sensitivity and
velocity to the supplied states and origin. Only times strictly greater than
75 enter the unchanged shared-feature calculation. These checks validate
representation and identity; they do not certify that fabricated arrays solve
the ODE. The runner must bind and replay the saved numerical evidence.

The adapter preserves the realized local direction and its negative. The
runtime must retain how that direction was obtained from the original tail
projection; the adapter does not recompute an eigenvector or choose its sign.
Its matrices and normalized rows satisfy the final model's phase, radius,
physical-group and direction conventions. The extra adapter schema field does
not enter training: the runner supplies the prediction API with matrices,
rows and ordered names. The historical training API remains separate.

A narrow evaluator identity defect was found before outcomes: a rehashed
prediction packet could change direction metadata or use boolean `True` in
place of candidate 1. The policy seal now preserves and validates complete
realized directions, canonical uniform entries, the local signed pair and
agreement across radii, and requires strict prediction identity types. The
reviewer's synthetic reproductions are regression tests. Missing directions
are allowed only for a no-prediction unavailable ledger, preserving failure
rows without enabling learned-policy qualification.

The fixed catalog contains 6/5/4/1 distinct policies at budgets 2/4/8/18.
All 6,144 declared readouts remain in an incomplete attempt. Selected-only
winner reconstruction rejects extra label IDs, uses candidate-ID ties, and
can verify all 5,760 non-full18 winner seals against later revealed labels.
The full18 endpoint is computed only at the full-reference stage. The runner
also must validate the exact winner identity census before granting that
reveal capability; a record count alone is insufficient.

Cohort summaries require the complete census, qualified predictions and
labels, and all seven explicit caller gates. Any required failure suppresses
all whole-cohort means and intervals while retaining available row diagnostics.
Means first average contexts within physical origins. The three signed
four-preview contrasts, five strata and two resolutions use the same 20,000
whole-origin draws at seed 20260915, yielding exactly 30 central 95%
descriptive intervals. They are not significance tests or fresh-population
confidence guarantees. Ranking changes across label resolutions remain
diagnostics; they do not select a preferred resolution or policy.

## Cost accounting closure

The pure amortization helper uses three fixed partitions: 192 singleton
context blocks; 24 blocks of eight contexts from one origin; and one block of
all 192 contexts. Within each block it charges selected candidate entries,
one reference per distinct origin, and one sensitivity per distinct
origin/phase when required. Final recipes and any supplied whole joint-setup
measurement are charged once per block. It divides the charged sum by the
block's decision count and averages blocks equally.

This does not amortize a new origin's reference across unrelated origins.
It does not divide a combined four-model or feature timer into invented
recipe/radius timings. Missing measurements remain unknown alongside the
known subtotal. Historical training-label and atlas replacement costs remain
unknown for policies requiring them. Coincident physical directions retain
separate candidate charges. These are task-seconds accounting tables, not
measured controller latency or evidence of net speedup.

## Validation and remaining boundary

With BLAS/OpenMP limited to one thread, the final feature, final-model and
evaluator suites passed **137 tests in 22.58 seconds**. The evaluator alone
passed **61 tests in 15.76 seconds**. The independent evaluator reviewer also
reran ten focused direction, identity and amortization checks: all passed in
2.39 seconds, and that review has no remaining blocker. These tests use algebraically fabricated features,
predictions and labels; the feature suite also replays the already declared
origin selection from old atlas/manifest metadata. No numerical integration
or production model fit occurs in these tests.

Synthetic coverage includes exact policy sets and ties, independent direct
scalar aggregation and bootstrap reconstruction, every required failure gate,
missing and nonfinite records, seal/identity tampering, selected-winner replay,
resolution/rank diagnostics and explicit cost formulas. The runner's source,
artifact and temporal access guards, ownership/deadline handling, actual
reference/sensitivity provenance, and full saved-array verification require
the separate integrated review. No final plan hash or execution claim is made
by this document.
