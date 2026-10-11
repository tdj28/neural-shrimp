# EXP-011: retrospective four-preview shortlist feasibility

14 September 2026. Exploratory development study, prepared after EXP-010
qualified the complete original 75-unit preview census. No model fitting or
new outcomes are permitted until this protocol, plan, all executable sources
and tests are locally reviewed, committed and pushed privately. This is not an
untouched test, a new geometry confirmation, or a 300-unit utility result.
The [design](shortlist-feasibility-design.md) and
[feature appendix](shortlist-feature-adapter-appendix.md) give the scientific
rationale and provenance audit.

## Cohort and endpoint

Use exactly the 3,456 original EXP-008 entries: 24 physical origins, two phases,
four radii, 18 candidate IDs per context, 192 contexts. Both historical roles
are now development. The twelve EXP-007 origins alias the twelve original
EXP-008 development origins; they do not add observations. Group by a common
dynamical-system identifier plus exact binary64 alpha/gamma coordinates,
never role, anchor spelling, phase or file identity. Preserve all 18 IDs even
where physical directions duplicate. Every policy's four IDs are distinct.

Labels are EXP-010's unmodified 75-unit phase-aligned aggregate RMSE at
`rk4-0.00125` (primary). Repeat the complete fit/tune/shortlist recipe with
`rk4-0.0025` as mandatory sensitivity. Do not choose a favorable resolution.
No transforms, clipping, target-window substitution, post-result feature
changes or fallback labels. EXP-010 completeness and the previously audited
full numerical-qualification result are mandatory input conditions; preparation
reads only a metadata-only qualification certificate. Full label-bearing
prior summaries are hashed without decoding, including during preparation.
All 16 previously fixed independent pairs remain diagnostics; their partial
coverage cannot produce a full independent 18-candidate oracle.

## Features and isolation

The new label-free adapter emits the exact ordered 37 shared names and six
geometry names listed in the appendix. The baseline receives only the 37
shared columns; the geometry model receives those 37 plus all six geometry
columns. No metadata becomes a feature. Parameter-response norms use direct
sensitivity/projection residuals, preserving the mathematical quantities and
avoiding cancellation in a Gram quadratic form. Original realized directions,
unperturbed certificates, saved sensitivity arrays and historical atlas are
bound. No preview, target, decision, model, or outcome file is an adapter input.

Run 24 isolated outer-fold processes. Each process receives label-free features,
source bindings and an explicit LabelStore capability that permits only the
other 23 physical origins' labels. The model API accepts only those training
arrays and held-origin features. All phases, radii and candidates of the outer
origin are withheld together. Each worker writes four model/prediction packets
(two schemas by two resolutions), exact ordered top-four decisions, fixed
controls and a hash-bound pre-reveal seal before any held-origin label reveal.
A separate evaluation stage runs after all fitting workers have terminated.
Its first accesses expose only each sealed four-entry shortlist; its later
all-18 reads are explicitly retrospective oracle evaluation. Separate training
and reveal access logs plus full packet replay verify this boundary. A failed
fold remains in the 24-origin denominator and cannot be replaced or resumed.

## Frozen ridge recipe and arithmetic check

For each outer model, select the ridge penalty by 23 inner leave-one-physical-
origin-out folds. Inner validation is mean raw-label squared prediction error
within the held origin, then an equal-origin mean over inner folds. The grid is
`[0.000001, 0.0001, 0.01, 1, 100]`. Select the first/smallest penalty on exact
CV-score ties. Standardization uses only current training data, with each
physical origin receiving equal total weight and each of its rows equal weight.
Center and divide by weighted population standard deviation; exact constant
columns have scale one. Preserve all columns. The intercept is the weighted
training-label mean and is unpenalized. Refit on all 23 outer-training origins.
Predictions are raw signed ridge outputs; do not clip them.

The primary solver applies an SVD to the weighted standardized design; it does
not form normal equations. Independently solve the augmented ridge design by
economic QR and triangular substitution at every grid value and inner fold,
plus the final training grid. QR selects its own minimum-CV penalty to expose
selection consequences. Save complete grids, scales, coefficients, CV errors,
held scores and shortlists for both arithmetic paths. Every inner and final
held prediction must agree within `1e-9 + 1e-7 * max(abs(SVD),abs(QR))`.
Also require exact selected-penalty equality and exact **ordered** four-ID
shortlist equality. Ordering within the same four-entry set can therefore fail
the numerical gate; report it separately from set changes and observed minimum
error changes. Full-18 ranking differences are recorded but alone do not fail
this gate. Retain all numerical disagreements and nonfinite/solver failures;
there is no fallback to QR, a different penalty, feature pruning or a new fit.
Full primary aggregate readouts are undefined if any required fit or arithmetic
gate fails; any available complete-cohort descriptive numbers are explicitly
unqualified.

## Selection, controls and readouts

For each held context, rank all 18 predicted scores ascending, breaking exact
ties by candidate ID. Select the first four, observe their true preview errors,
then choose their minimum, again breaking ties by candidate ID. Fixed controls
are axes `[0,4,8,12]`, diagonals `[2,6,10,14]`, and local coverage
`[0,8,16,17]`. Evaluate the full-18 minimum only in the retrospective evaluator.
Report selected RMSE, excess over full-18 minimum, deterministic full-18 winner
inclusion, inclusion of any exact minimizer, best selected true rank and mean
selected true rank for each policy and every context.

First average each readout within each physical origin's eight contexts, then
weight all 24 origins equally. Also report the four radii separately, retaining
all 24 origins and both phases. Report the paired shared-minus-geometry RMSE
contrast (positive favors geometry), and each fixed-control-minus-learned
contrast. Preserve all per-origin values and all 192 contexts. Descriptive
bootstrap summaries resample 24 whole origins with replacement 20,000 times,
NumPy default generator seed `20260914`, and report percentile 2.5/97.5 bounds.
The same resampling indices are used for each paired readout. These are
selected historical origins with possible spatial dependence and overlapping
outer training folds: the bounds are descriptive stability summaries for this
observed cohort, not confidence guarantees for fresh origins or confirmatory
hypothesis tests. No p-value threshold or model-selection decision follows them.

Compare both complete label-resolution recipes, their chosen penalties,
predicted-score gaps, ordered four-ID lists, four-ID sets, selected actions and
selected errors. Also evaluate each primary shortlist unchanged under the
comparison labels, separating label sensitivity from refitting sensitivity.
Retain all signed label/prediction changes, empirical maximum change envelopes,
best/second and fourth/fifth gaps, full rankings and exact tie/order changes.
Empirical envelopes are not certified integration-error bounds. Report SVD/QR
shortlist consequences in observed RMSE as well as score differences. All
available independent sentinel labels are retained with native differences and
shortlist membership; do not infer a full oracle from that partial panel.

## Costs, workload and execution

Each policy requires four candidate preview calls per context compared with 18
for the full reference: fourteen fewer calls, a 77.777...% nominal reduction.
This study reuses labels and performs zero new preview integrations. Measure
actual feature-loading/construction, fitting/prediction/selection, evaluation,
sealing and verification times separately. This does not measure end-to-end
preview-time savings or reacquire the old prerequisites. All learned policies
inherit the unperturbed 300-unit sensitivity prerequisite through features and
candidate IDs 16/17. Geometry additionally requires the historical atlas. Axes
and diagonal controls do not need sensitivity or atlas; local coverage needs
sensitivity. All policies need the reference waveform. The full-18 reference also
requires sensitivity because it includes candidates 16/17. Reused computations
are not free, and no net-speedup claim is permitted.

There are 24 outer tasks, 96 complete model recipes, 2,208 inner grid fits and
96 final grid fits. Each grid has five SVD/QR ridge solutions; final grids
retain all five for audit although only the selected predictions determine the
policy. A separate generated-noise benchmark on the local machine took 0.457818
seconds of wall time for one full 43-column, 3,312-row recipe including 23 inner
folds and the final grid. This resource-planning proxy gives approximately
43.95 seconds of summed recipe wall time for one 96-recipe pass. Full model
replay adds roughly another 43.95 seconds under the same proxy (87.90 seconds
combined), excluding input/feature and evaluation overhead. It is not a
research result or measured end-to-end
speedup. No GPU or paid call is needed. Cap workers at four and numerical
library threads at one before imports. The owned Linux process lifecycle is
reused unchanged from frozen EXP-009, with parent-death signals, signal-safe
launch registration, explicit joins/termination and complete timeout records.
Per-fold ceiling is 600 seconds; the execution and built-in replay ceiling is
3,600 seconds, starting after input/package/push preflight and the start record.
Final failure preservation and receipt inventory/sealing run
after disabling that alarm, so a timed-out attempt still gets its complete
terminal census and artifact receipt. Every partial packet, failure, stage and
attempt is retained.
Use an exclusive external attempt directory, normally
`/data2/neural-shrimp-data/raw/EXP-011/attempt-001`, never the checkout. No resume
or overwrite. Reserve 1 GiB before beginning. Validate the actual final resolved
output path, including symlinks. Record source and old/external input hashes,
exact clean pushed HEAD, platform/packages/thread settings, owner PIDs, all
expected terminal statuses and complete artifact inventory. Verify raw hashes,
training/reveal access boundaries, exact model/scoring replay, seals and full
summary reconstruction. Freeze revisions before outcomes; amendments after an
attempt must identify observed evidence and cannot silently repair old files.
