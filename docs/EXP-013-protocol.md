# EXP-013: fresh-origin preview loss versus charged budget

17 September 2026. Prospective **exploratory** executable protocol, written
before final old-data fitting or fresh dynamics. The complete source, this
protocol, machine plan and exact-plan local review must be committed and
privately pushed before execution. The runtime manifest records that full
commit and plan hash; this document does not invent their future values.

The experiment asks how selected 75-unit waveform error changes with budgets
of 2, 4, 8 and 18 previews at 24 fixed fresh physical origins. It compares
unchanged shared/geometry ridge recipes with fixed coverage controls. Four
previews remain the central comparison. There is no task-quality acceptance
margin, utility pass/fail rule, favorable-radius switch or outcome-selected
replacement policy. Numerical qualification below concerns solver agreement.

The [design record](fresh-origin-tradeoff-design.md) contains the origin
coordinates, earlier rationale, metadata calculations and post-EXP-012 CPU
decision. This protocol specifies the implemented stage boundaries and the
pre-outcome clarifications below. No earlier frozen experiment is modified.
The original EXP-008 strict Linux replay failure remains a failure.

## Scope, executable sources and freeze

The entry point is [run_fresh_preview.py](../tools/run_fresh_preview.py), with
[owned CPU runtime](../tools/fresh_preview_runtime.py),
[fresh feature adapter](../tools/fresh_preview_features.py),
[final old-data models](../tools/preview_final_model.py) and
[pure evaluator](../tools/fresh_preview_evaluation.py). They import unchanged
hash-bound numerical primitives. The complete transitive source/test/document
inventory is recorded by the machine plan's `source_files`; the exact ordered
feature names and historical input identity are inherited from the frozen
EXP-011 plan and validated before use.

`prepare` and `validate` read only historical atlas/origin metadata, source
files and artifact bindings. They perform no final fitting or new dynamics.
They independently reconstruct selection, all case/method IDs, controls and
counts. Historical label files can be hashed during preflight without
decoding new outcomes. Training later reads only the qualified historical
EXP-010 labels for the 24 EXP-008 physical origins.

`run` requires a clean source checkout, including no untracked files; every
bound source and plan must match the current commit. The live private remote
branch must resolve to exactly that commit. The committed local review at
`docs/EXP-013-local-review.json` must have status
`ready_for_exploratory_execution` and the exact plan SHA-256. Source/input
bindings are checked again after the producer. Runtime requires NumPy
**2.5.3** and SciPy **1.18.1** and records Python, platform, compiler, binary
hashes, numerical thread environment, invocation and resource observations.

This is a local review gate. The user-authorized Pro consultation was not
executed because the required enforceable [amount redacted] cap could not be established;
its incremental spend is **USD 0**. The prepared brief and compact not-run
receipt are source-bound. The separate local strategy analysis is not a Pro
verdict. No paid compute, paid review or public release is part of this run.

## Fixed cohort and complete work census

Reconstruct the 24 `fresh-00` through `fresh-23` origins with the unchanged
spatial selector on all 181 EXP-004 focus rows and core pixel `[53,60]`.
The exact physical union of EXP-006/007/008 origins contains 36 exclusions;
EXP-007's twelve origins alias EXP-008 development entries. Require distance
strictly greater than 0.04 in focus-rectangle-normalized coordinates and the
original component/crop-margin rule. Preserve all 1,703 eligible pixels,
recursive split/leaf membership and selected representatives in the plan.
Physical groups use exact binary64 parameter hex values, not rounded aliases.

Selection is conditional on the already observed atlas. Halos exclude prior
starting origins, not every prior perturbation destination. The resulting
origins may remain spatially dependent. Neither 24 representatives nor their
many phase/radius rows constitute a random independent population sample.
No origin, failed reference or inconvenient candidate may be replaced.

| Planned work | Required count |
|---|---:|
| Four old-data model recipes, handled by one owned model-stage task | 4 |
| Leave-one-old-origin validation folds / final ridge grids | 96 / 4 |
| Paired baseline/certificate tasks | 24 |
| Independent native baseline calls, two per origin | 48 |
| Numerical cycle certificates after both baselines retain nine peaks | 24 planned; ineligible prerequisites remain explicit |
| Unperturbed origin/phase sensitivity tasks | 48 |
| Candidate cases / decision contexts | 3,456 / 192 |
| Sham cases | 48 |
| Candidate native traces, three steps each | 10,368 |
| Sham native traces, three steps each | 144 |
| Independent panel cases / adaptive traces | 48 / 96 |
| **All required short traces** | **10,608** |
| Policy/context/resolution readouts | **6,144** |
| Non-full-18 selected-only winner records | **5,760** |
| Descriptive paired bootstrap intervals | **30** |

The staged terminal ledger has **10,681** top-level tasks: one joint model
task, 24 paired baseline/certificate tasks, 48 sensitivity tasks and 10,608
short-trace tasks. Baseline calls and fitted recipes are counts of work inside
those joint tasks, not additional ledger rows.

The native census entails 362,880,000 candidate steps, 5,040,000 sham steps
and 86,400,000 baseline steps: **454,320,000 RK4 steps**, plus adaptive
certificates, sensitivities and panel solves. Saved-evidence replay adds
analysis and old-data model refits, not new integration.

## Ordered stages and pre-outcome failure clarification

1. Build the unchanged CPU libraries and verify historical input bindings.
   Fit and seal all four qualified old-data models before fresh dynamics.
2. Attempt the full 24 paired baseline/certificate tasks. All references must
   qualify before the sensitivity stage starts.
3. Attempt all 48 unperturbed sensitivity tasks. All must succeed and pass
   input/feature validation before prediction and preview acquisition.
4. Build and seal all fresh features, all four prediction packets, the learned
   shortlists and fixed policies. Their numerical gates must pass before any
   perturbed preview starts.
5. Acquire every planned candidate/sham/panel trace, preserving worker and
   partial-failure evidence. A label-owning operator may store scores during
   acquisition; it cannot feed them back into fitting, features or selection.
6. Through the selected-only capability, reveal each non-full-18 policy's
   charged entries and compute its minimum. Seal **all 5,760 winners** before
   opening either resolution's full-18 evaluation reference.
7. Replay saved scoring, perform full numerical qualification, reconstruct
   both resolution readouts, and compute the fixed descriptive statistics
   only when every required gate passes. Seal costs, postflight and receipts.

**Pre-outcome clarification of the design:** a failed prerequisite stage
blocks the dependent cohort globally. A model failure blocks fresh dynamics;
any failed reference blocks all sensitivities/previews; any failed sensitivity
or fresh-prediction gate blocks all previews. No successful subset proceeds
as a replacement cohort. The current stage retains every available result;
the terminal ledger still includes every planned downstream task with an
explicit unavailable/not-started reason. An unavailable policy ledger still
retains all 192 contexts and all 6,144 diagnostic readout positions.

A short-trace failure does not select a smaller study. Preserve every terminal
trace status and available diagnostic. The complete selected-reveal/winner
sequence is performed only when all required preview tasks are complete.
No missing or numerically failed row is dropped to obtain a cohort mean;
all required qualification gates block every cohort aggregate and interval.
Any later rescue/refinement requires a separate amendment and freeze.

## References, sensitivities and realized interventions

Each baseline step independently starts from `[0.1,0.1,0.1]`: RK4 0.01 and
0.005, burn 10,000, measurement 2,000, event capacity 128, maximum peak period
32, minimum repeats 4, recurrence atol `2e-6` and rtol `2e-5`. Both must
retain nine peaks. The fine run seeds the unchanged independent certificate.

All certificate gates remain required:

| Certificate requirement | Frozen rule |
|---|---|
| DOP853 integration | rtol `1e-10`, atol `1e-12`; original bounded max-step and shooting recipe |
| Relative full-state closure and section residual | each `<=1e-8` |
| Starting section is an x3 maximum | section curvature `<-1e-8` |
| Nonstationarity | flow norm `>1e-7`, maximum coordinate amplitude `>1e-6` |
| Primitive orbit | exactly nine peaks; earlier interior return distances `>10*1e-8` in the original normalization |
| Neutral multiplier and normalized flow residual | each `<=1e-6` |
| Neutral-vector/flow alignment | `>=0.99` |
| Transverse stability | every transverse modulus `<1-1e-5` |
| Saved reference | exactly 2,701 uniform times from 0 through T; finite float64 `(2701,3)` states and exact tanh activations; matching parameters/section state |

Require verified, primitive and attracting with every named certificate gate
true. These are numerical certificates, not interval-arithmetic proofs.

Use phase indices **675 and 1,350** of the endpoint-excluded 2,700-point
orbit. Initial state is the exact corresponding saved state; the reference
epoch is `phase*T/2700`. Each unperturbed sensitivity uses DOP853 rtol
`1e-10`, atol `1e-12`, max-step `0.1`, over 0..300 at 3,001 exact samples,
with the initial 3-by-2 parameter sensitivity equal to zero. Retain solver
success, complete finite arrays, exact initial/grid/parameter identity,
activation sensitivity `(1-tanh(x)^2)*state_sensitivity`, and activation
velocity from the unchanged RHS. The tail is strictly `times>75`.

Compute and save the tail phase-projected Gram minimum eigenvector before
previews. Candidate entries 0..15 are the unchanged uniform sixteen
directions with exact coordinate axes; 16 is this realized local direction
and 17 its negative. Their signed pair, unit norm to `1e-12`, uniform
identities and across-radius direction consistency are checked. Duplicate
physical directions remain distinct, charged candidate entries.

At each origin/phase use radii **0.00025, 0.001, 0.004, 0.016** in the original
parameter coordinates. Destination is exactly the origin plus radius times
the saved direction. There is no crop-based candidate rejection. Shams have
zero parameter delta and are reference diagnostics, never action candidates.
Preserve initial state, destination and realized delta in each raw trace.
The 300-unit sensitivity is unperturbed information, not a 300-unit perturbed
target and not evidence of long-horizon action utility.

## The unchanged 75-unit objective

Every preview saves all **751 float64 samples on [0,75] at spacing 0.1**,
including the initial state and endpoint, with all three potentials. The
score applies tanh and the unchanged `waveform_errors` against the
endpoint-excluded reference at the preserved epoch. It uses 128 coarse phase
bins, smallest-index coarse ties, bounded refinement within one bin
(`xatol=1e-10`, `maxiter=64`), and the better coarse/refined objective.

The primary scalar is aggregate aligned RMS over all samples/components.
Save every component, synchronous, phase, window, search and cadence field,
plus the fixed-reference-phase RMSE. One constant fitted phase serves the
whole preview and every window. Do not trim transients, fit frequency or
rephase separate windows. The existing bounded activation metric and sampled
reference interpolation are preserved; no learned-label transformation or
prediction clipping is introduced. Absolute sham errors and cadence flags
are diagnostic, not new task-quality exclusion rules.

The objective is a 75-unit forecast measure. EXP-009's long-label limitations
and all original evidence remain separate. Better preview loss does not
establish 300-unit performance, memory, training stability, complete shrimp
boundary topology or usefulness in a trained neural network.

## Final old-data models and label-free predictions

Use the 24 historical EXP-008 physical origins and their 3,456 EXP-010
qualified labels at each resolution. The metadata-only EXP-010 qualification
certificate, historical receipt inventory and EXP-011 plan bind the source.
No fresh preview is available to the fitting API.

The four recipes are shared **37 fields** or those fields plus **six geometry
fields**, each using labels at 0.00125 or 0.0025. For each recipe:

- Leave out one historical physical origin in each of 24 validation folds.
  Fit scalers and coefficients using only its 23 training origins. Give each
  origin equal total weight, with an unpenalized intercept. Exact constant
  columns get scale 1 under the unchanged recipe.
- Evaluate every penalty `[1e-6,1e-4,0.01,1,100]`. Choose the lowest mean
  origin-validation MSE; the smallest penalty wins an exact tie. Refit every
  penalty on all 24 historical origins and retain the selected index.
- Retain direct SVD and independent augmented-design QR solutions for every
  inner/final grid. Neither path forms the normal-equation Gram matrix.
  Every compared score must satisfy
  `abs(svd-qr) <= 1e-9 + 1e-7*max(abs(svd),abs(qr))`.
  Require the same selected penalty. Coefficient differences are reported;
  no coefficient threshold substitutes for the declared score/action gates.

Fresh features are finite float64 matrices of shapes `(3456,37)` and
`(3456,6)`. Strict physical-group, phase, radius, candidate and realized
direction bindings prevent metadata substitutions. There is no fit using
fresh outcomes, radius-specific tuning, new feature or alternative learner.

For every fresh context retain all 18 predicted scores and both complete
orders, with candidate ID breaking exact score ties. Require score agreement
and **exact ordered top-2, top-4 and top-8 agreement** between SVD and QR.
A disagreement beyond those prefixes remains a diagnostic; full-order
identity is not an extra action gate. No fallback chooses the favorable
solver, penalty, radius or resolution.

## Policies, winner seals and statistics

Shared and geometry learners use their sealed ordered prefixes at budgets
2, 4 and 8. Add the exact controls below; full 18 is a single common endpoint.

| Budget | Control | Candidate IDs |
|---:|---|---|
|2|Alpha pair|`[0,8]`|
|2|Gamma pair|`[4,12]`|
|2|Diagonal pair|`[2,10]`|
|2|Local pair|`[16,17]`|
|4|Axes|`[0,4,8,12]`|
|4|Diagonals|`[2,6,10,14]`|
|4|Local coverage|`[0,8,16,17]`|
|8|Octagon|`[0,2,4,6,8,10,12,14]`|
|8|Local plus six angles|`[0,2,4,8,10,12,16,17]`|
|18|Full reference|`[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]`|

There are 6/5/4/1 policies at budgets 2/4/8/18, totaling 16. At both
resolutions and all 192 contexts, each of the 15 non-full-18 policies reveals
only its selected IDs and chooses the minimum observed RMSE, ties by
candidate ID. This yields 5,760 winner records. The complete winner packet
must bind the policy seal and reproduce every selected-only decision before
the two full-resolution reveals are allowed. Overlapping policy selections
remain independently charged; previously read labels cannot alter a shortlist.

The logs bind model seal before fresh dynamics, policy seal before previews,
selected-ID accesses, winner seal and subsequent full-18 accesses. They check
recorded artifacts and reviewed capabilities; they are not an OS-level proof
that no unlogged file could ever have been read. The raw acquisition operator
and the selected-only chooser have distinct roles.

Report selected RMSE, excess over the same-context finite full-18 minimum,
exact tie-broken winner inclusion, any-minimizer inclusion and selected-rank
diagnostics. First average phases/radii within each origin, then weight all
24 origins equally. Radius strata average the two phases per origin.

The central contrasts are geometry-b4 minus shared-b4, axes-b4 and
local-coverage-b4; negative favors geometry. Use one shared array of
**20,000 paired whole-origin bootstrap draws**, seed **20260915**, with
sorted canonical physical-origin order. Across five strata (pooled plus
four radii), three contrasts and two resolutions, compute exactly **30**
central 95% descriptive intervals using NumPy linear 2.5/97.5 percentiles.
Retain all 24 origin effects and their signs. These are descriptive stability
summaries in a selected, spatially dependent cohort, not confirmatory tests
or population confidence guarantees.

Retain signed resolution changes, complete actual candidate orders, winners,
shortlist/action changes, cutoff gaps and maximum observed score changes.
Resolution order equality is diagnostic, not a scalar qualification rule.
Observed change envelopes are not certified integration-error bounds.

## Complete numerical qualification

Integrate every candidate and sham at RK4 **0.005, 0.0025, 0.00125**.
Primary labels are 0.00125, with complete policy/statistical readouts also at
0.0025. Each native trace must have its exact expected step count (15,000,
30,000 or 60,000), complete grid, finite float64 `(751,3)` state array,
initial identity, unchanged intervention and solver metadata.

For every candidate and sham define

```
d1 = abs(RMSE_0.005 - RMSE_0.0025)
d2 = abs(RMSE_0.0025 - RMSE_0.00125).
```

Require **both** `d2<=1e-4` and `d2<=d1`, including cases near roundoff.
Between the two finest unaligned activation traces require whole-preview
RMS difference `<=1e-3` and maximum absolute component difference `<=1e-2`.
Keep scalar, non-increase and waveform failures individually visible.

The 48 independent panel cases are fixed by origin index `i=0..23` and
phase-list index `p=0,1`: radius index `(i+2*p)%4`, candidate `(i+9*p)%18`.
The plan contains every exact ID; this gives two cases per origin, one per
phase, twelve per radius and coverage of all 18 candidates. It is not chosen
from newly observed error maxima and is not an exhaustive independent solve.

Run each panel case twice with DOP853:

| Method | rtol | atol | max step |
|---|---:|---:|---:|
| `dop853-1e-11` | `1e-11` | `1e-13` | `0.05` |
| `dop853-1e-12` | `1e-12` | `1e-14` | `0.025` |

Require solver success, finite complete arrays and recorded work metadata.
Compare native finest against each adaptive trace, and the adaptive traces
against one another. **Every pair** must pass scalar RMSE difference
`<=1e-4`, activation RMS difference `<=1e-3`, and activation maximum
difference `<=1e-2`. In addition to separately optimized complete score
trees, retain each pair's errors at the native finest trace's saved phase.
That common-phase diagnostic does not redefine the primary label or gates.

Saved-array replay reconstructs every score/phase/window/cadence field.
Floating replay uses the unchanged `rtol=1e-12`, `atol=1e-14`; shapes,
keys, categories and identity fields must agree exactly. Raw arrays are
hash/byte bound, and reconstructed fresh feature arrays must agree exactly.
Model packet replay retains all four recipes, penalties, fold membership,
predictions, ordered selections and final readouts. This is an implementation
replay, not independent reintegration or retroactive repair of earlier data.

The seven required whole-cohort gates are references, sensitivities, models,
native candidates, native shams, independent panel and access chronology.
All must be explicit true booleans, with complete finite labels and valid
winner replay, before any cohort mean or interval qualifies. Numerical
disagreement may coexist with complete execution; neither means successful
scientific qualification. No threshold is relaxed after seeing results.

## Shared CPU envelope, timing and honest costs

Use the existing native CPU path for every baseline and preview; certificates
and sensitivities remain CPU. EXP-012 failed both the specified CUDA
compatibility and deployment-speed criteria, so no GPU fallback is enabled.
At most **four owned worker processes** run at once. Each numerical-library
thread variable is fixed to **1** before NumPy/SciPy import. Record the
compiler and unchanged `-O3 -std=c++17 -fPIC -shared` builds; no reduced
precision or fast-math substitution is introduced.

The guarded producer ceiling is **3,600 seconds**, including fitting,
fresh dynamics, saved scoring and aggregation. Per model/prerequisite task
the ceiling is **600 seconds**; per short preview **120 seconds**. Check
at least **8 GiB available RAM** and **2 GiB external disk headroom** at
launch and before stages; record load and capacities. These are shared
availability checks, not a machine reservation. No failure increases the
worker count, changes numerical methods or extends a timeout. Retain owned
PID/start/boot identities; terminate and join only this attempt's processes.

EXP-010's measured 139.887663-second producer and 311,040,000 native steps
support a rough capacity comparison, not a measured runtime for this new
mixture. The design's approximately 204.3-second naive step scaling and
provisional 5–15 minute range omit differing adaptive, model and I/O costs.
The resource inventory and conservative fixed ceiling remain the execution
bounds; timing results must come from the actual attempt.

**Pre-outcome timing clarification:** the implementation records each paired
baseline plus its certificate as one joint task, and all four old-data model
recipes as one joint task. It does not measure separate baseline/certificate
or recipe-specific durations. Fresh feature construction, prediction and
sealing are also a joint stage. Report those scopes exactly. Do not divide
a joint timer by two, four, origins or radii and call that component timing.
Worker records separately retain wall time, process CPU time, max RSS and,
for previews, integration, storage/hash and scoring durations. Parallel
stage wall includes worker startup/scheduling; task-wall sums are neither
elapsed stage latency nor CPU process time.

Report full acquisition cost separately from selected-policy accounting.
The latter charges all selected entries, including physically duplicated
directions, and the policy's prerequisites. References are required by all;
sensitivity is required by local controls, learned policies and full 18;
geometry additionally requires the atlas. All historical label/atlas
creation or current-host replacement costs remain **unknown**, never zero
merely because those assets were reused.

The fixed amortization blocks are:

- **H=1:** every context separately, 192 blocks;
- **H=8:** the eight phase/radius contexts of each origin, 24 blocks;
- **H=192:** the entire fixed cohort, one block.

Within a block charge each unique origin's paired reference task once, each
needed origin/phase sensitivity once and selected preview tasks individually.
The runner supplies **unknown** recipe-specific fitting and joint setup costs
to this accounting, because the recorded four-model and feature/prediction
stages do not measure those policy/block-specific costs. Their full measured
stage times are displayed separately and must not be apportioned into
invented recipe costs. Show known subtotals, every unknown component and
`null` complete totals whenever any prerequisite cost is unknown. The complete
block records reside in external `amortization.json`; `cost.json` and the
summary contain compact readouts. These are charged task sums, not measured
per-decision deployment.

Loss versus **charged preview budget** is the primary display. Selecting b
trace durations from full-18 acquisition does not measure b-preview
end-to-end latency. Do not claim net deployment speedup, money saved or a
task-derived acceptable quality loss from these accounting tables.

Failure-preservation ledger completion, postflight, inventory/receipt sealing
and the later standalone verifier run outside the guarded producer's main
computation timer so a timed-out attempt can still be preserved.
`producer_and_analysis_seconds` includes the initial finalization ledger
completion; `sealing_seconds` separately measures the final artifact inventory.
Amortization/cost construction and postflight have no individual timers in
this implementation. Do not invent those durations or imply the 3,600-second
ceiling includes independent later audits.

## Evidence, verification and commands

Use an exclusive, non-resumable external attempt directory, normally
`/data2/neural-shrimp-data/raw/EXP-013/attempt-001`. Reject in-repository
destinations, existing attempts and path escapes/symlinks. Preserve complete
raw arrays before derived scoring; retain native/adaptive partial failures,
model partial packets, worker records, append-only journals, access logs,
all terminal status rows, source/runtime provenance and process cleanup.
Missing output never implies a successful or nonexistent task.

The receipt lists the exact artifact census, relative paths, byte counts
and SHA-256 hashes. Source postflight or evidence verification failure
invalidates qualification; it never licenses changing old artifacts. Raw
results and later audit outputs remain outside Git. Git receives compact
hash-bound summaries and receipts. A sibling raw directory is durable local
storage, not an independently verified off-machine backup.

These commands describe the frozen workflow; `run` is allowed only after
the complete private freeze and exact-plan local review:

```bash
.venv/bin/python tools/run_fresh_preview.py prepare
.venv/bin/python tools/run_fresh_preview.py validate
.venv/bin/python tools/run_fresh_preview.py run --data-root /data2/neural-shrimp-data/raw --attempt attempt-001
.venv/bin/python tools/run_fresh_preview.py verify /data2/neural-shrimp-data/raw/EXP-013/attempt-001 --audit-output /data2/neural-shrimp-data/audits/EXP-013/replay-001
```

Preparation writes a new plan exclusively. Any pre-outcome implementation
change requires a regenerated plan and matching committed review; the old
plan hash is not reused. The read-only verifier writes only to a new external
audit directory. It reconstructs artifact/source inventories, stage/task
identities, baseline classifications and certificate gates/input identities, sensitivity/feature
identities, old-model fits, saved preview scores, all numerical gates,
selected-only winners, full readouts, bootstrap statistics, timing/cost
scope and final complete/qualified status. It does not integrate dynamics.

Independent post-run raw/identity and aggregate/cost audits are still
required before interpreting the outcome; same-implementation replay alone
does not supply those audits. Report actual host process visibility when
checking cleanup, and do not infer host absence from a restricted namespace.
Preserve every failed/incomplete attempt and numerical limitation in the
result and claim ledgers, regardless of whether learned or simple policies
look favorable.

The work remains on the privately pushed, locally reviewed
`codex/fresh-origin-study` branch. GitHub PR creation is pending usable
private-repository API authentication; SSH pushes work, but `gh` is not
authenticated and the available connector did not provide this repository's
PR access at the pre-freeze check. A private execution freeze is not a
main-branch merge and does not make the retained EXP-008 gate pass. Do not
merge around that gate or describe the uncreated PR as reviewed on GitHub.
