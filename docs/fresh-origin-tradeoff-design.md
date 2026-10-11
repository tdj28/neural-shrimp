# Draft EXP-013: fresh-origin preview loss versus budget

15 September 2026. Design proposal after EXP-011; **not an executable freeze
and no new outcome has been generated for this design**. EXP-012 is reserved
for the separate CPU/GPU engineering qualification. Implement, independently
review, test, commit and privately push a complete EXP-013 protocol, machine
plan and source before training final models or integrating any fresh origin.
The canonical integrity playbook §§2–7 governs that boundary. This proposal
does not modify any EXP-009/010/011 source or retroactively repair EXP-008.

## Recommendation and question

Run one bounded, explicitly exploratory comparison on **24 new physical
starting settings**, conditional on the already observed EXP-004 atlas.
Measure the complete 75-unit waveform-loss curve at budgets **2, 4, 8 and 18**
for unchanged shared/geometry ridge recipes and strong fixed coverage rules.
Keep four previews as the central comparison; show every budget and radius.
There is no invented acceptable-loss margin, superiority/noninferiority test,
pass/fail utility criterion or outcome-dependent switch between policies.
Numerical qualification thresholds below concern solver agreement only.

This is preferable to immediately redesigning the learner: EXP-011 already
established numerical feasibility, while fixed signed axes beat the geometry
learner in the pooled development mean. Local coverage won the three smaller
radii and geometry the largest, which motivates displaying all controls and
radii, not deploying that post hoc switching rule. Fresh losses can show
whether those descriptive relationships recur and how much loss each smaller
budget leaves relative to the full-eighteen minimum-error reference. They
cannot establish the original 300-unit action-utility claim, net deployment
speedup, a rigorous shrimp boundary, transfer to trained networks, or a
population-wide guarantee. The original EXP-008 strict Linux replay gate
remains failed and unchanged.

## Exact origin selection, completed without new dynamics

Use `select_information_anchors` in the frozen
`tools/hopfield_information_design.py`, with `block_count=24` and
`exclusion_radius=0.04`, on the complete EXP-004 attempt-001 focus grid and
core pixel **[53,60]**. Bind its receipt, manifest, `selected.json`, all 181
focus row files and every selection dependency in the new machine plan.
Do not use the old convenience loader's hard-coded twelve-origin census.

The exclusion set is the union of all origins in EXP-006, EXP-007 and
EXP-008 attempt-001 manifests, deduplicated by the exact binary64
`(alpha.hex(), gamma.hex())` pair: **36 physical origins**. EXP-008's twelve
old-development entries alias EXP-007, so aliases add no sample size. Bind:

| Prior manifest | SHA-256 |
|---|---|
| EXP-006 | `541b1fdbb0c551e89db0bdc47c3a4ee6a9f23d1857774bced4b3874b7e47cc3e` |
| EXP-007 | `573d817836d6c3271e685a45988be660743c104b47d96a5036414db4e6893ac7` |
| EXP-008 | `fb0e546947e0f36b77ff4978f60769db3541fc64330022bd664da22c20c321cd` |

Eligibility uses the old atlas's eight-connected nine-peak component through
that core. Its observed opposite-component pixel margin must be strictly
smaller than the nearest crop-edge distance. Distance from every excluded
origin, normalized by the entire focus rectangle's alpha/gamma widths, must
be **strictly greater than 0.04**. The 181×181 rectangle is alpha[1.056,1.08],
gamma[10.14,10.55]. Compute with saved binary64 coordinates; do not substitute
rounded decimal widths.

Partition all eligible pixels recursively along the widest normalized axis,
with alpha winning an equal-span tie. Keep coordinate ties intact; choose
the cut nearest the descendant-count-weighted median, with the lower cut
winning a tie. Select each leaf's pixel nearest its normalized centroid,
with row/column lexicographic ties. Preserve the complete eligible-pixel list,
leaf membership, centroids, splits and representatives in the machine plan.
Relabel the resulting entries `fresh-00` through `fresh-23` in leaf order;
physical groups use the system identifier plus exact parameter hex values.

The deterministic metadata-only calculation gives:

- 4,824 component pixels; 4,714 satisfy the crop-margin rule.
- The old-origin halos exclude 3,011 of those 4,714 pixels, leaving **1,703**.
- All 24 leaves are nonempty, with 65–86 pixels; sizes in order are
  `[72,67,75,73,67,69,65,86,69,71,77,70,74,73,67,66,69,71,69,68,68,76,67,74]`.
- The smallest selected-to-old normalized distance is
  `0.04006168083848473`; the smallest new-to-new distance is
  `0.028327886186623477`.

| ID | Grid row | Grid column | Alpha | Gamma |
|---|---:|---:|---:|---:|
| fresh-00 | 28 | 34 | 1.0605333333333333 | 10.203777777777779 |
| fresh-01 | 46 | 35 | 1.0606666666666666 | 10.244777777777779 |
| fresh-02 | 52 | 32 | 1.0602666666666667 | 10.258444444444445 |
| fresh-03 | 35 | 52 | 1.0629333333333333 | 10.219722222222224 |
| fresh-04 | 48 | 42 | 1.0616 | 10.249333333333334 |
| fresh-05 | 53 | 58 | 1.0637333333333334 | 10.260722222222222 |
| fresh-06 | 66 | 38 | 1.0610666666666668 | 10.290333333333335 |
| fresh-07 | 66 | 54 | 1.0632000000000001 | 10.290333333333335 |
| fresh-08 | 67 | 59 | 1.0638666666666667 | 10.292611111111112 |
| fresh-09 | 100 | 32 | 1.0602666666666667 | 10.367777777777778 |
| fresh-10 | 77 | 56 | 1.0634666666666668 | 10.31538888888889 |
| fresh-11 | 88 | 52 | 1.0629333333333333 | 10.340444444444445 |
| fresh-12 | 32 | 77 | 1.0662666666666667 | 10.21288888888889 |
| fresh-13 | 45 | 67 | 1.0649333333333333 | 10.242500000000001 |
| fresh-14 | 47 | 82 | 1.0669333333333333 | 10.247055555555557 |
| fresh-15 | 57 | 80 | 1.0666666666666667 | 10.269833333333334 |
| fresh-16 | 81 | 66 | 1.0648 | 10.3245 |
| fresh-17 | 63 | 82 | 1.0669333333333333 | 10.2835 |
| fresh-18 | 55 | 101 | 1.0694666666666668 | 10.26527777777778 |
| fresh-19 | 57 | 106 | 1.0701333333333334 | 10.269833333333334 |
| fresh-20 | 52 | 112 | 1.0709333333333333 | 10.258444444444445 |
| fresh-21 | 62 | 130 | 1.0733333333333335 | 10.281222222222222 |
| fresh-22 | 73 | 144 | 1.0752000000000002 | 10.306277777777778 |
| fresh-23 | 82 | 165 | 1.078 | 10.326777777777778 |

These are development-informed selections from an already observed numerical
period atlas, not draws from an unseen parameter population. The halos
exclude prior starting origins, **not every prior intervention destination**.
Blocks and their representatives can remain spatially dependent. No candidate
is excluded for being outside the atlas crop; preserve the existing geometry
coverage/censoring indicators. Do not shrink the halo, add origins, substitute
a neighboring pixel or select a different successful baseline after outcomes.
The 24-origin size matches the prior feasibility cohort and keeps complete
qualification affordable; it is not a power calculation or evidence of
independence. Equal-origin means describe these 24 fixed representatives.

## References, phases, candidates and endpoint

Attempt all 24 references using the unchanged baseline/certificate recipe:
initial state `[0.1,0.1,0.1]`; independent RK4 baseline steps 0.01 and 0.005;
burn 10,000 plus recording 2,000; event capacity 128, max peak period 32,
minimum repeats 4, recurrence atol 2e-6 and rtol 2e-5. Both runs must retain
nine peaks. Seed `hopfield_diagnostics.certify` from the fine run as in
`run_hopfield_drift.baseline`; retain every certificate gate, numerical
primitive-period and Floquet result. The certificate uses DOP853
rtol 1e-10/atol 1e-12, closure 1e-8, neutral tolerances 1e-6, flow alignment 0.99,
and transverse margin 1e-5. Require verified and attracting, and the expected
2,701-point orbit including its duplicate endpoint. These are numerical
certificates, not interval proofs. Failure leaves that planned origin present
and unavailable; no replacement or alternate initialization.

For each origin use phase indices **675 and 1,350** of the endpoint-excluded
2,700-point orbit. Initial state is that exact saved orbit state, and the
reference epoch is `phase*T/2700`. Reconstruct the fixed pre-preview sensitivity
at the unperturbed origin for elapsed 0..300 at 0.1 spacing, with parameter
sensitivity initially zero. Keep the existing DOP853 sensitivity solver at
rtol 1e-10, atol 1e-12 and max_step 0.1; retain its finite array/success checks. Use the existing
single-phase projection over times strictly greater than 75 to obtain the
local minimum-response eigenvector. Preserve its realized sign/direction
and all sensitivity states/metadata before any perturbed preview. This
300-unit unperturbed feature calculation is not a 300-unit perturbed target.

Use the unchanged 18 ordered candidate entries: exact-axis uniform 16 followed
by the realized local direction and its negative. Retain separately charged
candidate IDs even when physical directions coincide. Freeze all original
four native-weight radii: `0.00025,0.001,0.004,0.016`. Thus the full candidate
census is 24 origins × 2 phases × 4 radii × 18 entries = **3,456** labels in **192**
decision contexts. Add **48 zero-dose shams**, one per origin/phase, as
reference diagnostics; they are not action candidates or replacements.

Score the entire elapsed interval **[0,75]**, all 751 samples at 0.1 spacing,
using the unchanged `score_probe`/`waveform_errors`: tanh of all three states,
endpoint-excluded periodic reference, preserved epoch, one constant fitted
phase with grid 128 and the existing refinement. Do not fit frequency, trim
transients, change windows or independently rephase windows. Primary scalar
label is aggregate aligned RMS; retain the full score tree, synchronous and
window diagnostics. Shams' absolute errors are displayed as reference
checks, without introducing a new acceptable task-error threshold.

## Final training and prediction isolation

All 24 historical EXP-008 physical origins are development data. Their
3,456 labels qualified by EXP-010 train the final models; no new origin's
preview or target label enters training or tuning. Bind the metadata-only
EXP-010 qualification certificate and receipt-verified external label files.
Before fitting, use metadata-only preflight and hash source artifacts without
decoding fresh labels. Preserve the explicit pre-preview feature API and
exact ordered 37 shared fields plus six geometry fields from EXP-011.

Fit exactly **four final recipes**: shared and shared-plus-geometry schemas,
each at primary old-label step 0.00125 and mandatory sensitivity step 0.0025.
For each recipe tune only on the 24 historical origins using 24
leave-one-physical-origin-out folds. Each inner fit standardizes using only
its 23 training origins, equal-origin weights, an unpenalized intercept,
raw 75-unit RMSE labels, and the old penalty grid
`[1e-6,1e-4,0.01,1,100]`. Average validation MSE equally across origins;
smallest penalty wins an exact tie. Refit once on all 24 old origins.
This yields 96 inner folds plus four final grids; no fresh-origin nested
refitting is needed. Do not transform or clip labels/predictions, revise
features, tune a radius-specific model or choose a favorable resolution.

Reuse the frozen stable SVD and independently augmented QR primitives in
`preview_shortlist_model.py` through a new final-fit wrapper. Retain both
solutions for every penalty and every inner/final grid. Use the same score
agreement rule (atol 1e-9 plus rtol 1e-7 times maximum absolute score) and
require matching selected penalties. For each fresh context require exact
ordered top 2, top 4 and top 8 agreement between the two solvers; retain full
ranking differences and cutoff gaps as diagnostics. This expands the action
checks prospectively to the budgets actually used. Any failed numerical
check is retained and prevents a complete learned-policy result; no fallback.

Seal all four fitted model packets before fresh dynamics begin. After
baseline/sensitivity acquisition, seal all fresh features, all 18 predictions
per context, the learned shortlists and fixed controls before opening any
fresh perturbed-preview label. Distinct training and reveal APIs must audit
which IDs are read. Then for each policy/budget, reveal only its selected
candidate labels and seal its minimum-error action, ties by candidate ID,
before reading its full-eighteen evaluation reference. Reuse EXP-011's
chronology/identity checks; every action must be reproducible from its own
shortlist regardless of the acquisition order of raw traces. Qualification
and scoring can use a separate label-owning evaluator after those seals;
no code path may feed its data back into features, predictions or model fit.

## Fixed budget policies and readouts

The learned shared and geometry policies select the first b entries of their
sealed ascending predicted-score order, ties by candidate ID, for b=2,4,8.
Each then takes the minimum observed error among those b previews, again with
candidate-ID ties. The full 18 reference is the common b=18 endpoint.
Use these fixed controls, all declared before outcomes:

| Budget | Control | Candidate IDs |
|---:|---|---|
|2|Alpha pair|`[0,8]`|
|2|Gamma pair|`[4,12]`|
|2|Diagonal pair|`[2,10]`|
|2|Local pair|`[16,17]`|
|4|Signed axes|`[0,4,8,12]`|
|4|Diagonals|`[2,6,10,14]`|
|4|Local coverage|`[0,8,16,17]`|
|8|Uniform octagon|`[0,2,4,6,8,10,12,14]`|
|8|Local plus six angles|`[0,2,4,8,10,12,16,17]`|
|18|Full candidate reference|`[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]`|

These allow nested coverage curves (alpha/gamma→axes→octagon,
diagonal→diagonals→octagon, local→local coverage→local plus six angles)
without adding a policy selected by the results. Count duplicated fixed
endpoints once, and never average favorable controls into a composite.
There are 6 distinct policy definitions at budget 2, 5 at budget 4, 4 at budget 8
and one at budget 18: **16 readouts per context**, 3,072 per label resolution,
**6,144** across both resolutions. Coincidental agreement between learned
shortlists does not erase those separately declared methods.

For every policy show mean 75-unit selected RMSE and mean excess over the
same-context full 18 minimum, by all four radii and pooled. Average phases
and radii within each origin first, then give all 24 origins equal weight;
radius tables average the two phases per origin. Preserve every origin's
paired results, raw candidate score arrays, signed resolution deltas,
rankings, cutoff gaps and empirical score-change envelopes. Report exact
full 18 winner inclusion separately from error; close ties can alter rankings
without materially changing scalar loss. Empirical change envelopes are not
certified error bounds.

The central four-preview contrast table must include geometry-minus-shared,
geometry-minus-axes and geometry-minus-local-coverage RMSE, with negative
values favoring geometry. Use the same 20,000 whole-origin paired resamples
(seed 20260915) for all three contrasts, five strata (pooled plus four radii)
and two resolutions: 30 descriptive central 95% resampling intervals. Preserve
all signs, intervals and all 24 origin effects. These intervals summarize
stability within this spatially dependent, selected cohort; they are not
fresh-population confidence guarantees or confirmatory significance tests.
No utility acceptance decision follows from an interval crossing zero.

## Complete numerical qualification and exact workloads

Fresh labels have no saved coarse trace, so integrate **every candidate and
sham at 0.005,0.0025 and 0.00125**, without selecting difficult cases afterward.
The primary labels use 0.00125; every policy/aggregate is also evaluated at
0.0025. Match the initial state, destination, waveform convention and all 751
sample times exactly across methods.

For every candidate and sham retain the inherited native diagnostics:
`d1=abs(RMSE_0.005-RMSE_0.0025)`;
`d2=abs(RMSE_0.0025-RMSE_0.00125)`;
require `d2<=1e-4` and `d2<=d1`, plus whole-preview activation RMS difference
<=1e-3 and maximum component difference <=1e-2 between the two finest
traces. Report scalar and waveform failures separately, including a failed
non-increase test near roundoff. Do not relax thresholds or switch solvers
for a troublesome label. Ranking/action equality across label resolutions
is a diagnostic, not an added requirement for scalar qualification; retain
both complete policy readouts.

Add a metadata-only independent panel of **48 candidate cases**. For each
origin index i=0..23 and phase-list index p=0,1, choose radius index
`(i+2*p)%4` and candidate ID `(i+9*p)%18`. This gives two cases per origin,
one per phase, twelve per radius, and includes all 18 candidate IDs. Persist
the explicit 48 IDs in the machine plan before dynamics. Run each with both
DOP853 settings from EXP-010: rtol 1e-11/atol 1e-13/max_step 0.05 and
rtol 1e-12/atol 1e-14/max_step 0.025. Require both settings to agree with the
native finest trace and with one another under the same 1e-4 scalar and
1e-3 RMS/1e-2 maximum activation difference rules. Retain comparisons at a
common saved phase in addition to each trace's complete optimized score tree.
This panel is planned coverage, not an exhaustive independent solve of all
candidates and not selected from newly observed maxima.

| Work item | Count |
|---|---:|
|Baseline native runs|48|
|Numerical cycle certificates attempted after qualified baselines|Up to 24; all 24 planned|
|Unperturbed sensitivity contexts|48|
|Native candidate previews, 3,456 × 3 steps|10,368|
|Native shams, 48 × 3 steps|144|
|Independent preview traces, 48 × 2 settings|96|
|Total saved short traces required|**10,608**|
|Old-data final recipes / inner folds|4 / 96|
|Distinct fresh candidate labels / contexts|3,456 / 192|
|Policy-context-resolution readouts|6,144|

Required native step counts are 362,880,000 for candidates, 5,040,000 for
shams and 86,400,000 for baselines: **454,320,000 RK4 steps** total, plus
adaptive certificates, sensitivities and 96 independent preview solves.
This count excludes verification replay, which scores saved arrays and
replays models without reintegrating dynamics. It also excludes any
separately frozen EXP-012 engineering benchmark.

The full 24-origin result requires every reference, sensitivity, candidate,
sham, independent-panel trace and required model check. A failure, timeout,
missing record or failed numerical qualification retains its planned row and
makes the complete-cohort qualified comparison unavailable. Show the full
completion/failure ledger and individually inspectable diagnostics; do not
publish a complete-case mean as the planned result, replace an origin, omit
a difficult candidate from the full 18 reference or train on a successful
subset. Any further numerical study requires its own dated amendment and
freeze, with this attempt preserved.

## Compute scope, timing and fair costs

**15 September post-EXP-012 amendment:** all planned engineering batches
completed, but both numerical compatibility and deployment timing failed.
Therefore this proposal selects the existing native CPU backend for every
fresh native trace. Bind that choice in the still-pending executable freeze.
The conditional rule below was written before the engineering outcomes and
is retained to show how this choice was determined. See
[EXP-012 results](EXP-012-results-audit.md). No fresh dynamics or final model
fits preceded this amendment.

The exact scientific census is independent of the EXP-012 acceleration
outcome. **All 0.005 previews and all baseline runs remain CPU-only:** EXP-012
qualifies only 0.0025 and 0.00125. Use CUDA for those two finer preview steps
only if every EXP-012 parity/layout gate and its declared deployment timing
criterion passes; otherwise keep all steps on CPU. This choice must be bound
before fresh outcomes. EXP-013 must bind its selected backend and binary/source hashes,
precision, batch size, GPU ownership if used, and CPU reference comparison
rules **before fresh dynamics**. A GPU performance win alone does not qualify
its arithmetic. Avoid outcome-based CPU/GPU switching; any allowed fallback
must be frozen as a failure/availability rule and retain the failed attempt.
The independent adaptive solves and certificate/sensitivity code remain on
CPU. No paid resource is required or authorized by this proposal.

EXP-010's local 4-worker producer took 139.887663 seconds for 6,944 traces and
311,040,000 native steps, excluding its later saved-array replay. The new
native step count is 1.46065 times that reference, about 204.3 seconds by a
naive step scaling alone. **This is only a rough capacity check**, not a
prediction or total-runtime estimate: baseline kernels, adaptive work, I/O,
scoring and certificate solves differ. Allow a provisional 5–15 minute
producer range and a fixed 3,600-second guarded producer ceiling, with at
most 4 owned CPU workers and one numerical-library thread per worker; use
600 seconds per baseline/certificate/sensitivity task and 120 seconds per
short trace. No task or numerical failure increases those ceilings.

Reserve at least 8 GiB available RAM and 2 GiB external disk headroom at launch.
EXP-010 retained 219.45 MB for 6,944 traces; the new trace count is 1.52765 times
as large (roughly 335 MB by that historical byte ratio), before baselines,
sensitivities, models, summaries and audits. The 2 GiB reserve is a conservative
stop-before-launch capacity requirement, not permission to silently truncate
raw data. Root's measured current-resource/EXP-012 report must confirm that
these ceilings are sensible before the executable freeze.

Record separate wall and CPU timing for baseline acquisition/certification,
sensitivity, geometry/input construction, old-data tuning/final fits,
fresh-feature creation, prediction/shortlist sealing, integration, scoring,
verification and receipt sealing. Record actual parallel stage wall times
and per-task durations separately; summed task wall time is not elapsed
latency or CPU process time. Keep complete source/runtime/invocation and
resource receipts.

Report exact nominal preview counts 2/4/8/18 alongside observed loss, and
separate:

1. Scientific acquisition expense: the full 18 census at three steps, shams,
   independent checks, verification and all failed attempts.
2. A transparent deployment accounting table: selected candidate trace/score
   costs plus each policy's prerequisites. Axes/diagonals require the common
   reference; local controls additionally require sensitivity. Shared learners
   require reference, sensitivity, feature construction and training; geometry
   additionally requires the atlas. Full 18 includes candidates 16/17 and also
   requires sensitivity. Duplicated physical candidate entries stay charged.
3. Once-per-model versus once-per-origin/phase versus once-per-radius costs,
   with amortization shown at the fixed horizons 1, 8 and 192 decisions. Saved
   atlas acquisition and historical training-label creation are reused assets,
   never measured as zero-cost creation. Bind their historical receipts and
   keep any unmeasured current-hardware replacement cost explicitly unknown.

Selecting b entries from a batched full 18 timing is accounting, **not a
measured b-preview end-to-end latency**. EXP-012 throughput likewise does not
prove an EXP-013 policy speedup. Do not claim net deployment savings unless a
separately frozen direct timing comparison measures the same full prerequisite
and execution scope. The present study's defensible primary display is
loss versus charged preview budget, with honest measured-stage cost tables.

## Implementation and review gates

Implementation checkpoint: `tools/preview_final_model.py` now supplies the
pure final-old-data fit and label-free prediction interface, with 34 synthetic
and failure-path tests. It is an independently reviewed component only; the
complete study has no executable machine-plan/runner freeze yet. No production
fit or fresh outcome has been generated by these tests.

Create new EXP-013 files; import unchanged, hash-bound primitives without
editing prior frozen sources. Suggested components are a pure selector/plan
builder; a label-free fresh feature adapter; a final-fit wrapper around the
EXP-011 ridge primitives; and a staged runner/evaluator using the bounded
EXP-009 ownership scheduler. Keep every attempt exclusive at
`/data2/neural-shrimp-data/raw/EXP-013/attempt-001` (configurable only to an
external data root), with no resume, exact expected counts, append-only
journals, terminal failure rows, partial raw payloads and complete hashes.

Preflight must independently reconstruct all 24 origins/36 exclusions,
48 sentinel IDs, candidate/context/readout counts and every dependency;
prove the live private remote HEAD equals the clean committed source;
reject untracked source and input mismatches; and verify correct task/method/
raw-path mapping. Synthetic tests must cover blocked-origin exclusion,
aliased groups, no replacement, training-only scalers, exact policy sets,
future-label access rejection, winner-before-reference seals, failed-trace
retention, source/plan tampering, missing summary/record and owned-worker
cleanup after interruption/deadline. Postflight must verify all saved score
fields, arrays, model packets, access chronology, timing scope, artifact
census/hashes and historical host process cleanup. Independent raw/aggregate
audits are required before interpretation. Final sealing/auditing is timed
separately from the guarded producer so an expired attempt remains preserved.

A useful outcome is a complete and numerically interpretable comparison,
even if simpler controls dominate. If current learners consistently add loss
or cost, retain simple controls for this model and move any redesign back to
development. If learned policies reduce loss across fresh representatives,
retain that as exploratory transfer evidence and obtain a real task's quality
requirement before a utility confirmation. A radius-specific pattern remains
a hypothesis for another freeze; a numerically incomplete cohort first needs
a preserved refinement study. None of these branches licenses changing this
study after its target outcomes are visible.
