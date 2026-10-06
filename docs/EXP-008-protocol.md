# EXP-008: does the shrimp map improve a fixed learned action selector?

Status: prospective private exploratory protocol, 13 September 2026. No
EXP-008 trajectory existed when this protocol was drafted. EXP-004--007 are
prior development evidence. The complete code, plan, tests and local audits
must be pushed before this experiment starts. This is not a registry
preregistration, publication approval, or authorization for paid resources.

## Claim and competing explanations

The question is whether supplying an old shrimp map improves *usable prediction
and action selection* for a frozen learner beyond local dynamics and identical
simulator forecasts. Both learned methods receive the same weights, destination
coordinates, nonlinear coordinate/action terms, orbit period, transverse
Floquet radius, full activation reference statistics, local sensitivity and
18 short forecasts. The geometry method adds six old-map features. A third
learner receives mismatched geometry as a representation-size control; direct
minimum-preview selection is an additional baseline.

Geometry is a deterministic representation of coordinates and an existing map.
Even a positive result establishes neither new information conditional on all
coordinates nor a shrimp-specific topological mechanism. It is conditional on
this ridge learner, these explicit columns, the sampled interventions, access
to the model and cloned states, and a map already paid for. A null/mixed result
limits this proposed use; it does not erase the measured neural rhythms or
prove every possible learner/application useless. No surprise or population
error-rate claim is planned. Both prior experiments informed this design.

## Network, samples and timing

Reuse the published three-neuron autonomous ODE and unchanged numerical gates
from EXP-004--007. Connection changes are measured in native Frobenius norm in
(alpha,gamma); normalized distances are for sample selection only. Geometry
lives in this parameter plane. Period and Floquet stability describe a neural
orbit in state space at fixed weights; pixel margins are not basin radii or
continued bifurcation boundaries.

Development comprises all 12 EXP-007 origins, with new phase contexts and a
fully observed candidate set. Test comprises 12 new origins, selected without
new trajectories from the core's eight-connected nine-peak component of the
old EXP-004 focus map. Require nearest observed non-component distance less
than distance to the crop edge. Exclude normalized distance <=0.04 from every
EXP-006/007 origin. Recursively partition the remaining eligible pixels along
the widest normalized axis into 12 compact groups; coordinate ties stay intact.
Choose the nearest pixel to each group centroid, with row/column tie breaks.
The selector and complete pixel membership/split history are frozen in the
manifest. No substitutions. These are groups of eligible pixels with holes,
not unseen rectangular maps, independent populations, or exclusions around
all old destinations/trajectories.

Each origin must retain nine peaks from the common initial state (0.1,0.1,0.1)
at dt=0.01 and 0.005 after burn 10000/sample 2000 and pass the unchanged
independent attracting primitive-orbit certificate. Preserve every failure.
Use indices 675 and 1350 of the 2700 endpoint-excluded certified samples:
quarter- and half-cycle states. A pre-outcome review replaced initially proposed
675/2025 because the odd ODE can make antipodal interventions redundant. The
chosen phases differ by a quarter cycle. They remain paired contexts, not
independent replicates.

Use radii 0.00025, 0.001, 0.004 and 0.016. Every method gets the same ordered
18 candidate entries: 16 uniform angles (including exact signed coordinate
axes), followed by the two signs of the minimum local-sensitivity direction.
Keep and charge duplicate directions. Geometry never changes candidate access.
Local sensitivity uses fixed initial state, the independent unperturbed
0--300 variational solve, and one phase projection over 75<t<=300.

Every candidate has a separately executed 75-unit dt=0.005 preview, sampled
at 0.1 from the same initial state. Forecast features use this whole preview
and its same-phase-fit windows. The task outcome is activation RMSE over
75<t<=300, scoring all three neurons equally against the original cycle with
one constant phase shift fitted only on that interval. No frequency refit,
time warp, or per-window alignment. Target traces restart from the same
initial state and run to 300 at both dt=0.01 and 0.005. The intervals use
different samples but are dynamically dependent. Zero-dose traces are also
required at both steps and phases. Total: 3456 previews, 6912 candidate target
records and 96 zero-dose records, or 7008 targets. Baseline assays are separate.

## Learned selectors and chronological seals

The complete algorithm and features are frozen before development outcomes.
Collect development previews and exhaustive targets, then fit three ridge
models: shared baseline columns; baseline plus real geometry; baseline plus
shuffled geometry. Train labels are fine-step tail RMSE. Use equal origin
weights, unpenalized intercept, and train-only weighted standardization. The
fixed lambda grid is [1e-6,1e-4,1e-2,1,100]. Select minimum equal-origin
leave-one-origin-out prediction MSE, fitting scalers separately in each fold;
exact ties use the first/smallest lambda. Refit all development data once.
No post-test tuning, feature pruning, clipping, or calibration.

All 51 baseline and six geometry column names are executable constants. Shared
inputs include squared/cross coordinate and action terms and crop membership
and clearance. The old-map features are anchor pixel margin, signed destination
pixel margin, core-component membership, outward projection, crop membership
and a crop-censor flag. Outside the observed crop, geometric values are masked
with explicit unknown/censor flags, not extrapolated as nonperiodic. A crop
flag is available to the baseline too. A mismatched-geometry control uses donor
origin (index+5) modulo 12, separately within development and test, matched by
phase, radius and candidate index. Donor local directions can differ; this is
a fixed structured mismatch, not a randomization test or an exact null law.

Seal the complete development model packet and its input hashes before any
test previews. Then collect all test previews, apply the locked packet and
seal all test predictions/actions before any test target. Each learner chooses
its minimum predicted error among all 18 candidates; exact ties choose the
lowest candidate index. Direct minimum-preview selection uses the same rule
on actual preview error. Scores are not clipped; regression extrapolation
outside [0,2] is retained, though measured activation RMSE is bounded.
Collect all 18 test outcomes, irrespective of selected actions or model failure.
Record model and decision seals and their chronological order in raw evidence.

## Endpoints, controls and missingness

One primary contrast: equal-block mean action-regret reduction from adding
geometry, pooled with equal phase and radius weights, at dt=0.005. Regret is
the selected action's true tail RMSE minus the minimum among the same 18
candidate outcomes. The shared oracle cancels, so primary benefit equals
baseline-selected true RMSE minus geometry-selected true RMSE. Positive favors
geometry. This finite-candidate hindsight oracle is not a global optimum.

Report all 12 block values, 10000 paired-block bootstrap percentile intervals
(seed 20260913), both integration steps with the *same* fine-preview/model
choices, per-dose results, absolute selected error and regret/oracle headroom.
Intervals describe variation in these selected blocks, not independent draws
from a population. The two controls, prediction MSE, CV curves, action identity,
all preview/target scores, numerical sham floors and costs are diagnostics.
No extra primary comparison is selected after outcomes.

Any required development failure, including coarse targets and zero shams,
invalidates the learner packet; no complete-case fit or fallback. Any missing
or failed required target across all 7008 records invalidates the primary,
including required coarse-step controls and shams. A missing candidate cannot
be omitted from the oracle. Incomplete previews invalidate model/choice inputs.
Keep stepwise descriptive panels where mathematically defined, explicitly
separate from an undefined primary. Preserve unsuccessful baselines, solver
payloads, model/prediction failures, duplicate candidates and all attempts.
No replacement anchors, resumed attempts or outcome-contingent extensions.

## Numerical audit, displays and resources

Exact hashes bind raw arrays, sources, model packet and sealed decisions.
Saved waveform metrics retain the existing strict score replay and separately
bounded phase-minimizer portability check. Correlated standardized ridge
columns and lambda down to 1e-6 can amplify BLAS rounding; model arithmetic
replay alone uses rtol=1e-7/atol=1e-9, prospectively fixed. Cohorts, columns,
selected lambda indices, action IDs, hashes and measured error tolerances stay
exact/unchanged. A repeated fit is one verification implementation, not an
independent scientific audit.

Freeze three new empirical figure families: (1) old period map with every
attempted development/test origin, test pixel groups and exclusions, and
geometry feature coverage; (2) all policies' dose means at both steps, pooled
contrast, all-block contrasts, oracle headroom and development CV curves;
(3) the first declared test origin, phase 675, radius 0.016, fine step: all
three activation waveforms and all-18 candidate preview versus later errors.
Keep this example even if it contradicts averages or is not evaluable.
Predictions, plotted arrays, full denominators, costs and receipts accompany
the figures. Layout may be improved without changing selected quantities.

Use the local float64 native CPU engine, four workers and exclusive attempt
directory; stream one candidate batch at a time. EXP-007 used 137.51 seconds
for 4032 previews/2784 targets/48 sensitivities before verification. This plan
has fewer previews, about 2.5 times as many targets, the same 48 sensitivities
and twice as many baselines; the conservative hard wall ceiling is 3600 seconds,
including verification. No neural target is used for prefreeze benchmarking.
Report elapsed collection, fit time, sensitivity time, summed overlapping call
times, the separate old-map cost and exhaustive evaluation cost. Every deployed
selector uses 18 previews plus baseline/sensitivity features; the extra oracle
targets are evaluation expense, not a free deployed forecast. No equal total
cost or net speedup follows without counting map construction and training.
No RunPod resource or paid review is authorized or launched.
