# Next steps after EXP-008

## Current decision, 13 September 2026

The map helps the frozen ridge model predict candidate error in the observed
mean, but useful action control remains unestablished. EXP-008 lowers the
secondary all-candidate prediction MSE by 17.17%, while its primary paired
action benefit is 0.000857628 [-0.007701467, 0.006545434]. Direct
minimum-preview selection has 41.30% lower mean task error than the geometry
learner using the same eighteen forecasts. This is evidence against paying
for all previews and then crediting geometry merely for improving a regression
fit. It does not show that every use of the map is useless.

The next practical question is **can the map save preview calls while preserving
waveform quality?** A provisional next experiment would use **four previews
from the same eighteen actions**, with its shortlist chosen before any of
those previews. This is a research proposal, not an executed or frozen experiment. The numerical audit below must precede its target freeze.
The [raw-data audit](EXP-008-results-audit.md) and
[independent model audit](EXP-008-model-results-audit.md) explain why.

### 1. Establish trustworthy labels and stable fitting first

All required EXP-008 records complete, but numerical completion is not uniform
convergence. Across 3,504 coarse/fine pairs, the median absolute RMSE change is
2.91e-7 and the maximum is 0.476851. Selected actions are much steadier; that
does not certify unselected training labels or the finite-candidate oracle.
Freeze a separate refinement packet from the already observed discrepancies,
with a declared inclusion threshold, a complete prior-selected case census,
fixed comparison controls, tighter integration and unchanged waveform meaning.
Keep reference, phase and initial-state conventions explicit. Report this as
a follow-up on known difficult cases, not an independent representative test,
and preserve the original EXP-008 values and attempt unchanged.

Independent weighted SVD also exceeds the unchanged coefficient tolerance
for eight baseline and eight shuffled-model coefficients, despite matching
all penalties and selected actions. Future fitting should use a prospectively
specified stable QR/SVD implementation and a documented treatment of redundant
columns. Existing coefficients are not causal feature-importance estimates.
Use EXP-006--008 only as development evidence for any such changes; neither
clipping negative scores nor refitting and rescoring the same test cases can
be presented as a new prospective success.

### 2. Test a four-preview shortlist against equal-budget controls

Before new outcomes, freeze fresh origins/phases, the candidate set, shortlist
rules, training procedure, one primary contrast, missingness rules and cost
accounting. The provisional primary budget is four candidate previews of the
same duration. Compare a shortlist based on pre-preview features plus old-map
geometry with the same learner and pre-preview features without geometry.
Both receive the same coordinates, orbit period/Floquet diagnostics and local
sensitivity. Include strong simple four-action controls based on signed
coordinate axes and local/action coverage. The exact baseline set must be
chosen before outcomes, not selected from whichever control loses afterward.

**No feature from any candidate's new preview may enter shortlist selection.**
The existing EXP-008 learner cannot be reused unchanged because its inputs
include preview errors. Seal shortlists first, then evaluate their four
previews and apply the identical minimum-preview winner rule. Seal winners
before scoring later task outcomes. Separately retain the full-eighteen
minimum-preview reference and exhaustive outcomes for evaluation; their extra
information cannot enter the four-preview selectors. Keep all candidates,
failed conditions and duplicates in the evaluation record.

The primary endpoint should be paired later waveform error for geometry-four
versus the prospectively declared equal-budget baseline. A practical savings
claim additionally needs a predeclared acceptable quality loss against the
full-eighteen reference, set from a task requirement or development evidence
before held-out data. A secondary budget curve can include two, eight and
eighteen previews, if fully specified in advance. Do not choose a favorable
budget, tolerance or task horizon after outcomes.

Count actual preview calls, qualification, sensitivity, prediction, training,
map construction and amortization over decisions. The exhaustive oracle is
evaluation expense, not online information. Fewer previews alone do not prove
lower total cost if map or local-diagnostic overhead dominates. If geometry
fails to improve a same-budget shortlist or meet the declared quality/cost
criterion, retain the simpler forecast rule for this application.

### 3. Applications remain conditional on measured benefit

Only after a shortlist benefit survives a new held-out test should a sequential
disturbance experiment measure time within waveform tolerance, recovery time,
action effort and total cost. Then select one concrete trained recurrent-model
task with a frozen checkpoint, full evolving state and a genuine task endpoint.
The present autonomous three-neuron system does not establish training speed,
reasoning correctness, memory capacity or modern-model transfer.

The earlier continuation-checkpoint failure and separation of transients,
phase, frequency and waveform shape remain useful independent follow-ups.
The unreviewed multiple-memory preprint motivates a separate interference
hypothesis; it is not evidence that this shortlist mechanism works.

### Resources and release

EXP-008 collection and summary took 162.352 local seconds, including 23.770
sensitivity seconds and 0.079 fit/seal seconds; verification is additional.
The earlier map cost 140.54 seconds separately. Paid compute and review remain
zero. This agenda launches no new target execution, paid review or RunPod resource.
Any successor requires a complete local audit and prospective source/plan
freeze; paid reviews still require separate explicit human approval. Keep the repository private until its owner decides otherwise.

---

# Historical agenda after EXP-007; preserved below

## Decision after EXP-007, 11 September 2026

EXP-007 completed the proposed one-step hybrid test: 4,032 previews, 2,784
scored targets and 12 new qualified starts. All primary pooled intervals span
zero. Largest-dose hybrid error is lower than local in the mean, but its
interval spans zero too. Augmented search has lower mean error. Tangent and
coordinate policies make identical actions in every condition. The switch
must not be retuned and scored on these same anchors as a new success.

The shrimp's physical meaning is established in this network: repeating
activation waveforms, cycle durations, transverse contraction and stability
changes. The unresolved issue is incremental task value beyond ordinary
sensitivity and cheap action tests. A visible shape alone cannot answer it.

### 1. Test incremental predictive information before another controller

Treat both EXP-006 and EXP-007 as development evidence. Freeze new spatial
intervention blocks and phases, an identical candidate-action set for all
methods, and total cost accounting before any trajectories. Compare prediction
of later waveform/phase error using (a) local sensitivities, (b) short forecasts,
and (c) the same inputs plus period, transverse stability and boundary distance.
Primary target should be prediction and action regret on the same later task
window, with all candidate outcomes retained to define regret prospectively.

This directly asks whether geometry adds information. It avoids crediting a
geometric method for a coordinate action already available to the baseline.
Freeze the preview horizon and scoring horizon separately; the present example
shows that an excellent short-preview choice can have poor later output.
Report computation and map costs; no universal optimality claim.

### 2. Resolve what the signal error consists of

At prospectively fixed settings inside, near and outside the verified window,
separate immediate transient error, late waveform error, period/frequency
change and phase drift. Add fixed state-disturbance directions and phases.
A separate longer-settling measurement is essential: eventual nine-peak
retention in EXP-006 does not certify immediate fidelity. Independently refine
large step-sensitive cases and the old failed continuation checkpoint without
replacing or erasing the original evidence. Preserve unstable branches.

### 3. Conditional applications

Only if incremental prediction survives, freeze a sequential controller test:
time within waveform/phase tolerance, recovery time, action effort and total
cost under held-out disturbance streams. Test geometry as an admissibility
constraint alongside direct task control. Training/reasoning transfer remains
later work, with complete optimizer/latent state and an actual task endpoint.

The [Aguilera–De Martino preprint note](literature/2026-09-11-aguilera-demartino.md)
suggests a separate multiple-memory experiment: hold target, patterns, load and
coupling scale fixed while varying background spectral dispersion. Measure
crosstalk, waveform/frequency quality and retrieval from corrupted cues, not
only overlap magnitude. First clarify the response operator and inconsistent
printed update identities or obtain reproducible implementation details.
No memory-capacity replication, multi-memory shrimp or trained-model benefit
has been demonstrated here. This proposal is not an executable frozen plan.

### Resources and release

All work so far used local CPU and no paid reviews. EXP-007 collection and
summary took 137.51 seconds; verification is additional. The next bounded
protocol should again be validated locally. A RunPod job would require a
measured cost plan and explicit paid-compute authorization; none is launched.
Keep the repository private. Publication remains a human decision.

---

# Earlier waveform-utility agenda

## Historical agenda after EXP-006; preserved below

This section supersedes the historical EXP-004--005 agenda retained below. It is
a prospective agenda chosen after exploratory outcomes, not a frozen successor
experiment or spending authorization. Each new target packet must be complete,
locally audited, and pushed before its outcomes.

### What changed

EXP-006 measured an explicit task: preserve all three activation waveforms in
the immediate 0--300 interval after a connection-weight change, allowing one
constant phase offset. All 12 attempted anchors were eligible and all 4,320
trial-step records are complete. At radius 0.016, atlas RMSE is 0.301672 versus
0.466653 for the declared symmetric local-sensitivity comparator, with paired
benefit 0.164981 [0.013853, 0.326023] and 8/12 positive anchor effects. At both
smallest radii, local sensitivity wins at every anchor. The prespecified
across-radius local comparison is unresolved: 0.025738 [−0.027862, 0.083081].
Intervals describe variation across fixed selected anchors, not a population.

The atlas is not the best observed arm: `tangent_plus` has lower mean waveform
error at every dose, including 0.260795 versus 0.301672 at radius 0.016. Keep it
as an exploratory diagnostic now and include it prospectively as a competing
policy next time. Do not silently replace the frozen primary comparator with
an outcome-selected winner.

Eventual nine-peak retention and short waveform quality are different outcomes.
The former follows 10,000 settling and 2,000 measurement time units, whereas
waveform error covers the first 300. The atlas retains nine peaks in 188/192
long assays versus 308/384 across both local signs. Those counts do not replace
the waveform endpoint. The frozen `anchor-08` example favors both local signs,
although every shown arm eventually has nine peaks. Transients, phase, frequency,
and waveform shape have not been isolated as causes of immediate error.

See the [utility audit](EXP-006-utility-results-audit.md),
[continuation audit](EXP-006-continuation-results-audit.md), and
[raw receipt](../evidence/EXP-006/attempt-001/receipt.json). This fixed-network
task does not establish training or reasoning benefit.

### 1. Freeze a hybrid policy and stronger held-out comparisons

Test whether local sensitivity handles fine changes while geometric constraints
help larger changes. This hybrid is a new hypothesis prompted by EXP-006, not an
already demonstrated benefit. Fix the switch rule, constraint, actions, and
tolerances on development cases. Freeze untouched spatial blocks or anchor
settings and fresh phases for evaluation; retain all EXP-006 anchors as
development evidence rather than relabeling them as a new holdout population.

Compare the hybrid with the original local method, local methods given the same
sign/action-selection opportunities, a second-order or finite-probe sensitivity
method, and a directly measured task-quality map/search with equal total cost.
Add frozen `tangent_plus`, both tangent signs, and signed coordinate-axis controls.
Retain atlas-only, random-direction, and shuffled-geometry arms as diagnostics.
Select signs only from information permitted before the scored trajectory, never
from the better observed test outcome. Match native weight norms, observations,
decision times, action sets, tuning budgets, and all computational costs.

The frozen aggregate waveform-quality endpoint must beat the strongest relevant
baseline, with dose-specific effects and failures shown alongside it. Count atlas
construction, local/second-order sensitivity, probes, and failed trials. Report
cold-start and explicitly amortized costs, without free-map or unlimited-reuse
assumptions. Freeze exact counts, coverage, missingness, and cluster analysis
before generating new trajectories.

### 2. Test a coupled controller under held-out drift

Freeze a reference waveform, waveform/phase tolerances, and disturbance-stream
generator on development cases. Apply successive weight drift and state
disturbances so that each policy's actions change its subsequent observations.
All policies begin from common checkpoints and receive matched sensors and
action opportunities; evaluation streams and settings remain held out.

The proposed primary endpoint is the fraction of the evaluation horizon meeting
both frozen waveform and phase requirements, retaining all failures. Also report
recovery, control effort, cumulative error, and total computation. Compare with
equally funded tangent, coordinate-axis, local, second-order, and task-quality
controllers. This is the test needed to connect independent weight jumps to
sustained rhythm management. State changes at fixed weights and weight changes
remain separate disturbances; attraction does not remove neutral phase drift.

### 3. Resolve the failed checkpoint and refine crossing candidates

The four continuation branches retain 196 node records, including 122 unstable
records and five failed correctors. These are overlapping records with repeated
seeds, not distinct cycles. Geometric shooting agreement is 27/28. Near-edge
initial positive-gamma branch node 40 fails independent closure at 1.0591e-8
against the unchanged 1e-8 threshold; state and period agreement do not erase it.

Prepare a separately frozen local numerical successor using independently
controlled tolerances, resolution, and integration methods, while preserving
the original failed checkpoint. Refine saved crossing candidates and parameter
reversals; retain unstable segments and failed correctors. Adjacent arclength
nodes need not enclose a crossing's gamma at a fold. Do not claim exact boundary
intervals, complete two-parameter topology, or disappearance after correction
failure. Never relax the old threshold or reset the consumed attempt.

### 4. Transfer to one learned model only after beating the baseline

One concrete proposed transfer is a 16-state autonomous tanh network,
`dh/dt = -h + W tanh(h)`, with learned recurrent weights W and a learned three-output readout
`y = V tanh(h)`, trained on the target `(sin(t), sin(2t), sin(3t))`.
This is a proposed learned waveform model, not an existing result or a frozen
architecture. Pin its training samples, numerical solver, optimization protocol,
checkpoint selection, readout, and two learned recurrent-weight entries before
the transfer run. First verify its ordinary waveform quality and full-state
periodicity; do not assume successful training yields a useful cycle or a shrimp.
This supplies a direct bridge from handcrafted connectivity to learned weights.

Only after the hybrid passes the stronger fixed-network comparisons, test it
on that model's held-out disturbance streams. Measure full-state dynamics,
waveform quality, failures, and cost. Negative transfer remains a result. A
future fixed-answer reasoner requires another endpoint: a cycle desired for
rhythm generation can obstruct a solver. The earlier training/policy pilots
remain unchanged and supply no transfer benefit.

### Resource decision

EXP-006 used 423.866 seconds of local collection and analysis, excluding final
verification. Its 48 local-sensitivity solves consumed 26.627 seconds within
that total; the old atlas's 140.54 seconds is separate. No paid resources or
paid Pro review were used. Begin successors with bounded local plans. A later
GPU request requires profiling, numerical parity, and the specific cost and
approval procedure in [compute-plan.md](compute-plan.md). This agenda starts no
new experiment and supplies no spending authorization.

## Historical EXP-004--005 agenda, preserved unchanged

The remaining text records the earlier proposal. EXP-006 has now executed its
initial weight-drift and continuation stages; the current agenda above governs
the proposed successors. Statements below describing those stages as future
refer to the earlier research state.

Updated 2026-09-11 after EXP-004. The earlier plan is preserved in Git history.
This is an experiment agenda, not an executed utility result or authorization
for paid resources. Each target packet must be complete, locally audited, and
pushed before its outcomes.

## What is now established

EXP-004 evaluates 39,523 neural settings, including an independently initialized
201-point transect. It reproduces the published nine-peak shrimp-shaped window
in a fixed-weight three-neuron recurrent network. Independent shooting verifies
two nine-peak orbits and one eighteen-peak orbit. At the core, the full neuron
state and activation vector repeat every 37.63637538 time units; the dominant
transverse Floquet multiplier is 0.06506010. The nearby eighteen-peak orbit
repeats after 74.48169921 time units. Those are different network behaviors,
not graphical decorations.

The window is in the plane of two connection weights. A trajectory evolves in
three-dimensional state space; changing its state does not move its connection
weights into or out of a shrimp. Control changes and basin changes are different
interventions. The original grid-selected exterior becomes a nine-peak candidate
under both refinements. EXP-005 separately checked fixed controls using the
source paper's 10,000-unit settling horizon. Both distant comparison points
retained absent short periods and positive finite-time exponents at both steps.
After a 1e-6 state perturbation and 300 time units, the actual activation-vector
separation was 1.53e-6 at the core, versus 1.41 and 2.13 at the lower and upper
controls. This is a strong finite-horizon repeatability difference, not a task
quality or asymptotic-chaos theorem. The failed near-edge interpretation remains
in EXP-004; EXP-005 independently certified a nine-peak cycle there.

## 1. Measure whether the shape predicts tolerance to weight drift

**Question:** do broad bodies and narrow extensions differ in the connection
perturbations they tolerate, beyond knowing that their current orbit is stable?
This is the most direct potential use of the two-parameter geometry. A Floquet
multiplier concerns small changes of state at fixed weights; it does not measure
how far the weights can change before the attractor disappears or changes.

Continue the certified nine-peak orbit and locate its saddle-node and flip
boundaries. Retain failed continuation steps, unstable branches, and ambiguous
locations. Sample interior, extension, and boundary settings under a prospectively
specified rule, with multiple initial states and phases. Measure period, physical
cycle duration, neutral mode, transverse multipliers, basin capture, and control
margin. Distinguish branch stability from which branch a given initial state
reaches. A visual edge in a finite map is not an exact bifurcation boundary.

At those settings inject matched perturbations of the two weights, with the
control metric fixed before outcomes. Use the focus-rectangle widths for one
explicit dimensionless metric, and report perturbations in raw weight units too.
Measure retention of the nine-peak regime, waveform error after phase alignment,
phase drift, recovery time, and failure probability. Separately perturb states
at fixed weights to test orbital recovery. Do not infer weight robustness from
state contraction or a broad state basin from a wide parameter window.

**Decision criterion:** boundary distance or local window width must predict
weight-perturbation failure beyond period, amplitude, clean waveform quality,
and local Floquet multipliers on held-out settings. A null result is useful:
it says the expensive map adds little to simpler local diagnostics.

## 2. Test one concrete use: preserve a neural rhythm under drift

Treat the recurrent network as a rhythm generator with a fixed reference
activation waveform, specified allowable phase error, and a prescribed stream
of connection drift and state disturbances. Freeze the reference and tolerances
using development cases. Final disturbance streams and parameter settings are
held out. This defines a real measured control task, but remains a small-model
benchmark, not an established application in a trained modern network.

At shared saved checkpoints clone all policy arms. Compare a controller using
window geometry with an ordinary parameter search, a controller using complete
local recurrence/period/amplitude/stability diagnostics, and a sham with identical
diagnostic overhead. Match allowed weight changes, action opportunities, sensor
access, tuning budget, and total computation. Include random directions and a
shuffled-boundary control to test whether the direction supplied by the geometry
matters. A task-quality map with the same cost is an essential competing baseline.

**Primary endpoint:** fraction of the evaluation horizon satisfying the frozen
waveform and phase tolerances, counting every failure. Secondary endpoints are
recovery time, control effort, and total compute including atlas construction,
tuning, failed probes, and online diagnostics. Report the number of uses needed
to amortize mapping cost. Do not call improved recurrence alone useful control.

## 3. Determine whether the mechanism transfers to training or reasoning

The Sohl-Dickstein training map and the latent-reasoning basin map have different
evolving states and controls from this fixed-weight ODE. Pin one small published
model and reproduce its reported dynamics before trying to transfer a policy.
Include optimizer buffers, schedule phase, caches, or secondary latents in the
state when they evolve. A repeated scalar loss or decoded answer is insufficient.

Once a full-state periodic regime is verified, measure phase-resolved predictions,
held-out quality, correctness, settling time, and resource use. For a solver meant
to reach a fixed answer, a persistent cycle can be a failure mode; for a rhythm
generator it can be the desired behavior. Which interpretation applies is a
property of the task and readout, not the shrimp outline. Compare entry, exit,
averaging, and stopping only against equally funded ordinary methods. Preserve
all unresolved and divergent runs. The original training/policy pilots supply
no demonstrated benefit and remain unchanged.

## Resource decision

The actual EXP-004 grid completed locally with four CPU workers in 140.54 seconds.
The immediate continuation and controlled-perturbation work should therefore
start locally. A GPU is justified only by profiling the exact later workload,
checking numerical parity, and preparing a measured bounded cost plan under
[compute-plan.md](compute-plan.md). Paid execution and paid Pro review each need
their own explicit approval; none was used for the present numerical result.
