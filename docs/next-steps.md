# Next experiments: turn dynamical structure into a useful decision

**Design update: 2026-09-11, after the exploratory pilots.** This document is a
proposal, not an executable freeze or preregistration. It changes the next-step
roadmap after inspecting EXP-001 through EXP-003; those attempts and their
outcomes remain unchanged. New target outcomes require a separately audited,
committed and pushed implementation, inputs, manifest, analysis and failure
rules before execution.

The central question is where a verified change in dynamics becomes visible:
**full state → predictions or decoded answers → task quality and resource use →
a decision to stay, average, stop or leave.** A strong internal signal is a
reason to measure this chain. It does not guarantee that every readout preserves
the signal, or that a visible signal improves a task. The measurement framework
is in [observable consequences](observable-consequences.md); the
[exact scalar derivation](analytic-cycle.md) already illustrates a real
prediction consequence without adding an empirical result.

## What the existing evidence permits

The coarse neural pilot has no verified periodic candidates. Its unresolved
bounded cells are neither certified cycles nor demonstrated chaos. The
recurrence controller never triggered, so that pilot supplies no intervention
benefit. The Hénon control qualifies part of the numerical instrument; its
periodic orbits do not establish neural shrimps. Longer horizons and another
model may change what a new search can resolve, but cannot change those results.

Use a control-parameter plane, such as two layer learning rates, to locate a
periodic window. The evolving weights are training **state**; an initial-state
slice describes basins. A trajectory does not enter a learning-rate plane, and
an island in a two-coordinate state projection is not a shrimp certificate.

## A. Establish an actual neural periodic regime

Two complementary small controls come first.

1. **Analytic instrument control.** Implement the tied depth-two scalar neural
   map with prediction `y_hat = theta² x`, datum `(x,y)=(1,1)` and half squared
   loss. Its gradient update is
   `F(theta) = (1+2 eta) theta - 2 eta theta³`. The fixed point flips at
   `eta=1/2`; the exact same-sign period-two branch is attracting for
   `1/2 < eta < (sqrt(5)-1)/2`. Test full-state closure, primitive period,
   amplitude, multiplier and predicted phase readouts against the algebra in
   [analytic-cycle.md](analytic-cycle.md). This is an instrument fixture and
   toy neural example, not a two-parameter shrimp or practical-model result.
2. **Published training reproduction.** Pin the paper, author implementation
   if available, model, data, loss convention, normalization, initialization,
   optimizer, precision and plotted parameter range for the small-model
   period-doubling setup in Kalra et al.; see [literature.md](literature.md).
   First write a source-to-implementation table that identifies missing
   settings and any necessary assumptions. Independently recover the reported
   dynamical regime before testing a new utility claim. A mismatch stays a
   reported reproduction discrepancy.

**Gate:** both the analytic numerical contract and an actual reproduced neural
periodic regime must be available before attributing proposed readout patterns
to the target dynamics. If reproduction fails, retain the failure and resolve
that bounded problem locally. Do not launch a broad geometry-utility study on
unverified loss oscillations.

## B. Map what changes, at four measurement levels

Prepare a separate tanh search informed by the disclosed pilot. Extend the
horizon and refine bounded high-rate regions while independently varying task
and initialization. Include a pinned reproduction of Sohl-Dickstein's
architecture, scaling and no-bias/no-ridge choices as a distinct arm; the
existing biased, regularized teacher-regression model remains its own arm.

For each candidate, retain the full evolving state, phase-resolved predictions
on fixed input splits, training gradients, recurrence residual, primitive
period and stability. Continue the orbit equation and saddle-node/flip
boundaries before calling the window a shrimp. Preserve multistability,
nonhyperbolic cases, failed correctors, divergence and unresolved transients.

Across a resolved boundary, use signed distance in a prespecified scaled
control metric and plot the following on aligned axes:

| Level | Measurements | What a change could mean |
|---|---|---|
| State | Primitive period, cycle amplitude, multiplier, transient length | The update law supports a different attractor or relaxation rate |
| Readout | Phase prediction spread, prediction bias, phase gradients, decoded-answer changes where applicable | The internal change reaches a measured model output or training quantity |
| Task | Held-out error/calibration, quality-target attainment, divergence, failure and latency | The visible change affects what the model is used for |
| Robustness | Recovery after state noise, basin capture, failures under control perturbations | The operating regime is accessible and survives specified disturbances |

Run both common-initialization scans and explicitly labeled continuation from
an existing attractor. Their difference can expose basin selection; mixing
them would obscure it. Near-neutral stability can produce slow convergence,
but neither every boundary nor every readout must show a discontinuity.

**Gate:** identify a reproducible state-to-readout consequence or a measured
robustness/latency consequence. A null readout result narrows the mechanism:
check symmetry, finite input coverage and averaging before claiming complete
invisibility. If the neural map has periodic windows without established
shrimp organization, continue as a periodic-dynamics study and say so.

## C. Intervene to distinguish mechanism from association

Observational boundary plots locate a contrast. They do not establish that
using geometry improves control. At a shared saved state and intervention
time, clone the run into all policy arms. Keep task, initialization, history,
compute allowance, evaluation schedule and disturbance draws paired.

### C1. Does cycle phase matter to prediction?

Compare a frozen single-phase rule, a uniformly sampled phase, prediction
averaging over one complete verified cycle, and weight averaging where its
meaning is well-defined. Also compare an ordinary snapshot ensemble or
averaging policy with the same number of predictions, training updates,
validation accesses and storage allowance. Freeze phase selection using
training/development data, never the test outcome.

For squared error, averaging predictions is guaranteed to beat or tie the
average phase error algebraically. That fact alone is not a research finding.
The substantive question is whether cycle organization supplies better
snapshots, stopping times or predictive quality than an equally funded generic
averager. Report ensemble inference cost separately from a single averaged
weight checkpoint. Do not transfer the squared-error guarantee to decoded
accuracy or calibration.

### C2. Does a boundary tell us where to intervene?

At each prespecified intervention opportunity compare equal-norm control moves
along the estimated boundary normal, along the tangent, and in random
directions. Use both normal signs when testing entry versus exit. Define the
norm in advance, for example Euclidean norm of scaled log learning rates; all
arms receive the same dose and admissible-control constraints.

Add a **zero-dose sham** with the same diagnostic overhead and a
**matched-norm false-geometry control**, such as a normal from a shuffled
window. A zero-dose sham cannot be norm-matched to a nonzero intervention;
these controls answer different questions. Compare return toward the original
cycle, escape, subsequent readout quality and time-to-quality. Record actual
boundary crossing and basin capture separately. Tangency does not guarantee
non-crossing at a finite dose, and crossing does not guarantee useful escape.

### C3. Does interior geometry predict robustness?

Test common distributions of state noise and control perturbations, with
noise scale, injection phase and evaluation horizon fixed beforehand. Compare
interior and near-boundary settings matched as closely as feasible on clean
quality and local contraction. Report remaining mismatch. Measure basin
capture, phase slips, return time, failure probability and quality loss. A
large state-space basin and a broad control-parameter window are different
robustness properties; measure both.

**Gate:** an intervention changes a task-relevant endpoint under the paired
controls. If it changes only recurrence, call that dynamical control. If it
helps only through generic averaging or a generic learning-rate reduction,
credit that mechanism rather than shrimp geometry.

## D. Test whether geometry earns its measurement cost

Freeze a new held-out comparison of the geometry policy with recurrence,
loss-stagnation, local-stability, fixed-schedule and ordinary hyperparameter
optimization policies. Match available actions, intervention opportunities,
validation access, tuning trials and total resource budget. Include an
equally expensive local performance map, since moving toward a good setting
may need no bifurcation information.

For the predictive comparison, give the trajectory-diagnostic baseline all
available generic periodic information, including estimated period, cycle
amplitude, recurrence residual and local stability. Add only boundary
distance, width, direction and window organization in the geometry arm.
Otherwise a gain from knowing the period could be misattributed to geometry.

Use held-out tasks, independent initializations and held-out windows. Freeze
transfer rules on development cases. If a target window needs a fresh local
map, charge that construction to its evaluation; do not call it zero-shot
transfer. Report cold-start costs and an explicitly stated reuse horizon for
amortized costs. Include CPU/GPU time, diagnostic work, map construction,
extra inference, storage, failed runs and validation queries.

The primary decision endpoint should be specified before new outcomes, such
as held-out quality at fixed total cost or time to a predeclared quality
target with failures retained as nonattainment. Success means geometry beats
the strongest matched comparator by a prespecified practically meaningful
margin, with the planned uncertainty analysis. A tie or loss supports using
the cheaper diagnostic in the tested setting. It does not make the underlying
structure unreal.

Reserve an untouched scoring split distinct from training, development and
any online validation used to choose interventions. Policies and tuning may
not query its labels or scores. Final quality and quality-target attainment
are evaluated on that split using the frozen analysis; the target is not
used as an online stopping signal through repeated scoring queries.

## E. Extend to latent reasoning when the cheap gates justify it

Pin a small open recurrent checkpoint and task population. Log all evolving
state, including secondary latents, momentum, caches or schedule phase, plus
decoded answers, correctness, residuals and timeouts. A repeated decoded answer
does not certify a hidden-state cycle.

Relaxation and momentum can change stability while preserving the fixed-point
equation, if defined appropriately on an augmented state. Feedback-gain
changes generally define a different solver and need separate attribution.
Test early stopping, rescue, incorrect termination and disruption of
productive transients against equal-cost residual, stagnation and restart
policies. Do not select only difficult or successfully rescued examples after
the intervention outcomes are known.

## Immediate deliverables and resource gate

The next implementation packet is local and bounded: scalar analytic fixture;
source-to-code reproduction table; full-state and phase-readout logging;
boundary-following specification; paired intervention protocol; result-free
figure contracts; and one complete CPU run-time measurement. It must be
reviewed locally and pushed before its new target attempts.

No paid GPU is justified solely by the existing pilot, and none is authorized
by this proposal. Follow [compute-plan.md](compute-plan.md): profile the exact
path, verify any accelerator parity, prepare measured base/slow-case costs and
a hard ceiling, and obtain explicit approval for that particular paid run.
Paid Pro review remains optional and requires separate milestone approval;
routine research and manuscript work use local audits.
