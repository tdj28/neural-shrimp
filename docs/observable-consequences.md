# Where a dynamical structure becomes observable

**Status: 2026-09-11 conceptual framework and post-pilot exploratory design.**
The initial neural scan did not verify periodic cycles. None of the proposed
neural boundary, noise or utility effects below is a result of that pilot.

A verified attractor organization describes real behavior of the specified
update equations. The useful question is how much of that behavior survives
each measurement layer, and whether acting on it improves an actual decision.

## Four objects that should appear together in the paper

Let `s` contain every evolving variable, `lambda` the fixed controls during a
run, `F_lambda` the deterministic update, and `R(s;x)` the readout on input `x`:

\[
s_{t+1}=F_\lambda(s_t),\qquad
z_t(x)=R(s_t;x),\qquad
q_t=Q(z_t,y),\qquad
a=\pi(\text{available observations}).
\]

During full-batch gradient training, `s` contains weights and any optimizer
state, while learning rates belong to `lambda`. During deterministic recurrent
inference, the hidden state evolves and the weights are fixed. A state-space
projection, an initial-state basin slice and a control-parameter plane answer
different questions. Conventional shrimp organization belongs to the last of
these; its attracting orbits live in state space.

The first equation asks whether there is a stable periodic orbit. The second
asks whether its phases make distinguishable predictions. The third asks
whether those predictions matter for error, correctness, calibration or
latency. The last asks whether the information changes a useful action after
counting the cost of obtaining it. Evidence at one step does not automatically
establish the next.

## What must be measured

For a verified primitive period `p` with states `s_j`, define phase means and
spreads on a fixed evaluation population:

\[
\bar s=\frac1p\sum_j s_j,\qquad
A_s^2=\frac1p\sum_j\|s_j-\bar s\|^2,
\]

\[
\bar z(x)=\frac1p\sum_jR(s_j;x),\qquad
A_z^2=\mathbb E_x\frac1p\sum_j
\|R(s_j;x)-\bar z(x)\|^2.
\]

State coordinates and output units differ. Plot their spreads on separately
labeled axes; do not interpret their raw ratio as a dimensionless measure of
importance. Preserve coordinate definitions and any normalization. In a
finite sample, `A_z=0` means indistinguishability on the tested inputs and
precision, not all possible inputs.

Retain phase-resolved gradients, because optimizer dynamics may be evident
there even when loss is flat. If a state-space plot uses PCA or another
projection, preserve the full states for certification and label the
projection. Periodic loss, predictions or projected coordinates do not certify
full-state recurrence.

For squared prediction error, the exact decomposition is

\[
\mathbb E_{x,y}\frac1p\sum_j\|R(s_j;x)-y\|^2
=\mathbb E_{x,y}\|\bar z(x)-y\|^2 + A_z^2.
\]

Thus prediction averaging cannot be worse than average phase error under
this loss. It may remain biased, require more inference, lose to a selected
phase, or fail to beat ordinary equal-cost averaging. Averaging weights is a
different operation: a nonlinear network generally satisfies
`R(mean(s);x) != mean(R(s;x))`. Any correct phase-selection advantage must also
survive the cost and held-out evaluation of selecting that phase.

## A strong internal signal can be hidden by the readout

Three mechanisms are worth distinguishing rather than assuming away.

| Mechanism | Observable implication | Discriminating measurement |
|---|---|---|
| Exact parameter symmetry | Different parameter states implement the same function | Compare full states and predictions on a controlled input set; identify the symmetry explicitly |
| Readout-insensitive motion | A state change lies in a local readout Jacobian nullspace; finite changes can still have higher-order effects | Measure state displacement, output displacement and a local Jacobian approximation separately |
| Temporal averaging | Large phase oscillations cancel in the reported average | Plot phase traces, their mean and task error on the same input set |

For the tied scalar model `R(theta;x)=theta²x`, opposite parameter signs give
identical predictions for every input. That algebra does not claim that any
chosen sign-alternating trajectory is an attracting training orbit. An exact
sign-alternating orbit of this particular gradient map exists, but is
repelling; [analytic-cycle.md](analytic-cycle.md) gives the qualification.

The same model also has a genuine attracting period-two branch with distinct
phase predictions and a biased cycle mean. It provides a concrete
counterexample to both shortcuts: stable internal cycling need not be
prediction-invisible, and averaging a stable cycle need not recover the
correct prediction. These are exact statements about a constructed scalar
model, not experimental evidence about the tanh pilot or large models.

## The consequences most worth testing

| Hypothesis | Observable signature | Action it might justify | Alternative explanation to exclude |
|---|---|---|---|
| Phase error is wasting predictive quality | Phase prediction spread with a better cycle mean | Average predictions or use a validated stopping phase | Generic ensembling supplies the same gain |
| A poor attractor traps training or inference | Recurrence persists while task quality stalls | Change controls, restart or stop | A generic schedule/stagnation rule is equally effective |
| Useful interior settings are robust | Higher basin capture or lower failure under matched noise | Choose an interior operating setting | Better clean quality or stronger local contraction explains everything |
| A boundary predicts a useful escape direction | Equal-dose normal moves beat tangent/random moves on later quality or time-to-quality | Use a geometry-informed controller | Any favorable local performance direction would work |
| Near-neutral dynamics impose latency | Longer measured return or settling time tracks the local stability margin | Adjust damping or stop when a fixed readout is sufficient | Finite-horizon censoring or a difficult task produces the association |

These signatures may be absent. For example, a period-doubling bifurcation can
change state dynamics while a phase-averaged error remains smooth. A stable
incorrect answer can be reliable in the dynamical sense and useless for the
task. Conversely, a changing or sensitive state can support productive
computation. A controller must measure both helped cases and interrupted
productive cases.

## Causal contrasts and a useful null

Use a shared saved state to start each paired intervention, with common
inputs, opportunities, dose and resource allowance. Compare normal, tangent,
random and false-geometry moves in a frozen control-coordinate metric. Include
a zero-dose sham for instrumentation overhead. Test generic rate reduction,
stagnation, recurrence, local stability and ordinary hyperparameter search
with matched action and measurement budgets.

Geometry-specific benefit means that access to the boundary organization
improves the held-out decision beyond these cheaper or equally expensive
alternatives. Merely using a successful normal direction is insufficient if
it was selected using extra task outcomes. Charge the map-building queries
and freeze direction selection before held-out evaluation.

Include period and cycle amplitude in the trajectory-only diagnostic
baseline, along with recurrence and local stability. Geometry adds boundary
distance, width, direction and organization; it must not receive exclusive
access to ordinary periodic information. Keep final scoring data separate
from all online validation and tuning queries, so improved fit to queried
validation outcomes cannot stand in for held-out utility.

If a verified structure changes internal recurrence but all measured task
effects are explained by ordinary diagnostics, that is a useful null: the
model has interesting dynamics, and the simpler controller is sufficient in
the tested regime. The next-step protocol should stop claiming incremental
geometry utility at that point while preserving the dynamical result.

## Current conceptual figures and future measurement panels

The manuscript's conceptual figures are generated by
[build_concept_figures.py](../tools/build_concept_figures.py). They explain
different parts of the argument without adding empirical pilot outcomes:

1. **`conceptual-spaces`: controls → state → readout.** Three boxes identify
   which variables define the update, which evolve, and which are measured.
   The control box locates possible shrimp organization; the state box
   locates trajectories. The figure does not draw a measured shrimp,
   state-space orbit or initial-state basin slice.
2. **`conceptual-readout`: preserved versus hidden motion.** The exact
   constructed map `F(h)=-h`, initialized at `1`, alternates sign. Identity
   readout preserves the alternation while squared readout is constant.
   The cycle is neutral, not attracting; this is neither a trained neural
   example nor shrimp evidence.
3. **`analytic-training-cycle`: an attracting toy branch with visible
   consequences.** Exact scalar formulas show the two phase predictions,
   the three MSE readouts and the period multiplier as learning rate varies.
   The attracting interval is distinguished from the unstable algebraic
   continuation. The zero-MSE exact fixed-point solution is identified in
   the caption because zero cannot be plotted on the logarithmic MSE axis.
   This is the mathematical example derived in [analytic-cycle.md](analytic-cycle.md),
   with no new target training run.
4. **`conceptual-roadmap`: from observation to causal benefit.** Four stages
   connect verified periodicity, readout consequences, matched boundary
   interventions and a useful task decision. The annotations include
   alternatives when a stage fails to add information. It is a plan, not a
   completed result. The richer baseline contracts, including ordinary HPO,
   false geometry and generic averaging, are specified in
   [next-steps.md](next-steps.md).

Future empirical panels should align a verified control boundary with state
period/amplitude, phase predictions, gradients, task quality and noise
response on a common signed-distance axis. A separate state orbit and
initial-state basin slice could show how one fixed control setting produces
those outcomes. These are **future visual extensions**, not elements of the
current conceptual diagrams. Retain the necessary full states, readouts,
boundary estimates and paired control outcomes before generating such plots.
