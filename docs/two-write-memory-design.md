# Two-write memory under interference: revised exploratory design

17 September 2026. Design after the completed
[Astra Pro consultation and local adjudication](consults/2026-09-17-pro-adjudication.md).
**Under implementation; no new neural outcomes and no executable freeze yet.**
The final EXP-016 protocol, machine plan, tests, local review and privately
pushed source must all agree before execution. This document records the
scientific choices motivating that package, not an authorization barrier.

**Human-friendly TLDR:** The selected near-edge network remembers a small
disturbance longer than the core network. The next question is whether an
observer can recover that earlier input after a second input interferes with
it. We will also remove local timing or transverse components to see which
ones help that observer. Failure of one readout will not be called loss of all
memory, and two selected cycles cannot establish a shrimp-specific advantage.

## Question and fixed scope

Use the same fixed-weight three-neuron tanh system and the two previously
certified core/near-edge cycles. The primary estimand is the near-edge minus
core difference in first-bit balanced accuracy at the window ending at four
periods, under the identity intervention and an unknown balanced second bit.
This is the performance of a prespecified linear observer with known starting
phase, an external clock and independent Gaussian measurement noise. It is
not a capacity estimate, a trained-network result or a geometry-causal test.

Take 24 prospectively fixed phase fractions, shared across the two settings:
eight development phases followed by sixteen held-out phases. Use development
work for engineering measurement only; it cannot tune the dose, noise,
observer, primary window, gates or held-out sample count. No early stopping
for a favorable neural result. A fresh larger cohort requires another design.

At elapsed time zero write `s1 * epsilon * v`, where `v` is the unit local
phase-null direction from the reference state and adjoint, `epsilon=1e-4`, and
`s1` is -1, 0 or +1. Integrate to exactly one nominal period T. Relative to the
presealed reference at T, retain the entire deviation, its local timing
component, or its transverse component. Then add `s2 * epsilon * v`, with
`s2` again -1, 0 or +1. The second write follows the intervention. Both writes
use the same declared direction; its requested and realized values are saved.

Use `Q(T)=f(T) Z(T)^T`, `P(T)=I-Q(T)` without repairs. The reference flow is
evaluated at the actual nominal T endpoint; the adjoint's periodic-template
provenance and any normalization error remain explicit. Preserve both sides
of the event, replacement size, resulting displacement, oblique amplification,
phase leakage and nonidempotence. These are privileged state interventions,
not exact nonlinear isochrons or biological mechanisms. Identity executes the
same event pipeline. All zero-first-write, zero-second-write and reset-only
histories remain in the design.

## Observer and controls

Read 32 midpoint samples of all three activations over each cycle ending at
2T, 4T and 8T. Observation noise has fixed standard deviation `sigma=1e-4`
per activation sample. A short first-write window within the first eighth of
a period measures immediate detectability. Sample indices and all baselines
must be fixed in the machine plan.

Before any pulse trajectory, integrate and qualify full-state and all-nine-entry
variational references for every new setting/phase context. Predict the first
response A and second response B under the actual intervention schedule, with
the first response passing through the retention operator at T. For the
identity arm, seal a unit-norm observer proportional to
`(sigma^2 I + B B^T)^(-1) A`, oriented toward A. An undefined/zero-A observer
is explicit and cannot be silently rescued with held-out data.

Apply that same observer to second-present/absent and all retention arms.
Subtract only the presealed zero-write reference with the same event schedule;
the baseline never receives the unknown second-bit sign. Report all signed
noiseless margins and the analytic Gaussian sign probabilities, not sampled
noise-trial accuracies. Phase-conditioned observers and baselines all receive
the same phase/clock information.

Controls are the complete full-linear scheduled response, the existing
derivative-based clock observer, a separately presealed arm-specific linear
observer, and a one-mode approximation with the same second-write response B
and `A=rho*B`, where rho is the signed dominant real transverse multiplier.
The one-mode approximation matches second-write observability at that window;
it does not claim to reproduce the entire nonlinear network. Freeze extraction
of rho and any unsupported-complex-multiplier failure before target work.

The primary observer is a covariance/SNR construction, not Bayes-optimal for
binary interference. A synthetic one-mode four-level example must demonstrate
that poor linear accuracy can coexist with perfectly separable noiseless bit
pairs. The experiment consequently claims **linearly accessible first-bit
memory**, not all recoverable information. A nonlinear mixture oracle is
deferred rather than improvised after seeing an unfavorable result. The
arm-specific observer is secondary: component removal may reformat a signal
for the original observer without erasing it.

## Complete census and qualification

The complete crossed census contains
`2 settings * 24 phases * 3 first signs * 3 second signs * 3 interventions`
= **1,296 physical histories**: 432 development and 864 held out. Two native
resolutions for every history yield 2,592 traces. An independent DOP853 panel
containing all 27 histories at phase indices 0 and 8 for both settings adds
108 traces, for **2,700 traces**. Two full-state/tangent methods for all 48
contexts add 96 reference jobs. There are **2,796 planned jobs** before any
explicitly additional control computation; readout windows are repeated
measurements, not additional independent neural trajectories.

Use period-locked native RK4 steps T/32768 and T/65536, saving at T/4096.
Restart exactly at T rather than rounding it to an old fixed timestep.
Independent DOP853 uses the same requested sample/event clocks. Reference
checks use direct augmented DOP853 and RK45 integrations, with a fresh identity
tangent at T and no inversion of a collapsed fundamental matrix. Comparing
all transverse columns is embedded in the behavioral experiment.

The final protocol must encode full-trace state and activation agreement,
unsaturated signed-margin agreement at the primary noise scale, paired
accuracy agreement and reference/tangent agreement. Preserve the established
state/activation relevance scales of 1e-6/1e-7, standardized margin difference
.01 and accuracy difference .001; define the pulse-scaled operator comparison
explicitly rather than applying a unit-matrix tolerance without units.
All declared method pairs are retained. Saturated accuracy cannot waive a
trajectory failure. Reset-only zero-write drift must be checked separately
from the legitimate signal produced by a nonzero second write.

Any missing or failed case remains in the denominator. Qualified setting
averages require all their frozen prerequisites and histories; otherwise
withhold them and report conservative zero/one bounds with the failure census.
A paired bootstrap resamples phase blocks only and is descriptive conditional
on the two selected settings. Do not treat bits, arms, solvers or windows as
independent networks. Report effect magnitudes without inventing a minimum
useful accuracy threshold. Old EXP-013/014/015 failures remain unchanged.

## Execution and current implementation

Use up to four owned CPU workers, one numerical thread each. The prior local
425-trace acquisition took 178 seconds with four workers, but the new reference
and event work require their own measured timings; do not present an
unmeasured extrapolation as a guarantee. The final plan must freeze total and
per-task deadlines, retained failure behavior, RAM/raw-data ceilings, resource
checks, exact source/inputs, a new exclusive external attempt, reference seals
before pulse dispatch, and a saved-array verifier. Native and reference work
counts and storage costs must both appear in the ledger.

The tested GPU pipeline was slower and failed numerical agreement in EXP-012.
CPU is justified for this small stiff/sensitive system. Suitable future GPU
work remains authorized under [Skills.md](../Skills.md); no paid compute or
additional Pro request is part of this experiment.

The earlier [32-job independent tangent design](independent-tangent-qualification-next-design.md)
is preserved. Its numerical question is now addressed within this task;
standalone execution is deferred unless reference qualification exposes a
specific issue. Matched-geometry continuation, finite-dose sign reversals,
nonlinear bit mixing and trained-network transfer remain separate future
questions, not automatic expansions of this pilot.
