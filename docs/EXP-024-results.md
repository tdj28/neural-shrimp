# EXP-024: phase-dependent nonlinear input mixing

22 September 2026 local time. **Development and the numerically refined held-out
panel are complete.** Original attempts, failures and amendments remain intact.
The [compact summary](EXP-024-summary.json) binds source commits, receipts,
numerical checks, saved replay, independent arithmetic and figures.

**Human-friendly TLDR:** At some phases, the selected near-edge neural rhythm
does more than retain inputs: its internal trajectory combines two successive
inputs into a feature that a fixed linear readout can recover. This depends
strongly on when the inputs arrive and how large they are. The locally more
attracting core also contains a phase where finite inputs defeat the local
second-order prediction and its readout. Local stability, useful nonlinear
response and predictable finite-input behavior are distinct properties. These
results concern two fixed three-neuron networks; they do not show a unique
computational advantage of shrimp geometry or transfer to trained networks.

## Primary held-out result

At the prospectively selected near-edge, amplitude 0.01, early **state** window,
the sealed second-order reference readout achieved mean parity R-squared
**0.337901**, with a fixed phase-bootstrap 95% interval **[0.195261,0.490502]**
across 32 originally reserved phases. The descriptive affine oracle, computed
afterward from finite means, achieved0.352693. The former result is the primary
endpoint: its observer was constructed from sensitivities and saved before any
finite test trajectory, not fitted to those test means.

R-squared here measures reduction in noise-averaged squared error relative to
the constant zero parity predictor. It is **not classification accuracy**.
The primary noise model is iid Gaussian measurement noise with SD 0.0001 in
each sampled state coordinate. Initial phase and the external write clock are
known. The interval describes phase sampling at these fixed weights, not a
population of networks.

| Input amplitude | Near-edge early-state mean reference R² | Descriptive affine oracle R² |
|---:|---:|---:|
| 0.0001 | 0.0000859 | 0.0000859 |
| 0.001 | 0.090425 | 0.090532 |
| 0.01, primary | 0.337901 | 0.352693 |

The effect is very heterogeneous. At the primary amplitude,14/32 phases exceed
R² 0.05,12/32 exceed 0.5, and 7/32 exceed 0.9. The median is only0.00000375, and the
range is0 to 0.99533. A phase average should not be described as reliable
computation at every phase. Direct state interactions exceed ten times their
cross-method discrepancy in29/32 primary phases; this is a numerical diagnostic,
not a formal error bound.

![All held-out phases and input amplitudes](/data2/neural-shrimp-data/figures/EXP-024/heldout-001/phase-dependence.png)

## What this says about the mechanism

Balanced additive inputs have zero covariance with their parity. The direct
four-sign state contrast and positive performance of a **linear** state readout
therefore establish an interaction in the trajectory under this assay. A
nonlinear decoder or static tanh observation is not required for the primary
effect. The second-order sensitivity construction predicts most of the primary
affine recoverability, though not all of it.

The first-order and dynamical-cross-term-ablated state reference models have
zero parity covariance. Their zero readouts are mathematical controls, not
physical interventions on the ODE. Activation analyses separately retain the
static observation cross term and include exact tanh of additive reference
states. Removing a term from a response expansion does not prove that it can
be independently removed from a biological or trained network.

The median direct interaction scales approximately quadratically with input
amplitude, and predicted versus direct contrast norms generally agree over
many orders of magnitude. The figures retain conspicuous exceptions and label
numerically unresolved points. Norm agreement is weaker than vector agreement.
As a supplementary saved-array diagnostic, the median relative vector prediction
error among the 29 resolved primary phases is about 0.000940; the maximum is 0.1142.
This diagnostic does not replace the original full 32-phase endpoint.

![Direct and predicted interaction scaling](/data2/neural-shrimp-data/figures/EXP-024/heldout-001/contrast-scaling.png)

![Predicted versus direct interaction norms](/data2/neural-shrimp-data/figures/EXP-024/heldout-001/predicted-vs-direct.png)

## The core result changes the interpretation

The development panel suggested an almost inactive core. The full held-out
panel is more complicated: at amplitude 0.01, its early-state affine oracle
mean is 0.163030, but its sealed reference readout mean is −25404.70. The median
reference score is near zero, and six phases have negative scores. The mean
is dominated by a severe failure at **core--p21**, whose early-state reference
R² is about−812906 while its affine oracle is nearly1.

This does not mean the network has negative information or that all its core
phases are poor computational substrates. It means the fixed predictor can be
very wrong when the finite trajectory leaves the range accurately described
by the local expansion. We retain the extreme score: squared-error R-squared
has no lower bound. The prespecified paired mean difference and interval are
in the machine summary, but their enormous magnitude is dominated by this
failure and should not be advertised as a general near-edge advantage.

A post hoc algebraic diagnostic of the saved core--p21 early-state risk at 0.01
attributes roughly 483256 units to first-input leakage and 324276 to reference
centering error, with measurement-noise contribution about 0.00010. Thus the
catastrophic score is principally model/readout mismatch, not noise. No observer
was refitted and no case was removed. At the smaller amplitude 0.0001 the same
phase's later state readout reaches R²0.589, further ruling out a blanket
interpretation that this core phase cannot support readable interactions.

There is also a near-edge later-window exception: at 0.01 its reference mean
is 0.203598, interval [−0.161072,0.471243], while its oracle mean is 0.398086. One
phase scores−4.38175. The strong primary result does not establish uniformly
accurate second-order prediction at later times or every finite amplitude.

![Mean scores retain all negative cases](/data2/neural-shrimp-data/figures/EXP-024/heldout-001/parity-r2.png)

## Numerical qualification and retained failure

The first held-out attempt completed all 1668 jobs, but four analysis cells at
core--p21, amplitude 0.01 failed the frozen numerical comparisons. That attempt
remains **complete but unqualified**. Its saved-array replay passed; replay
agreement is distinct from numerical qualification.

The explicit [numerical amendment](EXP-024-refinement-protocol.md) froze and
privately pushed eight replacement solver histories: all four signs and both
methods for that one context/dose, with tighter tolerances and sixteen-fold
smaller step caps. Initial states, write directions, observer coefficients,
phases, endpoints and gates were unchanged. No original evidence was overwritten.

The resulting composite contains the original 1536 held-out solver-history slots
with exactly eight replacements, documented in a hash-bound provenance index.
All768 analysis cells now qualify. Maximum discrepancies are8.65e-9 in state,
2.38e-10 in contrast coordinates and 0.0001113 in R-squared, below the original
2e-7,1e-7 and 0.005 gates. The core approximation failure remains, so it cannot
be dismissed as the original solver discrepancy.

Composite saved-array replay and a separately implemented arithmetic audit
pass. The auxiliary arithmetic audit initially failed an inappropriate fixed
absolute1e-10 equality check on risks of millions; that failure is preserved.
With an explicitly recorded scale-aware arithmetic check, maximum normalized
score disagreement is3.41e-14. This correction did not relax any scientific
cross-solver threshold or change any outcome.

Development, held-out execution and refinement used four single-thread CPU
workers and approximately 236.5 seconds of compute wall time in total. All
registered owned workers are gone. There was no additional paid computation.
Raw trajectories remain outside Git; source, plans, receipts and this account
are privately checkpointed.

## Decision and remaining scientific questions

The original engineering continuation criterion passes for the primary cell.
Pro's first question now has a positive, scoped answer: recurrent state dynamics
can generate a linearly readable temporal interaction at these selected weights
and phases. This is progress beyond demonstrating recoverable memory with a
different decoder.

The next bounded extension is the recommended second-write separation T/2,
with the same principle of sealed reference observers and direct state controls.
Its protocol must explicitly distinguish the changed second-write phase from
changed input retention, preserve the finite-amplitude failures revealed here,
and use fresh execution bindings. The later portability test remains independently
initialized recurrent networks. EXP-023 cycle matching stays deferred.

We have not established basin switching, a specific bifurcation cause for the
core failure, process-noise robustness, unknown-phase decoding, a globally valid
phase-amplitude reduction, trained-network transfer or a computational advantage
unique to shrimp boundary geometry. Those require different measurements.
