# Track B design draft: does window geometry predict computation?

Draft, 6 October 2026, Program C Track B stages B0 and B1. **Not frozen; no
capacity outcome exists.** This records the instrument and split design so it
can be reviewed before any capacity measurement. Executable protocols for B1
and B2 will be frozen separately.

## The measurement, in one paragraph

The da Silva and Rech network is driven by a random input stream injected into
neuron 1, `dx/dt = -x + W(alpha, gamma) tanh(x) + eps u(t) e1`, with `u(t)`
held constant for `tau` time units per input. Every run starts at a random
phase of the autonomous attractor and discards a washout. A linear readout is
trained by ridge regression on one input stream and evaluated on an
independent stream, so the readout never knows the phase. Capacities are
held-out measures of how well the readout reconstructs past inputs (linear
memory) and their degree-2 functions (nonlinear processing), in the sense of
Dambre et al. (2012). The same pipeline runs at every parameter cell.

## B0 instrument

| Quantity | Definition | Purpose |
|---|---|---|
| Linear memory capacity | Sum over delays `d = 1..D` of held-out `1 - NMSE` for target `u[k-d]`, each clipped at zero, raw values retained | Fading memory of inputs |
| Degree-2 capacity | Same for Legendre `P2(u[k-d])` and products `u[k-d1] u[k-d2]` | Nonlinear processing |
| Consistency | Correlation of feature responses to one stream from two independent starting phases after washout (Uchida et al. 2004) | Whether responses forget the starting phase (echo-state behavior) |
| Conditional Lyapunov exponent | Largest exponent of the driven tangent dynamics along the driven trajectory | Generalized synchronization; neutral phase gives about zero |
| Autonomous descriptors | Verified period, dominant transverse Floquet multiplier, finite-time exponent | Local dynamical competitors |
| Input gain | Linear response norm of the readout features to a unit input | Matched access across settings |

Features: all three `tanh` activations sampled at `V` equally spaced times in
each hold interval (time multiplexing, as in single-node delay reservoirs),
plus a constant. Inputs are i.i.d. uniform on `[-1, 1]`. Ridge penalty, `V`,
`tau`, `eps`, washout, stream lengths and delay ranges are fixed in the frozen
plan before outcomes, chosen from synthetic validation only.

Synthetic validation targets with known answers: a scalar linear leaky
integrator (analytic memory function), an exact delay line (capacity equals
its length), a known quadratic system (degree-2 capacity), and the
Dambre bound (total capacity cannot exceed the feature count for a fading-
memory system). An autonomous oscillator with a neutral phase is a negative
control for consistency.

## B1 maps and the spatial split

High-resolution period and Lyapunov maps of the autonomous network are drawn
first. They are dynamical outcomes, so their plan is frozen and pushed before
they run, but they are not capacity outcomes.

The (alpha, gamma) rectangle is divided into square blocks. Blocks are
assigned to development or held-out sets by a seeded permutation, stratified
by autonomous composition (fraction of verified periodic cells, of the main
shrimp's domain and of chaotic candidates), using only the autonomous maps.
The split is frozen and pushed before any capacity measurement. Held-out
blocks are never used for design decisions.

## The confirmatory comparison (B3, to be frozen later)

The question is incremental: on held-out blocks, does adding geometry features
(shrimp membership, distance to the shrimp boundary, boundary type, position in
the spiral organization) improve prediction of each capacity beyond a model
using only local quantities (dominant transverse multiplier, finite-time and
conditional exponents, period, input gain, consistency)? The same regression
family and fitting budget are used for both models, fitted on development
blocks only.

**Circularity.** Distance to a bifurcation boundary and the dominant multiplier
share a factor: the multiplier reaches modulus one at the boundary. Geometry
earns credit only for prediction beyond that local quantity. Consistency and
the conditional exponent may also share factors with memory capacity; they are
listed as competitors, never as evidence for geometry.

**Controls.** Distant ordinary periodic windows and distant chaotic regions
with explicit chaos diagnostics are included through the stratified blocks.
A capacity pattern that recurs in distant windows, or that local quantities
explain at the frozen practical threshold, is generic.

## Open design items before B1 freeze

- Bounds of the rectangle (the EXP-004 overview `[1.055, 1.110] x [10.1, 11.3]`
  contains the focused shrimp and distant structures).
- Map resolution and integration settings; GPU use requires a qualified
  backend.
- Exact geometry feature definitions, including how boundary type is
  determined without continuation at every cell.
