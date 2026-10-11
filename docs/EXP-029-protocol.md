# EXP-029: autonomous survey and shrimp focus maps

Prospective exploratory descriptive survey, 6 October 2026, Program C Track B
stage B1. No outcome of these maps existed when this design was written; the
EXP-004 focus rectangle and the reproduced shrimp inside it are prior results.
This is a private Git freeze, not a preregistration. The machine plan
`experiments/track-b/EXP-029-plan.json` is authoritative where more specific.

## Purpose and claim boundary

Track B asks whether shrimp geometry predicts memory and nonlinear capacity
beyond local dynamical quantities, using distant ordinary-periodic and chaotic
controls. That requires knowing, before any capacity measurement, where the
network has equilibria, simple and complex periodic orbits and chaotic
candidates, and how strongly its periodic orbits contract. EXP-029 measures
only the autonomous network (no inputs). It supplies:

1. a global survey of the (alpha, gamma) plane to locate distant ordinary
   periodic regimes, distant periodic windows and chaotic regions;
2. a high-resolution map of the shrimp focus rectangle for geometry features;
3. per-cell local quantities (screened period, finite-time exponent,
   dominant transverse Floquet modulus) that geometry must beat later.

Permitted statements are descriptive: screened classes, periods, finite-time
exponents and multipliers on the stated grids, starts, horizons and tolerances.
No capacity, computation, shrimp-organization or chaos-theorem claim follows.
Chaotic candidates remain finite-horizon labels.

The spatial split and the Track B sampling design (stage B1b) are written after
EXP-029 and frozen before any capacity outcome. Autonomous maps are covariates
of Track B, not its target outcomes.

## Network and maps

`dx/dt = -x + W tanh(x)`, `W = [[1.5, 2.9, alpha], [-1.5, 1.18, 0], [gamma, -22, 0.47]]`
(da Silva and Rech 2022). Rows index gamma and columns alpha; axes are
inclusive evenly spaced grids.

| Map | alpha | gamma | Grid | Starts |
|---|---|---|---|---|
| global | 0.0 to 1.5 | 5.0 to 16.0 | 256 x 256 | A and B at every cell |
| focus | 1.056 to 1.080 | 10.14 to 10.55 | 768 x 768 | A at every cell; B at every fourth row and column |

Start A is `(0.1, 0.1, 0.1)` and start B is `(0.2, -0.1, 0.3)`, as in EXP-025.
The alpha range includes the uncoupled limit `alpha = 0`, where neurons 1 and 2
form an oscillator driving a stable neuron 3; the gamma range extends about
five units on either side of the shrimp. These ranges were chosen from that
structural reasoning and the EXP-004 rectangle, not from new outcomes. The
planned total is 757,760 cell runs.

## Integration and labels

Classical RK4 in float64 on CPU, `dt = 0.01`, settling for 2,000 time units
and measuring for 1,000. During measurement every local maximum of `x3` is
recorded as a full-state event by cubic Hermite localization (up to the last
160 events). The screened period is the smallest `p <= 32` such that each of the
final `4p + 1` events agrees with the event `p` later within
`2e-6 + 2e-5 * max(|a|, |b|)`, as in EXP-004; it counts `x3` maxima per cycle.
The largest finite-time exponent uses one tangent, aligned for 100 time units
and averaged over the remaining 900; tangent seed 0.

Labels: diverged (nonfinite or above 1e6; not expected for this bounded
network); equilibrium (no event during measurement and final vector-field norm
below 1e-8); periodic (screened period at least 1); chaotic candidate (no
recurrence, not an equilibrium, exponent above 0.005 per time unit, the EXP-005
threshold); unresolved (all remaining cells).

For every periodic cell, the state at the last recorded event is integrated
with its 3 x 3 variational matrix for one screened cycle time (RK4, same step,
a final partial step for the remainder). The monodromy eigenvalues are saved;
the multiplier closest to 1 is the flow direction and the largest remaining
modulus is the dominant transverse multiplier. The analysis reports the
largest neutral-multiplier error and Floquet closure error as numerical
quality indicators. These are screening multipliers, not shooting
certificates.

## Analysis and audit

The analysis labels every computed cell, reports counts, period histograms,
transverse-modulus quantiles, start A/B agreement (same class, and same period
for periodic cells) and draws every start-A cell of both maps. It refuses an
incomplete attempt or existing outputs.

A separately written audit (no imports of the autonomous engine or analysis
labels) samples up to 8 computed cells per label per job with
`numpy.random.default_rng(2907 + job_index)`, re-integrates each with scipy's
adaptive DOP853 (`rtol 1e-10, atol 1e-12`) and solver event detection,
screens events with its own code, estimates the exponent from a separately
integrated variational system, recomputes the label, and for sampled periodic
cells recomputes the dominant transverse multiplier from the saved event state
and cycle time (agreement tolerance 1e-4). Disagreements are reported, not
repaired; chaotic or near-boundary cells may legitimately differ between
integrators.

## Resources and retention

The engine runs at about 0.4 microseconds per cell-step while settling and 1.1
while measuring, measured on random artificial weight matrices under shared
load. With 14 workers (one BLAS thread each) and chunks of 4,096 cells the
estimated wall time is about 3 hours; the hard ceiling is 6 hours. The
preflight refuses to start unless the worktree is clean, the tracked default
plan is used, HEAD equals the live remote head, every bound source matches, the
one-minute load average is at most 16 and at least 16 GiB is available. EXP-029
does not start while EXP-028 is running. No GPU, paid compute or Pro review.

Exclusive raw directory: `../neural-shrimp-data/raw/EXP-029/attempt-001`.
Chunks, assembled maps, runtime records, errors and a SHA-256 inventory are
kept. A failed or interrupted attempt is preserved; any retry uses a new
attempt directory under a dated amendment.
