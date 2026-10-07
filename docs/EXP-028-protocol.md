# EXP-028: period-resolved trainability maps, existence pilot

Prospective exploratory pilot, 6 October 2026, Program C Track A stage A1.
No target outcome existed when this design was written. This is a private Git
freeze, not a formal preregistration or confirmatory study. The machine plan
`experiments/trainability/EXP-028-plan.json` is authoritative where it is more
specific than this prose.

## Question and claim boundary

In the plane of two per-layer learning rates for deterministic full-batch
gradient descent, do runs that neither converge nor diverge contain attracting
periodic orbits of the full parameter-space update map, and do equal-period
cells form connected domains that border chaotic candidates?

The state of each map is the complete parameter vector; the two learning rates
are fixed controls. Each grid cell starts from the same parameters and data.

The strongest permitted positive result: "In these specified networks, data,
initializations, horizons and tolerances, the learning-rate plane contains
numerically verified attracting primitive cycles of the training map within
connected equal-period candidate domains adjacent to chaotic candidates."
This would not establish shrimp boundary organization (stage A2), any training
benefit (stage A3), stochastic training behavior or behavior of large models.
The word "shrimp" is not used for any EXP-028 domain.

A null result means no such verified cycle under this pilot's grids and
horizons. It is not evidence that windows are absent elsewhere in the plane,
at finer resolution, at longer horizons or in other networks.

## Systems, data and initial parameters

| System | Network | Data | Loss | Parameters |
|---|---|---|---|---|
| S1 | One tanh hidden layer, width 8, input dimension 8, no biases, mean-field scaling `out = (1/8) w1 . tanh(sqrt(2/8) x W0)` as in the published Sohl-Dickstein notebook defaults | 72 inputs and targets, all standard normal, equal to the parameter count | Mean squared error | 72, all standard normal |
| S2 | One tanh hidden layer, width 3, input dimension 2, biases | 24 training examples from a random tanh teacher of width 3 (`make_teacher_regression`) | Half mean squared error plus ridge 1e-3 | 13, `TanhRegressor.initialize(seed + 100000, scale 0.2)` |

Seeds: S1 uses 28001 and 28002; S2 uses 28011 and 28012. Each (system, seed)
pair defines one map. S1 has as many examples as parameters and no continuous
symmetry, so interpolating minima are generically isolated; S2 uses ridge to
make minima isolated. Any multiplier with modulus within the stability margin
of one is reported as neutral or unresolved, never thresholded away.

## Learning-rate grids

For each map, compute the largest Gauss-Newton eigenvalue of the hidden and
output parameter blocks at the initial parameters, `lambda_h` and `lambda_o`.
Edge rates are `2 / lambda`. Each axis spans base-10 logarithms from
`log10(2 / lambda) - 1.5` to `+ 1.5` with 256 evenly spaced values, giving
65,536 coarse cells per map. These ranges use initial parameters and data only.

**Adaptive zoom, frozen rule.** After a coarse map completes, count in every
32 x 32 index window (stride 16 in both directions) the cells classified
periodic candidate with period at least 2 or chaotic candidate. Choose the
window with the largest count; ties go to the smallest row, then column. If the
largest count is zero, no zoom is run for that map and this is reported. The
zoom map has 256 x 256 cells spanning the logarithmic centers of the chosen
window's first and last rows and columns, an 8.2-fold refinement. Zoom cells
use the same initial parameters, data, horizons and rules.

## Iteration and screening

Every cell runs 20,000 full-batch updates in float64 on CPU with one BLAS
thread. A cell diverges when any parameter or the loss becomes nonfinite or a
parameter exceeds 1e8 in magnitude; it is then frozen. Full parameter states
from the final 384 updates are screened for the smallest period up to 64 at
which all of the last six blocks agree with the final block within
`1e-8 + 1e-6 * max(|a|, |b|)`. The largest finite-time Lyapunov exponent uses
one exact tangent per cell (Hessian-vector products), aligned for 1,000 updates
and averaged over the final 4,000; tangent seed 0.

Labels: divergent; fixed-point candidate (period 1); periodic candidate
(period 2 to 64); chaotic candidate (no recurrence and exponent above 5e-3 per
update); unresolved (otherwise, including slow convergence). The Sohl-Dickstein
convergence score is also computed from the first 100, 500 and 1,000 losses so
that the published binary map and these labels can be compared cell by cell.
Loss summaries (minimum, final, mean over the tail) are descriptive only.

## Certification and chaos diagnostics

For each coarse and zoom map, form 8-connected components of periodic
candidates with equal period at least 2. Components with at least 4 cells are
ordered by size (largest first), then by smallest row-major index; the first
16 are certified. The representative is the member nearest the component's
index-space centroid (ties to smallest row-major index), seeded from its final
state. `certify_cycle` solves `F^p(x) = x` near that state with the exact
Jacobian `I - P H`; tolerances: closure `atol 1e-9, rtol 1e-8`, root
tolerance 1e-10, stability margin 1e-7, candidate distance 1e-3 relative.
A component borders chaos if any member is 8-adjacent to a chaotic candidate.
For sanity, the largest 4 fixed-point components per map are certified with
period 1 by the same rule.

Chaos diagnostics: up to 16 chaotic candidates per map are drawn without
replacement by `numpy.random.default_rng(2810 + map_index)`, where coarse maps
have indices 0 to 3 and zoom maps 4 to 7 in plan order. Each is rerun from the
initial parameters for 30,000 updates with tangent seed 1, alignment 1,000 and
an exponent window of 10,000. Agreement requires both exponents above 5e-3 and
no recurrence in the rerun tail. These remain finite-time diagnostics.

## Decision and reporting

**Stage A1 gate.** Positive if at least one certified attracting primitive
cycle with period at least 3 represents a component that borders chaos.
Otherwise report the null within the stated scope. Either outcome is reported
with every map, label count, period histogram, component table, certificate,
failure, divergence, unresolved cell and chaos diagnostic. Figures show every
coarse and zoom cell. No cell, seed, map or failed certificate is removed.

The comparison with the published binary score is descriptive: it counts how
many cells the score calls convergent that these labels call periodic,
chaotic, unresolved or fixed, at each horizon.

## Numerical qualification, resources and retention

The batched engine matches the scalar reference implementation to 1e-13 on
synthetic inputs and reproduces analytic Hénon flip boundaries and the
published Hénon exponent through the same screening code. After the run, an
independent audit re-simulates 32 cells per coarse map with separately written
single-cell code, compares labels and periods, and checks that the
certification inputs are the saved final states. A label disagreement is
reported as a failed audit check and does not alter the saved labels; a
disagreement only in chaotic or unresolved trajectories after many updates is
expected from floating-point sensitivity and is reported separately.

Resources: 12 worker processes, one BLAS/OpenMP thread each, chunks of 1,024
cells. Throughput measured on an artificial fixture (seed 99999, outside this
plan) was about 17 microseconds per cell-update for S1 on one core, giving an
estimated 3 hours total. Hard wall ceiling 6 hours; on breach the runner stops
only its own workers and records a failed attempt. Expected peak memory is
below 8 GiB. No GPU, paid compute or Pro review. Live CPU, memory and GPU
checks precede launch; the preflight refuses to start, before any target
computation, if the one-minute load average exceeds 18 of the 32 logical CPUs
or less than 16 GiB of memory is available.

Exclusive raw directory: `../neural-shrimp-data/raw/EXP-028/attempt-001`.
Preserve every chunk, assembled map, zoom specification, certificate, chaos
rerun, error, exact source commit, plan hash, environment metadata and a
SHA-256 inventory. A failed or interrupted attempt is preserved and any retry
uses a new attempt directory under a dated amendment.
