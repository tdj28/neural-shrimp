# EXP-028: period-resolved trainability maps, existence pilot

Prospective exploratory pilot, 6 October 2026, Program C Track A stage A1.
No target outcome existed when this design was written. This is a private Git
freeze, not a formal preregistration or confirmatory study. The machine plan
`experiments/trainability/EXP-028-plan.json` is authoritative where it is more
specific than this prose. The design was revised before freezing after an
adversarial review; see the [design audit](EXP-028-design-audit.md).

## Question and claim boundary

In the plane of two per-layer learning rates for deterministic full-batch
gradient descent, do runs that neither converge nor diverge include attracting
periodic orbits of the full parameter-space update map, organized as
equal-period domains **embedded in chaos** rather than ordinary
period-doubling bands next to a lower-period domain?

The state of each map is the complete parameter vector; the two learning rates
are fixed controls. Every cell of a map starts from the same parameters and
data.

Strongest permitted positive: "In the stated networks, data, initializations,
horizons and tolerances, the learning-rate plane contains numerically verified
attracting primitive cycles of the training map whose equal-period candidate
domains are surrounded mainly by rerun-confirmed chaotic candidates and border
no lower-divisor periodic or fixed-point cells." This would not establish shrimp
boundary organization (stage A2), any training benefit (stage A3), stochastic
training behavior or behavior of large models. The word "shrimp" is not used
for any EXP-028 domain.

A null means no such verified embedded cycle under these grids and horizons. It
is not evidence that windows are absent elsewhere, at finer resolution, at
longer horizons or in other networks.

## Systems, data and initial parameters

| System | Network | Data | Loss | Role |
|---|---|---|---|---|
| S1 | One tanh hidden layer, width 8, input dimension 8, no biases, mean-field scaling `out = (1/8) w1 . tanh(sqrt(2/8) x W0)` | 72 standard-normal inputs and targets (dataset size equals the 72 parameters); standard-normal weights | Mean squared error | Descriptive: comparison with the published binary score |
| S1R | S1 with identical data and initial parameters | same as S1 | Mean squared error plus ridge 1e-3 | Full pipeline |
| S2 | One tanh hidden layer, width 3, input dimension 2, biases | 24 training examples from a random width-3 tanh teacher (`make_teacher_regression`) | Half mean squared error plus ridge 1e-3 | Full pipeline |

S1 follows the default configuration of the published Sohl-Dickstein (2024)
notebook, pinned at commit `1074e4ed16e2762e9253707b9bf63352734f31c6` with file
SHA-256 `c1cd5aa2...6d74` (full values in the plan): width 8, depth 2, dataset
size equal to the parameter count, mean-field parameterization, tanh, mean
squared error, standard-normal data and weights, float64. The paper's figures
used width 16 and more updates than the notebook default.

Artificial-fixture probes before the freeze showed that ridge-free S1-type
training drifts toward infinite weights while the loss slowly decreases, so it
may have no finite attractors. S1 therefore keeps the published setting for a
descriptive map only; S1R adds the same ridge as S2 so that minima are finite
and isolated. Neither S1R nor S2 has a continuous parameter symmetry; any
multiplier within the stability margin of modulus one is reported as neutral or
unresolved, never thresholded away.

Seeds: S1 and S1R use 28001 and 28002 (the same data and initial parameters for
each seed); S2 uses 28011 and 28012. Six coarse maps are run.

## Learning-rate grids

For each map, compute the largest Gauss-Newton eigenvalue of the hidden and
output parameter blocks at the initial parameters, including ridge. Edge rates
are `2 / lambda`. Each axis spans base-10 logarithms from `log10(edge) - 1.5`
to `+ 1.5` with 256 evenly spaced values: 65,536 coarse cells per map. Rows
index the hidden-layer rate and columns the output-layer rate.

**Adaptive zoom, frozen rule (full-pipeline maps only).** After the coarse
maps complete, score every 32 x 32 index window (stride 16 in both
directions). The primary score counts periodic candidates with period at least 3
that have a chaotic 8-neighbor; the secondary score counts chaotic candidates
with an 8-neighbor that is neither chaotic nor divergent. The window with the
lexicographically largest (primary, secondary) wins; ties go to the smallest
row, then column. If both scores are zero everywhere, no zoom is run and this
is reported. The zoom map has 256 x 256 cells spanning the logarithmic centers
of the window's first and last rows and columns. Zoom maps have indices 6 to 9
in plan order.

## Iteration and screening

Every cell runs 20,000 full-batch updates in float64 on CPU with one BLAS
thread. A cell diverges when the loss or any parameter becomes nonfinite or a
parameter exceeds 1e8 in magnitude; it is then frozen. The full parameter
states from the final 384 updates are screened for the smallest period `p` up
to 64 such that every pair of states `p` updates apart across the whole tail
agrees within `1e-8 + 1e-6 * max(|a|, |b|)`. The largest finite-time Lyapunov
exponent uses one exact tangent per cell (Hessian-vector products), aligned for
1,000 updates and averaged over the final 4,000; tangent seed 0.

Labels: divergent; fixed-point candidate (screened period 1); periodic
candidate (period 2 to 64); chaotic candidate (no recurrence and exponent above
5e-3 per update); drift candidate (otherwise, with parameter-norm growth above
1% over the final 4,000 updates); unresolved (all remaining cells, including
slow convergence). A screened period is a candidate: a slowly converging orbit
can pass at a multiple of its true period, which certification checks.

The Sohl-Dickstein convergence score is computed from the first 100, 500 and
1,000 losses exactly as in the pinned notebook, so the published binary map and
these labels can be compared cell by cell. Loss summaries (minimum, mean over
the tail and the loss at the state before the final update) are descriptive.

## Certification

For each full-pipeline map (coarse and zoom), form 8-connected components of
equal-label cells with at least 4 cells: periodic candidates of equal period at
least 2, and fixed-point candidates. The exterior ring of a component is its
set of 8-neighbors outside it. A periodic component is an **embedded
candidate** when its period is at least 3, no ring cell is a fixed-point
candidate or a periodic candidate whose period is a proper divisor of the
component's period, and at least half of its ring cells are chaotic candidates.
Other periodic components are cascade-adjacent or otherwise non-embedded. In
each list, components are ordered by size (descending), then smallest
row-major member; up to 16 embedded, 8 other periodic and 4 fixed-point
components are certified. The representative is the member nearest the
component's index-space centroid (ties to smallest row-major index), seeded from
its saved final state and saved realized rates.

Certification uses damped Newton iteration on `F^p(x) = x` with the exact
Jacobian `M(x) - I`, where `M` is the ordered product of the exact map
Jacobians `I - P H` along the orbit. Acceptance requires: an undamped Newton
correction at most 1e-10 relative; normalized closure at most one with
`atol 1e-9, rtol 1e-8`; distance to the observed state at most 1e-3 relative;
and separation of the orbit from itself at every proper divisor of `p` of at
least 1e-6 relative. "Attracting" requires every multiplier modulus below
`1 - 1e-7`. The certificate also records the norm of `M`, the condition number
of `M - I` and the eigenvalue condition number of the dominant multiplier. On
artificial fixtures the exact-Jacobian and finite-difference spectral radii
agreed to between 1e-12 and 2e-9, well inside the 1e-7 stability margin.

## Ring confirmation and chaos diagnostics

For every certified attracting embedded candidate, every chaotic ring cell is
rerun individually from the initial parameters, in row-major order, up to 48
cells: 30,000 updates, tangent seed 1, alignment 1,000, exponent window 10,000,
same screening. A ring cell is confirmed when the rerun again labels it a
chaotic candidate. The **confirmed ring fraction** is the number of confirmed
chaotic ring cells divided by the number of all ring cells; cells not rerun or
whose rerun failed count as unconfirmed.

Separately, up to 16 chaotic candidates per map (coarse and zoom, including
descriptive maps) are drawn without replacement by
`numpy.random.default_rng(2810 + map_index)` and rerun with the same settings, to
describe the robustness of chaos labels. These remain finite-time diagnostics,
not proofs of chaos.

## Decision and reporting

**Stage A1 gate.** Producer-positive if at least one component satisfies all of:
embedded candidate; certified verified and attracting with primitive period at
least 3; no proper-divisor or fixed-point ring cells; confirmed ring fraction at
least 0.5. **The gate is positive only if the independent audit confirms at
least one such component on its own evidence** (below). Otherwise the result is
reported as "producer-positive but unconfirmed" or as the null within scope.

Mixed outcomes are reported per system and map with these labels:
"confirmed in S1R only", "confirmed in S2 only", "confirmed in both",
"confirmed only in zoom maps", or "not confirmed". A positive for one system
does not transfer to the other. Cascade-adjacent periodic components, other
certificates, all label counts, period histograms, drift and unresolved cells,
random chaos diagnostics, the published-score cross tabulation and every failure
are reported regardless of the gate. Figures show every cell of every map.

## Independent audit

After the run, a separately written audit (no imports of the batched engine,
orbit certifier or selection helpers) verifies the receipt inventory and plan
hash, regenerates inputs, recomputes the zoom rule and zoom axes, re-simulates
36 stratified cells per coarse map with single-cell code and compares labels,
recomputes each certified component's membership, ring and composition with
`scipy.ndimage`, recomputes rates from the saved axes, rechecks closure,
primitivity and the spectral radius with finite-difference Jacobians (agreement
within 1e-6 and radius below `1 - 1e-7` required for qualification), and
re-simulates every ring cell rerun for producer-qualifying components. Its own
gate is the confirmation required above. Saved labels are never altered by the
audit; disagreements in chaotic or drifting trajectories after many updates are
expected from floating-point sensitivity and are reported separately.

## Numerical qualification

The batched engine matches the scalar reference implementation to 1e-13 and
reproduces analytic Hénon flip boundaries and the published Hénon exponent
through the same screening code with the plan's tolerances. An analytic
gradient-descent positive control (a ridge-regularized two-parameter product
model) reproduces its flip curve for equal and unequal rates and its exact
multipliers through the batched engine and the certifier.

## Resources, failure handling and retention

Resources: 14 worker processes with one BLAS/OpenMP thread each, chunks of
1,024 cells, all chunks of a stage scheduled together. Measured on artificial
fixtures: about 17 to 25 microseconds per surviving cell-update for S1-type
maps and about 1.7 for S2-type maps, giving an estimated 3 to 4 hours of wall
time. Hard wall ceiling 8 hours. The preflight refuses to start, before any
target computation, unless the worktree is clean, the tracked default plan is
used, HEAD equals the live remote branch head, every bound source matches the
plan, the one-minute load average is at most 16 and at least 16 GiB of memory
is available. A dead worker, a ceiling breach or a termination signal stops the
run promptly, terminates only this run's workers and still writes the runtime
record and receipt. Certification and rerun failures are recorded per task. No
GPU, paid compute or Pro review.

Exclusive raw directory: `../neural-shrimp-data/raw/EXP-028/attempt-001`.
Preserve every chunk, assembled map, zoom specification, certificate, rerun,
error, exact source commit, plan hash, environment metadata and a SHA-256
inventory. The analysis refuses an incomplete attempt and never overwrites an
existing summary or figure directory. A failed or interrupted attempt is
preserved; any retry uses a new attempt directory under a dated amendment.
