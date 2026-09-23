# EXP-024 bounded development execution

22 September 2026 local time. Prospective exploratory development stage under
the [EXP-024 design](EXP-024-protocol.md). This document freezes only the eight
development phases; no held-out trajectory is authorized by this executable.

**Human-friendly TLDR:** The run will test whether two inputs interact inside
the neural state, rather than only becoming interpretable through a nonlinear
readout. It first checks that our calculations predict known synthetic cases
and agree with direct perturbations. Development results will guide the next
frozen test; they are not new held-out evidence or a shrimp-specific claim.

## Exact inputs and chronology

Reuse both globally qualified EXP-018 reference pairs, including their canonical
section state, period and terminal adjoint. The inherited `old_inputs()` audit
checks the original receipt, complete qualification census, summary, source
commit, plan and byte hashes before execution. The development machine plan
embeds these bindings; finite EXP-019/021 outcomes are not loaded.

Use NumPy Generator(PCG64), seed 2026092201, to draw 40 uniform fractions. Indices
0–7 are the development panel shared across settings. Indices 8–39 are reserved
held-out metadata: this runner never integrates or evaluates them. No reuse of
the prior 24 phase fractions as a new holdout.

Integrate each canonical section through T, then integrate the adjoint backward
from the saved terminal gradient along this new dense reference. At each new
phase, form P=I-zz^T/(z^T z), select its largest-norm column (lowest index on a
tie), and normalize that column to a unit write q. No phase-gradient repair is
performed. Use the primary dense initial state and primary q identically for
both solvers and both fixed writes at 0 and T. Directions never depend on a
perturbed trajectory, sign or amplitude. Compare independently reconstructed
phase states and gradients before using the common context.

Sensitivity state order is x,u1,u2,v11,v22,v12, each length three. The second
fixed state translation adds q to u2 at T, with every other sensitivity
continuous. All 64 samples occur after the second write, including the sample
at T. Windows, time-major flattening, observation units and analytic Gaussian
measurement noise follow the parent design.

## Closed task census

| Stage | Jobs | ODE segments per job | Scope |
|---|---:|---:|---|
| Fresh phase contexts | 4 | 2 | 2 settings × 2 methods; all 8 phases from each dense solution |
| First/second-order references | 32 | 2 | 16 contexts × 2 methods |
| Finite-difference checks | 68 | 2 | First two development phases per setting; 17 histories each |
| Finite factorial histories | 384 | 2 | 16 contexts × 3 amplitudes × 4 signs × 2 methods |
| Total maximum | 488 | 976 segments | No automatic retry or additional panel |

For finite differences, each of four contexts gets one zero-write trace and
eight traces at each of h=0.001 and h=0.0005: (+h,0), (-h,0), (0,+h), (0,-h)
and the four two-write sign combinations. Use centered differences for both
first derivatives, both self second derivatives and the mixed second derivative.
Save results at both h; do not select whichever step passes. This development
check is distinct from the experimental amplitude comparison.

After every context and reference qualifies, serialize and hash all 672
phase/dose/window/model observer definitions (16 contexts × 42), reload them,
and record the barrier before generating any finite-difference or factorial
trajectory. The 42 include 36 first/second/ablated state or activation observers
plus six exact static-tanh-of-additive-state controls. Each observer uses only
reference sensitivities. Full-trajectory factorial means are used solely for
evaluation and the explicitly labeled descriptive affine oracle.

If finite-difference qualification passes, time 32 primary trajectories:
both settings, phases 0–3, amplitude 0.01, all four signs. Then execute the
remaining 352 already declared jobs. This is a balanced timing batch, not a
selection or effect-dependent expansion. No test-set or network-population
inference is made from the development panel.

## Numerical configurations and qualification

Primary: DOP853, rtol 2e-11, atol 2e-13, maximum step T/512.
Independent: RK45, rtol 2e-12, atol 2e-14, maximum step T/2048.
Use float64, SciPy dense sampling, one numerical thread per owned worker, and
the exact NumPy/SciPy versions recorded in the machine plan. The reference
sensitivities and finite trajectories use the same declared method settings.

All thresholds below are prospective numerical acceptance gates, not scientific
effect-size definitions. Both source reference pairs must already qualify.

- Phase base closure and cross-method state error: maximum absolute coordinate
  error at most 2e-7. Cross-method phase-gradient error at most 1e-6;
  absolute z·f−1 at most 1e-7 and z·q at most 1e-12.
- Across every reference coefficient m,A,B,C, amplitude, window, observation and
  model: cross-method maximum absolute coordinate error at most 2e-7.
- Centered finite differences: maximum absolute error divided by
  max(1, maximum absolute reference derivative), at most 0.001 for u1/u2 and
  0.02 for v11/v22/v12. Both steps must pass. Report errors separately, including
  cancellation and approximation limits; no post-outcome tolerance relaxation.
- Finite trajectories: maximum absolute state error at most 2e-7, direct
  factorial contrast coordinate error at most 1e-7, and absolute difference in
  R-squared under the same sealed observer at most 0.005 for every comparator.
- Record contrast norm, predicted norm, prediction-error norm and the
  cross-method contrast-error norm. Do not call a very small contrast resolved
  solely because its absolute error passes a gate.

Any failed prerequisite blocks later stages and leaves their jobs explicitly
not started. An incomplete worker, failed solver, nonfinite output, corruption
or timeout is retained as a failure. No automatic retry in this development
attempt (a deliberate stricter choice than Pro's optional one-retry suggestion).
Fixing a defective plan requires a dated amendment and a new private freeze.
Full execution with some failed final numerical comparisons remains complete
but unqualified; all rows and failures stay visible. Negative R-squared is not
clipped. Missing results are not imputed as zero.

## Analysis and stopping

Evaluate all 192 development context × amplitude × observation × window cells.
For each report first-order, second-order and cross-term-ablated reference
observer scores, an affine oracle constructed from finite means, and interaction
prediction/numerical discrepancies. Activation cells also report the exact
static nonlinear observation control. Removing the dynamical cross term retains
the observation's cross term and all self terms. This mathematical ablation is
not an intervention on the physical ODE.

Use the whole panel to assess scaling, the known-phase observation assumptions,
reference/oracle gaps and computational cost. Report the prospective primary
cell descriptively on development phases. No bootstrap interval or usefulness
claim from this small development panel. No automatic execution or enlargement
of the 32 held-out phases. A complete held-out machine plan must be separately
reviewed, pushed and referenced before those outcomes exist.

## Ownership, resources, storage and replay

Use the existing tested `two_write_runtime.execute` owned-worker controller,
at most four children, per-job wall cap 180 seconds and total wall cap eight
hours. On timeout/interruption terminate only owned processes, preserve partial
payloads and journal terminal status for all scheduled tasks. Its cleanup
receipt checks process start identity and absence after joining. No RunPod or
additional paid Pro call is involved.

Check shared capacity immediately before launch. The inherited resource check
requires 16 GiB available RAM initially, reserves 8 GiB and limits owned RSS
to 12 GiB and raw storage to 8 GiB. Workers also refuse starts below 8 GiB
available RAM. Recorded capacity on this date was about 23 GiB available RAM;
that observation is not a reservation. Prior GPU timing/qualification does
not justify transferring this pipeline to CUDA.

Store new attempts exclusively under `/data2/neural-shrimp-data/raw/EXP-024/`.
Bind source commit, plan and every retained artifact by hash. The runner verifies
a clean checkout, the exact source/review/plan committed at the supplied HEAD,
and the same HEAD at the authorized private origin before target calculation.
Preserve every failed/incomplete attempt. After completion, replay all observer,
finite-difference and analysis arithmetic from saved arrays without new
integration, and verify artifact inventory hashes and worker cleanup. Compact
results and receipts go in Git; raw arrays stay outside it.
