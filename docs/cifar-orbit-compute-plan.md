# CIFAR orbit continuation: bounded compute plan

**TLDR:** CIFAR has loss modulation, but no accepted orbit or crossing result
from the long legacy continuation. Before interpreting it, the corrected
runner must establish closure, a separated period-2 state and residual-qualified
Ritz pairs. Those pairs still describe an estimated partial spectrum; unseen
modes may change the stability interpretation.

This is an exploratory subset-A plan under the existing Phase 1 authorization.
Subset B and seeds 29000-29999 remain protected. No paid launch is implied by
this file. The root coordinator owns the paid-run plan, ledger and pod lifecycle;
all rates, proposed monetary ceilings and actual spend belong in the private
`cost-ledger.md` only.

## Corrected numerical gates

- Newton reports the residual of its returned state even after its last
  permitted update. A zero state has a defined normalization. Failed or
  nonfinite GMRES corrections are rejected.
- Every continuation point and bisection midpoint must pass the corrector and
  primitive-2 separation checks before invoking Arnoldi. A failed midpoint
  retains an unresolved bracket and never creates a crossing label.
- Arnoldi uses recorded seeded starts and checks each returned Ritz residual.
  Incomplete ARPACK convergence is unqualified. A small residual does not
  bound all eigenvalue errors in a nonnormal problem or prove spectral coverage.
- A numerical stability margin leaves near-unit values unresolved. A completed
  bracket gives only a fold/flip/Neimark-Sacker **candidate**, requiring matching
  dominant kinds at both ends. Transversality and nonlinear criticality are
  separate tests.
- Each new orbit attempt is exclusive. An fsynced append-only event journal
  records each Newton residual and each continuation/bisection result; accepted
  states have separate files. Atomic JSON checkpoints preserve progress if a
  later stage times out. Completed attempts have source/input hashes and a
  manifest. Abruptly killed attempts remain incomplete with their saved events.
- New defaults: one CPU thread, 600-second wall budget, eight Newton updates,
  GMRES restart 50 and four cycles, Arnoldi 100 iterations and 2,000 matvecs per
  search mode. The external launcher must add a process timeout as a backstop.

The original running checkout and its artifacts are untouched. Its unseeded
Arnoldi output, if any, lacks Ritz residual checks; its returned-state Newton
residual may be stale on exhaustion. Preserve it under those actual labels.
No new outcome repairs the old attempt retroactively.

## Measured representative CPU operations

The committed `tools/benchmark_cifar_orbit.py` measured the existing owned
state `sweep/state-lam0.01-eta0.00827753.npy`, subset A of 512 images,
3072-32-10 tanh model (98,666 parameters), lambda 0.01, float64, one thread,
nice 10. Source and exact input hashes are in the compact receipt below. The
benchmark makes no orbit or stability claim and never inspects the sealed
subset-B index list.

| Operation | Measured local cost |
|---|---:|
| Gradient, median of 3 | 5.268 ms |
| HVP, median of 3 | 17.842 ms |
| HVP, first measured call | 84.705 ms |
| Two-step JVP, cached first orbit point, median of 3 | 33.473 ms |
| One GMRES cycle, restart 5, six matvecs | 215.782 ms |

The five-vector GMRES cycle did not converge (`info=1`); its relative residual
fell only from 0.8994 to 0.8968. This is a convergence warning, not an estimate
of a complete corrector's iteration count. At measured local cost, 2,000
J2VP calls alone take about 67 seconds per Arnoldi mode; two modes would take
about 134 seconds before extra overhead. Eight Newton updates with four
50-vector cycles could add roughly 54 seconds of J2VP work, but neither this
arithmetic bound nor a larger GPU guarantees convergence.

Evidence: [operation timing receipt](cifar-operation-timing-receipt.json).
Fifteen focused tests cover analytic derivatives, dense-spectrum comparison,
seeded starts, false Ritz pairs, partial convergence, final-state residuals,
failed solver gates, budgets, bisection failure and incremental checkpointing.

The circle subcommand is explicitly disabled for this pilot. It requires a
separate review of end-to-end resource budgets before any fresh use. Orbit
inputs now bind the subset-A record, selected index hash and exact contiguous
normalized float64 x/y arrays (dtype, shape and byte hash). Corrected states
are fsynced and hash-bound before the first Arnoldi call, retaining a
spectrum-pending record after interruption. Invalid rate, step, tolerance,
margin or resource controls are rejected before creating an attempt.

## First H100 pilot, after local runner rehearsal

The earlier Phase 0 H100 measurement timed batched forward GD steps. It does
not measure HVP, GMRES orthogonalization, Arnoldi, or host/device transfers.
Do not apply its forward-step speedup to orbit continuation.

1. Prepare and locally rehearse a device-capable runner. The current SciPy
   wrapper calls CPU NumPy and the `Problem` owns CPU tensors; moving only the
   state to CUDA is insufficient. Keep data/state/HVP on the GPU and explicitly
   account for CPU Arnoldi/GMRES vector transfers, or qualify a native GPU
   Krylov implementation separately. Use the identical loss and float64.
2. Upload only subset-A tensors, the one saved state, deterministic tangent
   vectors and needed source. Bind all to hashes. Compare GPU gradient, HVP
   and two-step JVP against CPU reference outputs with disclosed tolerances
   before proceeding; failed numerical checks stop the scientific stage.
3. First time the actual H100 operations, including transfers and a bounded
   GMRES cycle. Record cold and warm costs, matvec counts, residual history
   and peak memory. This qualification stage has a five-minute process cap.
4. Only if qualification passes, try one corrector at the existing starting
   rate, with an eight-minute process cap and the bounded numerical settings
   above. If it fails, retain the reason and do not classify multipliers.
   If it succeeds, spend the remaining approved time on two seeded Arnoldi
   searches with explicit matvec caps. This initial pod is a feasibility pilot,
   not permission for a broad sweep.
5. Proposed initial pod lifetime ceiling: 20 minutes total, including startup,
   transfer and retrieval. The root coordinator must cost this at the current
   provider quote, reserve it within the remaining private Phase 1 ceiling,
   record exact task ownership, and install both process deadlines and pod
   teardown. Retrieve/hash-check every artifact, terminate in `finally`, then
   verify the task-owned pod is gone. No ownership action on other pods.

A successor continuation budget must use the measured H100 J2VP cost and the
observed Krylov convergence, not forward throughput or an assumed iteration
count. If Krylov convergence is the bottleneck, investigate preconditioning or
an alternate corrector before paying for more of the same work.


## Matched local device pilot, 8 October 2026 UTC

TLDR: device derivatives agree at the tested state and are faster locally;
this does not resolve the poor Krylov convergence or establish an orbit.
Source a3debab72195d5ac58bee5562cb59afd8b6466e8, subset A and same owned state,
float64, one CPU thread/nice10; source committed before all three attempts.
Two tiny dense Hessian/J2VP/complex-bridge oracle tests passed in1.89s.
CPU bit-identical to saved reference. CUDA maximum absolute gradient/HVP/J2VP
errors6.81e-17/5.90e-17/1.73e-18 pass atol1e-11+rtol1e-9*abs(reference).
CPU median gradient/HVP/J2VP5.14/16.34/31.75ms; RTX2000Ada2.09/5.81/11.09ms.
Bridge total averages32.11msCPU/11.44msCUDA, includingCUDA H2D.166ms and
D2H.251ms. SciPy remains on CPU. Six-matvec GMRES cycle197.19msCPU/74.42msGPU;
both info1, residual .899405 to .896782, no correction applied. PeakCUDA54.3MB.
Preparation1.327s, CPUprobe1.455s, CUDAprobe1.367s after imports; owned
PIDs4045355/4045636/4045871 exited, GPU empty afterward. All raw manifest
hashes and sizes checked. Exact records: docs/cifar-device-timing-receipt.json.
GPU output arrays were not retained in this timing-only pilot; CPU references
and parity metrics are retained, so independent GPU replay needs a fresh
explicit run. Three directions at one state are not general error bounds.

Immediate recommendation: root-owned CPU corrector-only, one existing state,
internal180s/external200s+5s grace, maxNewton8, GMRESrestart50/maxiter4,
unchanged GMRESrtol1e-10 and Newtontol1e-11; closure gate1e-8 and primitive-2
separation1e-7 unchanged. Zero Arnoldi. About1632 J2VP calls cost roughly52s
plus orthogonalization; failed GMRES stops unchanged. Preserve full returned
state/history even on failure. Root must commit mode/run card before launch.
Failure motivates solver inspection/preconditioning; H100 does not guarantee
convergence. Existing CPU path remains default; device benchmark is not a
complete device corrector. No paid hardware needed for this immediate check.
The H100 plan above is now a conditional fallback requiring separate review.


## Corrector-only CLI gate

`orbit --corrector-only` runs exactly one correction at the supplied binary64
rate, then exits. It cannot invoke multipliers, continuation or bisection.
Convergence and period-2 separation use the unchanged tolerances above;
success says `corrector_converged_separated`, never stable. Failure says
`corrector_not_accepted`; an internal timeout says `corrector_deadline` and
returns a nonzero process exit. Every accepted iterate and every freshly
computed residual has a full-state/hash/history checkpoint. If a deadline
occurs after a Newton update but before its residual evaluation, the latest
state survives with `residual_describes_saved_state=false` and a null current
residual. The older residual remains correctly scoped inside the history.

Forty focused tests passed, including successful/failed/collapsed correction
without multiplier calls, deadline-state preservation, and checkpoints before
residual evaluation. Root owns the following proposed run and resource check;
it has not been launched by this subagent. It requires independent mode review
before execution. Its output directory must be new and exclusive.
