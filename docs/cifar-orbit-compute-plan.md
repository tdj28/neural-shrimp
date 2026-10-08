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

## 2026-10-08 07:30 UTC amendment: diagnose the unresolved CPU linear solve

Status: source/controls prepared for independent review; **no target diagnostic
has run**. This is an exploratory local CPU amendment, not a paid launch or a
change to the orbit/spectrum criteria. Base integration is
`5e716f6a191e58022e17052277d4de90c44091c7`. The owned clean managed review
worktree is reused on branch `codex/cifar-linear-diagnostic`; other streams and
all historical evidence remain unchanged.

The corrector-only attempt in `docs/cifar-corrector-only-receipt.json`, source
`7de5dcfb`, stopped at its first GMRES solve with info4: restart50, four cycles,
rtol1e-10, and no accepted Newton update. Its closure4.9730550594621696e-6 fails
1e-8, while separation3.543241439719903e-4 passes1e-7. The saved receipt omits
inner residual history and the returned GMRES direction. Hence it does not
show whether a useful approximate direction was discarded, nor establish bad
conditioning or orbit nonexistence. The separate five-vector engineering
probe's0.8994-to0.8968 residual history cannot answer the200-vector question.

### Fixed diagnostic and merit rule

Let `R(x)=F_eta^2(x)-x`, `A=D(F_eta^2)(x)-I`. At the identical stored state,
repeat **one** unpreconditioned GMRES solve with the existing restart50,
maxiter4 and rtol1e-10; use explicit `callback_type='pr_norm'` so four means
restart cycles. Save every reported inner residual, operation count/timing and
returned direction p, even when the solver reports unsuccessful convergence.
Persist p before the independently evaluated true residual `e=A p+R`, so a
later deadline cannot discard it. Permit at most205 operator calls: the prior
200 inner products/four restart checks plus one independent final product.
No additional Krylov solve or budget increase is authorized.

Compute the true unpreconditioned forcing ratio `q=||e||/||R||`, and merit
`phi=||R||^2/2`, with directional derivative `d=R^T A p`. The declared inexact
forcing test is q<=0.5 with d<0. For an exact Jacobian the elementary bound
`R^T A p <= -(1-q)||R||^2` explains the descent test; floating output remains
numerical evidence, not a theorem about this target. Independently evaluate
all four fixed states `x+alpha p`, alpha=(1,1/2,1/4,1/8), each from the same
original x. A trial is an **inexact step candidate** only if finite, the true
forcing/descent tests pass, and
`phi(x+alpha p) <= phi(x) + 1e-4 alpha d`.
The absolute merit prevents a growing normalization denominator from masquerading
as improvement. Report relative step norm, absolute and normalized closure,
one-step separation, merit ratio, every failed trial and all full trial states.
Do not chain trials or automatically continue from a successful one.

Keep the closure screen1e-8 and primitive-separation screen1e-7 unchanged.
The existing corrector's stricter internal Newton stopping tolerance1e-11 is
also unchanged. **Every trial remains accepted_orbit=false**, even if those
screens pass. No stability, spectrum, Floquet Arnoldi, global continuation or
orbit-nonexistence claim follows. GMRES's internal Krylov iteration is the
existing linear solve, not a new Floquet eigenvalue computation.

This is standard inexact-Newton diagnostic logic, not a novel solver claim.
References: [SciPy GMRES callback/tolerance semantics](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.gmres.html)
and [Dembo, Eisenstat and Steihaug, Inexact Newton Methods (1982)](https://doi.org/10.1137/0719025).
No general convergence guarantee is asserted: the local theorem hypotheses
have not been established for this CIFAR target.

### Exact inputs, failure controls and resources

Consume only the previously prepared normalized subset-A reference bundle,
`/data2/neural-shrimp-data/explore/cifar/device-benchmark/reference-001`, and the
saved corrector-only attempt. Their manifest SHA256 values are fixed in code:
`414e8f2b1491a7f28cd1e1522376f7b008d7d3ac57bd29f69f0d5f4b953add38` and
`d02393831f0d0b207b06eec11f114fb798e459bfac6679f19dc5cd68ddf50dc8`.
State SHA256 is
`36066b4b1d657924bd6efdce83dc3fd2d06a7ab135ef78afb03fb87649dd9674`.
Eta0.00827753 and ridge0.01 retain their binary64 values; model/data shapes
and normalized x/y byte hashes must match. No archive loading, subsetB,
reserved seed or new dataset is needed. Before GMRES, replay initial closure/
separation and one existing gradient/HVP/J2VP reference-direction parity check.
Failure blocks the solve; the old and current source identities stay distinct.

The driver binds all executable source plus this plan/log to clean committed
HEAD, records the supplied exact independent-review commit, libraries and
SciPy GMRES implementation hash, and creates one exclusive output directory.
Retain direction, applied direction, true linear residual, baseline/trial
states and map images/residuals, fsynced history/events, all failures, runtime
and an exact output manifest. New files are capped at32MiB. No invalid/nonfinite
value can qualify a candidate. SIGTERM triggers best-effort manifest cleanup;
SIGKILL/storage/device failures can still truncate an attempt and must remain
visible. A review commit identifier records provenance; launch still requires
root's actual independent-review signoff.

CPU1, nice10; no GPU reservation, Astra, paid compute or paid review. At07:28UTC
load was3.54/3.84/3.04 and available RAM44GiB; recheck before execution. Earlier
CPU operation timing was about32ms per two-step JVP, and the comparable failed
corrector took10.115s total/8.770s correction. Estimate10–20s for this diagnostic,
not a new measurement. Bound internal time100s; external SIGTERM at115s plus5s
kill grace gives a120s hard ceiling. Transient GMRES basis is about40MB; historical
Torch process RSS was under1GiB, so budget2GiB RAM. No fallback/automatic retry.

Synthetic controls cover finite/invalid limits, explicit budget/deadline stops,
a failed GMRES solve with a useful partial direction, dishonest solver-success
and residual callbacks, nonlinear damping, absolute-merit versus normalized-
residual confounding, fixed-point/nonfinite rejection, immutable input and
source binding, retained-byte cap and direction persistence before true-residual
failure. Existing CIFAR CPU/device controls are rerun. Source commit, test
verdict and review must be reported before the target command is permitted.
