# CIFAR fixed restart sensitivity: implementation and prospective run card

8 October 2026 UTC. Root approved source implementation and synthetic controls
under design c891a74c7b85e37fb223b8f73490927b9078a76e. **Target launch remains
unauthorized until independent source review and a separate root instruction.**
No target/profile/model/gradient/HVP/JVP or new scientific solver call was made
while preparing this implementation. Tests use tiny algebraic fake operators,
synthetic arrays and owned subprocesses only.

## Frozen scientific settings

The decision file and JSON retain the exact inputs and original comparison.
The new tools/diagnose_cifar_restart.py uses the reviewed direction/trial core.
Only an opt-in explicit_zero_start argument was added to that core; its old
restart50/four-cycle settings, omitted-x0 behavior and scientific gates remain
unchanged. The new driver explicitly supplies zero x0, restart200/maxiter1,
rtol1e-10, atol0, no preconditioner and callback_type=pr_norm. A normal full
cycle uses200 inner actions, one cycle-end action and one independent action.
All attempts, including operations that throw, consume their caps:

| Operation | Maximum attempted calls |
|---|---:|
| A(v)=D(F²)(theta)v−v | 202 |
| Two-step JVP, including one reference direction | 203 |
| HVP, including one separate reference direction | 407 |
| Direct gradient, excluding HVP-internal differentiation | 11 |

The CountedProblemMixin wraps the unchanged Problem implementation. HVPs
invoke grad internally; a scoped depth counter excludes that internal work
from the direct-gradient count, while recording every HVP attempt/return.
No derivative path, batching, precision, model, data or rates are changed.
The source inventory includes package __init__, controls, models, cifar,
both diagnostic modules/tools, this launcher and both design documents.
A clean checkout and exact HEAD bytes are required before output creation.

Full regular-file inventories of the pinned reference bundle, prior corrector
and old linear diagnostic are checked. Exact library versions and GMRES
function hash must match. Replayed theta/map1/map2/R require identical dtype,
shape and C-order bytes, including signed zero. Sampled gradient/HVP/J2VP
parity uses the unchanged atol1e-11+rtol1e-9|reference| rule. Those returned
vectors are saved before checking parity. Every first-50 callback is saved
before its rtol1e-10/atol1e-12 control check. Any mismatch stops the diagnostic.
A return before50 callbacks is saved but remains explicitly unqualified.

The returned p and info are persisted before the independent A action. Its
returned Ap and e=Ap+R are saved before the core's finite/norm checks. Initial
and trial map1 is saved before computing map2, and every trial theta is saved
before evaluation. No Arnoldi basis or intermediate action vector is retained.
If interrupted inside GMRES without a return, no direction is invented.

True forcing0.5, strict descent, absolute-merit Armijo1e-4, four independent
alphas(1,.5,.25,.125), closure1e-8 and primitive separation1e-7 are unchanged.
SciPy's info, requested linear tolerance and separately tested true residual
remain distinct from the looser inexact-direction gate. Even a passing trial
has accepted_orbit=false and applied_to_continuation=false; spectrum is never
requested. Every outcome stops. No inexact Newton update is authorized.

## Resource and failure contract

CPU/BLAS/Torch intra/inter-op1; nice>=10; no GPU/paid/protected data. Internal
45s checks occur at operation boundaries. SIGTERM only sets a sticky flag;
the next budget boundary raises after the prior return can be persisted.
GNU timeout independently sends TERM at60s and KILL after5s. The launcher
also kills only its own session/process group at65s, including an exited
leader's pipe-holding descendants, and bounds a closed-pipe/live-leader case.
A one-second reap allowance can extend wrapper bookkeeping beyond65s; it
adds no scientific runtime. SIGKILL or OS failure can prevent final JSON,
but existing fsynced artifacts and launcher inventory are retained.

The 2GiB memory threshold is **monitored child process-group RSS**, polled at
0.1s, plus the worker's peak-RSS check before operations. It is not a kernel
hard memory bound: a transient overshoot is possible. The launcher itself
is outside that child group; its imported source-check interpreter memory is
additional. The driver saves peak worker RSS and the launcher saves maximum
observed group RSS, exact observed PIDs and any live owned PIDs remaining.
Foreign groups are never signalled. A guard hit makes the attempt incomplete
and cannot qualify the comparison; there is no retry.

Total raw32MiB is divided into31MiB science and1MiB launcher; stdout cap128KiB.
Scientific storage reserves1MiB for final failure/result/manifest records.
Per-file writes are exclusive and fsynced; records are never overwritten.
The launcher keeps its exact source, start/resources/argv, bounded combined
stdout, exit/ownership/RSS/raw inventory receipt and full manifest. Both
folders become read-only on finalization. Missing or inconsistent scientific
manifest prevents launcher success even after process exit0. Bytes from a
failed attempt remain preserved under that attempt, with no cap relaxation.
The fresh host/GPU-occupancy check is required in root's launch authorization
record; the launcher also captures current load and /proc/meminfo. It never
accesses CUDA. Resource-only checks do not authorize a target run.

## Exact prospective invocation; do not run without separate authorization

Use an exact reviewed clean source commit and a full independent review commit.
Their placeholders must be resolved and written into the launch authorization
record before execution. Root may integrate metadata while retaining identical
source-inventory bytes; bind actual execution HEAD separately from review.

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
nice -n 10 /data2/neural-shrimp/.venv/bin/python tools/launch_cifar_restart.py \
  --out /data2/neural-shrimp-data/explore/cifar/restart-sensitivity/attempt-001 \
  --launcher /data2/neural-shrimp-data/explore/cifar/restart-sensitivity/attempt-001-launcher \
  --source-commit EXECUTION_HEAD_FULL_SHA \
  --review-commit INDEPENDENT_REVIEW_FULL_SHA
```

Both directories must be absent. No directory has been created by this card.
The launcher creates one process only; no retry, extra cycle, rerun of the
baseline, profile or scan is allowed. All four diagnostic trials use the
original theta. Do not modify previous immutable receipts or historical status.

## Prospective independent saved-data audit boundary

Before reading outcomes, an independent auditor should freeze its evaluator
and card. It can bind this exact source/settings/runtime, complete raw and
launcher inventories, old manifests, exact theta/images/residuals and prefix;
recompute q,d, all trial merits/screens/decisions from saved arrays; and check
all attempted/returned counters against events, caps and interrupted statuses.
Separate recomputed arithmetic from any shared numerical primitives. The
basis and all200 Arnoldi action vectors are not saved, so a saved-data audit
cannot independently replay the orthogonalization or full solver trajectory.
Any new model/JVP evaluation would require a separately reviewed bounded scope;
it is not part of this source-only preparation. Report a failed control or
resource stop under its actual incomplete label, without changing thresholds.
