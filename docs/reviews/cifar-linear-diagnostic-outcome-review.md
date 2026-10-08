# CIFAR linear diagnostic: independent saved-outcome audit

**TLDR:** the selected CIFAR state still has no accepted period-two correction.
The returned approximate direction is a descent direction, and its two smaller
trial steps modestly reduce residual merit, but the true linear forcing ratio
fails the declared gate. This is a local solver diagnosis, not evidence of an
orbit, orbit absence, stability, basin structure or training benefit.

**Verdict: the saved outcome and all four reported decisions pass this audit.**
Producer execution source `a9830e714d259597a53cee6ef7be41a9f07bb205`;
source review `dc7b4fea57c26c0a9e6db1b255c0e1d44efc1799`.
Audit execution source `3471ab00cb8a71920970191117227e11ec997645`, committed
with a prospective card before numerical saved-array replay. This independent
review evaluates no CIFAR map, HVP, GMRES solve, new trial or continuation.

The audit checks all 31 manifest-bound raw files and 23 finite float64 vectors,
source hashes against the exact producer Git tree, both pinned historical input
manifests and their complete inventories, normalized subset-A x/y identity,
unchanged original state/rate/ridge, runtime/result mirrors, returned-direction
record, source-review identity and the installed SciPy GMRES implementation hash.
The fsynced events contain exactly 200 inner iterations and 205 operator calls,
consistent with `info=4` and no declared convergence. Raw size is 18,218,537
bytes, below the planned 32 MiB. CPU/Torch threads are one, nice is 10 and
recorded execution is CPU-only.

Norms and dot products were recomputed using Python `math.fsum` reductions,
independently of the producer's NumPy/BLAS reductions. All scalar comparisons
agree within relative 1e-12 and absolute 1e-20. Exact array checks confirm
`e=A*p+R`, each trial `x+alpha*p` from the original x, and every saved
residual `map2-trial`. The actual operator and map images are retained inputs
for these identities; they were not independently evaluated here.

The recomputed true forcing ratio is **0.8843136324964807**, failing the 0.5
limit. The merit derivative is **-4.688089663616572e-11**, so the direction is
descending but does not satisfy the declared inexact solve rule. The requested
linear tolerance also fails.

| Damping | Merit ratio | Armijo | Relative closure | Candidate |
|---:|---:|---|---:|---|
| 1 | 21.3925568722 | fails | 2.3006225590e-5 | no |
| 1/2 | 2.1372607360 | fails | 7.2724291133e-6 | no |
| 1/4 | 0.9816889639 | passes | 4.9282697878e-6 | no |
| 1/8 | 0.9539323048 | passes | 4.8576844290e-6 | no |

All four closure screens fail 1e-8; all separation screens pass 1e-7. Every
candidate decision is false because true forcing fails, even when Armijo
passes. All four records explicitly reject orbit acceptance and application
to continuation, and the aggregate candidate list is empty.

Two additional synthetic audit controls pass: changed forcing/decision fields
or residual arrays are rejected, as are an invented orbit acceptance and a
trial formed by chaining from another state. Audit runtime was 0.428075 seconds,
CPU one thread/nice 10, PID 15796, maximum RSS 100,196 KiB; the process exited.
No new target operator evaluation was performed.

The complete compact record is
`docs/reviews/cifar-linear-diagnostic-outcome-review.json`. Immutable raw audit:
`/data2/neural-shrimp-data/audits/cifar-linear-diagnostic/attempt-001`.
Producer raw evidence remains unchanged at
`/data2/neural-shrimp-data/explore/cifar/linear-diagnostic/attempt-001`.

This audit checks source/input bindings and saved-array arithmetic. It is not
an independent rerun of A*p or the nonlinear map and does not upgrade this
single exploratory diagnostic to an orbit certificate or a general convergence
result. Source-review limitations concerning cooperative deadlines and
best-effort finalization remain in force.
