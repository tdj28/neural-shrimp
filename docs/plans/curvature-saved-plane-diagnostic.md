# Saved-plane explanation after the negative curvature pilot

2026-10-09 UTC. **Adapted exploratory descriptive analysis**, proposed after
rootf78f420f reported that all six references passed physical/historical checks
but all six fixed rank-one/bulk pair errors exceeded.02, with false complex
predictions at both real controls and a false real prediction at the S1R
complex control. The author also displayed the heading and first two rows of
the retained outcome table during source inspection; no empirical arrays were
opened or evaluated during this preparation. This adaptation does not replace
or rescue the original failed engineering screen. It makes no prediction,
validated attribution, complete-spectrum fit, utility or new-theory claim.

## One fixed saved-array computation

Read only the immutable six analysis records from
`/data2/neural-shrimp-data/explore/curvature-mechanism/attempt-001-panel`, pinned
manifest SHA256
`898880f0bc1755896251f39e5abf806cc89f32a1558ed102fc0c954bd6d9ff4b`.
Verify the complete inventory, producer execution-source bytes and record/array
hashes. Keep the six slots in original order and the saved actual leading-pair
indices; do not select different eigenvectors, points or rank. Exact current
source and input bindings accompany the output.

For each reference normalize the two stored whitened top-curvature directions
and apply fixed Gram–Schmidt. Use the same aligned convention
`1-(u0^T u1)^2 <= 1e-10`, yielding a one-dimensional Q in that case. Otherwise
Q has two orthonormal columns. Retain Q and its orthogonality residual.
For each of the two **stored actual leading right eigenvectors** v, retain
Q^T v, QQ^T v, the uncaptured residual, and the fractions
`||Q^T v||² / ||v||²` and `||v-QQ^T v||² / ||v||²`. These are Euclidean energies
in whitened coordinates, not unwhitened parameter-space energy.

Retain the actual native target product action MQ, full leakage
`MQ-Q(Q^T MQ)`, its Frobenius norm, and compression C=Q^T M Q. For two-dimensional
C, compute its two roots by explicit trace/determinant quadratic algebra and
compare with the unchanged actual leading pair using the fixed unordered
maximum-error metric normalized by max(1,actual spectral radius). For aligned
one-dimensional Q, report the scalar compression but abstain from a two-mode
pair comparison. Near-repeated quadratic roots have no validated accuracy claim.
No eigensolver, SVD, QR, inverse, Hessian, model, solve or trajectory is called.

An additional source-before-replay request from root, motivated by the negative
pilot's angle/modulus errors, retains the four exact-algebra residual actions:

```
(M-Mmodel)v = (M-SbSa)v + Sb0 Ea v + Eb Sa0 v + Eb Ea v.
```

Use only saved matrices; preserve every term, their sum, the direct difference
action, their floating identity residual and per-vector norms (also normalized
by ||v||). These expose omitted phase terms and cancellation on the target
vectors; they are not causal attribution of eigenvalue error.

Low capture would accompany missing target eigenspace. High capture with a
more accurate actual compression would accompany a poor original coefficient
approximation within the retained plane. Leakage helps qualify this reading:
a noninvariant plane's compression roots need not be full-system eigenvalues.
No threshold is added to classify capture or leakage, and no rank/point search
follows automatically. The retained numbers support a qualified explanation,
not a new positive engineering gate.

## Execution and evidence

Prepared script: `tools/explain_curvature_plane.py`. Root owns its one saved-data
launch; this source card does not execute it. Proposed exclusive output:
`/data2/neural-shrimp-data/explore/curvature-mechanism/attempt-001-plane-explanation`.
CPU/BLAS1, nice>=10; simple internal10s and external15+5s limits,1MiB Store
(768KiB ordinary plus256KiB finalization reserve). These are operational saved-
arithmetic caps, not scientific thresholds. Preserve partial/failure records;
no automatic retry. Store retains all fixed slot IDs, source/runtime/PID/exit
status, arrays and full manifest; files freeze0444/directory0555. Root retains
its external command/log/exit receipt as with the preceding exploratory unit.

```
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 timeout --signal=TERM --kill-after=5s 15s nice -n 10 /data2/neural-shrimp/.venv/bin/python tools/explain_curvature_plane.py --out /data2/neural-shrimp-data/explore/curvature-mechanism/attempt-001-plane-explanation
```

Nine synthetic algebra controls passed0.24s: invariant complex plane despite
bad original coefficients, missing eigenspace, explicit leakage, scaling/sign
invariance, aligned abstention, nonorthogonal Gram–Schmidt, pair binding, zero
vector rejection, and all four residual-action terms. Decomposition APIs are
forbidden in the principal control. An earlier binding test failed because its
fixture aliased the pair and source spectrum; making the fixture independent
restored the intended tamper test without changing the binding requirement.
No target replay, profile or empirical matrix operation ran during preparation.
