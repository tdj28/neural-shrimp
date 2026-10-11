# EXP-017: finer sampling helps, but periodic-reference consistency still limits the task

17 September 2026. [Frozen protocol](EXP-017-protocol.md),
[compact receipt and all-phase diagnostics](EXP-017-summary.json).

**Human-friendly TLDR:** This study measured the accuracy of our cycle reference,
not new memory performance. Finer sampling reduced the largest coordinate error,
but nine starting phases still fail the check with both solvers. The earlier
finding remains: some near-edge cases retain readable input history and show
interference from a later input. We still cannot make the full-phase comparison.
The remaining numerical limitation concerns the inherited nearly periodic
reference, rather than evidence that the network has lost its memory signal.

All **102 jobs** completed in **60.58 seconds**: six orbit/adjoint reconstructions
and 96 one-period state/tangent continuations. Every one of the 48 original
phase contexts is present. All six base-method pairs agree under the frozen
state, activation and pulse-scaled full-tangent limits. All six reconstructions
also pass the original grid normalization, relative closure and adjoint
periodicity diagnostics. Those broad checks are insufficient to guarantee the
more sensitive normalization check at every event phase.

Exactly **39/48** contexts pass the combined finer-method event check. The two
methods fail at the same nine contexts: core phases 0, 3, 9, 10 and near-edge
phases 4, 7, 11, 15, 16. The threshold remains `abs(z dot f - 1) <= 1e-7`.
Every finer-method initial normalization check passes.

| Setting | Original event maximum, both old methods | Exact dense start, finer DOP853 | Exact dense start, finer RK45 |
|---|---:|---:|---:|
| Core | 2.9027e-7 | 1.6980e-7 | 1.6995e-7 |
| Near-edge | 5.9443e-7 | 1.6993e-7 | 1.7015e-7 |

The original maxima come from the separately bound
[EXP-016 reference diagnosis](EXP-016-reference-diagnosis.md). EXP-017's direct
old-versus-new event comparisons consume only the old primary DOP853 seal;
they do not rerun or reload the old RK45 trajectory. No original missing
finite-input history has been filled or requalified.

The finer DOP853 reconstruction's maximum state interpolation error at the 24
declared phases falls from **1.2353e-9 to 1.0606e-10** for core and from
**2.0495e-9 to 1.9900e-10** near the edge when the saved grid doubles from 4096
to 8192 intervals. Adjoint interpolation improves too. Yet the event defects
persist when starting directly from solver-dense samples, without saved-grid
interpolation. Both finer algorithms reproduce these remaining failures.
Interpolation therefore contributes, but it is not the whole explanation.
Nor does every individual phase improve: the primary core phase-0 event defect
rises from 1.5822e-7 to 1.6690e-7, while near-edge phase 11 falls from
5.9309e-7 to 1.5248e-7. Both remain above the same threshold.

This diagnostic retained the old section state, period and terminal adjoint
unchanged. Their small closure and periodicity errors can still matter when a
nominal periodic phase gradient is paired with the actual state one period
later. The evidence supports investigating that inherited reference consistency;
it is not a rigorous decomposition of all numerical error or proof that any
particular repair will pass. A global orbit/adjoint refinement needs its own
fixed plan and numerical verification. Merely increasing resolution again or
renormalizing each failed phase would not answer the same diagnostic question.

The run used four CPU workers with one numerical-library thread per process.
Sampled peak owned RSS was **710,725,632 bytes**. The immutable receipt binds
326 artifacts totaling **210,612,285 bytes**, stored externally at
`/data2/neural-shrimp-data/raw/EXP-017/attempt-001`. No paid compute, neural input
pulses, observer scores or fitted models were used.

Frozen source: `8fbd8ddcff5758526817e0f800a07f08c903a7c9`.
Plan SHA-256: `38a8d2aeb9f543a15cc7c3d0d471073b2ec095a669c8031acc837bd962e392f2`.
Raw receipt SHA-256: `7d8daa8d4bfe816247c568149b040758785c82d4fd4bb94a97b9c71b625b40f7`.

The frozen verifier reproduces the complete saved analysis and verifies that
all original owned process identities are absent. An independent saved-array
diagnosis passes all 28,447 evidence and categorical checks, including every
scientific pass/fail conclusion. It separately retains **309 failures among
2,004 strict alternate-arithmetic checks**, mostly in the alternative Hermite
derivative calculation; the largest absolute difference is about 9.56e-12.
There is no blanket independent arithmetic PASS and no scientific gate change.
Completion means the
diagnostic finished, **not** that all scientific checks passed. The original
EXP-016 study remains incomplete and unqualified, and no later eight-period
memory task is certified by this one-period diagnostic.
