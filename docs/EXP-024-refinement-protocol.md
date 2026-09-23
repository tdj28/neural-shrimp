# EXP-024 numerical refinement amendment

22 September 2026 local time, after inspecting the complete frozen held-out
attempt. **Preserve the original unqualified attempt; do not change scientific
endpoints, observers, phases, pulses, noise, bootstrap or acceptance thresholds.**

**Human-friendly TLDR:** The held-out panel descriptively retained the near-edge
interaction. It also revealed a core phase where the larger input causes the
second-order prediction and its fixed readout to fail dramatically. Strong
local attraction does not establish a reliable finite-input approximation.
Eight tighter solver histories will determine whether the numerical discrepancy
can be resolved. Very negative scores and the unexpected response stay in the
analysis. No cycle escape, basin switch or geometry-specific mechanism is
inferred from the discrepancy alone.

## Original evidence and reason for amendment

All 1668 original jobs completed in 178.93 seconds; saved-array replay passes.
Four of 768 analysis cells failed frozen numerical qualification. All four
are core, phase index21, amplitude0.01, covering both windows and both
observations. The original receipt is
`3625e1d1cf8f78d4848ad088b6414223555246654bcbfbc9417d3aba823940bd`.

The maximum state discrepancy is 1.2651e-6 (gate2e-7). The later state contrast
coordinate discrepancy is 3.3760e-7 (gate1e-7), and the largest readout-risk
discrepancy is0.11384 in an activation comparator (gate0.005). The corresponding
early state second-order reference R-squared is about−812906, while the
descriptive affine oracle is nearly1. This is retained evidence of a serious
local approximation/readout failure, not a reason to clip, remove or refit
that phase. The original complete-panel inference remains blocked.

An auxiliary independent arithmetic audit initially used an excessively strict
fixed absolute1e-10 equality tolerance. It differed from the production risk
calculation by6.98e-10 when some risks were millions, while oracle and contrast
differences were1.48e-14 and4.44e-16. Preserve that failed audit and assess
arithmetic with a stated scale-aware criterion separately. This auxiliary
arithmetic criterion is not the scientific cross-solver0.005 gate, which remains
unchanged.

## Exact bounded correction

Reintegrate **only all four signs at core--p21, amplitude0.01, with both methods**:
eight jobs and16 ODE segments maximum. Use the exact original saved initial
state, write direction, period and sampling times. Do not recompute any
reference, phase gradient or observer, and do not alter the768-history panel.
The new plan asserts that this one group accounts for every originally failed
cell. All other finite histories retain their original immutable arrays.

| Method | rtol | atol | Maximum step |
|---|---:|---:|---:|
| DOP853 primary | 5e-14 | 5e-16 | T/8192 |
| RK45 independent | 5e-14 | 5e-16 | T/32768 |

These caps give sixteen times more steps per period than the corresponding
original method, with tighter tolerances. Both are above SciPy's float64 rtol
floor. Keep the original state2e-7, contrast1e-7 and risk0.005 thresholds.
No automatic retries or selective sign/method replacement. If this pair fails,
preserve it and report an unresolved panel; do not average solvers or promote
the passing subset.

Use four single-thread owned CPU workers,180 seconds per job and1800 seconds
total, shared-resource checks and the existing owned cleanup controller. No
paid computation. Freeze and privately push the amended code and machine plan
before these new trajectories. Store them exclusively in
`/data2/neural-shrimp-data/raw/EXP-024/heldout-refinement-001/`.

Recompute the complete analysis using the1536 original solver-history slots,
replacing exactly these eight with the stricter pair. Record a hash-bound
per-history provenance index. Reuse the original sealed observers and fixed
bootstrap; do not adapt the score to the new results. If all cells qualify,
report a **numerically refined composite panel**, with both original and new
source commits and receipts. This does not retroactively qualify or overwrite
the original attempt. Verify saved-array arithmetic and composite provenance
without further integration before interpretation.
