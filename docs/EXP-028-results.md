# EXP-028 results: periodic windows embedded in chaos in gradient-descent training

7 October 2026. Prospectively frozen exploratory pilot (Program C, Track A,
stage A1). Freeze `91db3654`; plan SHA-256 `fbb6203c…6d19a`. Completed by a
pre-logged continuation after the original attempt reached its wall ceiling
(see "Execution" below).

**Human-friendly TLDR:** When two small neural networks are trained by plain
full-batch gradient descent with separate learning rates for the two layers,
the learning-rate plane contains many islands where training settles into an
exactly repeating cycle, surrounded by chaotic training. We verified 61 such
islands with independent checks, with cycle lengths from 4 to 58 updates
(including odd lengths 5, 11, 13, 17 and 29). The published binary
"trainable / not trainable" map hides all of this: most of what it calls
trainable is training that never converges. These islands are not yet shown
to be shrimps (that needs their bifurcation boundaries, stage A2), and nothing
here shows they matter for model quality.

## Stage-A1 gate: confirmed positive

The frozen gate required an embedded candidate component (period at least 3,
no fixed-point or lower-divisor neighbors, mostly chaotic exterior ring) whose
representative orbit is a verified, attracting, primitive cycle and whose ring
is at least half rerun-confirmed chaos, **and** independent audit confirmation.

| | Count |
|---|---:|
| Components qualifying on producer evidence | 62 |
| Components qualifying on independent audit evidence | 63 |
| **Confirmed by both (the gate's definition)** | **61** |
| Producer-only (ring fraction 0.507 producer vs 0.465 audit) | 1 |
| Audit-only (ring fraction exactly 0.50 on audit reruns) | 2 |

The three disagreements are all at the 0.5 ring-fraction threshold and arise
from re-simulating chaotic ring cells with different floating-point paths.

Confirmed components by map: S1R-28001: 1, S1R-28002: 6, S1R-28001-zoom: 13,
S1R-28002-zoom: 13, S2-28011: 9, S2-28012: 4, S2-28011-zoom: 9,
S2-28012-zoom: 6. Confirmed periods: 4, 4, 5, 6, 6, 8, 10, 10, 11, 12, 12, 13,
14 (x5), 16 (x4), 17, 18, 20 (x7), 22 (x4), 24 (x2), 26 (x4), 29, 30 (x5), 32,
34 (x4), 36, 38, 42, 44 (x2), 46 (x2), 50 (x2), 58.

**Permitted statement:** in these networks, data, initializations, horizons
and tolerances, the per-layer learning-rate plane of full-batch gradient
descent contains numerically verified attracting primitive cycles of the full
parameter-space update map whose equal-period candidate domains are surrounded
mainly by rerun-confirmed chaotic candidates and border no lower-divisor
periodic or fixed-point cells. This does not establish shrimp boundary
organization, any training benefit, or behavior of stochastic or large-scale
training.

## Certificates and diagnostics

- 154 certificates were attempted and all 154 verified: 85 embedded candidates
  (84 attracting, 1 unstable), 63 other periodic components (62 attracting,
  1 unstable) and 6 fixed points (all attracting). Every verified primitive
  period equals the screened period; no certification failed or errored.
- The independent audit rechecked all 154: closure, finite-difference spectral
  radius within 1e-6 of the exact-Jacobian value, stability consistent with the
  `1 - 1e-7` margin, primitivity, ring membership, seeds and rates all pass.
- 2,801 ring-confirmation reruns; 157 of 160 random chaotic candidates stayed
  chaotic on rerun with a different tangent seed and a longer window.
- Receipt inventory (276 files), plan hash, inputs, zoom rule and zoom axes all
  reproduce in the audit.

## The published binary score hides non-convergent training

For every coarse map, almost all cells the Sohl-Dickstein convergence score
labels "convergent" do not converge to a fixed point within 20,000 updates.

| Map | Score-convergent cells (1,000 updates) | Fixed point | Periodic | Chaotic | Drift | Unresolved | Late divergence |
|---|---:|---:|---:|---:|---:|---:|---:|
| S1-28001 (published setting) | 29,697 | 8 | 0 | 16,121 | 13,401 | 142 | 25 |
| S1-28002 (published setting) | 31,192 | 55 | 0 | 17,835 | 12,618 | 681 | 3 |
| S1R-28001 | 25,627 | 2,633 | 5,407 | 14,840 | 177 | 2,312 | 258 |
| S1R-28002 | 23,697 | 3,063 | 4,572 | 13,668 | 277 | 2,061 | 56 |
| S2-28011 | 27,659 | 1,666 | 11,836 | 7,233 | 73 | 6,780 | 71 |
| S2-28012 | 27,623 | 2,470 | 5,461 | 10,217 | 2,215 | 7,130 | 130 |

"Unresolved" includes slow convergence at small rates; "drift" means the
parameter norm still grew by more than 1% over the last 4,000 updates.

## The published setting (S1) has no finite periodic attractors

With no ridge, the Sohl-Dickstein notebook configuration produced zero
periodic cells in both seeds: training diverged, was chaotic, or drifted
toward infinite weights while the loss slowly fell. The few S1 cells labeled
fixed points are fragile: in the audit's independent single-cell
re-simulation, 8 of 13 sampled S1-28001 regular cells and 12 of 15 S1-28002
regular cells kept their labels; every disagreement is a cell labeled a fixed
point that diverged, became chaotic or stayed unresolved under the other
floating-point path.
This is consistent with long chaotic transients and riddled outcomes in
ridge-free training (compare Ly and Gong 2025). Adding ridge 1e-3 (S1R) made
periodic windows appear. S1 remains descriptive.

## Execution

attempt-001 (launched 18:05 local, 14 CPU workers) completed all coarse and
zoom maps and certification, then reached its 8-hour wall ceiling three
minutes into ring-confirmation reruns. Following an amendment logged before
any outcome was opened, attempt-002 (`tools/continue_exp028.py`) verified
attempt-001's receipt, reused all 266 map files byte-for-byte (no chunk
recomputed) and ran stage 3 in 927 seconds. Total map computation used
roughly 112 CPU-hours.

## Limitations

- Two small networks, two seeds each, deterministic full-batch training, one
  dataset per network, float64 on CPU.
- Screened periods use a 384-update tail; slowly converging orbits can appear
  at multiples of their period (certification guards the gate).
- Chaotic labels are finite-time; "embedded in chaos" means mostly
  rerun-confirmed chaotic neighbors at map resolution.
- Shrimp organization (fold and flip boundaries, cusp points), the unimodal
  MSS-order mechanism and any training consequence remain untested.

## Evidence

- Summary: [EXP-028-summary.json](EXP-028-summary.json)
- Protocol: [EXP-028-protocol.md](EXP-028-protocol.md); design audit:
  [EXP-028-design-audit.md](EXP-028-design-audit.md)
- Raw: `../neural-shrimp-data/raw/EXP-028/attempt-001` (maps, failed at
  ceiling) and `attempt-002` (continuation, complete)
- Audit: `../neural-shrimp-data/audits/EXP-028/validation-001/audit.json`
- Figures: `../neural-shrimp-data/figures/EXP-028/attempt-002/`

## Correction, 7 October 2026

The random-chaos rerun count above originally read "155 of 160". The frozen
summary (`docs/EXP-028-summary.json`, `random_chaos_diagnostics`: 157 confirmed of
160 reruns across 10 maps) and the raw diagnostics give 157; the text was a
transcription error found while building the manuscript's number macros. No
analysis, gate or conclusion depended on the miscounted figure.
