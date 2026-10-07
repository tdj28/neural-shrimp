# EXP-029 results: autonomous survey and shrimp focus maps

7 October 2026. Prospectively frozen exploratory descriptive survey (Program
C, Track B, stage B1). Freeze `6a162ce2`; plan SHA-256 `e2f8a007…4b18`; run
from `fc219a7a`; 757,760 cell runs in 6,769 seconds on 14 CPU workers.

**Human-friendly TLDR:** The three-neuron network's parameter plane has a large
chaotic tongue between two broad regions of simple (one-peak) oscillation.
Inside the tongue, periodic windows organize into the spiral pattern reported
by da Silva and Rech, including the reproduced shrimp. This gives Track B what
it needs: ordinary periodic regimes and chaotic regimes far from the shrimp, as
distant controls. Nothing here measures computation.

## Global survey (alpha 0 to 1.5, gamma 5 to 16, 256 x 256)

| Start | Periodic | Chaotic candidate | Unresolved | Equilibrium |
|---|---:|---:|---:|---:|
| A | 40,358 | 24,843 | 335 | 0 |
| B | 39,669 | 25,524 | 343 | 0 |

Simple one-peak oscillation is the most common periodic class (29,617 cells
for start A), filling the regions at small and large alpha. Periods 3, 5 and 7
are frequent inside the tongue (1,713; 4,329; 799 cells). Starts A and B give
the same class in 64,350 of 65,536 cells (98.2%) and the same class and period
in 64,115 (97.8%). No equilibrium exists anywhere in the surveyed plane.

## Focus map (EXP-004 rectangle, 768 x 768)

Start A: 104,159 periodic (period 9: 92,860; period 18: 5,357), 477,974
chaotic candidates, 7,691 unresolved. The start-B subsample (every fourth row
and column) agrees in class for 36,479 of 36,864 cells (99.0%).

## Numerical quality

The median neutral-multiplier error of periodic cells is 8e-9 (global) and
4e-8 (focus); 9 of 40,358 and 208 of 104,159 periodic cells exceed 1e-3, all
near boundaries with transverse multipliers close to 1. Floquet closure is at
most 9e-5.

## Independent audit (validation-001)

Receipt (all files) and plan hash reproduce. Of 96 sampled cells re-integrated
with scipy DOP853 (`rtol 1e-10`) and separately written event screening and
variational exponents, 92 keep their label; every sampled periodic cell keeps
its label and period (32/32). The four label differences are unresolved cells
near the 0.005 exponent threshold. Dominant transverse multipliers agree within
the 1e-4 tolerance for 30 of 32 periodic cells; the other two differ by
1.9e-4 and 3.3e-4 (both near-boundary focus cells) and are recorded as
exceeding the stated tolerance.

## Evidence

Summary [EXP-029-summary.json](EXP-029-summary.json); protocol
[EXP-029-protocol.md](EXP-029-protocol.md); raw
`../neural-shrimp-data/raw/EXP-029/attempt-001`; audit
`../neural-shrimp-data/audits/EXP-029/validation-001/audit.json`; figures
`../neural-shrimp-data/figures/EXP-029/attempt-001/`.
