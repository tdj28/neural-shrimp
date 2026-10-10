# Finite new-matrix N4 panel (exploratory, 9 October 2026)

The N4/28171 period5 and period7 families have valid activity orbits but have not met the classical finite shrimp-anatomy target. Their failed/incomplete morphology remains recorded. This new panel changes the matrix, without increasing state dimension or redefining a shrimp from a visual island.

Exactly six new matrices use seeds 28172–28177, dimension4, unchanged `inputs()` PCG64/SeedSequence convention, iid normal entries divided by sqrt(4), including diagonal entries. Both independent original starts and tangent vectors are retained for every cell; odd symmetry does not erase raw cases. No protected seed or held-out data is accessed. The discrete map and fixed-matrix (gain, leak) interpretation are unchanged.

Each matrix uses the same31×31 grid: gain geomspace(.6,24), leak linspace(.05,1), 12000 burn updates,1000 tangent alignment,8192 measured updates,513 full tail states,256-update growth blocks. This is11532 trajectories and21192 updates each. Exact input arrays are saved. No parameter continuation seeds the panel.

Retain every four-neighbor connected consensus-period component for primitive-period proposals3–64, all singleton/edge components included. For each component retain its grid-edge margin and two-cell halo's unresolved trajectories with positive (> .002 per physical time) finite-growth estimates in both measured halves, separately for either/both initial states. These are contextual finite-horizon diagnostics, not a chaos certificate.

At most eight components per matrix get bounded native qualification. Fixed ranking: margin at least two cells, then at least three halo cells with both-start positive-growth diagnostics, then larger component, lower period and lowest grid index. Each selected component uses its first/last lexicographic cell, both starts; at most192 total qualifications. The unchanged full-state correction, proper divisors, adjacent-phase closure, signed Floquet/eigen residual and attraction gates apply. All failed correction iterates and returns are saved. All unselected components remain available and explicitly unqualified.

The next candidate, if any, should have full-state qualifications, parameter margin and irregular context. Longer transients and local fold/flip anatomy must discriminate shrimp from tongue/ribbon alternatives; this panel alone establishes none of these geometries. No favorable outcome is required to retain a matrix.

## Measured resource screen

First run a separate9×9 N4 profile with seed28169 (excluded from search), the identical21192-update horizon and ten retained period64 map/eigensystem units. Projection is
`1.8 * 6 * (31²/9²) * (21192/21192) * batch_seconds + 192*8*period64_unit_seconds + 20`.
The matrix/batch and horizon factors are explicit. Projected runtime must be≤240s; conservative peak including three target tail arrays plus64MiB headroom must be≤512MiB; raw estimate must be≤256MiB. Profile source/environment and full inventory are pinned externally before target. Feasibility is independent of profile scientific behavior.

CPU/BLAS1, nice≥10; profile45s internal/60+5 external, panel240s internal/270+5 external. Worker448MiB plus64MiB headroom,256MiB scientific raw plus1MiB launcher. Constant-time conservative prewrite accounting reserves final metadata. Owned group cleanup is mandatory; unfamiliar processes are untouched. One profile and one panel, exclusive `new-matrices-profile-001` and `new-matrices-panel-001`. No retry or unrecorded expansion; changed plans require a dated amendment and root coordination.
