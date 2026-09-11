# EXP-005 saved-result verification portability

2026-09-11, after EXP-005 completed and before the result release. This is an
analysis/verifier amendment, not an integration rerun or changed experiment.
The exact pre-execution source remains at
`01209371d541aaa9d072a1addb3960aa3a74efd3`, archived under
`archive/EXP-005-source-v1`. Its original receipt and every artifact remain
byte-for-byte unchanged.

The post-run review found that analytic identity checks in the new verifier
and plotter required bitwise equality when recomputing tanh and Euclidean
norms. Those computations can differ by roundoff across ARM/macOS and x86/Linux
math libraries. Bitwise equality is appropriate for retained artifact hashes,
unchanged stored values, and lossless plotted-array reconstruction. It is an
unnecessary portability requirement for independently recomputed analytic
identities. No observed numerical target failure motivated this amendment.

Only these recomputed state-to-activation and Euclidean-distance identity
checks now use relative tolerance 1e-12 and absolute tolerance 1e-14, matching
the existing EXP-004 plot checks. All file hashes, saved-array checks, coordinates,
cell census, recurrence classification, period times, shooting closure, primitive
period, neutral-mode checks, transverse-stability thresholds, FTLE thresholds,
and experiment settings remain unchanged. No artifact was rescaled, replaced,
reclassified, or dropped. No new ODE trajectory was generated. The plotting
receipt binds the updated plotter and records the analytic replay tolerance.
