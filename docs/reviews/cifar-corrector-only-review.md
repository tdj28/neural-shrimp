# Independent CIFAR corrector-only saved-state review

**Pass: the attempt is accurately reported as unresolved, with no accepted orbit or stability inference.** Reviewed 2026-10-08 UTC against source `7de5dcfb5eb50ed7829bc3b69b43c6e2c4f889cd`, producer manifest and launcher logs. The [receipt](cifar-corrector-only-review.json) binds all identities.

An independently written NumPy chain-rule gradient for the 3072--32--10 tanh MLP reproduces the saved two-step relative closure as `4.97305505946193e-6` and one-step separation as `3.543241439719897e-4`. Differences from the producer are below `6e-19`. Closure fails `1e-8`; separation passes `1e-7`. The audit evaluated the two local map images needed for those residuals; it ran no new trajectory, correction or spectrum.

All six manifest entries and exact source blobs match. All four durable state files have the same SHA-256 as the input. The sole Newton history entry reports `linear_solver_failed`, GMRES `info=4`, before any accepted correction. The mode returns before multiplier computation, and events contain no spectrum stage. Launcher exit code is zero with reported scientific failure; wall time is 10.1154 seconds, within the 200-second external limit. This is solver-budget failure, not evidence of nonexistence of an orbit; no claim about matrix conditioning follows without more diagnostics.

The normalized subset-A arrays and index identity independently match. Only A was selected/evaluated; reading the raw CIFAR containers also reads other rows. The audit did not open the sealed B index record or consume reserved seeds. CPU/BLAS threads were one, nice level ten; independent replay took 0.366 seconds. No paid resource was used.

Audit source: `tools/review_cifar_corrector_only.py`, committed at `943b1b3b1f5bcb81affab5d5cce732659285f8cf` before replay. Retained output: `/data2/neural-shrimp-data/audits/cifar-corrector-only/attempt-001/review.json`. No blockers or required wording corrections were found.
