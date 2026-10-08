# Two bounded local gates — root launch decision

2026-10-08T11:04:48.354124+00:00. Exploratory; after independent source review and before outcomes.

Root reviewed the source and reran all 69 combined producer/independent synthetic
controls in 2.15 seconds. CIFAR's six bound source files match reviewed
17453ecc; utility's ten match reviewed075a76ec. Root's first source-inventory
report command used the wrong JSON key for the utility receipt and stopped;
the corrected command checks all ten byte hashes before writing this card.
There was no target launch or source change caused by that reporting error.

Authorize exactly one synthetic CIFAR resource profile under
`docs/plans/cifar-input-span-execution.md`, using review
97d0cad7bae51450dde17cd4f99dd1409c2b7458 and the independently inspected wrapper
`/tmp/neural-shrimp-launch-cifar-span.py`, SHA256
3c6c82bf15c55bb078981c62546bd796a19d9884d2f82a73e8960da5e12a8cc5.
The wrapper archives its source, owned processes and complete manifests.
Profile: CPU/BLAS1, nice10, internal5 seconds, external8+2 seconds; exclusive
`explore/cifar/input-span/profile-001` and companion launcher. The 64 MiB cap
includes the later target and both launcher records. A separate root decision
must inspect this fixed profile and pin its manifest before the saved-data
collection. This card authorizes no target projection, correction or solver.

Authorize exactly one four-gradient utility feasibility gate under
`docs/plans/utility-dose-feasibility.md`, source review
167d55bfe638ff788dae3343b62204ca960a6dbb. Use the committed launcher with the
exact clean integration HEAD recorded at runtime; all ten bound files must
retain the reviewed bytes. CPU/BLAS1, nice at least10, internal5 seconds,
external10+5 seconds, combined1 MiB. Fixed exclusive paths:
`explore/utility-dose-feasibility/attempt-001` and companion launcher.
No pulse, state update, new control, HVP or trajectory is part of this gate.
An independent analytic outcome auditor is being frozen before result reads.

Host check at10:59:33 UTC: load0.85/1.16/1.08,41 GiB RAM available;
GPU0 idle with15,947 MiB free and no listed compute application. These two
gates use no GPU. Foreign processes and swap configuration are untouched.
Record actual launch load and owned PIDs; no retries or paid resources.
Both source and this card must be privately pushed before execution.
