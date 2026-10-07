# EXP-028 design audit and pre-freeze revisions

6 October 2026. A separate agent adversarially reviewed the first EXP-028
design (local commit `999e5752`, never pushed or run) before any target outcome
existed. It ran the unit tests and small probes on **artificial fixtures only**
(S1-type seed 99999 and S2-type seed 99991, outside the plan), never on seeds
28001, 28002, 28011 or 28012. Those probes informed the revisions below; they
are design evidence, not EXP-028 results. Every change was made before the
freeze.

## Findings and resolutions

| # | Severity | Finding | Resolution |
|---|---|---|---|
| 1 | Blocker | The gate could pass on an ordinary period-doubling band: on the S2 fixture a period-4 component touching the period-2 domain certified as attracting | Frozen embedding criterion: period at least 3, no fixed-point or proper-divisor-period ring cells, at least half the ring chaotic; cascade-adjacent components are reported separately and cannot pass |
| 2 | Blocker | Ridge-free S1 drifts toward infinite weights (no fixed or periodic cells on the S1 fixture) and "unresolved" conflated drift with slow convergence | S1 kept as descriptive only; new S1R adds ridge 1e-3 with identical data and initial parameters; new drift label from norm growth over the final window |
| 3 | Important | The zoom rule saturated on period-2 plateaus and tie-broke toward regular regions | Zoom now ranks windows by period at least 3 cells bordering chaos, then by chaos-boundary cells |
| 4 | Important | The audit trusted producer rates, used a coarse radius tolerance, tested radius below 1 instead of the margin, and could not recompute the gate | The audit recomputes rates from axes, requires finite-difference radius within 1e-6 and below `1 - 1e-7`, rechecks primitivity, recomputes rings with `scipy.ndimage`, re-simulates ring reruns and recomputes the gate; a positive requires audit confirmation |
| 5 | Important | "Borders chaos" relied on single-window exponents | Every chaotic ring cell of each certified attracting embedded candidate is rerun (cap 48, row-major); the confirmed ring fraction gates |
| 6 | Important | Certification used a finite-difference Jacobian in the root solve, ignored solver success, had no primitivity margin; the screen compared only the last `6p` states | New `orbits.certify_orbit`: damped Newton with the exact chain-rule Jacobian, undamped-correction convergence test, divisor separation 1e-6, conditioning diagnostics; screening now uses the whole 384-update tail. The historical `certify_cycle` is unchanged |
| 7 | Important | The remote check trusted local refs; any plan path was accepted | Live `git ls-remote` must equal HEAD; only the tracked default plan runs |
| 8 | Important | A killed worker hung until the ceiling; interrupts skipped the receipt | `ProcessPoolExecutor` detects dead workers; termination signals and interrupts are recorded and the runtime record and receipt are always written (tested) |
| 9 | Important | Stage 3 was all-or-nothing | Certification and rerun failures are recorded per task |
| 10 | Minor | No analytic gradient-descent positive control; Hénon test used tighter tolerances | Added the two-parameter product-model flip-curve control through the batched engine and certifier; Hénon test uses the plan's tolerances |
| 11 | Minor | Incomplete source binding; scipy version unrecorded | Plan binds `__init__.py`, `controls.py`, `pyproject.toml`, `uv.lock` and new modules; runtime records scipy |
| 12 | Minor | Certification recomputed rates | Saved realized rate arrays are used |
| 13 | Minor | Analysis accepted incomplete receipts and overwrote outputs | Both refused (tested) |
| 14 | Minor | "Notebook defaults" unverified | Notebook pinned by commit and SHA-256 with a local copy; the plan records its use |
| 15 | Minor | Throughput unrecorded; per-map barriers idle workers | Throughput stated in the protocol; chunks of all maps in a stage run together |
| 16 | Minor | `final_loss` was the loss before the last update; figure called screened periods primitive | Renamed `last_evaluated_loss`; colorbar says "screened recurrence period" |
| 17 | Minor | Mixed-outcome language not frozen | Frozen per-system and per-map outcome labels |

## Pre-freeze fixture checks after revision

An opt-in end-to-end test (`NEURAL_SHRIMP_SLOW=1`) runs the complete revised
pipeline on the artificial S2-type fixture (seed 99991, 96 x 96 coarse cells,
64 x 64 zoom, 20,000 updates). It found two embedded period-4 components (ring
chaotic fractions 0.95 and 0.91 by labels; 0.80 and 0.78 confirmed by reruns),
both certified attracting and primitive (spectral radii 0.774 and 0.914). The
independent audit confirmed both, with finite-difference spectral radii
agreeing within 2e-9 and 4e-11. This shows the revised gate retains power on
this model family. It is not evidence about the planned seeds.
