# EXP-024 fixed held-out phase panel

22 September 2026 local time. Prospective exploratory held-out stage following
the completed development freeze `635a0b9e411933c3b77bc0f515b802a3b14be3d6`.
This is a private executable freeze, not a formal preregistration.

**Human-friendly TLDR:** The development panel suggests the selected near-edge
rhythm can combine two inputs into a feature a linear readout can recover,
while the core does so much more weakly. This run tests that finding on the
32 untouched phase draws reserved before development, with unchanged solvers,
write directions, amplitudes, observation windows, noise and numerical gates.
No claim of a shrimp-specific mechanism or generalization across networks follows.

## Development decision and unchanged method

All 488 development jobs and 192 analysis cells qualified; saved replay and an
independent arithmetic implementation passed. Runtime was 50.10 seconds with
four single-thread workers. At amplitude 0.01 the primary near-edge early-state
mean R-squared was 0.23096564 versus 0.00009291 at the core. The near-edge range
across eight development phases was approximately 0 to 0.904. These are
development observations, not the test-panel answer. The largest numerical risk
discrepancy was 2.90e-7. Details and hashes are in
[development results](EXP-024-development-results.json).

Proceed with the original 32 held-out phases. There is no change to the original
primary cell, amplitude range or direction rule based on these development
results. All scientific equations and configurations from the
[development protocol](EXP-024-development-protocol.md) apply. Keep state x
primary and tanh(x) secondary, phase-specific reference observers, both writes
at 0 and T, 32 inclusive samples per window, Gaussian measurement SD 1e-4,
and the same DOP853/RK45 comparisons. No process noise or label-dependent input
controller is introduced.

Use phase indices 8–39 from the original PCG64 seed 2026092201, without redraws.
The development source and executable remain preserved. The new runner is an
explicit lifecycle fork, with its differences locally reviewed: reserved phase
selection, census, aggregate statistics and separate output identity. No new
finite-difference tuning is done on test outcomes; the unchanged derivative
implementation already passed analytic controls and both finite-difference
steps on the fixed development checks. Cross-method references and every finite
test history are still checked throughout this panel.

## Exact census and chronology

| Stage | Jobs | Maximum ODE segments |
|---|---:|---:|
| Fresh dense phases, two settings × two methods | 4 | 8 |
| Sensitivity references, 64 contexts × two methods | 128 | 256 |
| Finite differences | 0 | 0 |
| Finite histories, 64 contexts × 3 amplitudes × 4 signs × two methods | 1536 | 3072 |
| Total | 1668 | 3336 |

There are 768 distinct factorial histories, each run with both methods; these
solver repeats are not extra independent observations. The 768 analysis cells
are 64 phase-setting contexts × three amplitudes × two windows × two observation
types. Phase is the uncertainty unit; signs, windows and solvers are nested.

Seal all 2688 reference observers (64 × 42) before any finite test trajectory.
The same 42-per-context definition includes first-order, second-order and
dynamical-cross-term-ablated state/activation observers plus the exact static
tanh-of-additive-state comparator. Test means only inform the labeled
descriptive affine oracle, never the deployed observer.

The 32-history timing batch is both settings, phases 8–11, amplitude 0.01 and
all four signs, primary solver only. The remaining 1504 jobs follow as already
declared; no outcome-dependent early-success stop or expansion.

## Primary analysis, uncertainty and decisions

Primary endpoint: mean across the 32 held-out phases of the **near-edge,
amplitude 0.01, early-window, state-coordinate** second-order reference
observer's noise-averaged parity R-squared. Report the paired core comparison
using the same phase indices. This endpoint was chosen before development.

Use 20,000 fixed-sample phase bootstrap draws from NumPy Generator(PCG64),
seed 2026092301, drawing 32 phase indices with replacement each time. Reuse
these draws for every descriptive cell and for the paired near-edge minus core
contrast. Report the 2.5th and 97.5th percentiles. These are conditional
phase-sampling intervals at two fixed networks, not between-network intervals.
Only the stated primary cell drives the prospective decision; other intervals
are descriptive and not adjusted for multiple comparisons.

Retain finite negative R-squared, every failed numerical comparison and all
original denominators. A missing, duplicate or unqualified cell blocks the
complete-panel primary inference; no favorable imputation. Report mean oracle
scores, score range, direct/predicted contrast norms, approximation error and
numerical error alongside the primary result.

The engineering continuation trigger is primary mean R-squared > 0.05, its
phase-bootstrap interval excluding zero, and at least one primary phase whose
direct contrast norm exceeds ten times its cross-method discrepancy norm.
This last diagnostic prevents a purely numerical interaction from driving the
decision; it is not a formal error bound. The 0.05 is a research-resource
threshold, not a universal definition of useful neural computation. Passing
supports planning the single T/2 write-separation extension recommended by Pro.
Failure, a reference/oracle gap, activation-only mixing or unresolved numerical
checks must lead to the corresponding narrower interpretation in the original
design. No extra phases or subsequent experiment launch is automatic.

## Resources, raw evidence and verification

Use four single-thread CPU workers, the same per-job 180-second cap, total
eight-hour cap, shared RAM checks and owned-process cleanup. Based on measured
development throughput, expect a few minutes, not hours; timing is evidence,
not a deadline extension. No accelerator or paid computation is needed.

Raw files go into a new exclusive
`/data2/neural-shrimp-data/raw/EXP-024/heldout-001/` directory. Source guard,
artifact receipts, observer barrier, complete terminal census and no-integration
saved-array replay are unchanged. In addition, replay the fixed bootstrap
aggregate. The development receipt SHA-256 is
`6f1488818683757a90181e0e99fcc48f1dd716d20ec1f33b0648d77f50b4bdef`;
the machine plan binds that receipt and compact development results. Commit and
privately push complete source, plan and local review before any held-out
integration. Preserve the frozen development protocol and every old attempt.
