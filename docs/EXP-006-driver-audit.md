# EXP-006 pretarget driver audit

Date: 2026-09-11. This is a local audit before new EXP-006 outcomes. All new
orchestration tests use stubs. Source-plan inspection reads retained EXP-004/005
evidence only. No paid review or new target trajectory was used for this audit.

The audit covers `run_hopfield_drift.py`, `prepare_hopfield_drift.py`, and the
generated EXP-006 plan. The metric and probe implementations were separately
checked on synthetic waveforms and non-target alpha=gamma=0 numerical fixtures.

## Checked behavior

- The plan expands to 4,320 unique trial-step records: 12 attempted anchors,
  four phases, two integration steps, and 45 records per phase-step (one zero
  dose plus four radii across eleven arms). All arms have the same raw Frobenius
  weight-change norm at a given radius.
- A full stubbed runner execution retains all 4,320 records when only one of
  twelve anchors is eligible. It emits 3,960 `baseline_not_evaluable` records,
  executes the remaining 360, and reports coverage 1/12. No anchor replacement
  or success-only deletion occurs.
- Baseline numerical steps restart independently from the declared common
  initial condition. Shooting uses the finest-step event seed, and is skipped
  unless both steps retain nine peaks.
- Short and long intervention trajectories share the exact baseline phase
  state and identical changed weights. Local sensitivity starts from that same
  phase state with zero initial parameter sensitivity. Each numerical step
  restarts independently; there is no phase or dose continuation in the assay.
- The waveform comparison uses candidate time plus the baseline quarter-cycle
  epoch. The certified nine-peak reference has 2,700 endpoint-excluded samples,
  so quarter-period phases are exact reference sample locations. A single
  global phase is fitted at the fixed original period, without frequency fitting.
- Clean-source and pushed-source gates reject before target-attempt directory
  creation or a target build. Plan preparation invokes 48 old-atlas direction
  calculations and no integration or shooting. Continuation seed keys and
  certificate fields match the retained EXP-005 schema.
- A synthetic long-integration failure preserves the completed short trace
  and a separate native-failure payload. The actual trace wrapper separately
  passed an artificial-overflow check preserving its initial row and progress.

The sixteen stub-only driver tests pass. Together with the ten waveform tests and
nine probe tests, the audited new numerical/driver checks total 35 passing tests.
These checks do not establish target scientific outcomes.

## Interpretation limits

The local comparator uses native alpha/gamma derivatives, which matches the
raw-weight metric. Its two signs are averaged prospectively; no sign is selected
from target outcomes. The atlas policy uses previously purchased information
from a finite, anisotropic pixel atlas and selects among 512 fixed directions.
Its score is an observed-pixel proxy, not a verified bifurcation distance. Atlas
creation and local-sensitivity cost must remain separately visible. This design
does not establish equal-budget superiority or a net speedup.

The task is preservation of a particular three-neuron activation waveform.
Because frequency fitting is forbidden, a stable but frequency-shifted orbit can
have large task error. Nine-peak retention and task preservation are distinct
outcomes. Conditioning on the prospectively specified eligible baseline subset
is explicit, with coverage reported against all twelve attempted anchors.

## Retention and verification findings resolved before freeze

The first audit identified that completing a raw trial payload overwrote the
already saved short-trace NPZ. This is repaired before the source freeze:
completed payloads write to an exclusively created separate partial NPZ, flush
and synchronize that file, then atomically replace the completed path. Failed
partial files remain as evidence and cannot be silently reused. An injected
write failure confirms that the earlier completed file remains byte-identical.

The raw verifier now reconstructs each declared arm from the frozen anchor and
saved local-sensitivity direction; it checks the exact weight delta, parameter
values, raw initial state and quarter-cycle epoch. It checks the sensitivity's
initial state and zero initial activation derivative, reclassifies periodicity,
and verifies the recorded period, residual, FTLE and statistics. Stub tampering
tests reject a swapped arm, wrong phase, altered initial state and altered RMS.

Raw hashes remain exact. Derived floating-point values and final summaries use
relative tolerance 1e-12 and absolute tolerance 1e-14; dictionary/list structure,
integer counts, booleans and strings remain exact. For waveform metrics, every
saved aggregate, component and window RMS is recalculated directly at the
recorded single global phase. No window receives an independent fitted phase.

Fresh phase minimization is a separate audit because the final digits of a
scalar optimizer's argmin and its evaluation count can vary across platforms.
The frozen bounded optimizer uses `xatol=1e-10` plus an internal stopping term
`sqrt(2.2e-16)*abs(phase)`. Its brackets lie within `[-1/128,1+1/128]`. The audit
uses a phase allowance of
`4*(sqrt(2.2e-16)*(1+1/128)+1e-10/3)`, covering two terminal brackets, and multiplies
it by the maximum RMS slope of the periodic linear reference interpolant.
Adding `1e-14 + 1e-12*abs(fresh_RMS)` gives the allowance for comparing the saved
phase's RMS with a fresh numerical optimum. This allowance changes only the
portability check on reoptimization: saved scientific scores remain tightly
bound to their actual recorded phase, and certificate, eligibility, missingness
and analysis rules are unchanged. Optimizer evaluation counts remain recorded
provenance. A synthetic constant-waveform test confirms that equivalent phases
and differing evaluation counts pass without pretending the phase is unique.
