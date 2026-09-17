# EXP-017: diagnose the phase-reference normalization failure

17 September 2026. Exploratory, outcome-selected numerical diagnostic after
[EXP-016](EXP-016-results.md). It does not requalify that incomplete study.

**Human-friendly TLDR:** Checked near-edge cases retain readable input history,
but reference errors left some starting phases untested. This study checks how
accurately we locate and describe the repeating activity pattern. It will not
measure new memory performance or turn numerical reference failure into a
claim about the network forgetting.

## Fixed question and inputs

The saved-evidence diagnosis finds eleven event-only normalization failures.
All 48 initial normalization checks and other reference-method gates pass.
Reference integrators agree much more closely than the observed event defect.
Separate state/adjoint interpolation error from inherited approximate periodicity
and integration error, retaining all 48 original contexts.

Use the two qualified reference packets from EXP-014 recovery attempt 002 and
the exact 24 phase fractions in the EXP-016 frozen plan. Keep each original
certified section state, period, parameters and terminal phase-gradient vector.
Bind the original receipts, packet bytes and all 48 EXP-016 context seals.
The latter provide the original Hermite initial state/gradient and primary
DOP853 integrated event state, with their original failed flags. Old-versus-new
event comparisons use that primary old state; this study does not reload the
old RK45 trajectory or recompute its event comparison. The preceding saved
diagnosis already retained the old two-method agreement. Do not consume finite-input
trajectories or decoder outcomes in the new calculation.

## Six base reconstructions and 96 event continuations

For each setting use all three methods, without selecting one after results:

| ID | Algorithm | Relative / absolute tolerance | Maximum step |
|---|---|---|---|
| baseline-dop853 | DOP853 | 1e-12 / 1e-14 | 0.025 |
| fine-dop853 | DOP853 | 1e-12 / 1e-14 | T/8192 |
| fine-rk45 | RK45 | 1e-12 / 1e-14 | T/8192 |

Each of six base jobs directly integrates state and all nine tangent entries
from the unchanged section state over [0,T]. Integrate the adjoint backward
over this same dense state trajectory, with the **unchanged original terminal
gradient**. No new eigenvector, scalar normalization, shooting correction or
replacement period is allowed. Save dense evaluations at all 24 exact phase
fractions, state/tangent/adjoint arrays and physical derivatives on both
4096- and 8192-interval grids, raw endpoints, method parameters and solver
diagnostics. This is reconstruction from an existing certificate, not new
primitive-orbit certification.

For each of the 48 contexts, independently continue each finer method's own
exact dense phase state for one period: 96 further jobs. Pair the resulting
physical event flow with that base's exact phase-gradient sample. Retain
complete event state and fresh-identity tangent traces on 8192 intervals,
with their derivatives. No
gradient is transported forward through a collapsing inverse tangent matrix,
renormalized, or forced to pass. All numerically completed bases feed their
planned continuations regardless of a diagnostic gate failure. Only absent or
incomplete prerequisite data block a dependent job, with its identity retained.

The ledger has **102 jobs** (six base jobs each contain a forward and a backward
solve). Failed, interrupted and not-started identities remain in the ledger.
No finite write, intervention, observer score or new noisy trial is computed.

## Saved-array analysis and interpretation

Report all six reconstructions, every pair of base methods, all 96 event jobs,
and all 48 paired contexts. Compare the 4096- and 8192-grid periodic cubic
Hermite estimates with each method's own exact dense phase samples, retaining
state, adjoint and physical-normalization discrepancies. Preserve endpoint
closure and adjoint periodicity instead of enforcing a periodic wrap on dense
samples. Compare old Hermite values and their event defects with each new
dense construction explicitly. Differences between old and new inputs are
diagnostics, not evidence that the original frozen task has passed.

The unchanged normalization criterion is `abs(z dot f - 1) <= 1e-7` at initial
and event states. Fine-method state and activation comparisons retain thresholds
1e-6 and 1e-7; full tangent differences are scaled by the original 1e-4 pulse
magnitude before the same state/activation relevance thresholds. Record raw
operator differences as well. Reference relative closure and adjoint relative
periodicity use the original 1e-8 and 1e-7 limits as labeled diagnostics.
Completion, arithmetic replay and scientific gates are separate fields.

If fine methods agree and exact dense starts remove the event defect while
the reconstructed coarse Hermite error tracks the original defect, interpolation
is a supported explanation. If defects persist with exact dense samples and
agree across methods, inherited closure/phase consistency needs further work.
If the finer methods disagree, solver reliability remains unresolved. Mixed
results stay mixed. No favorable-method selection, averaging away failures,
threshold relaxation or retrospective EXP-016 qualification is permitted.
A future two-write experiment must specify and freeze its own inputs and all
controls before any repeated finite-input outcomes.

## Local execution and verification

Use four CPU workers, one thread per numerical library, at most 180 seconds
per job and 1,800 seconds for the producer, plus 180 seconds for sealing.
Require 16 GiB available RAM at launch, maintain an 8 GiB host reserve, cap
owned sampled RSS at 12 GiB and raw artifacts at 8 GiB. These conservative
ceilings are not an estimate of consumption. EXP-016's 96 eight-period
reference jobs used about 476.75 summed worker seconds. Scaling their duration
suggests roughly 60 summed worker seconds for 96 one-period continuations,
plus reconstruction/adjoint work; this is only a planning proxy.

Inspect shared resources at launch, register exact owned PID/start identities,
retain partial solver output, bound worker lifetimes and verify cleanup. Use
the already tested owned-worker runtime; do not disturb foreign processes.
No GPU benchmark or paid execution is included. Place immutable raw evidence
under `../neural-shrimp-data/raw/EXP-017/attempt-NNN/`, exclusively.

Before new solves, freeze source, synthetic tests, this protocol, the complete
machine plan and local review to the private branch and verify the remote
commit. The verifier checks receipt inventory, historical/source bindings,
ledger/ownership chronology and recomputes summaries from saved arrays without
new integration. Keep every prior failure unchanged and include a behavioral
TLDR in the resulting report.
