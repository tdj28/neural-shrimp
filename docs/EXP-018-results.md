# EXP-018: the globally refined reference passes all 48 phase checks

17 September 2026. Exploratory numerical reference study. All 102 worker jobs
complete and all 48 paired phase contexts qualify under the prospectively
frozen rules. This is a new reference, not a repair of historical outcomes.

**Human-friendly TLDR:** Some previously checked near-edge cases retain a
readable input trace for several cycles, and a later input interferes with it.
We now have a repeating-orbit and phase reference that passes every planned
one-period check, resolving the numerical failures that left some phases
untested. This experiment measures no new memory performance. A fresh complete
eight-period memory experiment is still needed before comparing both settings
across the whole phase set.

## What changed and what passed

At each unchanged parameter setting, DOP853 and RK45 independently refine the
original x3-maximum section state and period. Both take one Newton update,
using two IVP evaluations each. Each resulting candidate is verified by both
methods without moving it. All eight verification paths retain nine maxima,
full-state primitive return separation, a unique neutral direction and
transverse attraction.

The DOP853 candidate was canonical in advance. Its newly verified monodromy
supplies one globally normalized phase vector per setting, used unchanged as
the terminal adjoint in both backward reconstructions. Every phase and event
gradient remains unrenormalized. All 96 one-period event continuations finish,
and every initial/event normalization and two-method agreement check passes.

| Frozen diagnostic | Core | Near edge | Required bound |
|---|---:|---:|---:|
| Qualified phase contexts | 24/24 | 24/24 | Every planned context |
| Maximum relative closure across four verification paths | 1.013e-14 | 1.636e-13 | <=1e-12 |
| Maximum relative adjoint endpoint mismatch across two methods | 7.468e-13 | 4.199e-12 | <=1e-10 |
| Maximum event normalization error, DOP853 | 2.992e-12 | 8.407e-11 | <=1e-7 |
| Maximum event normalization error, RK45 | 1.423e-11 | 1.343e-10 | <=1e-7 |
| Canonical dominant transverse multiplier magnitude | 0.06506010 | 0.85603323 | <0.99999 |

Across all 48 event-method comparisons, maximum state disagreement is
4.267e-12 and maximum activation disagreement is 2.663e-12. The largest
epsilon=1e-4-scaled full-tangent and activation-tangent disagreements are
1.391e-13 and 1.221e-13 respectively. The unchanged corresponding relevance
limits are 1e-6 and 1e-7. These checks use all nine tangent entries.

[EXP-017](EXP-017-results.md) left nine of these same 48 contexts failing in
both finer methods, with maximum event error about 1.70e-7. The new recipe
reduces the largest error to 1.34e-10 and all contexts pass. It jointly changes
global shooting, global adjoint construction, integration tolerance and step
density, so it does not isolate the causal contribution of each change.
The reused phase set is a regression census, not fresh held-out evidence.

## Behavioral interpretation and limits

The refined transverse multipliers preserve the earlier contrast: small
transverse disturbances contract much more strongly per cycle at the core
than at the selected edge. That supports the existing local explanation for
their different forgetting rates. Actual readable memory still depends on
the input direction, starting phase, observation noise and readout. The
multiplier alone is not a memory-capacity measurement.

The [EXP-016](EXP-016-results.md) behavioral observations remain individually
qualified examples: one post-hoc near-edge example has about 93.5% first-input
accuracy without a second input and 70.0% with it, under its fixed observer
and analytic Gaussian measurement-noise assumption. That incomplete study's
setting means remain withheld. New numerical references cannot relabel its
missing histories as completed or substitute for new finite-input trials.

EXP-018 establishes neither eight-period reference accuracy nor all-phase
validity. It does not complete the shrimp boundary organization, demonstrate
an advantage over ordinary slow dynamics, or establish transfer to trained
neural networks. No finite input, fitted decoder or new memory score is part
of this experiment.

## Evidence, resources and next experiment

The complete [protocol](EXP-018-protocol.md), executable plan, 100 passing
synthetic tests and [local review](EXP-018-local-review.json) were privately
pushed before target acquisition at source commit
`7c9b1977f5d12e4354ff41c537778a3a284a4027`. Plan SHA256:
`b1bd00f851bff8ef600745e09941682c6c58829203180b9e1cf238f9d3eef143`.

The run takes 128.41 seconds on four local CPU workers. It uses 116 IVP
integrations; all 208 planned slots remain visible, including 92 unneeded
shooting trials. The 333 raw artifacts total 207,116,405 bytes. Peak sampled
owned RSS is 666,304,512 bytes, counting shared pages per process. No paid
compute or additional Pro call is used.

Raw evidence is immutable at
`/data2/neural-shrimp-data/raw/EXP-018/attempt-001/`. Receipt SHA256:
`b4b56a8fbcc592285cebdbf15c2182a3173c62d3e770ebc1681cd9a142066ce1`.
The [compact summary](EXP-018-summary.json) retains all 48 contexts, candidate
and reference diagnostics, source bindings and resource observations. Frozen
verification confirms the byte inventory, saved-array replay, qualification,
full ledger and absence of all original owned process identities. Separate
saved-data audits are hash-bound in the compact summary.

The independent scientific/evidence audit passes 9,104 evidence and decision
checks. Its alternate arithmetic fails 10 of 638 strict comparisons (725
elements; maximum failing difference 7.5745e-14), with no qualification
decision changed. This reviewer authored the controller, but not the scientific
acquisition or evaluator. A separate algorithmic audit by the evaluator author
passes 63,395 identity, 66,081 declared-arithmetic and 6,998 gate checks; one
of 106 alternate-arithmetic comparisons fails (difference 3.8802e-14), again
without a gate disagreement. Neither audit is an overall strict-arithmetic
pass. Their original tolerances and failures remain recorded; no threshold
was relaxed to permit the next experiment.

The next study repeats the complete two-write task under a new freeze, using
the new canonical exact phase states and gradients. Both eight-period methods
must start from the same canonical state. All 96 fresh reference jobs and all
48 context/observer prerequisites must qualify before any of the 2,700 planned
finite-input traces runs. This global gate is an explicit prospective change
from EXP-016's per-context dispatch; failures and blocked histories remain in
the denominator. The original input controls, noise and readout definitions
remain fixed, and the reused phases retain their regression-study status.
