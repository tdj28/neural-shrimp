# Post-edge local scientific audit

Read-only independent review on 7 October 2026 (user time zone). This record
binds the snapshots reviewed, rather than claiming approval of later edits.
No paid service or experiment was used; tests and saved-evidence replays ran
on one local CPU thread with nice 10. Other agents' worktrees were untouched.

## Scalar mechanism

Reviewed commit `894ee5b8c967d06764a4f3071949e5ffbd7d3578`, including
`scalar_mechanism.py`, runner, analyzer, tests, receipt and the scalar addition
to `docs/explore-minimal-model-2026-10-07.md`. An isolated Git archive of that
commit was used for execution.

No blocking mathematical or implementation error found. Independently checked
the gradient, Hessian, composed second/third derivatives, cubic mean/difference
cycle equations, explicit balanced cycle, longitudinal/transverse multipliers,
Jury margins and minimum flip-rate formula. Positive-rate symmetric similarity
and the branch-specific real-spectrum argument are correct. The fixed regression
target is 1.2 in every retained grid case.

Seven scalar tests pass. The raw hash/receipt regeneration passes and reproduces
12 flip and five NS candidate outcomes; the three-point amplitude exponent is
0.4978966467268106. Direct checks of all five saved NS roots find distinct
period-2 points, complex unit-modulus multipliers, positive crossing slopes under
step halving and negative analytic coefficients. The minimum reported angular
distance from a strong resonance is 0.00676915340915446. The largest saved
augmented-root residual is 8.881784197001252e-16, and maximum analytic versus
finite-difference coefficient discrepancy is 2.823465905521516e-9.

Supported scope: exploratory numerical mechanism in the selected branch of a
two-weight model, and balanced-branch algebra. This is not a rigorous invariant
circle, a complete phase diagram, global necessity of ridge/asymmetry, or a
training/geometry benefit. Scalar cycle formulas reduce to Hofmann's balanced
cubic formulas under rescaling; cite that prior derivation when including them.

## Warmup and shared-state interventions

Reviewed commit `2c85cfbeaccf248b58fef2dfa5f7b8854042e1ce`, including `warmup.py`,
runner/checker, tests, receipt and the criticality summary. An isolated Git
archive was used. No blocking implementation error found.

Eleven warmup/criticality tests pass. The raw audit hashes 12 files and exactly
replays 704 sampled cell transitions and both per-update losses. Independent
checks also establish:

- Every fork has the identical common terminal state and the same comparison
  horizon; cell indexing correctly pairs the two starts with all three policies.
- All 82 frozen references close in both phases, with maximum absolute residual
  6.002143226879753e-16; reference loss means/maxima match recomputation exactly.
- Full-spectrum dominant log moduli match saved alpha exactly; the largest
  noncritical multiplier modulus is 0.9807054889982986.
- Every saved postcrossing peak, state-distance maximum and terminal loss ratio
  reproduces independently from raw arrays and the stated initial-orbit baseline.
- Losses refer to the two pre-update states of each pair, whereas saved states
  are pair endpoints. The checker and reported endpoints respect this indexing.
- Global reduction deliberately has no continued local orbit reference. Local
  departure is excluded for that arm; its loss recovery benchmark stays shared.

Supported scope: two previously inspected points of S2-28011 under matched
local multiplier offsets and two finite-horizon ramp schedules. Near-orbit
hard/soft behavior differs; ordinary hard initialization was already distant
from the local branch and is only an accessibility control. Reversal and
halving are different-sized interventions; this comparison establishes neither
an optimized controller nor geometry-specific advantage. Halving wins the
reported terminal-loss comparison. Loss spikes alone do not classify criticality.
No rigorous chaos, universal hysteresis, or transformer consequence follows.

## Manuscript snapshot

Reviewed `paper/gd-route/main.tex`, README, bibliography and reference ledger at
`a9e73a96`, compared with `bc9c526`. The new evidence table, heavy-ball qualification,
continuous-symmetry caveat, negative ancestry result and pending citation-content
labels materially improve accuracy. The following issues were sent to the
coordinator for correction; line numbers refer to this snapshot:

1. Lines 343/348: a unit-circle crossing is not the point where multipliers first
   become complex. Use "complex pair crosses the unit circle".
2. Lines 191-193: a real fixed-point spectrum excludes NS, but does not make every
   +1 crossing a fold. Qualify symmetry/degeneracy and the minimum/rate-scaling flip.
3. Lines 285-288: the implementation orders F-samples for odd periods and F^2
   samples for even periods. State q=p for odd p, q=p/2 for even p; p=2q is not
   universal.
4. Caption 412/body 426: separate projected trajectories do not demonstrate
   topological circle breakdown. Describe a broadened/irregular projected state
   with finite-time chaos evidence; update matching figure labels/alt text.
5. Lines 390-391: negative d and generic NS hypotheses imply invariant-circle
   onset, not necessarily irrational quasiperiodic motion; locked dynamics remain
   possible. Scope claims to numerical criticality and the tested trajectories.
6. New scalar integration should attribute the rescaled balanced cubic cycle and
   longitudinal multiplier to the existing cubic-map analysis, while identifying
   the actual new unequal-rate/full-state and fixed-target calculations.

The coordinator reported applying items 1-4 and is integrating the new reviewed
results; this record does not claim the resulting source/PDF has been rechecked.
Load-bearing primary-source spot-check:
[Hofmann 2026, arXiv v1](https://arxiv.org/html/2607.04993v1), especially the
explicit cycle and multiplier in section 2.3 and rotational-edge analysis in
appendix E.2. Version v1 was current when accessed. No claim that every other
reference's full text has been checked; the ledger's pending items remain open.
