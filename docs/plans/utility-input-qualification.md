# Saved-input qualification for the matched-pulse candidate

8 October 2026 UTC; exploratory design amendment to
[the candidate memo](utility-matched-pulse-candidate.md). Root authorized only
read-only auditing of consumed development artifacts. Base 8c26537e71ff12f8e6a4b7916775cc8160e3b8a1.
No model map, gradient, correction, HVP, numerical profile, pulse or protected
outcome is evaluated. Prior warmup outcomes and consequence summaries have been
seen; no equal-dose pulse outcome exists. Source/schema inspection already shows
that consequence tail states and warmup intermediate one-update states were not
exported. This is a qualification decision, not an experiment launch card.

## Fixed input and selection rules, before the inventory calculation

Only S2 seed28011 training-system artifacts are eligible. Do not read test-MSE,
probe, anneal, response-after-intervention or reserved-seed arrays. Use the warmup
attempt002 common traces at dwell100, both origins, with k the first saved pair
end having alpha>=0 (use recorded floating values; do not round them). Fork is
state[k,origin], after exactly 2(k+1) updates. Recovery is reference[0] with its
recorded log multiplier near -0.0015. Pin all binary64 coordinates and hashes.
Use the preceding 1000 completed pairs k-999..k for normalization, and the
recorded loss[k+1,origin,0] for loss at the fork (that loss precedes the next
update). Pair-end differences are only two-update net displacement, not the
one-update scale or oscillation amplitude needed for matching a period-two map.

A future control must lie at least **0.20 decades in log-rate infinity distance
from both local crossing controls**, and separately have an independently traced
ordinary periodic branch or the explicit chaos qualification below. Distance is
a candidate filter, never evidence of non-shrimp topology. Four controls, one
periodic and one chaotic per local point, must be distinct within each class.
Nominate only for future qualification; never call a summary label qualified.
Nomination sorts by maximum absolute log ratio of the saved tail mean loss to
both origins' local pre-fork means, then source path/row index. Record all eligible
candidates, not only nominees. Consequence candidates: stored status2/period2;
chaos candidates: nonperiodic, both stored FTLE estimates>0.005 and final median
on-attractor separation>1e-6. EXP028 alternatives: attracting numerically accepted
non-embedded period2 records, or the original/rerun chaotic status with both
FTLE>0.005. EXP028 retains complete periodic phases or a coarse final state, but
not the required qualification tail. Preserve this distinction.

Before any future pulse, both origins at a point must have common support with
its two controls: **each initial loss and pre-fork 1000-pair mean loss within a
factor2**, actual one-update norm at recovery rates within a factor2, all-update
oscillation RMS over 2000 updates within a factor2, and fixed 16-update maximum
amplification of the same two coordinate-direction perturbations within a
factor2. Define max/min<=2; nonpositive/nonfinite/missing values fail support.
These bounds cannot be widened after seeing a pulse. Missing saved variables are
unknown, never replaced by pair-end cancellation, sharpness or initial-perturbation
final separation. Retain every fork and continuous covariate even if support fails.

Control recovery maps are provisionally both rates multiplied by the matched
local point's hidden recovery/crossing-rate ratio; no optimization of a control
backoff. Record the regimes at both original and recovery controls. All four pulse
arms and their exact-displacement equality/25% group cap remain as in the memo;
no gradient is evaluated now, so dose feasibility is unresolved. Before launch
require relative dose error<=1e-12 and absolute floor1e-14 under float64, with all
arms' first-update increments computed from the identical state and gradient;
after the pulse, every arm takes16000 identical-recovery-map updates (8000pairs),
with the pulse row retained separately. All1000-pair endpoints remain unchanged.
Here equal dose means ||theta1_arm-theta1_sham||_2=R, not equality of the total
GD step lengths ||theta1_arm-theta0||_2. This clarifies the memo's ambiguous
“actual displacement” wording without changing its formula. Retain total step
lengths as competing generic explanations.

Ordinary periodic controls require full-state primitive closure with atol1e-9 /
rtol1e-8, relative correction<=1e-10, divisor separation>=1e-6, all multiplier
moduli<1-1e-7 and independently checked branch continuation away from the local
region. Chaos requires leading FTLE>0.005 per update at fixed4096/8192-update
horizons and each of four2048-update blocks, two fixed on-attractor perturbations
of norm1e-10 reaching separation>1e-6, no period<=64 under the EXP028 screen,
and a separately implemented or higher-precision refinement with the same label
and exponents agreeing within max(20 percent,0.001). No finite-horizon gate proves
mathematical chaos. These are future acquisition requirements, not computations
in this unit. Unsupported recovery regimes or insufficient common support stop
the specificity test; do not infer a different basin from finite nonreturn.

## This unit's bounded read-only audit

Implement tools/audit_utility_inputs.py with NumPy/standard-library only; no model
imports or numerical dynamics. Read exactly the warmup receipt/manifest/common
traces/references, the two existing S2 consequence slice metadata and selected
training-only array fields, and the EXP028 receipt, S2-28011 coarse/zoom archives,
periodic records and random-chaos records. Bind source metadata honestly: old
consequence metadata records a commit but not a clean-source assertion. Hash
files without evaluating their stored protected-outcome fields; never load those
arrays. Report missing fields and explicit provenance limitations.

Commit this card/code before deriving inventory. Run once, CPU/BLAS1 nice10,
internal30s/external40s+5, new output<=2MiB, no paid/GPU resource, under
/data2/neural-shrimp-data/audits/utility-inputs/attempt-001. All output/failure
stages remain. Synthetic controls cover fork timing/history, finite/support
rules, deterministic nominations and exact binary64 state identities. Report
actual audit I/O time separately from historical training throughput; it is not
a dynamics profile and cannot authorize a pulse or acquisition.
