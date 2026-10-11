# Utility next step: match the intervention first

8 October 2026; design only, adapted after the warmup and negative LM pilots.
Base `960ab535e1954e6f46dce21999338ddcf1bd23ee`.
**Go for design/qualification; no-go for target execution now. Defer the
frozen-curvature LM candidate.**

**TLDR:** Local geometry explains departure and accessibility in the small
network, but has not beaten ordinary backoff. The next defensible question is
whether the *direction* of an equal-sized rate impulse changes basin recovery
when subsequent rates are identical. This tests a mechanism, not a shrimp or
LLM benefit.

The [warmup summary](../explore-criticality-2026-10-07.md) and
[receipt](../warmup-intervention-receipt.json) show hard-point escape and soft-point
return, but halving both rates gave lower terminal loss. The retained reference
files give hidden-rate reversals of only **1.792%/2.293%**, versus **50% in both
rates** for the generic arm. Dose and time profile were unequal. Ordinary hard
initialization already missed the local branch before crossing. Manifest:
`dca4a1a821859714d6ebaa033a8f75f62f66a6e68f3cbd7c83801d046c64ec61`.
The [consequence pilot](../explore-consequence-2026-10-07.md) likewise attributes
most behavioral variation to generic dynamics.

## Minimal matched pulse

Reuse the two inspected S2-28011 hard/soft points, dwell 100, first pair-end state
with nonnegative log-multiplier, for both retained start origins: **four local
forks**. Pin exact state hashes/indices before execution. Record accessibility
before treatment using the existing full-state distance 0.01 rule and continuous
distances; never exclude an inaccessible case after seeing recovery.

Every arm first switches to that point's saved stable recovery rates
(`log|mu|=-0.0015`). Only its first update differs: sham/common recovery alone,
uniform additional 5% rate reduction, hidden-only reduction (the old reversal
direction), or output-only reduction (simple alternative allocation).

Match **actual first-step parameter displacement**. At the shared fork define
`s_h=eta_h*g_h`, `s_o=eta_o*g_o` in disjoint groups and
`R=.05*sqrt(norm(s_h)^2+norm(s_o)^2)`. Uniform 5% gives norm R; hidden/output-only
fractional reductions `R/norm(s_h)` and `R/norm(s_o)` give the same norm exactly.
Require finite nonzero group norms and reductions <=25%; an infeasible arm stops
positive interpretation without changing R. Verify dose numerically. After the
pulse every arm uses the **identical recovery map for 8000 two-update pairs**.
Subsequent differences therefore concern state response under a common map,
not a larger continuing reduction. Finite-horizon nonreturn is not proof of a
different basin. This is not a sustained-controller test.

Mechanistic endpoints: full-state phase-invariant reference distance, return
time, and distance<0.01 throughout the final 1000 pairs; retain failures and
continuous values. Return is defined only where a qualified periodic reference
exists, never by inventing a cycle for chaotic controls. Shared practical
endpoints across all settings: first 1000-pair mean loss, peak loss and final 1000-
pair mean, normalized by the preceding 1000-pair mean loss (floor 1e-12), fixed before forks.
A useful pulse must
beat **uniform and sham** by 5% on early mean loss with no worse peak or terminal
mean at each ordinary-start local fork; otherwise no useful-direction claim.
Near-orbit forks are mechanism controls, not substitutes for ordinary starts. Keep the two
start origins separate and show every fork. Accessibility interactions are
descriptive; near-orbit robustness does not imply recovery from ordinary starts.

## Specificity requires controls before outcomes

Follow [EXP-027](../EXP-027-status.md): two distant ordinary-periodic and two
distant chaotic development settings in the **same S2 network**, one of each
matched to each local point. The consequence slices offer candidates, not
qualification. Before pulse outcomes, pin coordinates/states, phase selection,
common recovery backoffs and matching tolerances for initial loss, update/
oscillation scale and transient sensitivity. No common support means no
specificity test, rather than looser matching.

Periodic controls require primitive full-state closure/stability and an
independently traced ordinary branch, not appearance. Chaos requires positive
leading FTLE at two fixed horizons and in blocks, on-attractor perturbation
growth, and precision/refinement checks; unresolved irregularity stays unresolved.
These are numerical qualifications, not infinite-time proofs. Record regimes
before and after backoff separately. Apply identical pulse arms/horizons:
**eight forks, 32 tails** total. An effect reproduced in these controls, explained
by contraction/transient gain/accessibility, or restricted to hand-placed states
remains generic. Any geometry claim needs further fresh blocked replication
against local predictors; a local success cannot waive those controls.

## Cost, launch gate and LM decision

Control coordinates and qualification are unpinned: **no target launch**. Next
would require reviewed committed code/run card, synthetic exact-dose/common-map
checks and a representative resource-only qualification/rollout profile on
consumed development material. Proposed CPU/BLAS 1, nice 10, float64 ceilings:
profile 60s, complete qualification+collection 180s, analysis 30s, cleanup 5s/process,
256 MiB new raw retention. The old 63.65s CPU 1 warmup is context, not a throughput
guarantee. Profile the entire required workload; stop on cap, failed control or
infeasible dose. Preserve all outcomes. No GPU, paid call, sweep, reserved seed
or protected data.

**Defer my [LM frozen-curvature candidate](explore-tiny-lm-frozen-curvature-candidate.md).**
The [independently audited broader-data pilot](../reviews/tiny-lm-broader-outcome-audit.md)
again loses to simple forecasts. A transported-derivative match would validate
expected calculus; frozen-curvature failure would confirm anticipated time
variation. Success at four tiny checkpoints would not justify a frozen
approximation in a larger model. Neither result changes a current controller or
resource decision. Preserve the design without running it; revive it only when
a specified larger-model method genuinely must choose frozen versus updated
HVPs at matched cost. Main S1R branch identification remains separate.
