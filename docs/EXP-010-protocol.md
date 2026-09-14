# EXP-010: qualify all original short-preview labels

Status: prospective execution design for a post-run numerical qualification,
14 September 2026. EXP-008 and EXP-009 outcomes motivated this packet. No
EXP-010 integration exists at drafting. Commit and privately push the complete
plan, source and tests before execution. Do not change any frozen EXP-009
file. This packet trains no model and establishes no new controller benefit.

## Question

EXP-009 leaves many 300-unit candidate outcomes numerically unresolved while
the old selected actions remain steady. Are the shorter, 75-unit preview
errors stable enough to consider as labels for a later pre-preview shortlist?
The qualification covers every original preview, so numerical agreement is
not selected from favorable new outcomes. It does not resolve the later task
error or authorize replacing that endpoint by preview quality.

## Fixed complete census

Use all 3,456 entries in the receipt-bound EXP-008 development and test preview
journals: 24 origins, two phases, four radii and 18 ordered candidate entries.
The two roles each contain 1,728 entries. All duplicate directions and their
distinct candidate IDs remain separate; each receives its own integrations.
Exactly 192 origin/phase/radius contexts contain all 18 candidates. Preserve
the old test policies' 384 assignments to 220 unique candidate previews as
an explicitly labeled diagnostic subset, without recomputing those choices.

Use the exact old raw preview initial state, destination alpha/gamma, control
delta and sample times. Bind the original receipt, manifest, both preview
journals, every raw preview, both reference-baseline files, old test decisions,
and every executable source dependency by SHA-256 and byte size. Validate the
complete expected key product before selecting any independent sentinel.
No failed/missing old record may be substituted or silently omitted.

The independent panel is selected only from original preview scores. Within
each `(role, radius)` stratum of 432 entries:

1. Select the largest original preview RMSE; exact score ties choose the
   lexicographically smallest `(role, anchor, phase, radius, candidate)` key.
2. Sort by ascending `(original preview RMSE, canonical key)` and select
   zero-based index 215, the lower median rank of the 432 entries.

Keep both selection labels and deduplicate only the independent panel. This
gives exactly 16 distinct sentinels before new outcomes. Their exact IDs and
old scores are retained in the machine plan. This is a fixed diagnostic
panel, not an independent random sample or a population error estimate.

## Numerical methods and unchanged endpoint

Every preview restarts from its exact old initial state and runs over
`0 <= t <= 75`, retaining all 751 samples at cadence 0.1, including both
endpoints. Run native float64 RK4 at `dt=0.0025` and `dt=0.00125`: 30,000 and
60,000 integration steps, with output strides 40 and 80. This supplies 6,912
mandatory native traces. Do not reuse a prefix of a 300-unit trace.

Run each of the 16 sentinels with both independent Python DOP853 settings
used in EXP-009: `rtol=1e-11, atol=1e-13, max_step=0.05` and
`rtol=1e-12, atol=1e-14, max_step=0.025`. These 32 traces are mandatory
regardless of native outcomes. Total workload is **6,944 traces**.

The label is the original all-neuron activation RMSE over the whole preview,
against the original certificate's 2,700 endpoint-excluded activation samples.
The phase epoch stays `phase_index * original_period / 2700`. Fit one constant
phase on the entire preview with the unchanged `waveform_errors` procedure
and 128-bin search; use that same phase for all component and window scores.
There is no time warp, frequency fit, independently aligned subwindow, new
reference orbit, new direction solve, changed initial condition or altered
task horizon. Retain synchronous, component, window, phase-search and cadence
metadata. Separately evaluate error at the saved original preview phase.

## Stability, rankings and all-required gates

For every case record the old 0.005 error and both new errors. Define
`d1=abs(old_rmse_0.005-new_rmse_0.0025)` and
`d2=abs(new_rmse_0.0025-new_rmse_0.00125)`. The descriptive scalar flag is
`d2<=1e-4 and d2<=d1`, with its two gates visible separately. The independent
scalar flag requires all three pairwise error differences among finest RK4
and both DOP853 results <=1e-4. These are numerical diagnostic tolerances,
not task noninferiority or proof of convergence.

Compare raw states and activations at equal recorded times over the whole
75-unit interval. Waveform agreement requires all-component activation RMS
difference <=1e-3 and maximum absolute component difference <=1e-2. Preserve
every scalar/waveform disagreement, including scalar agreement with different
waveforms. No post-outcome step halving or favorable solver choice is allowed.

Report complete counts and difference distributions by role, radius and the
original 220 selected-preview subset, with every denominator and failure.
Within each 18-entry context, retain all old/new errors and signed changes,
the complete ranking with candidate-index tie breaks, minimum-error choice,
best-to-second gap and fourth-to-fifth gap. Compare old and refined argmins,
rank positions and four-best membership. These diagnostics are necessary
because small absolute errors can still change a choice near a tie.

For each pair of numerical levels retain the empirical maximum absolute
candidate-score change, compare the earlier best/second and fourth/fifth gaps
with twice that change, and report the signed change in the minimum error.
Report whether the latter exceeds the 1e-4 numerical diagnostic tolerance
separately from an ordering change. These are measured finite-level
discrepancies, not bounds on the exact solution or task-quality tolerances.

Also report whether the minimum within the fixed signed-axis candidate set
`[0,4,8,12]` changes, using only already generated labels. This is a numerical
ranking diagnostic, not a fitted or deployed policy. Do not add candidate
features, fit a shortlist, infer future model performance, or select a
favorable radius/label definition from the results.

Every planned trace gets a terminal status, including unstarted, timeout,
nonfinite or solver failures. Preserve all available partial evidence. A
missing required case makes the corresponding full-group qualification and
context-ranking comparison undefined; descriptive completed-subset summaries
must state their exact denominator and cannot replace an all-required result.
No missing label is silently converted to task failure or zero loss. An
interrupted attempt remains consumed and is never restarted in place.

## Execution, ownership and evidence

Reuse the unchanged EXP-009 bounded scheduler, owned-child entry and cleanup
functions by import. The new worker implements the 75-unit endpoint; do not
modify the old worker or its hashed sources. Use at most four local CPU
workers, BLAS/OpenMP thread limits of one, a 60-second per-trace ceiling and
3,600-second target-execution ceiling. Receipt wall time includes building,
collection, summary and receipt preparation; it excludes the subsequent
saved-array verification. Record any separately measured full-process or
verification timing with its scope instead of calling receipt time an
end-to-end benchmark. Record all owned PIDs and verify
cleanup. Check current shared-server load, memory and output capacity before
launch; no GPU or paid resource is needed for this reference implementation.

The native workload is 311,040,000 RK4 steps. Original preview calls sum to
141.92287686 overlapping call-wall seconds on their old host, including
scoring. A simple sixfold step-scaling proxy is 851.53726113 summed call-wall
seconds, not a measured runtime or speedup on this server. Original raw
preview arrays total 70,436,011 bytes; unchanged sampling suggests roughly
twice that native raw volume plus metadata and independent traces. Reserve
2 GiB of external space.

The default exclusive output is
`/data2/neural-shrimp-data/raw/EXP-010/attempt-001/`; any configurable data root
must resolve outside the Git checkout. The run requires clean tracked and
untracked source, committed complete dependencies and plan, and a live origin
branch HEAD matching the exact source commit. Save the full manifest, build
and runtime provenance, all raw traces and records, append-only status/PID
journals, complete summary, hashes and receipt. Hash/audit the full census
and replay the complete score tree from saved arrays before interpretation.

This packet can qualify numerical labels for later design. It freezes no
four-preview utility study, claims no improved action quality, and does not
clear either EXP-009's unresolved long-target cases or the original EXP-008
strict replay qualification.
