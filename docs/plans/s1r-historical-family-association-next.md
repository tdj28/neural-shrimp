# Next S1R historical-family association question

8 October 2026 UTC. Design only: no implementation, target evaluation, profile
or launch accompanies this note. Basis: producer source
`db10bd1e1b25b0e24b1f0f8f97cb84035742c776`, outcome
`4e50f004ad6f5db80be0dea01fdf404eec71bd1b`, independent saved-outcome review
`dfc6de4c7a803b1e810d4842ca76ee61ed686a6c`.

**Smallest next question:** at fixed log10 output rate
`0.6271428571428572`, does a new reproduction of the historical hidden-1.10
bootstrap qualify as the same full-state period-two pair as the reviewed saved
lineage, when that lineage is continued back to exactly hidden 1.10? Reproduce
the hidden-0.95 bootstrap once in the same bounded unit and preserve it as a
separate initial-condition outcome; do not imply that its association is
settled before a connected common-parameter comparison exists.

## What the historical procedure actually did

`tools/explore_a3_ns_curve.py:one` independently regenerated
`SYSTEMS['S1R'](28002)` for every output slice. Its unchanged source was added
at `0e3e07e1723bf357ff24cba6d9906a1276dca9dd`; current file SHA-256 is
`97976e2b7660b0f914155acd9e7601b7ee202e8d5582f74f2ddfa9661cb8ae03`.
Each slice used one grid cell and the same deterministic theta0, followed by:

- Exactly 40,000 simultaneous float64 updates at hidden log-rate 1.10 or 0.95
  and the common output log-rate above; rates are `10.0**float(log_rate)`.
- `run_grid(..., tail_steps=384, ftle_steps=1000, ftle_align=200,
  sd_horizons=())`. Remaining defaults: maximum period 64, six repeats,
  recurrence atol 1e-8/rtol 1e-6, tangent seed 0, chaos threshold 5e-3,
  divergence threshold 1e8, drift threshold 1e-2 and screen batch 256.
- Continuation only if the recorded smallest tail period equaled two. The old
  ray then increased hidden log-rate with initial step 0.002, boundary bisection
  1e-10, and range 0.6 or 0.8 for starts 1.10 or 0.95 respectively.

The common output is historical row index 3 of both 36-row traces; the saved
criticality orbit used by the new pilot is criticality row index 2. Historical
trace hashes remain `3c69306e7427eadbe98fcd51132d7ccf4ba417a8eee80ed7d031db990194b30e`
and `3ae8ac5ccfcead978ff43953f064e71e4ff832418040ccb24314401b8dcb1ea7`.
At this output their reported hidden boundaries are 1.12556987032888 and
0.9770019149966538, with angles 0.12180206007240837 and 0.10677922599851965.
These are historical summaries, not full-state matching evidence.

The old run cards identify `959beee68b3a` **with a dirty worktree**. Their JSON
files retain neither bootstrap states, full paths nor boundary states. Thus a
new frozen run can reproduce the recorded procedure once per start; it cannot
prove byte-identical reproduction of unknown historical execution sources or
recover historical full-state ancestry from summary agreement. Do not run the
entire old tracer as the next experiment: its broad stability search is outside
this question, and its coarse summaries discard the evidence now needed.

## Proposed first unit

1. Bind the reviewed input/model identities and immutable local-pilot manifest
   `45c35991b7ec56f7d9888b6dcf28434396ff94f65de7bf90ba9822408c5f3c8b`.
   Reproduce each 40,000-update bootstrap exactly once, sequentially, from the
   original theta0 with one cell and the historical arithmetic path. Preserve
   accepted-update counts, every 1,000-step checkpoint, final tail and both
   terminal phases. The final tail contains states theta39617 through
   theta40000; loss values evaluated in those loop iterations describe
   theta39616 through theta39999. Keep these indices explicit. New durability
   hooks must not alter map arithmetic; verify this on synthetic short runs.
2. Retain the historical tail-screen label unchanged. If it is not period two,
   record `bootstrap_not_period_two` and do not force a root from that run or
   extend its horizon. For each period-two candidate, independently correct
   both phases and apply the reviewed full-state, primitive, correction-size,
   candidate-distance and complete-spectrum gates. Keep failed qualification,
   finite-horizon basin accessibility and numerical orbit identity distinct.
3. Requalify saved local node0004 at hidden 1.123569870288833. Continue it
   backward to **exactly** 1.10 at the fixed common output, using 12 declared
   equal hidden-coordinate macro links. Maximum nominal log step is below
   0.002; this is an explicit amendment from the first pilot's 0.0005 hidden
   step. Each link needs full-pair direct, two-half-step and reverse witnesses,
   the original phase-cost/cost-gap/candidate gates and all saved states. Keep
   phase order explicitly; no neuron permutation, sign identification or
   angle-based match is allowed. Refuse an unbridged gap.
4. Compare the continued endpoint and qualified 1.10 bootstrap only at the
   identical binary64 controls. Identity support requires the existing cyclic
   full-pair replay tolerance 1e-7 plus its separation/cost-gap checks. A minimum
   distance over the two cyclic phases above 1e-5, with both separately
   qualified, supports a numerically distinct pair at those controls; distances
   between these cutoffs remain unresolved. Distinct pairs support numerical
   coexistence there; **coexisting attractors** additionally requires both
   complete spectra to pass attraction. A correction/link failure or resource
   stop gives no association verdict and no proof of nonexistence.

Proposed cap: CPU/BLAS 1, nice 10, internal 100 s, external 115 s plus 5 s kill
grace, 64 MiB new raw data, no GPU/paid compute. Two bootstrap qualifications
plus saved-node requalification use at most six phase-corrector calls. Twelve
nominal witnessed links use 96 more. A global cap of 160 calls and at most 19
link attempts leaves bounded room for rejected/subdivided attempts. Retain the
existing four-halving ceiling per macro link and stop at 0.000125 hidden log
step; accepted substeps do not authorize extra global attempts. This cap is a
design proposal to implement and review, not permission to launch.

Existing measured cost is 2.511 s for 42 phase correctors and all evidence in
the local pilot; that is a cost reference, not a guarantee away from its
neighborhood. There is no retained precise single-cell 40,000-step warmup timing.
Before the target unit, commit a separate non-outcome profile: at the already
saved central state, time 2,000 independent one-map calls and 128 independent
JVP calls with a fixed coordinate direction, each call resetting to the same
state. No evolving trajectory, fitting or selection. CPU 1/nice 10, at most
10 s; retain timing and hashes. Project two warmups as 80,000 map calls plus
2,400 JVPs with a 2x safety factor, then add 30 s for the bounded corrector unit.
Proceed under the 100 s internal cap only if that estimate fits; otherwise
split into explicit bootstrap and bridge units with new committed cards. Do
not shorten the historical horizon, batch the two cells together or hide a
partial bootstrap rerun to make it fit.

## What the second family would still require

The 0.95 bootstrap is at a different hidden rate. Comparing it directly with
any 1.10/1.12357 state says nothing about coexistence or common branch identity.
A future explicit continuation contract must bring it, or the saved lineage,
to a common hidden rate with the same witnessed links and no gaps. At maximum
step 0.002, taking 0.95 to 1.10 requires at least 75 links / 600 phase-corrector
calls before bootstrap costs. This does not fit the proposed 160-call unit;
retain intermediate checkpoints in separately bounded units if that question
is pursued. Do not jump directly between distant roots. Continuation may pass
through unstable qualified period-two orbits; instability alone is not branch
loss, and the old stop-at-first-stability-loss rule cannot establish that longer
connection. Any generalization of the helper to a hidden-coordinate segment
needs fresh synthetic phase, gap, time and budget controls and source review.

Even a successful common-parameter comparison would concern the newly
reproduced procedures and the saved criticality lineage. Matching historical
boundary numbers or angles cannot fill missing full-state records. Adjacent
historical output slices remain separately initialized experiments.

## Manuscript wording now

The inspected traces and new one-lineage pilot do not establish several
coexisting period-two orbits at any common parameter pair. Use **distinct
continuation outcomes from two bootstrap procedures; full-state family
association remains unresolved**. The contribution bullet, S1R heading and
claim of rotation branches coexisting over the same rates need separate
common-parameter state evidence before using coexistence language. Do not
infer a general absence of coexistence from this evidence gap.

## Implementation amendment — 2026-10-08 09:00 UTC, before new outcomes

The source implementation is prepared from integration
`960ab535e1954e6f46dce21999338ddcf1bd23ee` on
`codex/s1r-bootstrap-association`. No new S1R profile or target has run. The
historical summary and reviewed local-branch outcomes above were already known;
this remains exploratory. `gdmaps.py`, `trainability.py`, `models.py` and the
historical tracer have identical committed bytes between the old named
`959beee68b3a` revision and that integration baseline; unknown old dirty changes
are not recovered by this check.

`tools/explore_s1r_bootstrap_association.py` has separate `profile` and `collect`
modes. Both bind the complete immutable 720-file local-pilot inventory, exact
model/data/theta0 identities, saved node0004, current source files/HEAD, and an
explicit independent source-review commit. The review SHA records provenance;
it does not by itself establish approval. Root must authorize each launch.
Output creation is exclusive. All raw states, checkpoint records, failures,
source identities and process/resource settings are retained in a final
hash-bound manifest when ordinary finalization completes.

The optional `iterate_and_screen` observer gets lazy copied snapshots only.
The historical gradient-with-loss, pre-update JVP, tangent normalization and
post-update tail arithmetic are unchanged. Synthetic tests compare result
arrays byte-for-byte with the frozen baseline, including an observer that
mutates every returned copy. A surviving 40,000-step cell makes 1,200 JVP calls
at pre-update states theta38800 through theta39999; only the last 1,000 growth
terms enter FTLE. Snapshots distinguish attempted map updates, accepted finite
updates, attempted JVP calls and successful JVP returns. A call interrupted
inside the operator counts as attempted, not returned. Divergence retains the
last finite accepted state separately from the historical NaN-frozen state.
Incomplete horizons receive no fabricated complete-tail classification.

Checkpoint cadence is every 1,000 completed iterations plus start and terminal.
Deadline checks occur before and after each loop, without copying the full tail
every time. SIGTERM sets a stop flag rather than raising asynchronously inside
an update; the next hook stops and preserves consistent update/state counters.
The terminal snapshot is written using the existing finalization reserve,
including when a nonterminal record exhausts its ordinary byte allowance.
SIGKILL, an uninterruptible numerical call, filesystem failure or exhausted
absolute capacity can prevent finalization; earlier fsynced records remain.
The external timeout and launcher must retain exit status/logs even if no final
manifest exists. This is cooperative durability, not a promise against power
loss or arbitrary process death.

Root clarified the profile cap to **8 seconds internal / 9 seconds external
plus 1 second kill grace**, preserving the proposed 10-second overall cap.
The profile times precisely the single-cell `gradient(..., return_loss=True)`
update and HVP-based JVP paths, with a fixed saved state and e0 direction.
Its 2,000 map calls and 128 JVP calls do not advance a trajectory. Projection
remains `2*(80000*map_seconds/2000 + 2400*jvp_seconds/128) + 30` seconds and must
be finite and at most 100. The factor accounts only heuristically for warm-loop,
screening and checkpoint overhead; corrector variability can still cause a
preserved collection timeout. A failed/incomplete/over-budget profile does not
permit a target run or a shortened warmup. A disclosed split requires a new
card and review. Collection retains the original 100 / 115 + 5 second limits.

The hidden-coordinate helper uses axis zero for widths and secants, exact final
binary64 controls, unchanged direct/two-half/reverse qualification, explicit
native and carried cyclic phases and a cap checked before creating link 20.
Twelve nominal macro links plus three two-phase qualifications cost 102 phase
calls; at most 19 attempted links plus those qualifications cost 158, below the
hard 160-call cap. The nominal macro width divided by sixteen falls below the
minimum 0.000125, so that minimum can stop after three attempted halvings; four
is a ceiling, not a promise to attempt all four. Failed edges and any remaining
gap are retained and cannot authorize an endpoint jump. Both rate bytes and
log-control bytes must agree before the common-parameter comparison.

No fresh seeds, hidden-start selection, horizon extension, lower-family bridge,
GPU, paid service, broad map or stability-boundary search is included. A matching
new upper bootstrap supports this finite witnessed numerical connection only;
it does not recover missing historical states or prove continuous interval
continuation. The lower bootstrap is a separately retained basin outcome until
a future common-control bridge is designed and reviewed.
