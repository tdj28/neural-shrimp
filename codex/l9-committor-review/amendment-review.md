# L9 amended-source disposition: 8f3e3431

**Verdict:** the amendment corrects the main statistical interpretation and the
split-dialogue bootstrap. Remaining admission gaps must be fixed or explicitly
excluded from the primary L9 analysis before its outputs are promoted. Nothing
in this review assesses the L9 outcomes.

Reviewed commit: `8f3e34310b5a572a53936675cbf3d86c7f698ea3`.
`tools/p1_committor.py` SHA-256:
`483395a1c6dbb297aa7afbb4d8977e9ab79761a4cd561d152c5dfbc193900378`.
`tests/test_p1_committor.py` SHA-256:
`a9777c073537687d531256f6cc38103cd0dcca17f071f60b3233bc3f0a3d046f`.
Exact remaining identities are in `amendment-source-manifest.json`; the six
dependency sources are unchanged from the original review. Root authorized
read-only Git exports from the owned Task6 worktree; no Git mutations occurred.
I read the delta, relevant surrounding source, and tests, but ran no producer
analysis/test/model and opened no L9 data, progress log, pod or job.

**Exposure timeline:** the amendment at research-log lines 21977–21984 and the
reported 13:20Z coordination disclose that the producer had seen hostile-share
progress for rollout turn 1, blocks 0 and 1 (.0515 and .0544), but no analysis
outcome or retrieved block file. This is limited outcome-related progress
exposure, not complete blindness. The reviewer received that disclosure through
source/coordination, not by opening L9 evidence. Preserve the distinction:
“before the analysis outcomes were read; two progress summaries had been seen.”

## Fixes accepted by source inspection

- Lines 3–11 now distinguish population-optimal true-committor rankings from
  finite-K realized AUCs, which are not guaranteed upper bounds. The original
  counterexample and binomial/tie argument remain in `pre-outcome-review.md`.
- Both calibration and AUC bootstrap callers use a single stratum
  (`402,414,444`). Thus one dialogue's positive and negative selected rows now
  receive the same multiplicity. Keep the fixed inclusion weights and describe
  this as a joint-dialogue bootstrap on the selected sample, not an exact replay
  of the original fixed-size row-stratified sampling experiment. This resolves
  the specific lost-dependence defect without proving every possible CI target.
- Lines 459–462 explicitly make the onset profiles descriptive, conditional on
  future outcomes, and remove mechanism identification.
- Lines 280–298 reject duplicate rollout keys, inconsistent shapes, nonfinite
  or out-of-range scores. Lines 301–326 bind available metadata to the states
  hash and array shape and enforce complete coverage by default. Lines 421–428
  implement the previously declared calibration stop and record overrides.
  These are substantive improvements over the original source.

## Remaining concrete actions

1. **Do not label missing-provenance data complete.** Lines 312 and 320–323 allow
   a missing `rollout.json` when `--allow-partial` is set, but `partial` is only
   `bool(missing_states)`. With all keys present and metadata absent, the result
   reports `partial=false` although the state-hash/metadata check never ran.
   Require metadata for the primary analysis. If a restricted analysis is
   retained, record both coverage and provenance explicitly and mark it
   non-primary even when every key exists. A partial or calibration-overridden
   analysis is descriptive exploratory evidence and cannot claim passage of
   the declared L9 gates or become a new confirmatory result. Add that exact
   complete-keys/missing-metadata regression.

2. **Bind actual K to the declared K=32, not just to self-reported metadata.**
   Lines 317–318 only compare metadata K with array K. Both can consistently be
   4 or 16 while passing that check; the new tests themselves use such values.
   H is bound to the selection design at line 324, but there is no equivalent
   run-card K binding. Admit the actual L9 analysis against a frozen expected
   32x3 contract and planned counts. Keep small synthetic fixtures configurable
   rather than weakening that run-specific admission. The figure hardcodes
   “K = 32” at line 532, so it must use the admitted value or actual saved K.

3. **Reject duplicate keys in the selection specification as well as in
   blocks.** `expected = {s['key'] ...}` at line 306 collapses duplicate selected
   states. The later state list retains duplicates and can double-count a row
   at lines 385–388. Check key uniqueness in `spec['states']`, consistency with
   `design.states`, and the admitted frozen selection/counts before filtering.
   This is a distinct regression from the existing duplicate-block test.

4. **Preserve failed analysis artifacts and correct the timeline.** The failure
   file at line 427 has no existing-file/exclusive guard; a second failed
   analysis can overwrite it. Use a fresh attempt/output path and refuse an
   existing failure file, as the successful file already does at 492–495.
   Replace “before any outcome was read” in line 10 and equivalent comments
   with the disclosed limited-exposure wording above.

5. **Keep the estimate and fidelity interpretation narrow.** The 1/K fit is
   still a heuristic, not a proven extrapolation to Bayes AUC. Existing JSON
   names `ceiling_inf` and `share_of_attainable` and the figure description
   (`452–456,519–520`) must not resurrect the withdrawn bound in a paper or
   caption. The denominator can be zero/nonpositive and bootstrap draws are
   silently omitted; disclose undefined counts and avoid interpreting the
   ratio as a fraction of a demonstrated maximum. A calibration pass is a
   consistency screen, not proof of identical conditional dynamics; a failure
   can arise from sampling variation. Keep the original stop rule without
   inventing a replacement threshold after looking at results. The required
   2000 re-scores remain an explicit invocation obligation, not the default.

The original review's scoped qualifications still apply: h=1/2 use the
h=3-selected population; the profile is late-onset/survivor-conditioned;
soft-target and hard-label fits have different support when labels are censored;
fit warnings/undefined draws should be retained; no actual asset/feature-source
authentication or runtime equivalence was independently tested here. These do
not call for a new experiment or a larger framework. They delimit what this
exploratory selected-state analysis can claim.

**Release:** suitable as a narrowed exploratory analysis after the remaining
primary-admission/preservation issues above are closed or explicitly excluded.
No empirical detectability ceiling, mechanistic classification, result pass,
new confirmatory evidence or model execution is certified by this source review.

## Final source-only closure — eef65a92, 13 October 2026

Root independently read the complete source/test diff at
`eef65a929c9dcbecdc7bcd4b42682e4b564845eb` and accepts closure of the four
specific remaining admission issues: separate coverage/provenance and primary
eligibility, the default frozen32/3/1706 run contract, selected-key uniqueness
and count checks, and exclusive failure output. Heuristic field names,
undefined-draw counts and the limited-progress timeline wording were corrected.
This is source-inspection acceptance only: no producer test, model, analysis or
outcome was executed or read in that review. Actual data/assets, fidelity,
explicit rescore2000 invocation and exploratory-scope obligations remain.
The authoritative correction commit time is13:14:50UTC; Claude disclosed that
the earlier13:20 and13:25 communication headings were approximate timestamps.
