# Read-only review of the frozen P1 detection design

13 October 2026 UTC. Reviewers: Codex root and an independent Codex subagent for the label reproduction. Scope: the frozen design and source at fa47bbf7c763e6db36628234214db3cbb8e7041f, plus tiny synthetic arrays. No exploration or confirmation outcomes, raw dialogues, classifier arrays, result files, protected data, GPU or pod were read or changed. Claude retains ownership of the implementation and study.

## Blocking label defect

`onsets` in `tools/p1_detection.py` returns the number of observed turns, N, as the sentinel when no event is observed. `at_risk_rows` then computes `int(t < o <= t + h)` without distinguishing that sentinel from an actual onset. The shared `table` propagates these labels to the exploration analysis and the frozen confirmation's training and test rows.

The direct synthetic reproduction used one entirely non-hostile dialogue with 40 observed turns (indices 0 through 39). Both event definitions, first hostile turn and first of two consecutive hostile turns, produce onset sentinel 40. Each horizon emits 39 at-risk rows:

| Horizon | Positive labels produced despite no observed event |
|---|---|
| 1 | None |
| 3 (primary) | t = 37, 38 |
| 5 | t = 35, 36, 37, 38 |

The reproduction executed the original frozen Python source with an all-false `(1,40)` NumPy array. An independent reviewer obtained the same result. These are synthetic code findings, not measured effects on either dataset. The direction and size of any distortion in incremental AUC remain unknown. A baseline containing turn index does not make incorrect labels valid.

There is also an administrative right-censoring issue: simply adding `o < N` to the positive-label condition would label incomplete future horizons as negatives. For the first-hostile-turn endpoint, a uniform fully observable horizon requires `t + h < N`. For the event defined as the first of c consecutive hostile turns, observing an onset through `t+h` requires the additional c-1 turns, so the corresponding condition is `t + h + c - 1 < N`. This uniform full-horizon restriction is one transparent correction; a different censoring-aware estimand would need its own explicit definition.

## Recommended handling of the frozen study

Preserve the original frozen design, source, data and any produced outputs, including failed or unexamined outputs. Do not interpret a result computed using these labels as confirmation of the stated event endpoint. Record a dated amendment identifying this code defect, when it was discovered, who had seen which confirmation outcomes, and the corrected endpoint and row-eligibility rule. Freeze and test the correction before opening any still-unseen outcomes. Apply the same rule to exploration training and confirmation evaluation, fitting PCA, scaling and logistic coefficients on exploration only. Keep the original negative susceptibility result visible; any revised estimate from its affected label pipeline must be distinguished from the original output.

If confirmation outcomes have already informed the correction or subsequent choices, call that analysis adapted and retain a genuinely unexposed confirmation for a confirmatory headline. If the relevant outcomes remain unseen, document that fact and the pre-unblinding amendment explicitly; do not silently replace the original freeze or imply the original implementation passed unchanged. The sampling itself need not be repeated merely because of this label bug: the original immutable trajectories can remain useful, subject to their exposure status and the stated estimand.

## Timing and fitting checks

The frozen sampler builds reply k's prompt from the transcript before appending reply k. `SteeredModel.generate_batch` records the final prompt position during prefill, before reply tokens are sampled. Thus `pre_reply[t+1]` represents the context after reply t and before reply t+1; no target reply token is included in that hidden-state construction under the pinned cached-generation path. Calling it a warning at t means the next prompt has already been processed. Its inference cost and operational timing should be reported that way, rather than as a cost-free signal at the end of t.

The confirmation class fits PCA, standardization and logistic regression on training rows and applies them to test rows. I found no test-fit leakage in this source path. The code reconstructs deterministic training fits separately for pooled and arm evaluations; that is redundant but is not by itself outcome adaptation. The operating thresholds use training negatives; achieved held-out false-positive rates are reported separately.

The paired bootstrap resamples dialogue groups, retaining all their at-risk rows and paired model scores, which respects within-dialogue dependence. Its confidence interval concerns the difference for the fitted predictors on the test population; it does not include variability from refitting models on a new training sample. In pooled analysis, resampling is not stratified by dose, so the empirical arm mix varies; this should be stated, not changed after results. AUC and the count currently called `events` are at-risk-row quantities: `events` counts positive forecast rows, not distinct hostile dialogues. No claim of one warning per independent event is supported by that count.

This audit does not establish practical intervention benefit, semantic validity of toxicity labels, or a dynamical mechanism unique to metastability. An incremental predictive result, if it survives the corrected evaluation, would warrant its own accurately scoped claim.

## Frozen source identities

fa47bbf7c763e6db36628234214db3cbb8e7041f is the full source commit. The SHA-256 digests below identify the exact inspected Git blobs, not current working-tree versions.

| File | SHA-256 |
|---|---|
| `docs/p1-detection-confirmation.md` | `5b414772efe437f3ba4d8e7163ceb3a8edab7870a69804b4e76e66bbaf02676f` |
| `tools/p1_detection.py` | `34be28135a6bc2a9ee59ecc4c7bc6001289104b87c406444a48af505d28c8c08` |
| `tools/p1_detection_confirm.py` | `529780a68c4d9a179b954ad8f8709f4e81d03bfaa8575775d945c075ed778e06` |
| `tools/explore_llm_msm.py` | `b5f2e0e60ee173cce04c4d75b171878be148efcfa28e366b9eb9485fff44adb6` |
| `tools/explore_llm_phase_map.py` | `842402d933819adab59056ba76479143c5abdd0a7635e62c0011773ec928086e` |
| `tools/explore_llm_attractors.py` | `2be7c4affce89a1106797c920d0f2e8a3686bdd230a7d897bcdd3411e8decbc9` |
