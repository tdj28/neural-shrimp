# Task11: P1 robustness under a second classifier

Status: source preparation only, no target analysis or profile admitted.
Authority: Claude 15:45Z assignment as superseded by the 15:49Z split, source
1491f37972e706426d3343ce5b69fa91ab89da01. Root assigned CPU-only independent
analysis; inference owns Git. Use only this new directory and its owned raw
root. No scoring, model/GPU/pod calls or edits to producer data.

## Fixed question and evidence scope

Recompute the existing P1 comparisons on the same saved replies with
s-nlp/roberta_toxicity_classifier, full revision
048c25bb1e199b98802784f96325f4840f22145d. Both classifiers use the saved float32
score **strictly greater than 0.5**. The original score is the toxic column of
unitary/toxic-bert at 4d6c22e74ba2fdd26bc4f7238f50766b045a0d94.
No threshold tuning or selection of favorable runs. This is an exploratory
measurement-robustness study on consumed dialogues, not a new confirmation or
a replacement for the original frozen L8 claim. The second instrument is a
different trained classifier, not ground truth or statistically independent
human annotation.

All44 arms are literal in definitions.py:12 capture +14 history +7 L6 +7 L10
+2 L7 training +2 L8 test. A target requires their complete immutable inventory,
exact sample arguments and a recorded producer-writer closure. Every input is
hashed before any array load and again before success. Presence alone does not
authorize use. Old evidence stays untouched. All agreement rows are reported,
including zero/low-prevalence and contrary rows.

## Fixed analysis and claim references

References are paper/llm/main.tex at1491f379: capture803–831, history834–853,
L6 control871–883, L10 shown-reply result885–897, L8 confirmation926–933.
Definitions also come from the existing Task6 audit, p1_controller.py and the
frozen p1-detection-confirmation.md including its censoring amendments. Source
definitions were read; producer analysis modules are never imported.

1. All44: paired 2x2 label counts, overall agreement, first-only/second-only
   fractions and both shares. These measure classifier agreement, not accuracy.
2. Capture: all12 rows, total/late shares, never-hostile counts, conditional
   zero-based median onset (plus explicitly censored-at40 median), adjacent
   hostile transition numerator/denominator, and geometric1/(1-p) summary.
   The latter is a geometric approximation, not a fitted survival-time model.
   Preserve both seed rows and random/cooking controls, regardless of outcome.
3. History: all14 seed rows and the11 fixed pooled histories; four contrasts
   ramp minus fresh.04/.05 and held.04/.03 minus fresh at the same dose.
   Late means use35–39; transition origins30–38. Each contrast resamples
   dialogues independently,4000 draws, seed0 reset, matching Task6 conventions.
4. L6: unchanged Task6 controller reducer, original seven-arm order, shared
   seed0 RNG,4000 dialogue draws. Report shown share5–39/35–39, actual mean
   counter-dose, and triggered-minus-interpolated constant response. Dose
   outside the observed0,.02,.04 range is flagged, not extrapolated silently.
5. L10: seven shown-reply arms only, same windows,4000 draws, seed0 in the
   declared L10 order. Report shown-response efficiency, actual mean dose and
   regeneration fractions (5–14,30–39 and5–39). This new shown-only estimand is distinct from the
   producer's would-be efficiency. Never infer second-classifier labels of
   discarded original replies or robustness of the paper's would-be/basin
   prevention claim. Lexical side effects do not change merely by relabeling.
6. L8 exploratory primary: first hostile event, horizon3. For each classifier
   separately, regenerate risk rows and labels and score-derived baseline
   features from that classifier. Preserve other baseline features, the
   layer18 state, train/test arm partition, PCA20 fit only on L7, train-fitted
   scaling/logistic C1/max_iter5000, and all solver warnings/iteration counts.
   No D1/D2/token features, missing-probe imputation or L8 tuning. Use the
   original corrected full-horizon censoring and strict event boundary.
   Fit baseline and baseline+D3 on L7 only; score L8 unchanged. Report pooled
   primary plus both dose rows,2000 dialogue-bootstrap draws seed0/order from
   Task6, all valid/degenerate counts, and training-negative90th-percentile
   thresholds with realized test FPR/sensitivity. Save every score and ID.
   Scientific admissibility also requires all logistic fits to converge; failed
   fits retain their scores/warnings but get no positive/negative primary verdict.
   If training has one class, report unestimable models without selecting new
   rows; if a test subset has one class, retain that row with undefined AUC/CI
   and no success verdict. Valid rows use the unchanged Task6 estimator. This
   explicit degenerate-row handling does not turn missing evidence into a null.
   A zero-row pooled test is returned explicitly before fitting/scoring sklearn.
   This substitutes the measurement instrument in both labels and baseline,
   not an isolated label-only perturbation of frozen original-model weights.

The original classifier is recalculated as a control with identical own code.
All numeric first/second values are reported, not just binary conclusions.
Fixed conclusion markers: increasing main-dose share/persistence, four history
contrast signs/positive CI lower bounds, each L6/L10-shown efficiency sign/CI,
and L8 pooled positive CI lower bound. A null or changed conclusion is retained.
No target-derived thresholds or additional model selection are permitted.

## Reuse, exposure and limitations

task6_reducers.py is byte-identical to the separately authored Task6 reducer.
task6_detection.py has only its own reducer import renamed. reuse-provenance.json
binds originals. Shared scientific dependencies make this independent of Claude's
analysis implementation, not independent of the earlier Codex implementation.
The launcher is the already reviewed Task6 direct-child supervisor with only the
new root/mode ceilings changed. It never starts descendants from the worker.

Prior original-classifier outcomes and the assignment summaries are known.
During source search the research log also exposed producer L9/L10 summaries;
no L9 raw data or analysis is in scope. At preparation, inference relayed Claude's
16:00 summary:44 second files ready; agreement.62–1, mostly first-only;
kindness.563→.201 and random/cooking reductions. These are disclosed producer
summaries, not independently checked results. They do not change this all44
plan or thresholds. At16:05 Claude supplied a44-file immutable manifest and
declared the df64645f scoring writer exited0 with no later writes. Its exact
source feac2a796e4aa673e2d7f44654a9dc06b2070416 and manifest
c90331389be2b2a6496c1b02925d99766252d945ea5a7b1c3698a5547af044ad
are bound in the loader. This is producer-attested writer closure; our admission
separately verifies every actual score-file hash, without rescoring the model.

## Resource and release plan

No substantial run until Root approves the source/card and inference privately
checkpoints it. Proposed profile001: CPU1/nice10,4GiB AS/RSS,120s,64MiB raw.
It runs both instruments' synthetic44-arm descriptive work, one full896-column
PCA plus two logistic fits, and2000 bootstrap draws on39936 synthetic rows.
Fit workload is a timing sample, not a worst-case convergence guarantee.
Proposed actual runs are descriptive001 and l8-primary001, sequential, each
CPU1/nice10,4GiB AS/RSS,300s,64MiB. Forecasts are2x descriptive+30s and
4x profile fit+8x bootstrap+30s; memory2x profile peak+512MiB must fit4GiB.
Cards remain prepared until actual measurements and immutable inputs are reviewed.

Worker+supervisor reserve2 auxiliary CPU slots. Coordinate with Task8 so the
shared auxiliary total stays≤4; Task1's7 slots and frozen execution checkout
are untouched. Startup MemAvailable≥20GiB; runtime≥16GiB, sampled aggregate RSS
watchdog4GiB, worker hard AS4GiB. External parent deadline is worker time+10s,
then only the birth-bound own direct child receives TERM and, if necessary,
KILL after5s. Parent reaps and records unresolved ownership. Sampled RSS is a
watchdog, not a kernel hard limit. All attempts and failures are exclusive and
preserved; no automatic retries. No worker creation is authorized by this note.

Frozen source hashes and private commit are carried in each card. Inference
verifies blobs/private push before granting the unchanged-source run lease.
Post-run receipts retain all inputs/outputs, runtime library/thread versions,
warnings and owned closure. Root independently reviews results before paper use.
