# Independent Task6 saved-result review — 2026-10-13

Verdict: the primary frozen L8 AUC benefit reproduces. All 30 L7 rows and 18 L8 rows are present; their AUCs agree exactly with a separate rank-based calculation on the saved scores. The L7 susceptibility confidence interval in the manuscript has a stale upper endpoint, and the threshold prose needs to identify positive prediction windows rather than distinct capture events. No model was fit, bootstrap resampled, or raw dialogue recensored in this review.

The launch owner authenticated and reaped L7-002 and L8-001 before this review. Producing source: `5ea521e8cfddef48d0be5b818599a3f51968c1a2`. Original results are `/data2/neural-shrimp-data/codex/llm-misalignment-audit/l7-002/result.json` (SHA256 `ec9d56a32388e24392e97fb2b67eb76aff484ea5e335e907680963cdf9649186`) and `l8-001/result.json` (SHA256 `78d36e635c022dcc60a2957905d5d56f28f50ee3c830c2cacaa64f59c384ad89`). I rehashed all 17 inventoried files in each worker result and pinned both completed comparison reports. The separate owner's audit covers original source/Git/process provenance.

## Complete row and fit checks

Every condition uses the complete-horizon rule `t+h+c-1 <= 39`, with pre-onset rows and the no-event sentinel excluded as a positive label. The following counts are independently checked against the saved label/group arrays; the maximum turn is the frozen rule's limit, not an independently reconstructed raw turn column.

| Study | Run length c | Horizon h | At-risk windows | Positive windows | Negative windows | Dialogue groups | Maximum eligible t |
|---|---:|---:|---:|---:|---:|---:|---:|
| L7 | 1 | 1 | 7,885 | 368 | 7,517 | 467 | 38 |
| L7 | 1 | 3 | 7,679 | 965 | 6,714 | 467 | 36 |
| L7 | 1 | 5 | 7,461 | 1,429 | 6,032 | 467 | 34 |
| L7 | 2 | 1 | 14,717 | 181 | 14,536 | 502 | 37 |
| L7 | 2 | 3 | 14,066 | 491 | 13,575 | 502 | 35 |
| L7 | 2 | 5 | 13,407 | 751 | 12,656 | 502 | 33 |
| L8 | 1 | 1 | 14,931 | 771 | 14,160 | 937 | 38 |
| L8 | 1 | 3 | 14,573 | 1,982 | 12,591 | 937 | 36 |
| L8 | 1 | 5 | 14,184 | 2,868 | 11,316 | 937 | 34 |
| L8 | 2 | 1 | 30,202 | 347 | 29,855 | 1,010 | 37 |
| L8 | 2 | 3 | 28,852 | 948 | 27,904 | 1,010 | 35 |
| L8 | 2 | 5 | 27,478 | 1,439 | 26,039 | 1,010 | 33 |

All 192 logistic fits have empty warning lists: 180 L7 fits used 8–16 iterations; 12 L8 fits used 8–13, versus the fixed maximum 5,000. Their retained settings agree: C=1, lbfgs, tol=0.0001, intercept on, no class weighting, no warm start, l1_ratio=0. The recorded scikit-learn version is 1.9.1. All 36 PCA fits use 20 components on 896 features. The unchanged automatic PCA solver selects randomized for c=1 and covariance_eigh for c=2; this is a shape-dependent solver choice, not outcome selection. L7's five deterministic group folds are exactly reconstructed from the saved group IDs and seed0; train/test groups are disjoint, every fold contains both classes, and all recorded training counts and feature counts agree. L8's six pairs of fitted models/PCA use the matching L7 table counts, and one unchanged pair of thresholds is shared across pooled and per-dose test rows for each condition.

Every row reports 2,000 valid of 2,000 seed0 dialogue-bootstrap draws, hence zero reported undefined draws. All confidence intervals are finite and ordered. The original arrays of individual bootstrap draws and raw onset/turn columns were not retained in these score files: CI generation and complete-horizon selection therefore rely on the separately reviewed, authenticated source, not an additional raw recensoring or resampling claim. Full original row values, fold counts, fit checks, comparisons and hashes are retained in `/tmp/task6-independent-saved-review.json`.

## Accepted primary values and manuscript corrections

L7, first hostile turn/h3: baseline AUC 0.6415017880818211; D3 state AUC 0.67743814255573; increment 0.03593635447390886, 95% CI [0.013650960485748606, 0.0580568266974773]. D2 susceptibility increment is -0.0021064174927959423 with CI [-0.005904733623256173, 0.0015023174538074465]. The manuscript at lines859–861 should therefore show `-0.002 [-0.006,+0.002]`, not an upper bound of +0.020. D2 remains unconfirmed; D3 was an exploratory secondary predictor in L7.

The frozen L8 primary, first hostile turn/h3 pooled, is baseline AUC 0.6560151481673557 versus state AUC 0.6892950500978507: increment **0.033279901930495015**, CI **[0.017959708662791424, 0.04816594372526904]**. Its lower endpoint is positive. Both primary per-dose intervals are also positive: alpha0.04 gives +0.0376982031444274 [0.013051283288452632,0.06212640968522155]; alpha0.045 gives +0.0297219018004915 [0.009150078972415024,0.050205252241522384]. Pooled first-hostile h1 and h5 increments are +0.03576230883656861 and +0.036738843261636234, respectively, with positive intervals. The secondary two-consecutive pooled intervals are positive at all three horizons, but three per-dose secondary intervals include zero; do not generalize the primary per-dose conclusion to every secondary row.

The saved field `events` equals the sum of positive window labels, not a count of distinct onset events. At the frozen training-negative thresholds [-1.4365622322971054,-1.243446183184403], the L8 pooled h3 operating point is:

- Baseline sensitivity 27.49747729566095% of positive windows, held-out FPR 10.555158446509412%.
- State sensitivity 29.061553985872857% of positive windows, held-out FPR 10.308950837900087%.

Suggested prose: “At thresholds fixed from the training negatives, the state model flags 29.06% of positive three-turn prediction windows versus 27.50% for the baseline, with held-out false-positive rates of 10.31% and 10.56%, respectively.” This is neither distinct-event recall nor a comparison at exactly equal held-out FPR. The existing baseline28% quotation is narrowly outside nearest-integer rounding (27.4975%, only0.0025 percentage points below27.5%); using the precise values avoids ambiguity.

At alpha0.04, sensitivity rises 10.87%→16.56% while FPR rises 4.68%→6.56%; at alpha0.045, sensitivity falls 43.31%→40.94% while FPR falls 18.95%→15.67%. Thus the AUC benefit is positive at both doses, but the single frozen threshold does not dominate in sensitivity at each dose. The L7 22.798%/26.839% figures are also positive-window sensitivities and use a descriptive pooled-negative OOF quantile, not the L8 frozen training thresholds.

## Reference discrepancies and provenance boundary

All 85 L7 machine-reference mismatches are confined to the two-consecutive event. Current complete-horizon rows at h1/h3/h5 are14,717/14,066/13,407 versus legacy15,038/14,392/13,739; h3/h5 positive windows are491/751 versus496/762. Every discrepant row remains in the comparison report. The frozen independent audit explicitly applies amendment2.

`docs/p1-detection-confirmation.md` records amendment1's `t+h<=39` rule and amendment2's additional observation for c=2, applied after confirmation outcomes. `docs/research-log.md:21538` records amendment1-era L7 c2 h3 D1≈0.038/D3≈0.031; line21674 says amendment2 recomputed confirmation secondaries. The legacy `p1-detection-corrected/detection.json` has only args and primary metadata, with no source commit or label-version pin. Its being an amendment1-era reference is therefore supported by the history and row discrepancies, but is **not authenticated file-specific provenance**. Claude should replace or reconcile that old L7 reference with a source-bound amendment2 output; the 85 mismatches must not be called exact reproduction.

All three L8 machine discrepancies are restricted to pooled c2/h5: AUC/increment differences 2.6687930865e-8, upper-CI difference1.4365396386e-8, below manuscript precision. They are retained as numerical discrepancies, without a proven backend-cause attribution. The primary rows match the machine reference within the declared1e-10 comparison tolerance.

The checked manuscript has two numerical discrepancy rows (D2 CI and the borderline28% operating point). The further wording correction about positive windows is substantive for interpretation. No evidence here establishes calibration, reliable event-level alarms, or universal transfer beyond this fixed model/direction/prompt/data design.

## Audit-development disclosure

The first saved-only helper refused because I initially assumed that every PCA fit used the randomized solver. Reading the complete diagnostics showed the unchanged automatic solver correctly chose covariance_eigh for the larger c=2 training tables. I corrected only my audit assertion; the original helper and refusal note remain in `/tmp/task6-independent-saved-review-before-pca-check.py` and `/tmp/task6-independent-saved-review-initial-refusal.txt`. No producer source, result, fitted model, bootstrap draw or scientific threshold changed.
