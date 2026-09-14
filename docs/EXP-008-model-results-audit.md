# EXP008 independent model and outcome audit

Date: 2026-09-13 (local time). Source
`3c5604497ec57b19d62fbb595da7b190102f4b33`; manifest SHA-256
`fb0e546947e0f36b77ff4978f60769db3541fc64330022bd664da22c20c321cd`.
Only the completed `EXP-008/attempt-001` was read. No frozen experiment file,
target, threshold, feature or selected action was changed. No integration or
paid call was performed.

**The primary geometry benefit remains unresolved.** Its fine-step paired
mean is **0.00085762764**, with descriptive 95% interval
**[-0.00770146732, 0.00654543378]**: eight positive blocks, three negative and
one exact zero. The direct minimum-preview control performs better than the
geometry learner in the pooled diagnostic comparison. Geometry has lower
mean all-candidate prediction MSE, which is a different, secondary outcome.

**One independent numerical comparison fails and remains visible:** weighted
SVD reconstruction exceeds the prospectively stated coefficient replay
tolerance for eight baseline and eight shuffled-model coefficients. All
chosen penalties and all 288 learned action selections remain unchanged.
This is a sensitivity of the independent coefficient reconstruction, not a
failure of the frozen same-algorithm replay, which the root verifier completed
successfully. It does not justify silently relaxing either check.

## Independent implementation and data separation

`tools/audit_information_models.py` imports only standard-library modules and
NumPy. It invokes neither the experiment fitter nor any integrator. It checks
all **10,650 receipt-listed artifacts** by hash and byte count, the fixed source
and manifest identifiers, and the model/feature/target/decision seal bindings.
All 3,504 development and 3,504 test targets complete, as do 1,728 development
and 1,728 test preview/feature records. All unique expected IDs are present.

The model packet contains precisely the development rows and their twelve
groups, with 51 shared columns and six optional geometry columns. Test origins
are absent from that group/row census. The independent audit reconstructs the
development input digest, including the fine-step labels and the cyclic
five-origin donor shift, paired by phase, dose and candidate entry. The same
structured mismatch is reconstructed separately for test features; no labels
are permuted.

Recorded UTC chronology is consistent and content-bound:

- Model sealed: `2026-09-14T02:58:48.993087+00:00`.
- All test decisions sealed: `2026-09-14T02:59:28.530216+00:00`.
- Test target phase started: `2026-09-14T02:59:28.539437+00:00`.

Every test target binds the decision seal. These records support the private
execution sequence; they are not a public registry timestamp. Source inspection
confirms that fitting receives only development labels and test selection
receives only features, previews and the sealed development packet.

## Folds, numerical conditioning and coefficient caveat

The independent standardizer averages explicit within-group moments rather
than using the original weighted matrix reduction. Each of the 36 held-out
folds excludes its validation group from the scaler. Weighted SVD reconstructs
ridge solutions and validation scores for every declared penalty, instead of
solving the experiment's normal equations. Every fold scaler and all 180
validation-loss entries agree within their unchanged comparison rules.

| Learner | Chosen lambda | Minimum LOAO MSE | Final design rank / columns | Regularized normal condition number |
|---|---:|---:|---:|---:|
| Baseline | 0.000001 | 0.03374042729 | 46 / 51 | 13,494,907 |
| Geometry | 0.01 | 0.02422021481 | 50 / 57 | 1,512 |
| Shuffled geometry | 0.000001 | 0.03885763127 | 51 / 57 | 14,779,432 |

The largest condition number across the full fold/penalty grid is
15,784,178. These designs contain algebraically dependent columns even though
no column is constant. Selected-penalty CV gaps to the runner-up are
0.00404492, 0.00102090 and 0.00248485, respectively. All independent selected
lambda indices match. The largest independently reconstructed fold-loss
discrepancy is 2.01e-10, within the frozen model-arithmetic tolerance.

For the final coefficients, the unchanged comparison is rtol=1e-7 and
atol=1e-9. **Eight of 51 baseline coefficients and eight of 57 shuffled
coefficients fail it; geometry has no coefficient mismatch.** The largest
absolute coefficient difference anywhere is 7.52e-8. The original saved
normal-equation solutions have relative backward residuals at most 7.65e-17:
the discrepancy is consistent with sensitivity of an ill-conditioned
representation, rather than a large residual in the recorded linear system.
Individual standardized coefficients should not be treated as causal feature
importance or accurately identified physical parameters.

Prediction and action consequences are much smaller:

| Learner | Maximum test prediction difference, SVD vs saved model | Smallest best-to-second predicted-score gap | Changed chosen actions |
|---|---:|---:|---:|
| Baseline | 1.32e-9 | 3.70e-6 | 0 / 96 |
| Geometry | 7.23e-14 | 5.58e-6 | 0 / 96 |
| Shuffled geometry | 1.46e-9 | 5.46e-5 | 0 / 96 |

There are no exact minimum-score ties. Every within-condition action gap
exceeds twice its largest independent prediction discrepancy. Thus this
specific independent reconstruction changes no selected action and no
selected-outcome contrast. This is a measured numerical check, not a proof of
robustness to arbitrary implementations or altered features. The failed
coefficient comparison is retained in the audit output rather than reclassified
as a pass because the decisions are stable.

## Outcomes from the sealed choices

The audit independently reconstructs all eighteen candidate outcomes per test
condition, each policy's selected error and finite-candidate regret, equal
phase/dose/block averages, and the fixed 10,000-draw paired-block bootstrap.
It also reconstructs the secondary all-candidate prediction MSE. Scientific
aggregate comparisons use rtol=1e-12 and atol=1e-14. No alternative phase fit,
new waveform score, refit or new selected action is introduced.

| Fine-step pooled quantity | Baseline | Geometry | Shuffled geometry | Minimum preview |
|---|---:|---:|---:|---:|
| Selected activation RMSE | 0.017823672 | 0.016966044 | 0.016402038 | 0.009958855 |
| Absolute candidate regret | 0.010590556 | 0.009732929 | 0.009168922 | 0.002725739 |
| All-candidate prediction MSE | 0.022640656 | 0.018753462 | 0.024346924 | Not this endpoint |

The mean finite-candidate oracle error is 0.007233116. Its common value
cancels from paired regret reductions; regret improvement is not an additional
independent validation of selected-error improvement.

At the smallest dose, geometry improves on the learned baseline by 0.00745121
[0.00228043, 0.01400802]. At the largest dose the effect reverses in the mean:
-0.00814152 [-0.04059677, 0.01186726]. Neither observation replaces the
prespecified pooled comparison. Block `test-block-11` contributes -0.04041369
to the primary and remains included.

The pooled minimum-preview-minus-geometry contrast is **-0.00700718911
[-0.01473183538, -0.00161119171]**, with only one block favoring geometry.
Shuffled-minus-geometry is -0.00056400630 [-0.00989201037, 0.00706994347].
Thus a simpler decision rule using the same eighteen already available
forecasts has lower selected error here; the geometric learner has not
established incremental control value beyond that rule. This comparison does
not constitute a measured total-cost speedup.

Geometry's pooled prediction MSE is lower than baseline in seven of twelve
blocks. Its smaller overall prediction error is a useful secondary point
estimate, not a resolved primary action advantage or a universal prediction
claim. The policies choose identical candidate IDs in 52/96 baseline/geometry
conditions; they are distinct rules but do not provide 96 distinct action
comparisons. All other declared controls remain reported.

## Extrapolation, steps, shams and limitations

Predictions were never clipped. Baseline predictions range from -0.27732038
to 1.30282203, geometry from -0.07720030 to 1.00189441, and shuffled geometry
from -0.25804078 to 1.39305572. Respectively **386, 259 and 340 of 1,728**
predictions are below zero (22.34%, 14.99%, 19.68%); none exceeds two.
The selected prediction is negative in 44, 33 and 38 of 96 conditions.
These are regression ranking scores, not physically valid negative errors.
Their visible extrapolation must not be repaired and rescored on these same
outcomes as though the repair were prospectively tested.

The coarse pooled primary is 0.00085746727
[-0.00770149649, 0.00654526921], preserving the unresolved conclusion. Across
all 1,752 test coarse/fine pairs, however, absolute RMSE differences have
median 2.29e-7, mean 0.01740466, 95th percentile 0.10827143 and maximum
0.43927640. Development pairs have mean 0.02777974 and maximum 0.47685138.
There is no uniform pointwise convergence claim. Selected-policy test pairs
are much closer: maximum differences are 1.25e-5 for baseline, 2.08e-6 for
geometry, 9.998e-4 for shuffled geometry and 1.25e-5 for minimum preview.
Small differences among selected cases do not validate every training label
or every unselected candidate.

All 96 zero shams complete. Fine development sham RMSE has mean 6.51e-5 and
maximum 6.89e-5; fine test shams have mean 6.77e-5 and maximum 8.16e-5.
The collection/summary receipt records 162.35249 seconds; model fitting/sealing
records 0.07915 seconds within the execution. Atlas construction, orbit
qualification, sensitivities, shared previews and verification are separate
cost components; the fit time is not a complete policy cost.

The geometry columns encode a deterministic representation of coordinates
and an observed parameter map. The mismatch control is structured donor
information, not an exact randomization null. The results concern one fixed
ridge family, one network, a limited action set and new interventions within
an already observed atlas. They establish neither information-theoretic
novelty given coordinates nor a shrimp-topology mechanism. Improved period
retention, trained-task utility, general retrieval capacity and sequential
controller benefit remain unmeasured. New settings, learners or thresholds
require a new prospective assessment.

## Reproduction

```sh
python tools/audit_information_models.py evidence/EXP-008/attempt-001 \
  --out /tmp/EXP-008-model-independent.json
```

The audit refuses to read outcomes before a complete receipt and refuses to
write inside the immutable attempt. Its output retains all independent fold
losses, coefficient mismatch details, condition numbers, action margins,
selected actions and reconstructed comparisons. This audit uses the saved
RMSE labels; it does not independently integrate trajectories or redo orbit
certification. The frozen verifier separately checks saved waveform metrics.
The first strict coefficient comparison raised the reported mismatch; the
audit was then made to retain those same failed comparisons while continuing
the other checks, without changing their tolerance or the experiment.
