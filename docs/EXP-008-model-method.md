# EXP008 fixed model, action selection and exhaustive regret analysis

Prospective method, 2026-09-13. Implementation:
`tools/hopfield_information_model.py`. No neural trajectories are integrated
by this module. All tests use synthetic matrices, labels or saved-record
schemas. The complete experiment driver, feature construction, spatial design,
failure rules and source must be frozen separately before new trajectories.

## Inputs and chronology

Development consists of twelve declared EXP007 origins, phases **675 and
1350**, four radii (0.00025, 0.001, 0.004, 0.016) and eighteen candidate
entries: sixteen uniform native-unit directions with exact coordinate axes,
then both signs of local sensitivity. Duplicates remain separate entries.
There are 1,728 fine-step development action labels. The driver must provide
the independently generated complete expected row and group ID census; passing
only the rows that survived is not permitted. Test blocks and their labels are
never arguments to fitting or model selection.

Both learners receive the same explicit baseline columns: coordinates,
phase, period, transverse Floquet stability, local sensitivity, proposed
action and its 75-unit forecast summaries. The root driver freezes their exact
column names, units and deterministic nonlinear coordinate/action terms,
including destination coordinates and crop information available equally to
both methods. This module does not silently expand or select features.

The geometry learner appends only the driver's declared old-map geometry
columns. The shuffled control appends the same columns from a fixed cyclic
donor shift of five among twelve anchors, separately within development and
test, matching phase, radius and candidate index. The driver constructs and
audits the donor arrays and records the permutation metadata; the regression
module does not infer a permutation from outcomes. Base features and labels
are never shuffled.

The required chronology is: collect complete development previews and labels;
fit and durably seal the model packet; obtain allowed test features and
previews; durably seal **all** test decisions; then generate test targets.
The driver enforces ordering and exclusive attempts. A SHA inside a JSON
packet verifies content integrity; by itself it does not establish chronology.

## Fixed estimator and tuning

All three learners use linear ridge regression on the supplied columns with
an unpenalized intercept. Each training group has total weight 1/G, and rows
within a group share that weight equally. The same weights define feature
means, feature variances and the fitted label mean. Constant columns have
scale one. For standardized matrix Z, weights W and centered response y,

`beta(lambda) = solve(Z.T W Z + lambda I, Z.T W (y - weighted_mean(y)))`.

The only candidate penalties are **[1e-6, 1e-4, 1e-2, 1, 100]**. Twelve
leave-one-anchor-out folds each fit their standardizer using only the eleven
training anchors. Each penalty's validation objective is mean squared
prediction error, averaging validation rows within their held-out anchor and
then averaging anchors equally. Select the smallest objective; an exact tie
selects the first, smallest penalty. No tolerance creates approximate ties.
The tuning objective is prediction MSE, **not** validation action regret.

After tuning, fit on all development groups. Each packet retains feature
names, row/group census and input digest, every fold's training groups,
scaler and penalty errors, chosen penalty, final scaler, intercept,
coefficients, weighted training MSE and conceptual solve counts. There are
65 ridge solves per learner (12 folds plus final fit, five penalties each).
The driver measures actual elapsed costs separately. No outcome-responsive
hyperparameter, feature or horizon change is permitted.

Predictions are linear scores with no clipping; extrapolated values can lie
outside the physical RMSE interval [0,2]. Their ranking is used for action
selection. This avoids creating artificial ties by clipping. Actual labels
and preview RMSE must be finite and within [0,2]; invalid required development
data or a solver failure invalidates the whole learner packet, without
dropping rows, substituting a penalty or fitting a replacement model.

## API

```python
packet = fit_information_models(
    base_features, geometry_features, target_rmse, group_ids,
    shuffled_geometry_features=shuffled,
    base_feature_names=base_names, geometry_feature_names=geometry_names,
    row_ids=row_ids, expected_row_ids=planned_row_ids,
    expected_group_ids=planned_development_ids,
    shuffle_metadata={"shift": 5, "donor_ids": donor_ids})

prediction = predict_information_models(
    packet, test_base, test_geometry,
    shuffled_geometry_features=test_shuffled,
    base_feature_names=base_names, geometry_feature_names=geometry_names)

decision = select_candidate_actions(prediction, preview_scores)
```

The final call consumes one condition's eighteen scores in candidate-ID order.
For batched prediction, the driver slices all three score arrays to the same
condition, retaining `status` and `model_packet_sha256`. Optional explicit
candidate IDs permit reordering but require all IDs 0–17 exactly once.
Every policy chooses its minimum score; exact ties use the lowest candidate
ID. Policies are `baseline`, `geometry`, `shuffled_geometry`, and
`min_preview`. All receive the same eighteen actions and forecasts. A failed
prediction has no learner fallback; the separate valid minimum-preview
diagnostic can remain defined. Every preview score and prediction is retained.

## Exhaustive target analysis

`summarize_information_outcomes(candidate_records, decision_records,
block_ids=planned_test_blocks)` takes test-only rows with this schema:

- Candidate: `block`, `phase`, `radius`, `candidate`, `dt`, `status`, `rmse`.
- Decision: `block`, `phase`, `radius`, and `policies[name]` containing
  `status` and selected `candidate`; `predictions[learner]` retains eighteen
  linear scores. Optional `candidate_ids` supplies their explicit order.

Twelve blocks, two phases, four radii, eighteen entries and two numerical
steps require **3,456 test candidate outcomes** and 96 decision contexts.
Both steps reuse the same sealed decisions. Zero-dose shams and development
outcomes are separate driver assays and must not enter this test summary.
The fine step 0.005 on samples **75<t<=300** is primary; step 0.01 is the
numerical comparison. Neither is an independent statistical replication.

For each complete condition, the empirical oracle is the minimum of all
eighteen actual candidate RMSEs, with all tied oracle indices retained.
Absolute regret is selected RMSE minus that minimum. This oracle is limited
to the fixed finite candidate set; it is not the optimum over all directions.
The primary paired reduction from geometry obeys the identity

`baseline_regret - geometry_regret = baseline_RMSE - geometry_RMSE`.

The common oracle cancels. The primary therefore measures a paired task-error
improvement, not an independent additional source of evidence. Absolute regret
is a secondary description of distance from the best evaluated action. The
implementation computes the paired benefit directly from selected RMSEs to
avoid subtracting separately rounded regret means.

First average the two phases and four doses equally within each block, then
average the twelve blocks equally. Report each dose as well. The fixed
10,000-draw bootstrap, seed **20260913**, resamples whole paired test blocks,
with descriptive 2.5th/97.5th percentiles. Phases, candidate actions and
numerical steps are not independently resampled. Positive
`baseline_minus_geometry` favors geometry; shuffled and minimum-preview
comparisons remain diagnostics and do not replace the primary after outcomes.

Duplicate or unplanned row/decision IDs are errors. Missing entries and failed
or invalid candidate metrics remain in the census. **Any missing or failed
required candidate**, even an unselected action, makes that condition's oracle
and policy evaluation undefined. If any required fine-step block contribution
is undefined, the primary remains undefined; no complete-case subset or
reduced candidate oracle is substituted. Both numerical steps are required
controls: **any required coarse-step candidate failure also makes the primary
undefined**, while retaining the complete fine-step panels as diagnostics.
The outer driver additionally gates its release primary on all 7,008 planned
development/test targets, including zero-dose shams; this module receives only
the 3,456 test candidate records and cannot validate that larger census.
Original observed values remain visible even when a complete-condition
aggregate cannot be formed.

## Prospectively declared secondary prediction MSE

For each of the three learned predictors, score **all eighteen candidates**
against their later measured tail RMSE, regardless of which action was chosen.
The secondary MSE is `mean((predicted_tail_RMSE - measured_tail_RMSE)**2)`.
It measures squared prediction error of the task statistic, not the original
activation-waveform RMSE itself. The unchanged fine-preview predictions are
evaluated separately against fine and coarse target measurements.

Average candidates equally within a condition, then phases, radii and blocks
equally. Retain results by dose and pooled, plus each block's MSE. Every learned
policy is reported; a favorable predictor cannot replace the frozen primary
action comparison. This endpoint uses the already sealed prediction arrays;
no refit, calibration, clipping, horizon change or test-label feedback occurs.

Report the minimum and maximum prediction and the fractions below zero, above
two, and outside [0,2]. Bounds zero and two themselves count as in range.
Fractions use the same candidate/phase/dose/block weighting. The range is the
minimum/maximum across the complete declared prediction set, not an average
of its endpoints. These diagnostics make extrapolation and negative linear
scores visible; they do not truncate values before MSE calculation.

Output locations are `steps[dt].panels[dose_or_pooled].prediction[learner]`
and `prediction_per_block[learner]`; each condition also retains its prediction
metrics. A missing, failed or nonfinite required outcome or prediction makes
the affected learner's MSE undefined, with no survivor average. Missing
predictions also make the full-range/fraction diagnostics undefined. If all
forecasts exist but a target fails, their range remains available and clearly
has a separate status from the undefined MSE. An overflowing squared error is
retained as an undefined numerical metric, not serialized as a complete NaN.
This secondary does not alter the primary's previously fixed gates.

## Validation and claim boundary

Synthetic checks cover analytic ridge recovery, fold-specific scaling without
validation leakage, equal group weights under duplicated within-group rows,
geometry/shuffle separation in synthetic data, sealed-packet mutation,
schema/failure handling, exact and near ties, and immutable prediction.
Exhaustive synthetic outcomes validate the complete census, equal phase/dose
weights, paired-block intervals, exact oracle cancellation and failure of an
unselected action invalidating the primary. An injected solver failure does
not trigger another penalty or fit. Secondary tests independently check all
four weighting levels, both numerical steps using the same forecasts,
candidate reordering, strict range fractions, missing-data propagation and
unclipped prediction-error overflow.

Any result is conditional on this network, observed atlas, fixed feature
representation, ridge family, finite action set and simulator-preview budget.
It is not an information-theoretic novelty theorem, proof that the shrimp's
topology causes usefulness, universal policy optimality, or equal-total-cost
superiority. Map construction, feature extraction, fitting and prediction
costs must be reported alongside the shared forecasts and action outcomes.
