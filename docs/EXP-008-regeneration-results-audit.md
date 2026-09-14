# EXP-008 saved-trajectory regeneration results audit

14 September 2026. The complete saved-trajectory regeneration finished with
19 retained numerical comparison failures. All required cases were processed;
all fitted penalties and selected actions were unchanged; every regenerated
target label equaled its saved JSON value. The original primary result is
therefore unchanged and remains inconclusive. This is post-run development
evidence, not a replacement for the failed strict EXP-008 verifier.

## Evidence and independent checks

The auditor ran from private-pushed source
`d276782c229aa33e2819ec89f15df4e23c7f7834`, following the
[frozen method](EXP-008-regeneration-method.md). It completed in
256.6340431569988 seconds with the expected exit code 1,
`completed_with_comparison_failures`. It performed no integration.

The full 24,027,658-byte report is retained outside Git at
`/data2/neural-shrimp-data/audits/EXP-008-regeneration/attempt-001/report.json`.
Its SHA-256 is
`127f0a80b103957622a906562fb7b05adf64e4350207acd8c3bc228abd03e749`.
The [compact machine-readable summary](EXP-008-regeneration-summary.json)
preserves the complete 19-failure list, denominators, maxima and source bindings.

This results audit independently checked the report hash, the auditor bytes
against the named Git commit, the unchanged original receipt hash, unique
feature/target IDs and both role censuses, exact target-label equality, and
the policy/primary consequences directly from JSON. The completed auditor's
preflight and postflight checks both bound all 10,650 original artifacts
(728,224,123 bytes), 186 source inputs and 45 source files. Those full raw-file
rehashes are the auditor's checks; the results inspection did not rerun the
numerical auditor or duplicate those full rehashes.

| Required material | Received | Unavailable |
|---|---:|---:|
| Sensitivity contexts | 48 | 0 |
| Preview feature rows | 3,456 | 0 |
| Target and zero-sham rows | 7,008 | 0 |
| Learned decisions per chain | 288 | 0 |
| Minimum-preview decisions per chain | 96 | 0 |

Each role contains 1,728 unique feature rows and 3,504 unique target rows.
Clean source/tree identity and original evidence bindings remained unchanged
through postflight. There were no identity, raw-input, census or binding
failures.

## What failed and what changed

Three comparisons exceeded the original arithmetic tolerance
`rtol=1e-12, atol=1e-14` when checking regenerated minimum-eigenvalue metadata:

| Context | Saved minimum eigenvalue | Regenerated | Absolute difference |
|---|---:|---:|---:|
| development-07, phase 675 | 0.7335663650477718 | 0.733566365047011 | 7.608358387756198e-13 |
| development-10, phase 675 | 0.05132747603026058 | 0.051327476030508024 | 2.4744095661333176e-13 |
| development-10, phase 1350 | 0.05125129824884155 | 0.05125129824864416 | 1.9739071488444893e-13 |

These are failures in this expanded diagnostic's metadata comparisons. They
must not be described as the old strict verifier's exact failure locations.

The remaining 16 failures are standardized model coefficients compared using
the unchanged `rtol=1e-7, atol=1e-9` rule: eight baseline and eight shuffled-
geometry coefficients. Both fail at indices 1, 3, 4, 6, 32, 44, 45 and 46.
The geometry model has no coefficient failure. Maximum coefficient differences
over all coordinates are 7.78938658019257e-8 for baseline,
8.737802148495177e-14 for geometry and 7.812242075466713e-8 for shuffled
geometry. The largest absolute differences need not be tolerance failures,
because the declared rule includes a relative term.

All 176,256 shared-feature values passed their arithmetic comparisons. Some
low bits changed: 7,549 values in development and 7,775 in test. The largest
shared-feature difference was 9.582612481295882e-15, in development-07,
phase 1350, radius 0.016, candidate 17, column 35 (`local_unprojected_rmse`).
Geometry features also passed; their largest difference was
1.734723475976807e-18. The regenerated development-data hash differs from the
saved hash, as expected when even one input bit changes; both are retained.

Freshly reconstructed minimum directions were bitwise identical in only
5 of 48 contexts; their maximum component difference was
3.0531133177191805e-16. The original realized directions and raw interventions
were preserved exactly when evaluating stored trajectories. This report does
not assign those trajectories to slightly changed regenerated directions.

All 10,464 freshly optimized aggregate RMSE values and phase offsets equaled
their saved JSON values exactly. All 7,008 regenerated target labels were
identical, including development, test, both old integration steps and zero
shams. The broader metric comparison passed, with a maximum component-level
numeric difference of 6.1409888675936775e-21.

These results do not erase the earlier strict shared-feature failure. This
auditor recalculates activation sensitivity and velocity from saved state
arrays before building Grams; the older saved-input path starts from stored
activation-sensitivity arrays. Passing this regenerated feature comparison
does not establish that the old path now passes or that all platforms agree.

## Penalties, actions and primary result

The selected penalties stayed at 1e-6 for baseline, 0.01 for geometry and
1e-6 for shuffled geometry. All 288 learned actions and all 96 minimum-preview
actions stayed unchanged in both comparisons: saved models applied to
regenerated inputs, and regenerated models applied to regenerated inputs.

| Learner | Maximum score change, saved model/new inputs | Maximum score change, new model/new inputs |
|---|---:|---:|
| Baseline | 9.270362255620057e-15 | 1.247690506467336e-9 |
| Geometry | 2.220446049250313e-16 | 7.543965452327939e-14 |
| Shuffled geometry | 9.103828801926284e-15 | 1.3901160778040378e-9 |
| Minimum preview | 0 | 0 |

Each chain includes all 5,184 learned candidate scores. The smallest saved
learned best-to-second gap is 3.701027002200874e-6. Every condition's gap
exceeds twice its observed score change. This is measured decision agreement
on these inputs, not a new universal score-error tolerance.

All three outcome panels have exactly the same primary result: saved actions
with saved labels, saved actions with regenerated labels, and regenerated
actions with regenerated labels. Baseline-minus-geometry mean benefit is
**0.0008576276406187631**, with the original descriptive whole-block 95%
interval **[-0.007701467321172137, 0.006545433784693783]**; 8 of 12 blocks are
positive. The interval still includes zero. Independently reconstructing the
fixed-action means from the 96 contexts reproduces this effect to final
floating-point rounding.

Fine-step pooled selected RMSE remains approximately 0.01782367 for baseline,
0.01696604 for geometry, 0.01640204 for shuffled geometry and 0.00995885 for
minimum preview. Numerical regeneration supplies no new action-utility
advantage over minimum preview. Prediction-score low bits change, but the
secondary mean prediction-MSE pattern remains: approximately 0.02264066 for
baseline versus 0.01875346 for geometry. This remains a prediction result on
observed data, distinct from useful action selection or net compute savings.

## Claim boundary and next work

This closes the narrower question left by saved-feature-only audits: under
this server's numerical environment, rebuilding readouts, metrics, features,
development fits and decisions from the original stored trajectories leaves
the selected actions and primary effect unchanged. It does not resolve
trajectory integration convergence, requalify reference orbits, validate new
directions, establish general platform portability or repair the original
strict gate. Numerical refinement of the stored trajectories' underlying
integrations is a separate study. Any subsequent preview-budget study must
retain these observations as development knowledge and use a new frozen
design with explicitly qualified labels.
