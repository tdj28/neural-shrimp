# Four-preview design appendix: feature provenance and adapter contract

14 September 2026. Read-only source/manifest audit for the
[shortlist design](shortlist-feasibility-design.md). No model fit, integration
or frozen-code edit was performed. The proposed recipe remains exactly
37 shared features plus six geometry features.

## Exact shared schema

Names and zero-based historical positions come from
[`BASE_FEATURE_NAMES`](../tools/hopfield_information_features.py#L5).
Their derivations are in
[`base_features`](../tools/hopfield_information_features.py#L21).

| Index | Exact feature name | Inputs |
|---:|---|---|
| 0 | `alpha` | Origin alpha |
| 1 | `gamma` | Origin gamma |
| 2 | `destination_alpha` | alpha + radius × direction_alpha |
| 3 | `destination_gamma` | gamma + radius × direction_gamma |
| 4 | `alpha_squared` | alpha² |
| 5 | `alpha_gamma` | alpha × gamma |
| 6 | `gamma_squared` | gamma² |
| 7 | `phase_sin` | sin(2π × phase_index / 2700) |
| 8 | `phase_cos` | cos(2π × phase_index / 2700) |
| 9 | `period` | Certified unperturbed orbit period T |
| 10 | `log_rho` | log of certified transverse spectral radius rho |
| 11 | `log_rho_per_time` | log(rho) / T |
| 12 | `radius` | Planned perturbation norm r |
| 13 | `log_radius` | log(r) |
| 14 | `radius_squared` | r² |
| 15 | `direction_alpha` | Declared candidate direction component |
| 16 | `direction_gamma` | Declared candidate direction component |
| 17 | `delta_alpha` | r × direction_alpha |
| 18 | `delta_gamma` | r × direction_gamma |
| 19 | `delta_alpha_squared` | delta_alpha² |
| 20 | `delta_cross` | delta_alpha × delta_gamma |
| 21 | `delta_gamma_squared` | delta_gamma² |
| 22 | `alpha_delta_alpha` | alpha × delta_alpha |
| 23 | `alpha_delta_gamma` | alpha × delta_gamma |
| 24 | `gamma_delta_alpha` | gamma × delta_alpha |
| 25 | `gamma_delta_gamma` | gamma × delta_gamma |
| 26 | `rectangle_contains_destination` | Indicator that all four crop clearances are nonnegative |
| 27 | `signed_crop_clearance` | Minimum signed destination-to-crop-edge distance |
| 28 | `initial_u1` | Certified orbit activation at the chosen phase, component 1 |
| 29 | `initial_u2` | Certified orbit activation at the chosen phase, component 2 |
| 30 | `initial_u3` | Certified orbit activation at the chosen phase, component 3 |
| 31 | `orbit_std_u1` | Population standard deviation of 2,700 reference activations, component 1 |
| 32 | `orbit_std_u2` | Same, component 2 |
| 33 | `orbit_std_u3` | Same, component 3 |
| 34 | `local_projected_rmse` | Predicted linear-response RMS after removing one global phase direction |
| 35 | `local_unprojected_rmse` | Predicted linear-response RMS without phase removal |
| 36 | `local_phase_cycles` | Phase-projection coefficients dotted with delta, divided by T |

Crop bounds are `(alpha_min, alpha_max, gamma_min, gamma_max)` =
`(1.056, 1.08, 10.14, 10.55)`. Sharing this rectangular metadata does not expose
the period-component raster to the baseline model. The historical function
requires phases 675/1350 and the four declared positive radii; zero shams are
not candidate feature rows.

The certificate comes from the independent unperturbed baseline assay and
shooting/Floquet calculation, not a perturbed preview or target. Its endpoint
is excluded before computing standard deviations; phase index selects the
initial activation, not a preview-fitted phase. See
[`baseline`](../tools/run_hopfield_drift.py#L78) and the certificate fields in
[`hopfield_diagnostics.py`](../tools/hopfield_diagnostics.py#L222).

For fields 34--36, `prepare_context` starts the variational system at the
certified phase state with zero initial parameter derivatives. It integrates
the **unperturbed** model over 0--300 using DOP853 and builds the tail projection
on `75 < t <= 300`. These are computational prerequisites, not labels from a
perturbed 75--300 target. See
[`prepare_context`](../tools/run_hopfield_information.py#L170),
[`local_parameter_sensitivity`](../tools/hopfield_drift_probe.py#L142), and
[`phase_projected_gram`](../tools/hopfield_drift_probe.py#L106).

The old formulas for fields 34/35 are square roots of directional Gram
quadratic forms. The proposed numerical version should instead form the
directional response and take its RMS, retaining the same mathematical
quantities. For flattened activation sensitivity S, velocity v, projection
coefficients c, projected columns P = S - v c, and delta d, these are
`sqrt(mean((P d)^2))`, `sqrt(mean((S d)^2))`, and `(c dot d)/T`.
This is a separately versioned numerical-method choice requiring its own
freeze and tests; do not silently replace the saved EXP-008 values.

## Exact geometry schema

The six names and implementation are in
[`hopfield_information_design.py`](../tools/hopfield_information_design.py#L25).
`build_atlas_geometry` reads the original EXP-004 focus-map period raster,
finds the core pixel's eight-connected nine-peak component, and measures
native-coordinate distances between observed pixels of opposite component
classes. A disconnected nine-peak component is outside this chosen component.

| Index | Exact feature name | Derivation |
|---:|---|---|
| 0 | `anchor_raw_margin` | Origin's distance to nearest observed pixel outside the core component |
| 1 | `destination_signed_pixel_margin` | Nearest destination pixel's signed opposite-class distance; positive inside the component, negative outside; zero encoding outside crop |
| 2 | `destination_in_core_component` | Membership of that nearest pixel; zero encoding outside crop |
| 3 | `outward_projection` | Candidate unit direction dotted with the origin's nearest-exterior outward unit vector, clipped to [-1,1] |
| 4 | `destination_inside_crop` | Destination is inside the observed rectangle |
| 5 | `destination_margin_crop_censored` | Outside crop, or absolute signed pixel margin reaches/exceeds destination's nearest crop edge |

Nearest-axis-pixel ties choose the lower index. Exterior-neighbor ties use the
explicit source tolerance and deterministic minimum index. Outside-crop zeros
in fields 1/2 are paired with inside-crop=0 and censored=1; they are not known
zero margin or a verified nonperiodic destination. Censoring remains visible.
`outward_projection` is a unit-vector dot product, not radius-scaled distance.
The inside-crop feature is shared rectangular information already represented
by shared field 26. The other geometry fields depend on the old raster. These
are historical representations of coordinates, not new information conditional
on coordinates and the old map, and not certified dynamical boundaries.

## Leakage findings and compute dependencies

No direct perturbed-preview or perturbed-target label enters the 37 proposed
mathematical features. However, the existing `base_features` API
**unconditionally reads `preview_metrics`** before returning all 51 fields.
Calling it and slicing the first 37 requires forbidden information access.
A new pre-preview builder must have no preview/target argument and return
exactly the audited named schema.

Historical fields 37--47 and 49--50 are preview descendants and must be
excluded: `preview_rmse`, `preview_synchronous_rmse`, `preview_phase_sin`,
`preview_phase_cos`, `preview_u1_rmse`, `preview_u2_rmse`, `preview_u3_rmse`,
`preview_first_window_rmse`, `preview_last_window_rmse`, `preview_window_change`,
`preview_rmse_squared`, `preview_times_local`, and `preview_times_radius`.
Historical field **48, `local_projected_mse`, is not leakage**: it is the square
of shared field 34. It remains excluded solely to preserve the fixed 37-field
recipe. A name-based deny rule for `preview_` alone is insufficient to enforce
an exact feature schema or prevent future derived aliases.

Candidate IDs 0--15 are predetermined uniform directions. IDs 16/17 are both
signs of the minimum local-sensitivity direction. Consequently every learned
policy comparing all eighteen inherits the 300-unit unperturbed-sensitivity
prerequisite through its candidate set, even if fields 34--36 were removed.
The direction-derived coordinate, delta and geometry features inherit that
dependency for IDs 16/17 too. Fixed axes `[0,4,8,12]` and diagonals
`[2,6,10,14]` require neither this sensitivity solve nor the atlas; fixed local
coverage does require sensitivity. All policies still require the reference
waveform for preview scoring. Geometry learners additionally require the old
atlas; retaining it as a historical input does not make its acquisition free.

Historical `fit_development` selects labels from `*-dt0.005` target records
and requires the original 51-field/twelve-development-group contract.
Historical `decide_test` consumes all eighteen preview scores. Neither is an
appropriate data-loading or shortlist-selection entry point for the new pilot.
A generic, side-effect-free numerical solver can be reused only through the
new explicit preview-label and feature interface.

## Exact physical-origin aliases

Both manifests were compared directly. Each pair below has exactly equal
binary64 alpha/gamma coordinates and identical old-focus row/column. The
EXP-008 `original_origin_id` field explicitly records the EXP-007 name.

| EXP-007 origin | EXP-008 origin/block | Focus row | Focus column |
|---|---|---:|---:|
| holdout-00 | development-00 | 108 | 22 |
| holdout-01 | development-01 | 87 | 177 |
| holdout-02 | development-02 | 37 | 91 |
| holdout-03 | development-03 | 22 | 24 |
| holdout-04 | development-04 | 41 | 90 |
| holdout-05 | development-05 | 109 | 18 |
| holdout-06 | development-06 | 90 | 173 |
| holdout-07 | development-07 | 22 | 23 |
| holdout-08 | development-08 | 46 | 92 |
| holdout-09 | development-09 | 24 | 24 |
| holdout-10 | development-10 | 81 | 50 |
| holdout-11 | development-11 | 41 | 56 |

EXP-008 `information-00` through `information-11` have block IDs
`test-block-00` through `test-block-11` respectively and no EXP-007 coordinate
match. The union contains **24 physical origins**, not 36. Group on an
explicit common dynamical-system identifier plus exact `float.hex(alpha)` and
`float.hex(gamma)`, checked against the grid coordinates and declared aliases.
Do not group on experiment ID, role, block-name spelling, phase, certificate
hash or source-implementation commit: those can differ at the same physical
origin. Retain those quantities separately as provenance/context keys.

EXP-007 phases are 337, 1012, 1687 and 2362; EXP-008 phases are 675 and 1350.
The feature formulas generalize to those phase indices mathematically, but
the old EXP-008 function rejects the EXP-007 phases. Pooling them requires an
explicit adapter version and qualification; it cannot be achieved safely by
renaming experiment IDs or silently clipping phases.

## Proposed adapter and leakage-test contract

1. A read-only provenance loader validates receipt/source/manifest hashes and
   produces a sanitized input bundle from origin/action specifications,
   baseline certificates, saved unperturbed sensitivity arrays/directions,
   and the old atlas. It does not expose mixed `*-features.json` rows, preview
   paths/scores, target journals, old model packets, decision packets, outcome
   summaries or EXP-009 results to feature construction.
2. A pure builder accepts that bundle and an explicit feature-schema version.
   It emits only a 37-column shared matrix, a six-column geometry matrix and
   non-feature metadata: row ID, origin group, phase, radius, candidate ID,
   source bindings, certificate/sensitivity/atlas hashes, and dependency
   costs. Geometry is passed only to the geometry learner; origin IDs,
   eligibility flags, role labels, selection groups and provenance do not
   become model columns. Preserve the original realized candidate directions.
3. A separate qualified-preview label store is keyed by complete row identity,
   source receipt and resolution. No implicit fallback to the old .005 label
   is allowed. Outer/inner training adapters expose only training-origin
   labels. The held-origin feature bundle is label-free. Seal model/input
   hashes, eighteen predicted scores and four distinct candidate IDs before
   the evaluator can reveal the selected four labels. The all-eighteen oracle
   belongs to a later evaluation stage only.
4. Test construction and inference with a filesystem/API spy that rejects any
   preview/target/decision/summary access. Sentinel objects must raise on access
   to forbidden keys. Mutating or removing every held-origin preview/target
   value must leave feature bytes and shortlists unchanged for a fixed fitted
   model. Training labels may affect the model; this test specifically isolates
   held-origin and inference-time leakage.
5. Enforce exact names/order/width, reject unknown columns and all preview
   descendants, and assert that field 48 stays excluded as a deliberate recipe
   choice. Check all 3,456 row identities, 192 contexts, eighteen entries per
   context, four unique shortlist IDs and duplicate-direction accounting.
6. Prove alias-group separation in both CV levels; fit scalers and select
   penalties only on permitted groups. Test same-origin rows under different
   experiments/roles/names to catch accidental splitting. Any geometry-control
   donor mapping must obey the same fold restrictions.
7. Test certificate endpoint exclusion, phase-index normalization, exact
   realized directions/deltas, crop boundary and censoring semantics, direct
   response-norm numerics, and label-resolution joins separately. A historical
   parity checker may read old mixed feature rows, but must run outside the
   label-free builder and must not feed its results into selection.

Source identities inspected: shared-feature implementation SHA-256
`4e616e317f7af3e859f4ea3d9947042b0dd79323682eaa9d97dc7c625fd39e2a`;
geometry implementation
`41e71719b9a17b7b8e8322c77945d0a38f4f2d189aa6fbbc884202708dea8e35`;
EXP-007 manifest
`573d817836d6c3271e685a45988be660743c104b47d96a5036414db4e6893ac7`;
EXP-008 manifest
`fb0e546947e0f36b77ff4978f60769db3541fc64330022bd664da22c20c321cd`.
