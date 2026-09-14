# Design recommendation: four-of-eighteen preview shortlists

14 September 2026. Design recommendation only. No model was fitted, no
shortlist result was computed and no new integration was run to prepare this
document. This is not an executable freeze or authorization to start a study.
The immediate prerequisite is the separately planned EXP-010 preview-label
qualification. EXP-009's unresolved long-horizon labels must not become the
training target for this feasibility question.

## Question and usable endpoint

Can pre-preview shared features plus old-map geometry identify four candidate
entries whose minimum observed 75-unit preview error approaches the minimum
among all eighteen, more closely than a matched shared-feature model and
simple fixed four-entry sets?

This asks whether geometry can reduce preview acquisition for a short-horizon
simulator objective. It does not establish later waveform preservation,
training/reasoning utility, or reduced total wall time. The original
300-unit task remains a separate unresolved question.

For a context, seal a four-entry shortlist using only pre-preview information.
Then reveal the four observed preview errors and choose their minimum, breaking
exact ties by original candidate ID. The eighteen-candidate reference is the
minimum over the entire numerically qualified preview table. Main descriptive
metrics are shortlisted minimum error, its excess over the eighteen-entry
minimum, and the matched shared-minus-geometry contrast. Report inclusion of
the full-set best candidate and top-four rank quality as secondary diagnostics;
rank errors alone need not imply material task-error differences.

## Data and chronology

Use EXP-008's 24 physical origins, two phases, four radii and eighteen entries
as the primary development table: 192 contexts and 3,456 labels. Both original
roles are now known development evidence. Any cross-validation is retrospective
development evaluation, not an untouched test or confirmatory result.

EXP-007 can be a supplementary check only after its preview labels and feature
conventions are separately qualified. Its twelve origins are the same physical
origins as EXP-008's original development origins. They are not twelve extra
independent origins. Its additional phases and 21-entry candidate set also
require an explicit harmonization plan; only the first eighteen entries map
to this proposed action set. Prefer completing the uniform EXP-008 feasibility
table first rather than quietly pooling incompatible or unqualified records.

Freeze one recipe before fitting: leave one physical origin out, with every
phase/radius/candidate from that origin excluded from training and all inner
tuning. If additional experiments are ever pooled, group by physical origin
coordinates across experiment IDs. For each outer fold, select the ridge
penalty by inner leave-one-origin-out equal-origin prediction MSE, using a
fixed grid and training-only preprocessing, then fit once and seal all
held-origin predicted scores and four-entry shortlists before reading that
origin's preview labels. Retain every fold, failed fit and missing label.

## Features, models and controls

Use an explicit named whitelist equivalent to EXP-008's shared columns 0--36.
These include coordinates, action/destination terms, reference-orbit features
and local sensitivity. Exclude every preview-derived field, including all
preview errors, phases, windows, squared scores and cross-products. Avoid
column-position-based filtering in the implementation: the whitelist must be
independently audited against feature provenance. No 75--300 target label,
target diagnostic or decision chosen using those targets is allowed as input.

The two primary learners predict raw 75-unit activation RMSE with the same
group weighting, intercept, feature standardization, penalty grid and solver.
One uses shared inputs, the other adds the original six geometry features.
Use numerically qualified direct projected-response norms for local features
and a stable weighted-design ridge implementation, with SVD and augmented-QR
cross-checks. Do not choose a target transformation, clipping rule, model family
or diversity constraint after viewing outer-fold results. The initial recipe
is ordinary raw-error prediction, unclipped, taking the four lowest predicted
scores with candidate-ID tie breaks.

Use these fixed, equal-preview-count controls, fixed before fitting:

| Shortlist | Candidate IDs | Meaning |
|---|---|---|
| Signed axes | 0, 4, 8, 12 | Four coordinate directions |
| Rotated uniform | 2, 6, 10, 14 | Four evenly spaced diagonal directions |
| Axis/local coverage | 0, 8, 16, 17 | Signed alpha plus both local directions |
| Full reference | 0--17 | All eighteen previews |

Calling the signed axes a separate four-entry uniform control would duplicate
the same policy. The rotated set above makes that comparison distinct. The
coverage set is a fixed simple comparator, not optimized on observed errors.
Preserve duplicate physical directions under distinct candidate IDs and charge
all four declared entries, as in EXP-007/008. A deduplicated acquisition policy
would be a different prospectively specified study. A mismatched-geometry model
is optional supplementary work; it needs a fold-safe donor rule and should not
delay the two-learner feasibility comparison.

## Numerical qualification and required tests

Complete EXP-010 before an executable shortlist freeze. Its full 3,456-case
preview table must preserve old/refined labels, both native refinement levels,
all independent-panel results, all failures and per-context candidate IDs.
Before assigning a canonical label, report step-to-step changes, method
disagreements, within-context best/runner-up gaps and boundary gaps relevant
to shortlist rankings. Do not silently choose whichever resolution favors a
selector or drop difficult labels. Cases that remain numerically unresolved
must be marked unresolved; a complete all-required feasibility endpoint may
therefore remain undefined.

Because observed refinement differences are not rigorous error bounds, label
uncertainty should be reported as an empirical numerical sensitivity envelope.
Re-evaluate each fixed shortlist's minimum across both refined label tables,
and all available independent labels, retaining action changes and excess-error
changes. A small label difference can reverse a near tie without materially
changing shortlisted error. Conversely, a stable RMSE mean can hide materially
unstable rankings. Keep these two consequences separate. Do not turn numerical
qualification tolerances into an unapproved task-quality noninferiority margin.

Required synthetic and structural checks include:

- A feature-access test that rejects all preview/target descendants and proves
  the shortlist is unchanged when hidden labels are altered or unavailable.
- Physical-origin separation in outer and inner folds, including EXP-007/008
  aliases; training-only scalers, penalties and any donor mapping.
- Exactly four distinct candidate IDs, deterministic ordering/ties, preserved
  duplicate directions, and exactly four charged preview entries.
- Explicit incomplete-result behavior; no complete-case fit, skipped candidate,
  saved-label fallback, resumed attempt or partial oracle.
- SVD/QR agreement on scores and shortlist consequences, including nearly
  tied fourth/fifth scores, constant columns and collinear synthetic designs.
- A synthetic example where rank changes but shortlisted excess error is
  negligible, and another where rank instability materially changes that error.
- Reconstructable prediction, shortlist, reveal and winner seals, plus raw
  input hashes and all eighteen reference-label identities.

## Interpretation, costs and fresh evaluation

Aggregate over physical origins with equal phase/radius weights, report each
origin and radius, and retain all outer-fold outcomes. Any interval is
descriptive for this selected development set. Existing historical labels
and overlapping training folds do not justify a new confirmatory p-value.

Four rather than eighteen previews reduces charged preview entries by fourteen
(77.8%). That accounting alone does not establish net compute savings. Measure
feature preparation, unperturbed 300-unit sensitivity, atlas construction or
amortization, fitting, prediction, and preview/scoring costs separately. Fixed
axes and rotated uniform sets need neither a local-sensitivity computation nor
the atlas; charge each policy only its actual prerequisites and report both
shared experimental execution cost and standalone policy cost.

A fresh held-origin utility freeze is premature now. First qualify preview
labels, then run this bounded retrospective feasibility comparison with its
recipe frozen before fitting. Only if the result has numerically resolved
shortlisted-error differences and plausible standalone cost should a new
prospective pilot select untouched origins and collect its own data. Even that
pilot would support only the explicitly chosen short-horizon objective; it
would not repair EXP-009's longer-horizon uncertainty or establish general
neural-computation utility.
