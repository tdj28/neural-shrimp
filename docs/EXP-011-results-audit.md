# EXP-011 retrospective shortlist results audit

14 September 2026 local time; execution completed 15 September UTC. This
[protocol](EXP-011-protocol.md) was frozen before fitting on already observed
development data. The result is numerically reproducible but does not establish
an incremental geometry benefit or establish the original 300-unit
action-utility claim.

The geometry learner's mean selected 75-unit RMSE is **0.0107899111**, compared
with **0.0162660638** for the shared learner. However, the fixed axes shortlist
has lower mean error, **0.0085689882**. The paired shared-minus-geometry mean is
0.0054761527, with a descriptive 95% bootstrap percentile interval of
[-0.0026527594, 0.0156184633]. Across origins, geometry improves 11, worsens 11
and ties two. This is mixed development evidence, not a fresh confirmation.

Exact compact values, radius and resolution summaries, numerical diagnostics,
timings and artifact bindings are in [EXP-011-summary.json](EXP-011-summary.json).
The complete 5.36 MB producer summary, model packets, predictions, all context
readouts and access logs remain outside Git at
`/data2/neural-shrimp-data/raw/EXP-011/attempt-001/`.

## Provenance, completeness and verification

The exact source freeze is
`64712453153ade24d2b22cf811f7a83811304de4`, privately pushed on
`codex/server-numerical-followup`. The runner verified the remote head at
01:26:07.807 UTC, started at 01:26:07.831, and sealed its receipt at
01:28:01.916 on 15 September. The invocation used the project Python 3.12.14
interpreter, NumPy 2.5.3 and SciPy 1.18.1. The
[design audit](EXP-011-design-audit.md) records the pre-outcome review.

| Artifact | SHA-256 |
|---|---|
| Receipt | `cdfcdeaab031e708624aafcdb0cfa3191b84bed73b644ad7726870dde165da35` |
| Frozen manifest | `a32c5283a2674ddaec01ba4cb508cf473f0be95f5b1724fa4ef7468a3c9b3f08` |
| Full producer summary | `9e010f9561ddd737e74f5cf7f965cefe1f6041cb8ef0e54fad546b0e74bd8776` |
| Terminal records | `26e0bc8c6c0c4da1acf0d8726d25e03f8ca072361cc94b7fe53c96e73ea244f7` |

This audit checked all 346 receipt-listed artifact byte counts and SHA-256
values: 94,844,269 bytes excluding the receipt. Including the receipt, the
attempt has 347 files and 94,907,306 bytes. The manifest binds 21 source and
dependency files, 281 feature inputs, the qualification certificate, six
external qualification/artifact/audit files and all 6,944 qualified label
records. The certificate is both an input binding and a committed source
binding; preparation and fitting do not decode a label-bearing outcome summary.

All 24 outer folds complete. All 96 recipes and every required numerical gate
pass; there are no omitted, failed or substituted folds. Each outer fold holds
out one physical origin with all eight phase/radius contexts. Its tuning uses
23 inner physical-origin folds, yielding 2,208 inner grids and 96 final grids.
Both historical roles are development; the EXP-007 aliases add no origins.
The complete denominator is 24 physical origins, 192 contexts and 3,456
separately charged candidate entries. All 18 IDs remain, including duplicate
physical directions.

The built-in saved-model replay reproduces all 96 recipes. Root's separate
frozen verifier also exited successfully, checking the 346-artifact inventory,
24 outer folds, 96 replayed recipes, feature reconstruction, training/reveal
access logs, prediction and winner seals, selected labels, all-18 evaluation
and complete summary replay. Its invocation was
`uv run --no-sync python tools/run_preview_shortlist.py verify --data-root /data2/neural-shrimp-data/raw --attempt attempt-001`.
This separate verifier's wall time was not measured and is not included in the
producer's reported elapsed time. The producer and its verifier share frozen
code; their success is distinguished from an independent implementation audit.

The independent saved-evidence audit passes **76,598 checks with zero
failures** in 5.9616123 s. It checks all 346 raw artifact bindings, source and
label identities, 24 folds, 96 model packets and 2,208 inner folds. Independent
elementwise prediction arithmetic replays 3,179,520 inner predictions, 22,080
inner MSE entries, 27,648 held predictions, 1,536 complete rankings and 1,920
policy-context scores. Maximum held-prediction disagreement is
8.9928064995e-15; maximum inner-MSE disagreement is 3.0878077872e-16.

All 48 training-access events and 320 reveal events match their permitted
identities and chronology. The last prediction seal precedes the first held
reveal by 1.364514597 s, and every selected-four winner is sealed before
full-18 reveal. This independent audit imports no producer modules and
performs **no coefficient refits, feature reconstructions or integrations**:
it replays from saved coefficients and permitted labels. Its isolation
conclusion combines saved logs/bindings with the independently reviewed frozen
source, and is not an operating-system proof that no unlogged read occurred.

The report SHA-256 is
`782326c1787535b333298f3b3171099eef60a57056cc93f4ee91c9eeedc91150`
(16,203 bytes); the script SHA-256 is
`5f1ae5f73684f5e112bad4143f5181aa67711659e0b1b07f240e3714ee6a95bc`
(27,656 bytes). Root preserved and hash-verified byte-identical copies in
`/data2/neural-shrimp-data/audits/EXP-011/post-run-audit/`, with the script also
in `tools/postrun_audits/`. Exact metadata is in the compact summary.

Root separately verifies owner cleanup using a complete host-permission
`ps -ef` inventory at 01:44:26 UTC: all 24 recorded fold PIDs and producer
parent 1716452 are absent. No process was killed by this audit. An earlier
aggregate-audit process conclusion used only sandbox `/proc` visibility;
that draft is preserved with an explicit metadata-only scope correction.
The host inventory is the cleanup evidence. No experiment, raw record or model
was changed by this correction.

The final independent statistical audit passes **17,742 checks**. All 380
frozen bootstrap intervals and the complete aggregate trees match exactly,
without a numerical tolerance. Policy/context readouts are rebuilt from saved
labels and packet predictions, with no producer imports, coefficient refits or
new integration. All 768 SVD/QR context comparisons preserve complete rankings,
and all 384 resolution-by-schema context comparisons preserve rankings and
actions; no selected penalty changes. The corrected report incorporates the
separate host inventory above. Every statistical field remains exactly equal
to the preserved initial report; only the operational metadata and its checks
were corrected.

The corrected statistics report is 691,467 bytes, SHA-256
`109247ecb2510abbfb4f4fc118057ee36edebaa0d90f3bcd9bba58eca140be92`;
its 20,605-byte script has SHA-256
`2e416da9a911a5570919098955d979aa1821e0bbb72ae6c670aad3e181f8f98b`.
Root copied and hash-verified the corrected report and script in the external
post-run audit directory, and the corrected script in `tools/postrun_audits/`.
The initial sandbox-only script/report and final-run log are also preserved
externally. Root's old/new JSON comparison confirms that only the operational
checks and worker-termination fields change, plus the explicit amendment;
every numerical/statistical field is identical. The compact summary preserves
the final hashes, original draft hashes, durable paths, correction scope and
historical host-check binding.

## Selected-preview error and radius dependence

Each policy predicts or specifies four candidates before seeing held-origin
labels, then selects the observed minimum among its four. A winner seal is
written before the subsequent all-18 reference evaluation. The learned models
use the exact 37 shared columns or those 37 plus six geometry columns. Ridge
penalties, weighting, feature construction and raw-label endpoint were fixed
before fitting; no clipping, replacement model or favorable-resolution choice
was introduced.

Values below first average eight contexts within each origin and then weight
all 24 origins equally. The oracle is the minimum over this finite 18-entry
candidate set, not an optimum over all possible perturbations.

| Policy at dt=0.00125 | Mean selected RMSE | Mean excess over full 18 | Full-18 winner included |
|---|---:|---:|---:|
| Shared learner | 0.0162660638 | 0.0100910249 | 96/192 |
| Geometry learner | 0.0107899111 | 0.0046148722 | 96/192 |
| Fixed axes `[0,4,8,12]` | 0.0085689882 | 0.0023939493 | 47/192 |
| Fixed diagonals `[2,6,10,14]` | 0.1199681977 | 0.1137931588 | 0/192 |
| Fixed local coverage `[0,8,16,17]` | 0.0124640987 | 0.0062890598 | 137/192 |
| Full 18 reference | 0.0061750388 | 0 | 192/192 |

Winner inclusion alone gives a misleading policy ordering here. Local
coverage includes the best candidate most often, but has higher mean selected
error than axes. The magnitude of missed opportunities matters as well as
their frequency. Equal-origin error is the relevant declared comparison.

| Radius | Shared | Geometry | Axes | Diagonals | Local coverage | Full 18 |
|---|---:|---:|---:|---:|---:|---:|
| 0.00025 | 0.0041992794 | 0.0023103371 | 0.0004420231 | 0.0038378393 | 0.0002797168 | 0.0002552256 |
| 0.001 | 0.0053063324 | 0.0085276617 | 0.0017056421 | 0.0136954400 | 0.0010456720 | 0.0009564590 |
| 0.004 | 0.0337937488 | 0.0120512098 | 0.0061920771 | 0.0684533804 | 0.0039849264 | 0.0036166148 |
| 0.016 | 0.0217648945 | 0.0202704356 | 0.0259362105 | 0.3938861309 | 0.0445460794 | 0.0198718560 |

Every radius retains all 24 origins and both phases. Local coverage has the
lowest policy mean at the first three radii; geometry has the lowest policy
mean at the largest. Geometry is worse than the shared learner at radius
0.001, while its largest mean improvement occurs at 0.004. These observations
do not authorize selecting a favorable radius or inventing a combined policy
after the fact.

The frozen bootstrap uses 20,000 paired whole-origin resamples, seed 20260914,
and percentile 2.5/97.5 bounds. The origins are selected historical locations,
may be spatially dependent, and have overlapping outer training sets. The
intervals describe this development cohort; they are not confirmatory
confidence guarantees or a hypothesis-test decision. The lower aggregate
axes mean likewise does not establish universal superiority: its largest-radius
mean is worse than geometry's.

The independent audit's explicitly post-result concentration diagnostic finds
that the three largest positive origin contrasts (`development-06`,
`development-01`, `information-11`) contribute 140.55% of the net positive sum;
the remaining origins contribute a negative sum. This explains why a positive
mean can coexist with 11 positive, 11 negative and two tied origins. Every
origin remains in the declared readout; this diagnostic neither excludes
origins nor defines a new successful subgroup.

## Numerical sensitivity

All 96 model packets pass the independent SVD/augmented-QR arithmetic checks.
The largest inner-validation prediction discrepancy is 1.7413848141e-13, the
largest final held prediction discrepancy is 8.5265128291e-14, and the largest
coefficient discrepancy is 2.1945556483e-11. Every selected penalty and ordered
four-entry shortlist agrees; no full-18 predicted ranking differs in any of
the 96 packets. The frozen prediction tolerance is
`1e-9 + 1e-7 * max(abs(SVD),abs(QR))`; exact penalty and ordered-list agreement
are additional gates. Passing these checks concerns arithmetic reproducibility,
not predictive quality.

The complete repeat at dt=0.0025 selects exactly the same penalties, ordered
shortlists, shortlist sets and observed actions as dt=0.00125. No learned
choice changes in any of the 192 contexts for either model. Mean selected
errors at dt=0.0025 are 0.0162660630 for shared and 0.0107899111 for geometry;
the paired mean is 0.0054761519. The largest context selected-error change is
1.0383799520e-7 for shared and 1.7822227028e-9 for geometry. All 192 true-label
rankings are unchanged, although the largest individual label difference is
5.8180722264e-6. Empirical differences are not certified integration-error
bounds.

The shared model chooses penalties 1e-6, 1e-4 and 0.01 in 10, 13 and one outer
folds respectively. Geometry chooses them in two, five and 17 folds. These
counts, and the individual fold assignments, are unchanged across resolutions.
Both SVD and QR make the same choices.

All 16 previously fixed independent sentinels are retained, with two DOP853
scalar labels each. The largest absolute finest-RK4 versus independent error
difference is 4.4164272239e-10. This is a partial panel, not independent
retraining or an independent complete 18-entry oracle. EXP-010's prerequisite
certificate separately covers all native preview scalar/waveform checks and
all 16 independent scalar/waveform checks. No new integration was performed
for EXP-011 or this audit.

## Resources and next research decision

The producer receipt records **114.0677945 seconds**, including its built-in
model and score replay. Saved-input feature construction takes 2.5608387 s,
parallel fitting/prediction/selection 16.2429480 s, label reveal/evaluation and
summary 15.0280734 s, and built-in replay 78.8140655 s. Receipt inventory adds
0.0760638 s. These stages do not include source/input/package/push preflight.
Summed worker recipe-stage times are 40.7472470 s for grid fitting, 0.8341469 s
for inner scoring and 0.0374669 s for held prediction/shortlisting; summed
worker times are not elapsed attempt time. Four local workers each use one
numerical-library thread. Paid compute and paid review are zero.

The policy budget is four previews versus 18, fourteen fewer nominal calls
(77.777...%). Actual new preview integrations are zero because all labels are
reused. Saved-input feature timing does not measure sensitivity integration,
atlas acquisition or complete deployment cost. All learned policies and the
full-18 reference inherit sensitivity through candidate IDs 16/17. Geometry
also needs the atlas; axes and diagonals need neither prerequisite. No net
speedup has been measured.

**Recommendation:** do not expand the geometry model on the strength of this
result. Prepare a fresh-physical-origin 75-unit comparison with fixed axes and
local coverage as primary practical controls, preserving the current learned
recipes and full-18 reference as comparisons. Before any new labels, freeze
the new-origin selection rule and sample size, a practically acceptable excess
error for the four-preview budget, all-candidate numerical qualification,
complete failure handling, radius readouts and measurement of reference,
sensitivity, atlas, feature, fitting and preview costs. Origins must be
physically disjoint from both old experiments, with no reuse through aliases.
A fresh test of a trained learner also needs its complete training/final-fit
recipe frozen before held-origin labels. Do not combine the best observed
policy at each radius without declaring a new development-derived policy and
then testing it independently.

Shortlisting may remain useful even if a fixed control proves preferable to
learning. A reproducible, practically worthwhile fresh-origin result is the
next evidence needed; a positive retrospective mean alone is insufficient.
The original 300-unit action-utility claim remains unestablished. Separately,
the original EXP-008 strict Linux replay gate remains failed and unchanged;
the separately frozen EXP-011 arithmetic path does not retroactively pass it.
EXP-009's
953 unresolved long-target cases in its selected 1,369-case cohort remain
unresolved. Stable 75-unit preview labels and shortlist arithmetic do not
repair those 300-unit labels or establish long-target control utility.
