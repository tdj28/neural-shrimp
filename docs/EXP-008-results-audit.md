# EXP-008 independent raw, feature and outcome audit

**Platform status update, 13 September 2026:** the results below were audited
locally on macOS. Linux CI at release `1d12ebeda040` does not pass the frozen
verifier. [PR-run log](EXP-008-linux-pr-failure.txt) records a baseline
coefficient difference of 3.68269315e-8 at the unchanged 1e-7/1e-9 rule; the
[push-run log](EXP-008-linux-push-failure.txt) records an earlier shared-feature
difference of 1.39784018e-14 at the unchanged 1e-12/1e-14 rule. Each run stops
at its first failure. Local replay and local SVD action agreement do not
establish full Linux portability. The strict gate remains failed and private
PR7 remains unmerged; a separate platform diagnostic does not replace it.

A subsequent two-worker Linux diagnostic preserves all three penalties and all
288 learned actions after refitting on saved features; maximum prediction
differences are 1.39e-9 and 1.96e-9. Feature/coefficient failures persist. This
is saved-input decision agreement, not a full regenerated-pipeline pass; see
[the platform results and complete reports](EXP-008-platform-status.md).

Date: 2026-09-13. Experiment source
`3c5604497ec57b19d62fbb595da7b190102f4b33`; manifest SHA-256
`fb0e546947e0f36b77ff4978f60769db3541fc64330022bd664da22c20c321cd`.
The independent auditor was written after source freeze and imports no
experiment runtime, feature, selector or analysis helper. It reads completed
saved evidence only and generates no new neural trajectories.

## Numerical conclusion

**The primary action-selection benefit remains unresolved.** Adding the six
old-map features changes pooled fine-step selected activation RMSE from
0.01782367172 to 0.01696604408, a paired benefit of 0.00085762764 with a
95% descriptive paired-block interval [-0.00770146732, 0.00654543378]. Eight
blocks favor geometry, one ties and three favor the baseline. The comparison
with shuffled geometry is also unresolved. These findings neither establish
incremental task value nor prove all uses of geometry impossible.

The direct minimum-preview policy has lower observed error than either learned
selector: 0.00995885497. The diagnostic preview-minus-geometry contrast is
-0.00700718911 [-0.01473183538, -0.00161119171]. Its mean error is 41.30% below
the geometry learner's under the same 18-action and preview access. This
control is essential: improving a particular regression representation does
not establish an advantage over simply choosing the best short forecast.

Geometry does improve the observed pooled prediction-MSE mean, from
0.02264065594 to 0.01875346156 (17.17% lower); shuffled geometry gives
0.02434692397. This is a secondary observed predictive result, not a replacement
for the primary utility endpoint or proof of a general predictive advantage.
Predictions of many candidates and the decision to choose one candidate are
different objectives. All extrapolations remain unmodified: 386/1728 baseline,
259/1728 geometry and 340/1728 shuffled predictions are below zero. None exceeds
two. The respective prediction minima are -0.27732038196, -0.07720030140 and
-0.25804077749. No post-outcome clipping or recalibration was applied.

## Independently reconstructed selected error and regret

Every cell averages two phases within each of twelve test blocks. Pooled
cells then average all four radii equally. The target consists of 2,250 elapsed
samples on 75<t<=300, with all three activation components and one fitted
constant phase. Error and regret use the same exhaustive eighteen-action
candidate set; the finite-set oracle is not a global optimum.

| Radius | Baseline error | Geometry error | Shuffled error | Minimum-preview error |
|---|---:|---:|---:|---:|
| 0.00025 | 0.012969838 | 0.005518633 | 0.008767470 | 0.000302092 |
| 0.001 | 0.013298666 | 0.009579848 | 0.011697102 | 0.001179119 |
| 0.004 | 0.006783589 | 0.006381585 | 0.006900985 | 0.005028178 |
| 0.016 | 0.038242594 | 0.046384110 | 0.038242594 | 0.033326030 |
| pooled | 0.017823672 | 0.016966044 | 0.016402038 | 0.009958855 |

Positive baseline-minus-geometry benefit favors the map representation. The
primary is the pooled row; dose-specific rows are retained diagnostics.
Intervals use 10,000 paired-block resamples with seed 20260913, reconstructed
without the experiment's summary helper.

| Radius | Baseline minus geometry | 95% interval | Positive blocks |
|---|---:|---|---:|
| 0.00025 | 0.007451205 | [0.002280433, 0.014008023] | 8/12 |
| 0.001 | 0.003718818 | [-0.000818945, 0.008816107] | 6/12 |
| 0.004 | 0.000402004 | [-0.000638652, 0.001647778] | 2/12 |
| 0.016 | -0.008141517 | [-0.040596771, 0.011867264] | 4/12 |
| pooled | 0.000857628 | [-0.007701467, 0.006545434] | 8/12 |

The small-dose 0.00025 contrast is positive, but direct preview selection has
still lower error there. At the largest dose, geometry has higher mean error
than the baseline and its interval crosses zero. The unfavorable final block
is retained: pooled baseline-minus-geometry for `test-block-11` is
-0.04041369207. Omitting it would change the scientific answer.

Pooled finite-action oracle error is 0.00723311551. Regrets are baseline
0.01059055621, geometry 0.00973292857, shuffled 0.00916892227 and direct preview
0.00272573945. Independently computed selected-error and regret differences
satisfy the shared-oracle cancellation identity. The other pooled descriptive
contrasts are baseline-minus-shuffled 0.00142163394
[-0.00287101045, 0.00482526041] and shuffled-minus-geometry -0.00056400630
[-0.00989201037, 0.00706994347].

## Evidence, feature and decision verification

- All 10,650 receipt artifact hashes and byte counts pass, totaling 728,224,123
  bytes. All 186 old-input artifacts and 45 frozen source files match their
  manifest hashes. The attempt manifest is byte-identical to its frozen Git
  version. No experiment source or raw artifact was edited by this audit.
- All 24 baseline origins are eligible: twelve development and twelve test.
  Their saved common-initial-state classifications are nine at both steps,
  and their saved independent certificates report verified attracting cycles.
  Orbit arrays have 2,700 endpoint-excluded samples and activation/state
  consistency is checked. This audit does not perform a new shooting solve.
- Every planned preview and target is present: 1,728 previews plus 3,504 targets
  per role, totaling 3,456 previews, 6,912 nonzero targets and 96 shams. All
  7,008 target records and all previews completed, with no numerical,
  selector, baseline or timeout failures. No origin or action was replaced.
- All 10,464 raw preview/target traces have the prescribed native parameter
  updates, initial orbit states, phase epochs, time grids and step sizes.
  All eighteen ordered unit directions, including exact signed axes and
  both local directions, are reconstructed; duplicate actions remain charged.
  Saved local Grams are recomputed from the tail sensitivity arrays, with
  zero initial state sensitivity verified at all 48 contexts.
- Independent periodic interpolation using `numpy.interp` reconstructs all
  10,464 saved-phase metric blocks and 517,248 aggregate, per-neuron and
  per-window RMS values. Maximum discrepancy is 4.44e-16 under unchanged
  rtol=1e-12, atol=1e-14 checks. Window errors reuse their recording's single
  saved phase. No reference-spectrum cadence diagnostic is flagged.
- All 3,456 rows of 51 baseline columns and six geometry columns are separately
  reconstructed from permitted inputs. The geometric component, raw distances,
  deterministic nearest-pixel/normal ties, crop flags and masked values are
  checked from the old atlas, independently of the feature implementation.
  The baseline receives crop location as well as local period and stability.
- Development and test shuffle matrices use the exact separately indexed
  cyclic donor shift five, matching phase/radius/candidate index. The exact
  development-data digest—including both geometry matrices, labels, row IDs
  and groups—matches the sealed model packet. No test label enters that digest.
- Model, feature, decision and target-start seals have exact hashes and valid
  chronological order. All saved model coefficients are independently applied
  to test features; every prediction, selected candidate and selection score
  agrees. All 96 direct-preview decisions are reconstructed from their eighteen
  preview errors. The separate model/CV audit owns independent ridge refitting
  and conditioning; this audit does not claim to provide that additional check.
- Independent phase/radius/block aggregation reproduces all selected-error,
  regret, contrast and bootstrap panels at both steps. It also reproduces the
  complete per-block prediction-MSE/range panels, without success-only deletion.
  The primary all-target gate passes and all four role/step sham groups have
  24/24 coverage.

In the test set, geometry is outside the observed crop for 0/432, 0/432, 6/432
and 156/432 candidates at ascending radii. The corresponding crop-censored
counts are 0, 0, 32 and 218. Outside membership and distance are `null` in
raw diagnostics, paired with the declared numerical mask. They are not known
nonperiodic outcomes. Development outside counts are 0, 12, 64 and 164; its
censored counts are 16, 29, 100 and 266. This coverage limitation is retained.

Geometry chooses the same candidate as the baseline in 52/96 contexts,
as shuffled geometry in 41/96, and as direct preview in 36/96. The fixed
illustration, `information-00`, phase 675, radius 0.016, fine step, chooses
candidate four (+gamma) for every policy. Its actual tail RMSE is
0.02290668240 for all four. The identical plotted waveforms are the correct
frozen example, not grounds to choose a more favorable illustration.

## Step sensitivity, costs and claim boundary

Across all 3,504 matched coarse/fine target pairs, absolute tail-error
changes have median 2.9145e-7, mean 0.02259219794, 95th percentile 0.13050512116
and maximum 0.47685138414. In the test half alone, the maximum is 0.43927640371.
The exhaustive candidate trajectories are therefore not uniformly converged.
The selected-policy errors are substantially steadier: maximum differences
across 96 actions are 1.2458e-5 for baseline, 2.0795e-6 for geometry,
9.9981e-4 for shuffled geometry and 1.2458e-5 for direct preview. These are
separate diagnostics, not an excuse to erase unstable candidate results.

The coarse-step primary benefit is 0.00085746727
[-0.00770149649, 0.00654526921], again unresolved; the coarse preview-minus-
geometry contrast is -0.00700727401 [-0.01473181583, -0.00161133141]. The primary
fine step was fixed before outcomes. Fine test-sham mean/max are
0.00006773553510/0.00008162572718; fine development-sham mean/max are
0.00006514206661/0.00006889217208. Both groups have 24/24 required values.

Collection and summary took 162.35249138 seconds before the producer's final
verification. Forty-eight sensitivity solves use 23.77027405 solver seconds
and 2,568,600 function evaluations. Model fitting plus writing takes
0.07915050 seconds. Preview and target calls sum to 141.92287686 and
323.14900859 overlapping call-wall seconds respectively; these cannot be
added to elapsed collection time. All models receive eighteen previews per
decision. Development labels and all eighteen later test outcomes are
training/evaluation expenses, not free online forecasts. Old atlas construction
and training must be included in any application cost claim. Paid compute and
paid review are both zero.

The result concerns a fixed ridge representation, two phase contexts, one
three-neuron network and a previously observed parameter atlas. Test blocks
partition eligible pixels with holes; old-origin halos do not exclude every
previous destination or imply independent populations. Atlas features are
functions of coordinates and the known map. Period/Floquet features describe
state-space orbits at fixed weights, while the shrimp outline is in parameter
space. This experiment does not certify long-time period-nine retention at
perturbed targets, state-space basin sizes, improved trained-task performance,
a sequential controller, memory capacity or net computational savings.

The raw/feature/action/outcome audit passes, and the frozen producer verifier
successfully checked all 7,008 targets and 3,456 previews. The result is mixed
and the primary utility claim remains unestablished. The completed separate
[model audit](EXP-008-model-results-audit.md) retains an independent numerical
limitation: eight baseline and eight shuffled-model coefficients exceed the
unchanged replay tolerance under weighted SVD reconstruction. Geometry's
coefficients pass. The chosen penalties agree and all 288 learned actions
remain identical. This coefficient sensitivity is not reclassified as a pass
because the actions are stable, and standardized coefficients should not be
interpreted as identified physical parameters or causal feature importance.

## Reproduction and audit-development note

Run the retained post-freeze auditor with:

```sh
python tools/audit_hopfield_information.py --out /tmp/EXP-008-independent-check.json
```

It defaults to this repository's `evidence/EXP-008/attempt-001`; `--attempt`
and `--repo` can override locations. Output inside the immutable attempt is
rejected. It reconstructs the stored phase, not a new optimum or trajectory.
The first harness pass expected the older three-field panel schema; it was
corrected to audit the frozen additional prediction/range panels as well.
That was a new-auditor schema correction, with no changes to experiment
source, measured values, thresholds or the consumed attempt.

The following hashes bind this independent packet. Audit wall-time metadata
will change on a rerun; the reconstructed scientific quantities do not.

- `tools/audit_hopfield_information.py` SHA-256 `539f56e0f2e6c93d090dbbab8348eb5b1a0e15a5dd9ee7393bbc7405befb11ac`.
- `docs/EXP-008-independent-check.json` SHA-256 `4e7acd997a7ab795314fff4d63581c52995cbe7dd72b29011e2378fa2b1e7972`.
- Experiment receipt SHA-256 `ddb22c78a6b5f11a023088fd90f3403d25a2797e00624f01c316171ce35706fb`.
