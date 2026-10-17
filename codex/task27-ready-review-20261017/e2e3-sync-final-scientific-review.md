# Task 27: E2/E3 and coupled-loop synchronization — saved-evidence review

17 October 2026. This is a review of the sole completed independent saved-data calculation, not another calculation or native experiment. The finite results support the reported E2 flip reduction, the absence of a detected E3 voting gain in this cohort, and approximate transverse-law agreement. They do not establish an enforced gain cap, equivalence of voting methods, a global synchronization threshold, a power-law distribution, or a uniquely identified roundoff/bubbling mechanism. The concrete qualifications below should accompany delivery.

## Evidence and coverage

The immutable output directory is `/data2/neural-shrimp-data/codex/task27-ready-review-20261017/run-001/e2e3-sync`. Its `report.json` is SHA256 `d65c49dbbc9d67d0cbf493f8cd872de625dd9d6e351c440468fffd64fa6fdf91`; the sibling supervision receipt is `fd9a04dd14f6d6d453643f4f79333a04626e56532233e3aee0001f9a41c7df39`; the owner closure is `6b319c225c957231fd91d1897a5b7cd062195fccefafc5521e8e58bfab5254d8`. Executed source commit: `5cc02f5ee6d34d1339a045aa367df78fbf45d7ce`. Executed driver/reducers: `454ad22cc7faa7dced4a6ff8ab4cc98b8b6f0c8674eb5064095e212a692bfb48`, `b4d442ddc51ac24a853b47b778a2594ace2d4998982f709fafe6edded2ccf261`, and `adc754093b7d8f011fe677254289692012002ef1812cdf858f7792c6fd17eebc`.

Inference directly observed session 2127 reap with exit 0 and authenticated nine captured source blobs, all six outputs, and absence of the original worker/supervisor identities, groups and sessions. My role was source and saved-evidence reading. A single bounded stdlib presentation traversal checked the five scientific-output pins against the report and retained their full saved dispositions. It did not import a producer/reducer, open original experiment arrays, resample, or call a model. Its full coverage file is `coverage.json` SHA256 `0a29a64f6ac3a1635a5b8d93fe6d07467882aafe3c7c7b3af637541b8c4a82a8`; the readable projection is `summary.json` SHA256 `cba7dd561de3e54d78e0d1a23100de2031d5eec0ebc318a64fdd21c8b38f17aa`; all sweep rows are in `all-sweep-rows.jsonl` SHA256 `ece9964346e4d484f135941d99d3030dc24b80cb401b3ffa1ec6035a3a78668c`.

Coverage includes all five E2 cohorts (1,024 rows each), eighteen E3 cohorts (512 each), every cohort/selection/finite-value gate, 512 E3 problem dispositions, 21 transverse rows, nine sweep scopes and 165 pair rows, 330 laminar opportunities, eighteen periodic dispositions, and all three precision ladders. The 28 cohort scopes in the presentation include the five E2 cohorts twice for the two analysis baselines; they are not 28 independent experiments. All 23 distinct cohorts have complete coverage, no duplicate/missing records or qualification issues, and no conflicting gold labels. The source-selection checks pass. The 174,592 saved soft0-versus-uncapped field comparisons are exact. All 41 bootstrap streams have 10,000 defined finite draws; their disposition/minimum/maximum records were read, with the complete draws retained in the immutable outputs rather than claimed as individually eyeballed.

Publication comparison: **3,823 exact, 101 different, 390 inherited, 29 publication-wrapper and 57 own-only nodes**, with no missing-own producer node. All 101 nonzero records were read. Four are E2 Monte Carlo interval endpoints below; the other 97 are synchronization TV/OLS floating differences, largest absolute `1.3322676295501878e-15`. No common E3 value differs. The 390 inherited fields include 330 decoded text tails and 60 native precision/index fields; these were not independently regenerated. The 57 extra nodes contain diagnostics, grid and coverage witnesses. This is not bit-exact reproduction of every publication field.

## E2: paired flip reduction, incomplete cap enforcement

The fixed cohort has 512 GSM8K **training-set** problems, allocated as 16 problems in each of 32 prompt-length clusters, each evaluated in layouts 8 and 16. The developmental selection changes and later outcome-informed constant-temperature control remain disclosed. Cluster bootstrap intervals are conditional on this fixed allocation, not a randomized-population causal guarantee.

| Arm | Layout flips / 512 | Mean accuracy | Unmet-cap share of saved steps | Maximum saved gain |
|---|---:|---:|---:|---:|
| Uncapped, T=.6 | 142 (.27734375) | .2158203125 | not applicable | .8897 |
| Cap .37 | 118 (.23046875) | .236328125 | .0092926025390625 | 539.12372 |
| Cap .10 | 119 (.232421875) | .2333984375 | .010711669921875 | 529.64307 |
| Cap .015 | 121 (.236328125) | .2412109375 | .0105133056640625 | 529.64307 |
| Constant T=.5 | 135 (.263671875) | .232421875 | not applicable | 1.05986 |

All three caps pass the frozen descriptive conditions against the uncapped baseline: the flip-difference cluster interval lies below zero, and the accuracy-difference lower endpoint exceeds −.03. The flip differences and intervals are respectively `−.046875 [−.0859375,−.0078125]`, `−.044921875 [−.080078125,−.009765625]`, and `−.041015625 [−.072265625,−.0078125]`. Accuracy differences are `+.0205078125 [.005859375,.03515625]`, `+.017578125 [.0009765625,.03515625]`, and `+.025390625 [.0078125,.0439453125]`. Exact McNemar p values are `.032272020733093106`, `.06923250717353216`, and `.08713322647534709`; the latter two must not be represented as passing that separate .05 test.

Constant T=.5 versus uncapped has flip difference `−.013671875 [−.048828125,.021484375]`, so the flip condition fails, despite an observed accuracy increase. Against T=.5, **all three cap flip conditions fail**: `−.033203125 [−.076171875,.009765625]`, `−.03125 [−.068359375,.005859375]`, `−.02734375 [−.0703125,.013671875]`. Their accuracy intervals still satisfy the frozen −.03 descriptive rule. Nondetection is not equivalence or proof that lowering temperature is the complete causal explanation.

The gain is the source's finite power estimate of the truncated readout covariance divided by temperature, not the composed-map Jacobian gain. Entropy/gap precede the per-row cap adjustment. Bisection and the temperature floor do not guarantee meeting the cap, particularly with ties. The approximately .93–1.07% unmet shares and very large maxima above make “until the proxy fell below a cap” false as a universal description. Saved gains and temperatures are rounded to five decimals; rounded-record diagnostics do not recover unavailable pre-rounding decisions. All recorded boundary ambiguities, missing answers and zero median gain among met steps remain in the saved coverage.

The disclosed independent schedule uses the five persisted arms in fixed order and resets the seed for the T=.5 baseline. Its historical second-baseline CLI order is unattested; no schedule was searched to match the summary. All four discrepancies occur there (own minus published, proportions):

| Endpoint | Own | Published | Signed difference |
|---|---:|---:|---:|
| Cap .37 − T=.5 flip CI lower | −.076171875 | −.078125 | +.001953125 |
| Cap .10 − T=.5 accuracy CI upper | .017578125 | .0185546875 | −.0009765625 |
| Cap .10 − T=.5 exactly-one-correct CI lower | −.048828125 | −.05078125 | +.001953125 |
| Cap .015 − T=.5 flip CI upper | .013671875 | .015625 | −.001953125 |

The extra uncapped-minus-T=.5 contrast is own-only, retained, and does not repair or replace publication contrasts.

## E3: plurality, realized perturbations and negative cases

All 512 problems are common to all eighteen runs, with no common-set exclusion. All 4,096 realized perturbations are present and finite. Their relative norms are **.0007423411589115858 to .0009954804554581642**, median `.0008273515559267253`; the requested .001 is not the realized displacement after bfloat16 casting.

Soft plurality accuracy is `.232421875`; mean single-perturbation accuracy `.230224609375`; unperturbed soft accuracy `.220703125`; greedy `.853515625`; sampled plurality `.904296875`; mean single-sample `.847900390625`. All seven contrast intervals are retained:

| Contrast | Difference | Cluster-bootstrap interval |
|---|---:|---|
| Soft vote − soft single | .002197265625 | [−.009521484375, .01318359375] |
| Soft vote − soft0 | .01171875 | [−.001953125, .02734375] |
| Soft vote − greedy | −.62109375 | [−.671875, −.56640625] |
| Soft vote − sample vote | −.671875 | [−.716796875, −.625] |
| Sample vote − sample single | .056396484375 | [.03515625, .080322265625] |
| Sample vote − greedy | .05078125 | [.025390625, .078125] |
| Soft single − soft0 | .009521484375 | [−.002197265625, .021484375] |

Thus the soft-vote improvement was not detected in this cohort; it is not proved zero. The source uses **plurality**, ignores missing answers as votes, and breaks a final count tie by the answer that first attained that tied count in run order. Its distinct-answer sensitivity nevertheless includes `None`. Mean answer-change rate `.201904296875` is averaged across eight perturbations, not the effect of one particular displacement.

All six sensitivity strata (distinct counts 1–6; sizes 300,144,48,16,3,1) remain. Their vote accuracies are `.3`, `.1388888888888889`, `.08333333333333333`, `.25`, `0`, and `1`: the coarse association is not globally monotone, particularly in the tiny tails. Soft pass@8 is `.2890625`, sampled pass@8 `.96484375`. Missing-answer and hard-token-budget dispositions were not dropped: perturbed soft runs have 16–23 missing answers, soft0 24; greedy has one missing and 75 budget-limited rows; samples have 0–5 missing and 71–88 budget-limited rows. These finite records do not prove that sensitivity can never aid reasoning.

## Synchronization: finite thresholds and complete negative evidence

The diagonal linearization is valid: along a synchronized orbit, `u'=(1−2c) DF u`, so the asymptotic transverse exponent is `λ1+log|1−2c|`. For positive λ1, the lower-threshold formulation applies on `0≤c<.5`; over `[0,1]` the negative-exponent interval also has an upper boundary `1−c*`. A negative typical transverse exponent is not uniform or global synchronization, nor proof of basin access or absence of transversely unstable embedded orbits. The recorded finite-twin rate is also not an exact finite-time tangent identity: its relative-norm boundary and different finite trajectory/horizon are not eliminated by the asymptotic derivation.

| Cell | Saved tangent λ | Ten-block SE | c* | Transformed λ±2SE endpoints |
|---|---:|---:|---:|---|
| 0 | .09750146083565657 | .009507328678060272 | .04644949178533753 | [.037742869888796715,.05499212429175021] |
| 1 | .4370358776552274 | .016012518563217646 | .17702587205696707 | [.16651520974182848,.18720526282530636] |
| 2 | .19590131660017163 | .014211896694399517 | .08895332117855786 | [.07710218607762787,.10047234436366159] |

These block SEs are descriptive, not validated universal confidence intervals. All 21 measured/predicted comparisons are finite. The maximum absolute discrepancy is **.05060468906338727** (cell 1, c=.15, signed measured-minus-predicted negative). The measured range is `−.8138673900098818` to `.4359541116747819`; the paper's outward `−.82` to `+.44` and `<.051` bounds are valid.

All nine finite scan brackets are below. `(a,b]` means the saved all-higher-tested-couplings rule with the final 500 steps, not a continuous solved boundary. Secondary 1e−6 tolerance and refinement are disclosed post-outcome changes.

| Cell / start or refinement | Primary 1e−8 | Secondary 1e−6 |
|---|---|---|
| 0 near | (.04,.05] | (.04,.05] |
| 0 far | (.20,.22] | (.20,.22] |
| 0 refine | (.045,.05] | (.045,.05] |
| 1 near | (.22,.23] | (.19,.20] |
| 1 far | (.18,.20] | (.18,.20] |
| 1 refine | **no passing bracket through .19** | (.18,.19] |
| 2 near | (.15,.16] | (.13,.14] |
| 2 far | (.14,.16] | (.10,.12] |
| 2 refine | (.12,.15] | (.12,.15] |

Nonmonotonicity remains material: cell 0 far has synchronized windows interrupted by failures at .14 and .20; cell 1 near passes primary at .20 then fails at .21/.22; cell 2 near passes at .14 then fails at .15; cell 2 refine passes secondary at .105 then fails .11/.115/.12. A selected first pass cannot replace the retained all-higher rule. The refined cell-1 bracket is compatible with its transformed block-error range/grid resolution but does not contain the point estimate .1770259, and its primary criterion fails.

All 165 distances/records/tails qualify and have no nonfinite or negative values. Independent final-state versus saved-last-distance evaluation has **85 nonzero floating differences**, maximum absolute `6.661338147750939e-16`; none was discarded. All exact row values remain in the companion JSONL.

Only **nine of 165 records** meet the frozen at-least-five-bursts rule: cell 1 refinement at .16/.165/.17/.18 has 13/14/12/13 bursts; cell 2 at .09/.095/.105/.11/.115 has 10/15/5/10/6. All other records fail that rule; cell 0 has at most two. This supports finite intermittency/bursting observations, not every-near-threshold behavior or an identified bubbling mechanism.

Of 330 threshold-specific laminar opportunities, **11 have fits and 319 fail the ≥100 eligible lengths rule**. All 216 right-censored last laminae are flagged and excluded from the fitted lengths; no actual first lamina is left-censored here, although the source would retain such a first lamina. Every defined descriptive log-density OLS slope is listed, including those outside the headline selection:

| Cell, c | Threshold | Slope | Eligible lengths / bins |
|---|---:|---:|---:|
| 1,.14 | 1e−4 | −1.3959164828772723 | 123 / 5 |
| 1,.15 | 1e−4 | −1.8323709234221734 | 182 / 7 |
| 1,.16 | 1e−4 | −1.7236533739121525 | 237 / 11 |
| 1,.16 | 1e−6 | −1.7593425620318526 | 132 / 8 |
| 1,.165 | 1e−4 | −1.6188098600498124 | 166 / 10 |
| 1,.165 | 1e−6 | −1.6359446986237247 | 146 / 9 |
| 1,.17 | 1e−4 | −1.5210298829106625 | 171 / 11 |
| 1,.17 | 1e−6 | −1.6099052211170644 | 156 / 8 |
| 1,.18 | 1e−4 | −1.423784444692585 | 108 / 9 |
| 1,.18 | 1e−6 | −1.3370563916757126 | 108 / 8 |
| 2,.095 | 1e−6 | −1.0743740875414463 | 136 / 7 |

The selected cell-1 .16–.17 slopes are compatible descriptively with −3/2; there is no fitted uncertainty, cutoff validation, distribution comparison, or established power law. Cell-2 bursts above its estimated typical threshold are compatible with bubbling, but the relevant orbit exponents were not computed.

All eighteen periodic scalar dispositions/joins pass their saved-field checks: fourteen nonnull period reports and four no-return results remain. The selected nonsynchronized periods include 4,6,8,10,12; the two additional synchronized reports have period 2. Largest reported closing error is `2.1879119082519402e-13`, so `<2.2e-13` is valid for those saved reports. The complete native return ladder and states were not saved here; primitive-period, stability/attraction and native derivative claims are inherited rather than newly proved. The four no-return cases are not converted into periodic successes.

Native precision fields likewise remain inherited. In each of the three cells, all four saved bitwise batch/identical-row equality predicates are false, with maximum reported relative batch discrepancy `5.965527347392245e-15`, below the original 1e−12 separate-row trigger. The finite-difference ratio recomputations match the saved ratios, but the three successive decade ratios are approximately `(99.97,100.01,77.98)`, `(99.91,99.96,77.45)`, and `(99.96,99.89,9.159)`. Thus the first two decades support h² scaling; a universal “100× down to 1e−5” claim is too strong. The original transverse run's `tools_dirty=true` stays disclosed; captured source chronology does not manufacture an exact clean historical loaded-source assertion.

## Concrete prose changes at the frozen source

References below are to `/tmp/task27-ready-source-001/tree`, commit `32c2f84fe59ab2dbc943886ab0b5bd517c55f0da`.

1. **`paper/llm/main.tex:2466–2467` — narrow the consequence.** Replace the unconditional threshold wording with: “Along the synchronized orbit, typical transverse perturbations decay when λ⊥<0; on 0≤c<1/2 this gives c>c*. This local condition need not imply synchronization from every start.”
2. **`main.tex:2477–2478` — preserve the criterion change and resolution.** Say “The refined first-cell bracket (.045,.05] is consistent with c*. In the second cell the secondary 1e−6 rule gives (.18,.19]; the primary 1e−8 rule did not pass through .19.” Avoid equating a grid bracket with the point estimate.
3. **`main.tex:2479–2480` — replace established power-law wording.** “The selected laminar records gave rough log-binned slopes between about −1.8 and −1.5; these descriptive fits do not establish a power law.”
4. **`main.tex:2485–2489` — qualify mechanism/native claims.** “No explicit noise or mismatch was added; recorded roundoff-level batch asymmetry provides a plausible seed.” Roundoff-alone causation and unstable-orbit bubbling were not isolated. On-manifold equality follows algebraically; finite token histograms do not prove an invariant-measure statement. Periodic closing errors are saved native reports; attracting/primitive claims do not acquire independent multiplier/return-ladder verification here. The promised per-step entropy endpoint is absent and cannot be audited by substituting token TV.
5. **`main.tex:3064–3067` — state cohort and incomplete enforcement.** Describe the fixed length-clustered 512-problem training cohort and the cap as an attempted proxy cap with approximately 1% unmet saved steps. Do not call the proxy the full composed-map gain.
6. **`main.tex:3071–3075` — retain finite-vote interpretation.** Use “plurality vote”; “we did not detect an accuracy gain (+.2197 percentage points; interval −.9522 to 1.3184 points)”; and “averaged over the eight realized perturbations, 20.1904% of answers changed.” Report the realized perturbation range above. The broad diversity/accuracy association has sparse, nonmonotone tail strata.
7. **`docs/research-log.md:28805` — qualify h² range.** First two tested decades scale by about 100; the final decade enters a rounding regime, especially cell 2 (factor 9.159).
8. **`docs/research-log.md:28847` — correct the threshold comparison.** Couplings .09 and .095 are **above** the updated cell-2 point estimate `.08895332117855786`, while inside its descriptive two-SE range and below the earlier published-spectrum threshold. Do not call them below the updated c*.
9. **`docs/research-log.md:28841` — use actual extrema when presenting bounds.** For the three selected cell-1 1e−4 fits the extrema are `−1.7236533739121525` and `−1.5210298829106625`; “between −1.72 and −1.52” excludes an endpoint. Use approximate values or outward `[-1.73,-1.52]`. The paper's broader approximate −1.8 to −1.5 description is compatible.

No further numerical computation is required for these findings. The original outputs, all differences, failed criteria, inherited fields and preparation history remain unchanged.
