# Task 17 — theory, loop-gain, return-map and novelty scope

Reviewed frozen source `6fcd6ba78c641c2a4b8dcd9ab8a87f32778f2940`, packet manifest `e39408a60b07cc6b353b933c0b3e50465d4704d7667af87e0a3f9c44cdb636bf`. All paper locations below refer to `paper/llm/main.tex`, SHA256 `47452e4083ccb1dec00639ebd0c386588359bb7a3b51453319991d7de2d9189b`. The paper last changed at 08:53 UTC; some accepted Task19 results are newer. This is a source/report and targeted primary-literature review, not a model run, raw-array audit or numerical reproduction. No producer file was changed.

The factorization and finite loop-gain findings survive. The principal corrections concern unconditional universality/temperature language, promotion of bracketed cusp candidates to located nondegenerate cusps, and promotion of scalar fits to a demonstrated one-dimensional reduction. The accepted adverse and amended results should remain visible.

## T01 — Make the temperature and tied-embedding assumptions consistent in the headlines

**Locations:** 37–40, 93–96, 104–109, heading 257, 268–269 and 1730–1732. The qualified body at 259–267 is substantially correct.

The covariance form in equation (gain) applies to tied input/output embeddings. The general chain-rule formula at 231 is correct for untied embeddings; its middle factor becomes a cross-covariance after an output-linear factorization. The identity-network Hopfield comparison likewise requires the tied case. Use “covariance, or cross-covariance for untied embeddings” in short general statements and keep the tied condition with the Hopfield comparison.

The pointwise zero-temperature limit does not establish an interval of small positive temperatures without chaos. A temperature-dependent attractor may approach token ties, where the derivative need not vanish. The body now states this correctly. High-temperature contraction follows for the fixed-prefix, fixed-steering smooth map with a network derivative bounded on the compact embedding hull. It is not an unconditional conclusion for every possible softmax-readout architecture, hard routing rule or growing-memory deployment.

Suggested headline: “The factorization proves contraction at sufficiently high temperature for this smooth fixed-prefix map. At zero temperature the finite-memory greedy map is eventually periodic; exclusion of chaos at small positive temperature additionally requires a uniform logit gap.” Describe the observed intermediate-temperature bands as finite sweep findings. At 1730 cite the general formula, not only the tied covariance equation, and retain the smoothness/bounded-derivative and fixed finite-memory assumptions.

The heading “Chaos needs intermediate temperatures” can become “High-temperature contraction and the nonuniform greedy limit.” No new computation is needed.

## T02 — Replace universality as a blanket guarantee with the actual cited scope

**Locations:** 85–89, 139, 1562–1574; “analogy exact” at 1568.

The cited random-network and period-doubling results do not say that arbitrary high-dimensional nonlinear two-parameter maps generically contain chaos, shrimps and Feigenbaum cascades. Sompolinsky–Crisanti–Sommers studies a specified random asymmetric recurrent-network ensemble and large-network limit; Kadmon–Sompolinsky explicitly makes the transition depend on architecture and transfer-function shape. Feigenbaum's result concerns a class of one-dimensional recursions with a specified critical-point order. Gallas studies shrimp domains in the Hénon map, with a broader conjectural interpretation, not a theorem for all dissipative two-parameter systems. The later sentence saying the structures merely can occur is appropriate but does not undo the earlier blanket wording. [Sompolinsky et al.](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.61.259), [Kadmon and Sompolinsky](https://journals.aps.org/prx/abstract/10.1103/PhysRevX.5.041030), [Feigenbaum](https://link.springer.com/article/10.1007/BF01020332), [Gallas](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.70.2714).

Suggested replacement: “Chaos and periodic windows occur in many nonlinear feedback systems. Random-network theory and familiar period-doubling and shrimp structures motivate looking for them here, but do not imply that this trained map realizes them.” The chain-rule factorization is exact; the identification with a random-network gain transition is an analogy, not an exact transfer of that ensemble's theorem. Keep finite-order ratios and the evidence for this particular map.

## T03 — Report cusp-candidate segments rather than six solved nondegenerate cusps

**Locations:** 42–44, 116–120, 193–197, 620–629, 631–636, 671–674, 689–692, 1208–1216, 1316, 1579–1583, 1640–1649, 1761 and 1815.

The V11 log at 22331–22335 records normal-form coefficients at fold points bracketing a sign change of the quadratic coefficient. Nonzero cubic coefficients at the two endpoints do not themselves establish nonvanishing at an intervening zero, a nonzero derivative there, or a rank-two unfolding at that zero. The U1 log at 22496 gives the unfolding Jacobians at P6 point 51, where b=14.75, and P7 point 13, where b=31.4. Their determinants 2737 and 53.5 are legitimate nearby-fold numerical measurements, not measurements at solved b≈0 cusp points.

The accepted independent Task14 endpoint report explicitly supports a **candidate cusp segment**, not a located nondegenerate cusp. It checks original points 13/14, their coefficients and sign change; it does not locate b=0 or test rank-two unfolding there. The source supplement confirms the producer's use of the original indexed fold point: `tools/cusp_unfolding.py` 73–79. This distinction applies at numerical rung 1; it is not a demand for interval proof or a proof assistant.

Suggested replacement: “Six fold-curve segments have bracketed quadratic-coefficient sign changes and nonzero endpoint cubic coefficients, supporting cusp candidates. At nearby fold points in two windows, the estimated two-parameter unfolding is nonsingular and gives the prospective local tests below.” Keep the measured determinants and endpoint coefficient agreement. Remove “located numerically nondegenerate” from the priority and evidence-ladder claims unless separate evidence actually locates and tests those points.

Authority: `codex/cusp-crosscheck/endpoints-001-result.md`, SHA256 `41e4070e23fd4ecd5cfed59c627c3e6bfa56c24ba93232c2caf731d2ebd7f875`; source supplement manifest `2ddce5b722cd3defc1f9a30c40ea1378e9777d482cb1f30332e4b1e6f8e34907`.

## T04 — Describe the prospective cubic test as seeded-root recovery, not exhaustive orbit counting

**Locations:** 42–44, 96–97, 116–120, 195–197, 631–636, 1213–1216, 1581–1583 and 1644–1645.

The exact source `tools/cusp_unfolding.py` (SHA256 `de36e9f39ae50bbbe21ecd66d4e6c17e5dfba2d4106ed8e9cce9b08828f64e8b`) saves predictions before solving, which supports its prospective character. Lines 94–95 seed Newton only from the cubic's predicted roots plus zero; lines 96–112 retain converged distinct solutions and compare their number and stability with the prediction. This checks recovery of predicted branches under those seeds. It does not exhaustively exclude additional nearby solutions.

Suggested replacement: “At twelve frozen offsets, Newton solves initialized from the cubic's predicted roots and the base state recovered matching numbers of distinct solutions at all twelve; predicted stability patterns matched at eleven. The search was not exhaustive.” Preserve **12/12 recovery-count agreement and 11/12 stability agreement**, including the failed stability row. “Predicted how many periodic orbits exist” is stronger than this implementation establishes.

## T05 — Preserve the loop-gain result, correct its bounds and its interpretation

**Locations:** 109–110, 243, 278–290 and 1691–1695.

Task18 supports expanding directions at every step together with a contracting eight-step cycle. Full step Jacobians **do** compose exactly by the chain rule; individual norms or spectra alone do not determine the ordered product. At 109 and 1693 use “every step expands some direction.” At 1692–1694 replace the statement that step Jacobians mislead or explanations do not compose with: “Stepwise norm or spectral summaries can be inconclusive; the ordered Jacobian product captures the cycle's linear stability.” Uniform stepwise contraction is itself a sufficient stability condition, so “stability cannot be read from the steps” is too absolute.

The very small second-to-first cycle singular-value ratios support strong local linear compression, not a proved one-dimensional nonlinear invariant set. At 288 say “strongly anisotropic linear return” rather than treating an effective one-dimensional stripe orbit as established by that ratio.

| Paper wording | Accepted value | Correction |
| --- | --- | --- |
| 243: at most 1.3e-13 | producer maximum 1.3417296425324618e-13 | approximately 1.34e-13, or a genuine upper bound such as 1.35e-13 |
| 282: norm between .06 and .64 | own extrema .05947348924 and .6437022935 | approximately .059–.644 |
| 285: ratio from 2e-6 | minimum approximately 1.716e-6 | approximately 1.7e-6–2e-4 |
| 286: reproduced to 8e-14 | own directional maximum 8.265928e-14 | approximately 8.3e-14 |

The signed eigenvalues +.01104015587 and −2.96117454e-7, step norms 1.24008–20.90778, product-of-norms 1.06799e7, and phase-dependent cycle norms are consistent with the accepted reports. Do not replace signed eigenvalues with only moduli in this explanation. The completed own run originally exited 1 during final status serialization; separately authenticated saved-summary recovery supplies the result. Producer p8 arrays were unavailable; p8b aligned at forward shift zero. These provenance limits belong in the reproducibility note, not a claim of an originally successful own terminal status.

The newer producer alignment test in log 23158–23168 reports **0/16 unstable Haar realignments**, all more contracting than the actual cycle. This is a disclosed producer result, not independently reproduced here. It does not contradict the matrix-product result, but it contradicts a causal story that special alignment was necessary for stability in this intervention. Incorporate that adverse test if discussing the alignment mechanism; do not quietly promote the loose norm bound to such a mechanism.

Authority: accepted own-results report `117a17a07df99b96db47e81532d840d407029ef42476f28489d4d9ff46578b96` and comparison review `3ce67ac96b5aa29b873f96707a698c44ab5693210f4d7123d63bdd89cfd5e1ea`.

## T06 — Replace demonstrated one-dimensional reduction with fitted scalar approximation

**Locations:** 143–145, 1241–1263, 1268–1273 and caption 1279–1285.

The scalar fits and approximate exponent agreement are useful evidence. They do not establish an injective observable, a one-dimensional invariant curve, or smooth conjugacy of the full state dynamics. The paper correctly conditions its coordinate-change statement at 1251–1252, but the unconditional assertions at 1244–1248 and 1261, and the figure captions, exceed it. A low-residual polynomial is not evidence that the observable is invertible. For a genuine smooth coordinate change, finite-time exponent differences also contain an endpoint term involving its derivative; asymptotic invariance does not require identical finite estimates.

A concrete analytic counterexample is F(x,y)=(4x(1−x),4y(1−y)), observed through x. Its scalar return is exactly unimodal, and under the usual product invariant measure its scalar exponent ln 2 equals the full largest exponent, yet two dimensions expand and the observable discards y. Even perfect scalar/full agreement cannot establish one-dimensional conjugacy by itself.

Suggested replacement: “On the sampled phase-6 record, a scalar return relation is well approximated by a polynomial with one quadratic maximum. Across many sampled cells, scalar estimates approximately reproduce the full-record twin estimates, with retained exceptions. This supports an empirical scalar description, not a proved one-dimensional reduction.” In captions use “separated sampled phase clouds” and “fitted return curve”; a PCA picture and finite pairwise separation do not establish globally disjoint invariant bands. The correct own minimum sampled cross-phase distance is .09832051568; .10053639036 is the different centroid statistic. The accepted independent check supports the former, but the original plotting centroid calculation alone does not.

At 1250 retain “a single stationary point,” not automatically “a maximum”: the original procedure gives 59 such cells but only 36 with a quadratic maximum. At 1284, 2000 steps at period 28 give 71 or 72 phase observations and 70 or 71 return pairs, not 72 complete repetitions. A fitted maximum on the sampled range does not by itself establish the hypotheses of Feigenbaum universality.

## T07 — Add the accepted independent checks without replacing historical outcomes

**Locations:** 1246–1259 and 1279–1285. These are new evidence to incorporate, not necessarily errors at the paper's 08:53 cutoff.

The original **49/62** and disclosed amended **51/62, median absolute discrepancy 3.7%** reproduce. The amendment is outcome-informed: adopting one rule before computing the amended aggregate does not undo prior inspection of the outlier and degree/spline trials. Keep it exploratory on the same records. Random-fold value CV on adjacent trajectory pairs does not establish independent future generalization or derivative/log-derivative accuracy.

The independent complete saved-sequence analysis gives original **49/62, median 4.196231146%**; amended **51/62, 3.658777310%**; purged-contiguous sensitivity **51/62, 3.414727798%**. All 65 records and the three nonqualifying cells remain. There are **60 fold-rank warnings** and selected degree-24/rank-23 phase fits in cell 48; those specific phases do not enter that cell's qualifying aggregate. Counts agree, but 2070/4140 producer comparison leaves differ numerically, maximum 2.46935055e-8. Do not call the complete computation bit-exact or uniformly full-rank.

| Independent trajectory | Full-state twin estimates | Original / amended / purged scalar discrepancies against own twin mean |
| --- | --- | --- |
| Primary, observed period 7 | .08887170325 / .08826492488 | +2.3096% / +1.5378% / +1.7041% |
| Fixed optional cell 04, observed period 10 | .05406364647 / .05428446970 | −5.8974% / +.2459% / +.2459% |
| Fixed optional cell 13, observed period 28 | .00663487521 / .00554844752 | +18.0811% / +18.0303% / +18.0303% |

All 2000 token IDs match the corresponding producer record in each own trajectory. No state return passes the fixed detector through lag 64; this is a finite tested bound. Primary producer twins .09109385979/.09065116593 remain distinct from the own estimates. Own cell 04 changes the original selected phase from producer 1 to 6, and its qualified fits have both a maximum and a minimum. Own cell 13 retains a selected phase-19 minimum, early negative running twin means, and a persistent >10% mismatch despite tiny value error. The new trajectories do not overwrite the all-cell historical counts or original cell-13 approximately +66% discrepancy. The stress cases are fixed exploratory checks after producer-summary exposure, not a fresh representative or blinded test.

Authority: accepted Task19 final synthesis `66feba6283cd4ac70738d17699a0bee64c7fec8386a9d8ff434da9fac129914f`. It retains all phase, candidate, fold, rank, arithmetic and disagreement records; no repeated audit is needed.

## T08 — Preserve the new delay test's failed criterion and its actual selection procedure

**Potential addition to:** 1253–1261. **Source/log:** `tools/return_map_delays.py` 76–99 and research-log 23158–23168. This test is not yet described in the frozen paper.

The new producer log reports **30/61** failures of its declared ≥20% delay-improvement criterion. It also reports small absolute normalized errors. Small error is not an independently measured numerical noise floor, so “at numerical noise” cannot erase that failed criterion. The source chooses the phase with the smallest scalar error **on the held-out segment itself** (92–93), then compares delays there. The final best-phase score therefore uses the evaluation segment for selection. It also loops over all eligible nonconstant phases without enforcing the previously declared single-valued-phase gate. Do not describe this as a clean independent test of scalar sufficiency on preselected qualifying phases.

Suggested wording if included: “An exploratory temporal split gives small value-prediction errors at the test-selected best scalar phase, but 30/61 cells fail the prespecified relative delay-improvement rule. These results do not establish scalar dynamical sufficiency or a numerical error floor.” This is a source-level limitation, not an independently reproduced 61-cell result or a request for a new experiment.

## T09 — Correct the nearby literature grouping and narrow priority language

**Locations:** 75–76, 168–175, 193–202 and 1579–1585.

Luo–Yu is grouped with Jacobian/fixed-point stability analyses, followed by the claim that the grouped analysis concerns fixed points. Its primary abstract instead concerns finite-depth score-margin geometry and endpoint-answer preservation on Huginn/Ouro, using completed trajectories. Move it to a separate sentence about finite-depth prediction/readout dynamics; do not characterize it as a fixed-point/Jacobian study. [Luo and Yu, arXiv:2609.21383](https://arxiv.org/abs/2609.21383).

Kim is correctly acknowledged as a close derivative-based normal-form precedent: the primary record explicitly gives parameter-free fold passage and spectral-gap predictions in a task-trained looped transformer. Keep that comparison. Restrict the new priority statement to the specified released pretrained map, objects and finite tests; it should not revert to “first derivative-based dynamical prediction in a transformer.” Remove the unsupported solved-cusp part identified in T03 and qualify orbit counts as in T04. The targeted-search limitation at 200–202 is appropriate. [Kim, arXiv:2608.26556](https://arxiv.org/abs/2608.26556).

## Retained theory and smaller wording points

The Hopfield comparison at 246–250 is supported by the cited update/energy analysis. The symmetric-positive-semidefinite argument at 251–253 is also sound for the stated smooth map on a convex domain: write F=∇Φ with Φ convex and V(x)=||x||²/2−Φ(x). Convexity gives V(F(x))−V(x)≤−||F(x)−x||²/2. On the compact invariant hull, this excludes nontrivial recurrent cycles; it does not require a norm bound below one. This analytic observation supports the manuscript's argument, rather than creating a new computation gate. [Ramsauer et al.](https://arxiv.org/html/2008.02217).

The attention citations at 298–301 are appropriately qualified: Dong et al. concerns pure attention and the roles of skip connections/MLPs; Geshkovski et al. studies idealized attention dynamics. Their results do not need to be removed. [Dong et al.](https://proceedings.mlr.press/v139/dong21a.html), [Geshkovski et al.](https://proceedings.neurips.cc/paper_files/paper/2023/hash/b2b3e1d9840eba17ad9bbf073e009afe-Abstract-Conference.html).

At 292–296, the derivative formula does not prove semantic irrelevance: a direction's geometry affects both factors and can correlate with its semantic interpretation. Prefer “The gain mechanism is not specific to concept-labelled directions; the finite norm-matched controls also find chaos under random directions.” Root's companion note handles 1719–1720: positive finite exponents do not prove mixing, forgetting all initial information or inability to stabilize/control a loop. At 1714 replace “never settles” with “did not settle over the recorded horizon.”

The previously duplicated figure-label issue does **not** persist in this packet: the inspected return-map and hidden-band figures use distinct labels. No correction is requested for that old issue.

## Review boundaries and disposition

This review does not repeat the old complete bibliography audit. Targeted primary records above support the specific new claim-scope checks; they do not establish exhaustive novelty. The general covariance and conditional high-temperature reasoning, signed Task18 multiplier result, retained 49/62→51/62 amendment, and the finite Task19 agreements remain useful findings. The report requests faithful wording and incorporation of already accepted results, not new model runs, cusp solving, statistical tests or stricter proof infrastructure. Root owns final paper-review synthesis; Inference alone owns adoption/checkpoint/delivery. Huginn, certificate, behavioural and large-model/search-scale findings are supplied in the parallel notes.
