# Claim ledger

Status: private exploratory program, updated 2026-09-14 through EXP-011 and the saved-input audits. Original training claims retain their meanings. A small fixed-weight recurrent shrimp is numerically reproduced. The retrospective four-preview geometry learner improves the observed shared-feature mean but is worse than the signed-axis control in the pooled mean; its paired improvement remains uncertain. Modern training/reasoning benefit and net compute savings are not established.

**Current release qualification:** EXP-008 is a private working diagnostic
draft, not a fully reproduced release. PR #7 remains unmerged. Local frozen
verification passed, but release candidate
`1d12ebeda0400c179e54dd36e647a835f7e0d91d` fails Linux PR run 34803148459 at
baseline coefficient `[1]` (absolute difference 3.68269315e-8; combined
`rtol=1e-7, atol=1e-9`) and Linux push run 34803133930 earlier at shared feature
`[34]` (1.39784018e-14; combined `rtol=1e-12, atol=1e-14`). See the
[PR](EXP-008-linux-pr-failure.txt) and [push](EXP-008-linux-push-failure.txt)
logs. The push does not complete feature/waveform replay; the PR reaches fit
replay. Later [PR](EXP-008-platform-linux-pr.json) and
[push](EXP-008-platform-linux-push.json) diagnostics at
`bb4ecd25cf5c025064e7bd200b6acd898953d0e5` pass exact evidence bindings and
preserve all three penalties and all 288 learned actions in **saved-feature
refits**. Shared-feature failures remain 2/0 and baseline-plus-shuffled
coefficient failures 8+9/10+10 (PR/push); geometry coefficients pass in both.
This is bounded agreement, not complete regenerated-input or integration
reproduction. The strict gate has not passed; original failures remain evidence.

| ID | Claim | Current status | What would warrant it | What does not warrant it |
|---|---|---|---|---|
| L1 | Neural systems can exhibit shrimp windows | Published precedent | da Silva and Rech (2022), cited in the literature file | Generalization to every neural architecture |
| L2 | Training and latent inference require different state/control definitions | Mathematical setup | Explicit update equation and complete evolving state | Treating an initial-state slice as a learning-rate map |
| E1 | This tiny training map has a nontrivial periodic orbit | Not demonstrated by EXP-002; all finite tails unresolved | Full-state primitive-period recurrence and independent closure | Repeated scalar loss or predictions |
| E2 | That orbit is locally attracting | Not demonstrated by EXP-002; no neural candidate verified | Period Jacobian product with spectral radius separated below one | Finite-time near recurrence, a plot, or a large observed basin alone |
| E3 | The tiny training parameter window has shrimp organization | Not established; training-map continuation not implemented | Connected verified isoperiodic window and continued bifurcation boundaries, with convergence checks | E1 + E2 at one pixel, visual similarity, or a Hénon control |
| U1 | Recurrence improves intervention timing beyond scalar stagnation | Not demonstrated by EXP-003; recurrence never intervened | Paired complete trajectories, equal action opportunities and doses, held-out cases | Beating a fixed learning rate alone |
| U2 | Shrimp geometry supplies a useful predictive/control representation | EXP-008 lowers observed all-candidate prediction MSE by 17.17%; primary selected-action benefit unresolved, and direct minimum preview has lower error; see I1--I6 | Matched feature/policy comparison with geometry's representation, ordinary features, task endpoints and costs distinguished | New information conditional on coordinates, a topology-specific mechanism, or control utility inferred from lower prediction MSE alone |
| U3 | The method provides net compute savings | Not established | Matched total budgets including scans, tuning, diagnostics, and failures; declared amortization | Equal optimizer-step counts, conditional rescue rates, or wall time excluding atlas cost |
| U4 | A stable cycle is beneficial or harmful | Task dependent; not assigned by dynamics alone | Independently measured correctness, validation quality, latency, and robustness | Negative Lyapunov exponent or low orbit residual alone |
| T1 | Positive relaxation preserves fixed points of a deterministic map | Algebraic fact under stated assumptions | `G_alpha(h)=h` iff `F(h)=h` for `alpha>0` | Preservation of cycles, basins, speed, or correctness |
| T2 | An attracting neural training cycle can change predictions and averaging bias | Exact result for the tied scalar toy in [analytic-cycle.md](analytic-cycle.md) | Direct orbit equations, primitive separation, multiplier, and readout algebra | Transfer to the tanh pilot, shrimp geometry, or generalization |
| T3 | Prediction averaging reduces average phase squared error | Exact variance decomposition; residual bias remains in the scalar toy | Compare squared error before and after averaging predictions on the same inputs | Superiority to a convergent optimizer or an equally funded generic averager; a guarantee for decoded accuracy |
| X1 | The approach benefits modern latent reasoners | Future extension, no evidence here | Exact checkpoint/task specification, full-state diagnostics, complete paired inference trials | Tiny feedforward training results or analogy to a preprint |

## Allowed outcome language

- **Positive diagnostic pilot:** “In these exploratory cases, the recurrence-triggered rule improved [receipt-backed endpoint] relative to [named controls]. This does not establish a contribution from shrimp geometry.”
- **Geometry-specific positive:** Only after a new frozen holdout study: “On the specified held-out task distribution, adding the measured geometry improved [endpoint] beyond [complete baseline set], under [cost regime].”
- **Null:** “This search/controller did not demonstrate [claim] within the specified ranges, horizons, precision, sample, and budgets.” Do not convert a finite search into nonexistence.
- **Mixed:** State quality and computation effects together. Do not describe faster wrong answers as useful reasoning.
- **Implementation defect:** Preserve original data and claim state; issue a dated correction and bind the repaired run to a new code/configuration hash.

## Claims explicitly outside scope

No first discovery of neural shrimps; no universal requirement to avoid chaos; no proof that all reasoning slowdowns arise from saddles; no semantic interpretation of hidden-state geometry without task evidence; no large-model speedup from a tiny deterministic model; no claim that a private pilot is preregistered or confirmatory. The existing literature is a starting map, not an exhaustive novelty certification.

## 2026-09-08 initial pilot evidence update

Source revision: `bd3d733b51e9f1ae9247ba712633aca023e7afac`, pushed before execution. Raw receipts: [EXP-001](../evidence/EXP-001/attempt-001/receipt.json), [EXP-002](../evidence/EXP-002/attempt-001/receipt.json), [EXP-003](../evidence/EXP-003/attempt-001/receipt.json). Full interpretation and generated empirical counts appear in the manuscript's pilot-results discussion. The descriptive report directories are `evidence/reports/EXP-001`, `EXP-002`, and `EXP-003`.

- The Hénon control supplies positive numerical-orbit evidence for the instrument. It supplies no new neural result and no shrimp topology certification.
- The neural scan found no verified attracting nontrivial cycle. Unresolved finite tails and operational divergence remain reported; global nonexistence is not inferred.
- The recurrence policy did not trigger. Independently comparing every paired fixed/recurrence raw artifact showed equal complete state, loss, and learning-rate arrays (including NaNs at matching positions), equal statuses and update counts, and equal terminal predictive losses. This establishes identical realized dynamics in these pairs, not general equivalence between active controllers.
- Divergent utility runs occur under every policy. The predeclared all-finite mean is therefore undefined for each policy; no successful-only average is promoted to the primary result.
- No net benefit, useful modern neural shrimp, or need for paid scale-up has been demonstrated. The next local exploratory stage should first identify and verify a neural periodic regime, then test task benefit under controlled interventions.

No original failure, trial, or consumed attempt was discarded or reset. This update is an outcome-status update, not a retrospective change to the original pilot design.

## 2026-09-11 visual and theoretical update

Six empirical figure sets re-express existing evidence, including explicitly
post-run saved-state diagnostics and finite-case illustrations. Three diagrams
and one exact algebraic plate are separately labeled. None changes E1–E3 or
U1–U3. T2–T3 add a constructed example and a mathematical identity, not a new
empirical result or novelty claim. Figure receipts and the
[visual guide](visual-revision.md) bind the intended interpretations.

## 2026-09-11 actual recurrent-network reproduction

| ID | Claim | Current evidence and limit |
|---|---|---|
| R1 | This three-neuron recurrent network has a shrimp-shaped periodic window | EXP-004: 39,322 grid cells and 201 independently initialized transect points reproduce the published shape. Selected orbits are verified; exact boundary topology and every pixel are not certified. |
| R2 | The shape corresponds to repeating neural activity | Two nine-peak cycles and one eighteen-peak cycle pass full-state shooting, primitive peak counts, neutral-flow checks, and transverse stability. Actual tanh activations and state waveforms are stored and plotted. |
| R3 | Locations within the window have different local stability | Dominant transverse multipliers 0.06506010, -0.46513914, and 0.65612133 at the three certified orbits. These are per-cycle local state responses, not basin size, weight-drift tolerance, or task performance. |
| R4 | The originally selected exterior is a validated chaotic control | Rejected: coarse period 0 becomes 9 at both refinements. The failure remains in all result tables and evidence. EXP-005 is a distinct prospective longer-horizon comparison. |
| R5 | Full activation readout preserves the exact state period | Componentwise tanh is injective for finite states; therefore the full activation vector has the same exact primitive period. Floating-point saturation affects numerical visibility. This does not apply automatically to a scalar, averaged, or decoded readout. |

Evidence: [EXP-004 receipt](../evidence/EXP-004/attempt-001/receipt.json),
[protocol](EXP-004-protocol.md), and [next steps](next-steps.md).
R1–R5 do not change the original EXP-002 training or EXP-003 utility outcomes.

EXP-005 update: the two fixed distant controls retain absent short recurrence
and finite-time exponents 0.123–0.139 at both steps after a 10,000-unit settling
horizon. The core and previously failed near-edge comparison independently
certify as nine-peak cycles. Under the one frozen 1e-6 perturbation, activation
separation at time 300 is 1.53e-6 at the core versus 1.41 and 2.13 at the two
irregular controls. This supports a concrete finite-horizon state-to-output
sensitivity contrast, not global chaos, basin robustness, causal task benefit,
or a comparison averaged over perturbation phases and directions. Original
EXP-004 classifications and failures remain unchanged.

## 2026-09-11 waveform utility and continuation update

EXP-006 source `ddb568e6cd67ef2cfbdb45e05710cd86025ef548` was pushed before its
exclusive attempt. All 12 attempted anchors qualified and all 4,320 trial-step
records are complete. See the [raw receipt](../evidence/EXP-006/attempt-001/receipt.json),
[utility audit](EXP-006-utility-results-audit.md), and
[continuation audit](EXP-006-continuation-results-audit.md).

| ID | Claim | Evidence and permitted interpretation |
|---|---|---|
| D1 | Atlas margin improves immediate waveform preservation over the declared symmetric local comparator at the largest tested dose | At native radius 0.016, RMSE 0.301672 versus 0.466653, about 35.4% lower; paired benefit 0.164981, descriptive 95% interval [0.013853, 0.326023], positive at 8/12 anchors. This is a fixed-sample, dose-specific comparison, not a universal geometry advantage. |
| D2 | Atlas margin is generally superior to local sensitivity | Not established: both smallest doses favor local sensitivity at every anchor. The prespecified across-radius benefit is 0.025738 [−0.027862, 0.083081], unresolved. |
| D3 | Atlas margin is the best observed direction policy | Rejected: the `tangent_plus` diagnostic has lower mean waveform error at every dose, including 0.260795 versus 0.301672 at radius 0.016. This is an exploratory arm comparison, not a replacement primary endpoint. Tangent and signed coordinate-axis policies belong in a future frozen baseline set. |
| D4 | Eventual nine-peak retention guarantees immediate waveform preservation | Rejected as a utility inference. Atlas retention is 188/192 versus 308/384 across both local signs after 10,000 settling plus 2,000 measurement units; waveform error instead measures the immediate 0--300 interval. The fixed `anchor-08` example favors both local signs although all shown arms eventually have nine peaks. Transients, phase, frequency, and shape are not isolated. |
| C1 | Continuation retains periodic branches beyond attraction | The four branches retain 196 node records, including 122 unstable records and five failed correctors. Counts include overlapping branches and repeated seeds. They are not counts of distinct cycles. |
| C2 | Every continuation checkpoint has an agreeing independent geometric check | Rejected: agreement is 27/28. Near-edge initial positive-gamma node 40 has independent closure 1.0591e-8 against the unchanged 1e-8 threshold; it remains failed despite state and period agreement. |
| C3 | The complete shrimp boundary or exact bifurcations have been located | Not established. Local gamma sections provide crossing candidates and reversals. Adjacent arclength-node gamma values need not enclose the crossing gamma at a fold. A failed corrector does not prove disappearance. |

The paired benefit sign is comparator RMSE minus atlas RMSE. Intervals use the
frozen anchor-cluster bootstrap and describe variation among the fixed selected
anchors, not a random population or confirmatory hypothesis test. No failed
condition was dropped, replacement anchor selected, threshold relaxed, or
consumed attempt reset. The frozen illustration is retained even though it
favors the comparator.

This establishes a bounded waveform-preservation experiment, not a completed
coupled controller, net speedup, or trained-model transfer. Collection and
analysis took 423.866 local seconds, excluding the final verification pass;
26.627 seconds of local sensitivity are included in that time, and the earlier
140.54-second atlas cost is separate. Paid compute and paid review spend remain
zero. A hybrid local/geometry policy and stronger baseline tests are proposed
in [next-steps.md](next-steps.md), not reported as successes.

## 2026-09-11 EXP-007 and preprint update

| ID | Claim | Evidence and permitted interpretation |
|---|---|---|
| H1 | The hybrid transfers as a clearly superior policy | Not established: all five primary pooled intervals include zero. Largest-dose local-minus-hybrid benefit 0.144887 [−0.045524,0.353194], positive at 6/12 new starts. |
| H2 | The tangent benefit independently requires detailed shrimp geometry | Not supported in this sample: tangent and axes choose exactly the same actions in 192/192 conditions, with identical 384 target state arrays across two steps. These are gamma-only actions; no global optimality claim. |
| H3 | More short previews guarantee better later waveform quality | Rejected for the frozen example: augmented search tail RMSE 0.589753 versus local 0.037763 and hybrid 0.059668. This is an explicitly selected example, not an overall superiority claim for local. |
| H4 | All individual target scores are numerically converged | Not established: coarse/fine absolute RMSE differences have median 1.303e-7 and maximum 0.234864. Aggregate unresolved comparisons persist. |
| L1 | The new preprint establishes shrimp control or infinite finite-network capacity | No. The versioned unreviewed work motivates a separate multiple-memory interference hypothesis. Our exact independent-spin checks expose printed inconsistencies; implementation impact remains unknown and its numerical capacities are unreplicated. |

EXP-007 source `ac098c3e3833daaf27c7dc2869d45e8b7085f948` was pushed before
its only attempt. All 12 baseline attempts qualify; all 4,032 previews and
2,784 target records complete. No failed attempt was reset or substituted.
The new starts exclude neighborhoods of EXP006 origins but reuse its old
atlas; this is not an unseen-network or unseen-map result. The independent
audit and exact receipt are linked from the README.

## 2026-09-13 EXP-008 learned-representation update

Source `3c5604497ec57b19d62fbb595da7b190102f4b33` was pushed before its only
attempt and is retained as `archive/EXP-008-source-v1`. All 24 attempted origins
qualify, 3,456 previews complete, and all 7,008 candidate/sham target records
complete. Development fits and all test choices were sealed in the declared
order. Read the [raw-data audit](EXP-008-results-audit.md),
[independent model audit](EXP-008-model-results-audit.md), and
[generated summary](../evidence/information-paper-summary.json).

| ID | Claim | Evidence and permitted interpretation |
|---|---|---|
| I1 | Old-map features improve prediction for this frozen representation | Observed fine-step all-candidate MSE falls from 0.022640656 to 0.018753462, 17.17% lower; seven of twelve blocks improve. This is a secondary mean, not a resolved population prediction claim. |
| I2 | Lower prediction MSE establishes better action selection | Not established: primary pooled benefit is 0.000857628, descriptive 95% interval [-0.007701467, 0.006545434], with eight positive blocks, three negative and one tie. Choosing the lowest-error action is different from predicting all candidate errors accurately. |
| I3 | The geometry learner beats the direct forecast rule | Rejected in this sample: minimum-preview RMSE is 0.009958855 versus geometry 0.016966044, 41.30% lower. Preview-minus-geometry benefit is -0.007007189 [-0.014731835, -0.001611192]. All policies have the same eighteen actions and previews. This is a diagnostic fixed-sample comparison, not net speedup. |
| I4 | Finite-action regret independently confirms task improvement | No: the common eighteen-action oracle cancels, so paired regret reduction equals paired selected-RMSE reduction. The oracle is neither a global optimum nor an independent endpoint. |
| I5 | Every candidate label and coefficient is numerically converged | Not established: maximum coarse/fine candidate RMSE change is 0.476851 across 3,504 pairs. Independent local SVD exceeds the unchanged coefficient tolerance for eight baseline and eight shuffled coefficients; geometry coefficients pass. Chosen penalties and actions agree in that local audit, and selected-policy step differences are much smaller. The separate Linux failures remain unresolved. |
| I6 | EXP-008 establishes shrimp-specific information, trained-model benefit or cheaper control | Not established. Geometry is a deterministic representation of coordinates and a previously observed atlas; the fixed ridge family, cloned states and simulator access limit the claim. No trained model or sequential controller was tested, and all eighteen previews were paid for. |
| I7 | EXP-008 is fully reproduced on Linux and ready to merge | Not established: strict checks fail and PR #7 remains unmerged. Two later Linux diagnostics preserve all three penalties and 288 learned actions using saved-feature refits, while retaining feature/coefficient failures. Their completion does not replace the strict gate or establish general portability. |
| I8 | Learned action choices survive the two measured Linux saved-feature refits | Supported within this scope: PR 34804593644 and push 34804589708 have maximum prediction differences 1.3874261184e-9 and 1.9615957192e-9, against minimum saved action-score gap 3.7010270022e-6. Saved-model score replay passes rtol=1e-12, atol=1e-14. No completely regenerated-input fit, waveform-metric replay or integration is performed by this diagnostic. |

The first declared example remains `information-00`, phase 675, radius 0.016,
fine step. Every policy selects candidate 4 and has RMSE 0.0229066824; curves
correctly coincide. The unfavorable `test-block-11` contribution (-0.040413692
pooled) remains in all results. Negative predictions, crop masks, step-sensitive
unselected targets and the independent coefficient mismatch are not repaired
and rescored as prospective successes.

Collection and summary used 162.352 local seconds before verification,
including 23.770 sensitivity seconds and 0.079 fit/seal seconds; the prior map
costs 140.54 seconds separately. Paid compute and review remain zero. The
next provisional question is whether geometry can select a useful four-action
shortlist **before** paying for previews, against equal-budget controls and the
full-eighteen reference. It first requires Linux portability and numerical-label
review, followed by a new freeze;
it is not a successful result or an executed experiment.

## 2026-09-14 server numerical follow-up

Arithmetic method/source was pushed at
`b3cf51bb88df75eccec4b19e0f7a15976b3a9b21`; the regeneration audit and
EXP-009 executable plan were pushed at
`d276782c229aa33e2819ec89f15df4e23c7f7834` before their outcomes. These are
post-run numerical studies of previously observed cases. Original data,
model choices, thresholds and failed comparisons remain unchanged.

| ID | Claim | Evidence and permitted interpretation |
|---|---|---|
| N1 | Directional cancellation contributes to the two known feature failures | Supported on saved sensitivity tensors: direct residual-norm evaluation is much closer to wider-precision and Decimal checks at the failed cases. This diagnoses arithmetic on saved inputs, not integration error or universal superiority of one solver. |
| N2 | Original choices survive a complete saved-trajectory input regeneration | Supported on this server: all 48 contexts, 3,456 previews and 7,008 targets regenerate; all aggregate waveform scores, three penalties, 288 learned actions and 96 preview actions remain unchanged. Nineteen eigenvalue/coefficient checks fail. New ODE/sensitivity integration and the original strict gate are separate. |
| N3 | Selected-policy task results are stable under finer target integration | Supported for the original 220 unique selected actions in EXP-009: all meet its predeclared native RMSE-stability rule. Largest new-step difference is 3.822096e-6; largest old-fine-to-finest change is 6.521438e-5. The mean baseline-minus-geometry contrast is 0.000857638205. This is post-run numerical sensitivity, not new held-out utility evidence. |
| N4 | Finer integration resolves all candidate labels | Rejected at the tested resolutions: 952/1,012 prior-selected sensitive cases still exceed 1e-4 native RMSE disagreement; one more fails the nonincreasing-difference criterion. Across the full selected cohort 416/1,369 pass both native RMSE criteria and 953 remain unresolved. Independent-panel RMSE agreement holds for 13/19 cases; 12/19 pass all waveform comparisons. This selected sample cannot estimate a population failure rate. |
| N5 | The numerical follow-up establishes a practical geometry advantage | Not established. Original control choices and their small unresolved contrast persist. Prediction across all candidates requires explicit treatment of label uncertainty; net preview savings still need their own frozen comparison. |
| N6 | The complete 75-unit preview table is numerically qualified for retrospective development | Supported by EXP-010 at the tested resolutions: 3,456/3,456 native scalar and waveform checks pass, 16/16 independent sentinels agree, and all 192 complete candidate rankings remain unchanged. Maximum finer-step RMSE difference is 5.818072e-6. This is not a qualification of EXP-009's longer outcomes, a new shortlist result or a net compute-saving claim. |

Full new raw reports and traces are under `/data2/neural-shrimp-data/`, with
compact hash-bound summaries in the repository. See the
[arithmetic](EXP-008-arithmetic-results-audit.md),
[regeneration](EXP-008-regeneration-results-audit.md), and
[refinement](EXP-009-results-audit.md) audits. All 2,776 EXP-009 traces complete;
no failed attempt was overwritten, paid resource launched or public release
made.

## 2026-09-14 local / 2026-09-15 UTC: retrospective shortlist feasibility

EXP-011 source `64712453153ade24d2b22cf811f7a83811304de4` was privately
pushed before fitting. The 24 historical origins are all development evidence;
each outer fold excludes its origin from fitting and inner tuning. Predictions
and four-entry shortlists are sealed before their labels, and chosen winners
are sealed before full-eighteen oracle evaluation.

| ID | Claim | Evidence and permitted interpretation |
|---|---|---|
| S1 | The frozen four-preview recipe runs with qualified numerical behavior | Supported on this server: all 24 folds and 96 recipes complete; all SVD/QR prediction, selected-penalty and ordered-shortlist gates pass, as does complete built-in model replay. Both label resolutions retain the same shortlists and actions. This is not general cross-platform certification. |
| S2 | Geometry improves the matched shared-feature shortlist | Observed pooled RMSE is 0.0107899 versus 0.0162661 (33.67% lower). The paired contrast is 0.00547615 with descriptive bootstrap bounds [-0.00265276, 0.01561846]. Its sign varies with radius; no fresh-origin confirmation or guaranteed increment follows. |
| S3 | Learning or geometry beats strong simple four-preview controls | Not established. Signed axes have the lowest pooled four-preview RMSE, 0.00856899, versus geometry 0.0107899 and local coverage 0.0124641. Local coverage has the lowest mean at the three smaller radii, and geometry at the largest. Choosing a radius-dependent winner now would be a new, outcome-informed policy. |
| S4 | Four previews preserve full-eighteen quality or save total runtime | Not established. Full-eighteen mean RMSE is 0.00617504; even the best pooled four-entry control adds 0.00239395. Four versus eighteen means fourteen fewer candidate calls by design, but this study reuses all labels and measures no new preview acquisition, sensitivity or atlas acquisition cost. No task-quality margin was specified. |
| S5 | Shortlist results qualify long-target control, modern training or a shrimp-specific mechanism | Not established. The endpoint is 75-unit preview RMSE in the known fixed-weight system. EXP-009's unresolved 300-unit labels, spatial dependence, learned-model generalization and topology-specific alternatives remain separate questions. |

See the [EXP-011 protocol](EXP-011-protocol.md),
[results audit](EXP-011-results-audit.md) and
[compact summary](EXP-011-summary.json). Preserve the original EXP-008 strict
gate and every prior failed comparison.
