# Claim ledger

Status: private exploratory program, updated 2026-09-11 with EXP-004--006 recurrent-network evidence. Original training claims retain their meanings. A small fixed-weight recurrent shrimp is numerically reproduced. Waveform utility is dose dependent; the aggregate advantage over the declared local comparator remains unresolved. Modern training/reasoning benefit is not established.

| ID | Claim | Current status | What would warrant it | What does not warrant it |
|---|---|---|---|---|
| L1 | Neural systems can exhibit shrimp windows | Published precedent | da Silva and Rech (2022), cited in the literature file | Generalization to every neural architecture |
| L2 | Training and latent inference require different state/control definitions | Mathematical setup | Explicit update equation and complete evolving state | Treating an initial-state slice as a learning-rate map |
| E1 | This tiny training map has a nontrivial periodic orbit | Not demonstrated by EXP-002; all finite tails unresolved | Full-state primitive-period recurrence and independent closure | Repeated scalar loss or predictions |
| E2 | That orbit is locally attracting | Not demonstrated by EXP-002; no neural candidate verified | Period Jacobian product with spectral radius separated below one | Finite-time near recurrence, a plot, or a large observed basin alone |
| E3 | The tiny training parameter window has shrimp organization | Not established; training-map continuation not implemented | Connected verified isoperiodic window and continued bifurcation boundaries, with convergence checks | E1 + E2 at one pixel, visual similarity, or a Hénon control |
| U1 | Recurrence improves intervention timing beyond scalar stagnation | Not demonstrated by EXP-003; recurrence never intervened | Paired complete trajectories, equal action opportunities and doses, held-out cases | Beating a fixed learning rate alone |
| U2 | Shrimp geometry adds predictive/control information | EXP-006 supplies dose-specific exploratory waveform evidence; pooled local advantage unresolved; see D1--D4 | Matched feature/policy comparison versus loss, recurrence, stability, and ordinary performance-map baselines | U1, attractive fractal images, or a better-tuned controller |
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
