# Claim ledger

Status: private exploratory program, updated 2026-09-11 with visual analysis and an exact toy derivation. The original meanings of claims below are preserved; outcome statuses cite the retained evidence. No entry is evidence that a modern neural-network shrimp has been found in this repository.

| ID | Claim | Current status | What would warrant it | What does not warrant it |
|---|---|---|---|---|
| L1 | Neural systems can exhibit shrimp windows | Published precedent | da Silva and Rech (2022), cited in the literature file | Generalization to every neural architecture |
| L2 | Training and latent inference require different state/control definitions | Mathematical setup | Explicit update equation and complete evolving state | Treating an initial-state slice as a learning-rate map |
| E1 | This tiny training map has a nontrivial periodic orbit | Not demonstrated by EXP-002; all finite tails unresolved | Full-state primitive-period recurrence and independent closure | Repeated scalar loss or predictions |
| E2 | That orbit is locally attracting | Not demonstrated by EXP-002; no neural candidate verified | Period Jacobian product with spectral radius separated below one | Finite-time near recurrence, a plot, or a large observed basin alone |
| E3 | The parameter window has shrimp organization | Not established; continuation not yet implemented | Connected verified isoperiodic window and continued bifurcation boundaries, with convergence checks | E1 + E2 at one pixel, visual similarity, or a Hénon control |
| U1 | Recurrence improves intervention timing beyond scalar stagnation | Not demonstrated by EXP-003; recurrence never intervened | Paired complete trajectories, equal action opportunities and doses, held-out cases | Beating a fixed learning rate alone |
| U2 | Shrimp geometry adds predictive/control information | Proposed main question; not implemented | Matched feature/policy comparison versus loss, recurrence, stability, and ordinary performance-map baselines | U1, attractive fractal images, or a better-tuned controller |
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
