# Research opportunity and what the evidence leaves open

**The largest opportunity is to connect real dynamical instability to a finite-budget computation—not merely to demonstrate further chaos.** I recommend one next experiment: **a paired, task-level reproducibility test of continuous-feedback inference under batch-layout changes, with controlled perturbations establishing whether feedback amplification explains any discrepancies.**

This upgrades REPRO from “trajectories eventually separate” to a question frontier inference researchers can act on: **does an otherwise irrelevant execution choice change a returned answer within a useful inference budget?**

The supplied evidence already supports a substantial dynamical phenomenon. It leaves three important bridges open:

- **Asymptotic instability versus finite-horizon relevance.** Separation at step 1500 need not matter at step 32. Conversely, transient amplification can matter even when the asymptotic exponent is negative.
- **The autonomous one-token map versus actual inference.** Growing-context Soft Thinking is a different, generally nonautonomous process. A fixed-window implementation has a larger state than the current embedding alone. Its relevant sensitivity includes the retained history/cache.
- **State divergence versus computational consequences.** Hidden chaos establishes that leading-token repetition can conceal instability; it does not establish answer instability, accuracy loss, or an alignment failure.

MECH can explain where instantaneous gain resides, but entropy, stable rank, and separate gain factors do not establish which changes *cause* sustained expansion. Products of Jacobians depend on orientation and temporal ordering, not just their individual magnitudes.

**Most referee-vulnerable claim:** any behavioral conclusion built on the unestablished within-task −4 to −5 percentage-point residual. Among the established dynamical findings, “every attractor tested has one positive exponent” is a defensible sample description, not a theorem about these models. Finite-time chaos labels and low-dimensional estimates also should not inherit the rigor of the separate interval certificate.

For framing, retain the current title, **“Bifurcations and chaos in language models fed their own output.”** A defensible opening is: “Continuous-output feedback can turn language-model inference into a low-dimensional nonlinear dynamical system; steering and temperature can move these systems between periodic and chaotic regimes.” Add reproducibility consequences only at the horizon and implementation actually demonstrated. Move extensive map atlases, additional precision-convergence plots, and negative shrimp-search detail to appendices; retain the operational definition, representative universality evidence, mechanism, and strongest consequence experiment in the main narrative.

# Ranked experimental directions

The ranking below emphasizes discriminating power and actionable knowledge. For practical consequences specifically: **upgraded REPRO > task-tested stabilization envelope > BP confirmatory**. MECH ranks highly as the explanation-building companion.

### Common experimental rules

Use independently generated **problem instances/prompts** as sampling units. Temperatures, numerical variants, offsets, time points, and repeated trajectories are paired observations within a prompt—not additional sample size. Separate generator seeds and operand/template families between development and held-out sets; freeze settings before held-out execution.

Use deterministic answer extraction and exact arithmetic scoring; no model judge is needed. Parsing failures count as incorrect answers. Record execution failures separately, retain the planned denominator, and give worst-case bounds for missing paired outcomes. More than 5% unexplained execution failures invalidates an operational comparison until repaired.

Report prompt-level paired confidence intervals. Small pilots are feasibility and effect-size screens: with zero events among 24 independent prompts, the approximate upper 95% bound is still 12%, not “negligible risk.”

## 1. Finite-budget reproducibility and answer instability

**Question and hypothesis.** Do batch-layout changes, holding model weights, dtype, masks, and decoding policy fixed, cause more answer disagreement in continuous-feedback inference than in ordinary greedy inference? The falsifiable hypothesis is excess disagreement at a prespecified short horizon, accompanied by measurable feedback amplification.

**Strongest competitor.** Answers change because a different numerical implementation changes logits, not because recurrent instability amplifies small differences. Another plausible outcome is genuine chaos that remains computationally irrelevant at useful horizons.

**Minimum experiment.** Use eight development and 24 held-out arithmetic problems on 0.5B. Compare singleton execution with execution alongside three fixed distractor sequences. Test two fixed temperatures selected from existing unsteered maps, FP64 and native BF16 execution, and the existing 16-token-window Soft Thinking implementation. Use ordinary greedy inference as the strong practical baseline. The detailed matrix appears below.

Regime assignments must be measured for the actual prompt if used in analysis; existing-map labels are only nominal. Do not infer the windowed implementation’s exponent from the one-token map.

**Controls.** Include identical-layout reruns, a one-step/no-feedback comparison, injected same-map perturbations, and known periodic/chaotic cells. Validate exact-answer scoring with deliberately correct, incorrect, and malformed outputs.

**Primary estimand.** Excess canonical-answer disagreement, in percentage points, for BF16 batch-layout changes in soft feedback versus greedy inference, averaged over the two fixed temperatures and then over prompts. Accuracy change and invalid-answer rate are separate outcomes.

**Interpretation and decisions.**

- **Positive:** operational sensitivity is established for this implementation and budget. Controlled amplification can support, but not automatically prove, the proposed mechanism.
- **Null:** useful evidence of short-horizon robustness; not a refutation of long-run chaos.
- **Mixed:** trajectory divergence without answer disagreement establishes a projection/horizon boundary.
- **Invalid exposure:** if layout changes produce no numerical difference, the execution comparison is robust, but it does not test amplification of an introduced disturbance.

Expand to 96 held-out problems if answer discrepancies occur or controlled perturbations reveal a clear horizon threshold. If only long-run divergence appears, revise the consequence claim rather than extending trajectories indefinitely.

**Resources, estimated:** 2–6 H100 GPU-hours for the initial 0.5B matrix and controls, depending on measured throughput; API cost $0. Cheapest useful pilot: four development problems, both layouts, one temperature.

## 2. Orbit-preserving readout/network crossover

**Question and hypothesis.** Does steering create local instability mainly through altered token competition, altered network sensitivity, or their interaction?

The strongest competing explanation is **state redistribution**: the gain split changes because steering visits different states, rather than because either factor independently creates expansion.

**Minimum decisive intervention.** At the same states on a recorded steered reference orbit, evaluate logits with steering off and on. Construct a four-arm, orbit-preserving replay that crosses:

| Arm | Reference readout probabilities | Input-dependent logit response |
|---|---|---|
| 00 | Unsteered | Unsteered |
| 10 | Steered | Unsteered |
| 01 | Unsteered | Steered |
| 11 | Steered | Steered |

A concrete implementation is:

\[
G_{ab,t}(x)=\bar x_{t+1}+E^\top\!\left[
\operatorname{softmax}\!\left(
\frac{z_a(\bar x_t)+z_b(x)-z_b(\bar x_t)}{T}
\right)
-\operatorname{softmax}\!\left(\frac{z_a(\bar x_t)}{T}\right)
\right].
\]

All arms preserve the reference orbit. Their derivatives cross the probability-dependent readout with the network’s logit Jacobian. Arm 11 reproduces the original steered map when the reference trajectory is generated by that map.

Use four development and 12 held-out prompts, one block and one steering pair initially, and 256-step measurement windows after the existing burn-in procedure. Repeat windows within prompts without treating them as independent.

**Baselines and controls.** Compare against the actual unsteered and steered maps, the running static logit-bias intervention, and cheap entropy/top-two-gap predictions. Arm 11 must reproduce ordinary tangent and twin growth. An attracting periodic orbit supplies the negative control. Same-arm replay is the implementation sham.

**Primary estimand.** Paired change in finite-time log growth, in nats per step, including the readout × network interaction; denominator: held-out prompts, not individual iterates.

A readout-only crossover reproducing full expansion supports readout sufficiency **along the controlled orbit**. Expansion only in arm 11 supports interaction. Null crossover effects challenge the proposed factor attribution. Failed replay/tangent agreement invalidates the intervention.

Expand to a second block and temperature only after replay agreement and a reproducible contrast. Abandon scalar entropy/rank explanations if they fail while crossed operators discriminate.

**Claim boundary:** this is a causal test of local expansion, not proof that the crossed surrogate has an autonomous chaotic attractor.

**Resources, estimated:** 2–6 H100 GPU-hours; $0 API. Cheapest pilot: one known chaotic and one periodic reference trajectory. This is the single additional MECH test I would prioritize.

## 3. A task-tested stabilization envelope

**Question and hypothesis.** Can feedback stabilization reduce answer instability without sacrificing accuracy or merely reproducing greedy decoding?

A periodicity-only envelope cannot answer this: a stable wrong answer is not a successful inference method.

**Intervention.** Test soft-to-hard feedback mixing,

\[
x'=\eta E^\top p+(1-\eta)e_{\arg\max p},
\]

with fixed \(\eta\in\{0,0.5,0.9,1\}\), plus an adaptive schedule developed separately. Within a region of constant argmax, the one-token feedback derivative scales by \(\eta\); token boundaries and retained context prevent a global contraction claim.

This is preferable to assuming ordinary damping will work. For \(F_\eta=(1-\eta)x+\eta F(x)\), a real unstable multiplier \(\mu>1\) remains above one.

**Experiment.** After Direction 1 finds a relevant sensitivity, use 16 development and 64 held-out arithmetic problems. Compare the adaptive schedule with the best development-selected fixed mixture, ordinary greedy decoding, and the original soft method at matched forward-pass budgets. A schedule permuted across prompts controls for simply inserting hard steps.

**Controls and metric.** Known unstable cells test whether hardening can alter measured amplification. A one-step constant-output feedback control tests that the pipeline detects suppression of sensitivity. The primary endpoint is exact-answer accuracy difference versus the strongest fixed baseline, in percentage points per problem; disagreement and compute cost are secondary.

Expand if the pilot shows at least a 10-point reduction in disagreement without more than a five-point observed accuracy loss; this is a screening rule, not a noninferiority conclusion. Abandon the adaptive method if a fixed mixture dominates it in both accuracy and reproducibility.

Positive results establish a useful controller, not that chaos universally harms reasoning. Reduced divergence with worse accuracy demonstrates a stability–computation tradeoff. No dynamical effect means the chosen intervention did not control the relevant state.

**Resources, estimated:** 2–6 H100 GPU-hours for a bounded pilot; $0 API. The dependency is substantive: do this only where finite-budget sensitivity exists.

## 4. Within-task predictive value of the loop regime

**Question and hypothesis.** Does a prompt’s loop regime predict ordinary-decoding errors within one tightly controlled task family, beyond cheap difficulty measures?

**Strongest competitor.** Regime membership proxies for operand length, carries, tokenization, or other task difficulty.

Use a frozen arithmetic generator, preregistered covariates, and fixed temperature. Compare a covariate-only predictor with covariates plus continuous finite-time growth and, secondarily, the categorical chaos label. Use development/held-out splits by operand ranges or expression structure. Baselines should include one-pass answer confidence/logit margin where available; length-only prediction is the cheap baseline.

A planted error process tied to known difficulty validates the analysis pipeline; within-stratum shuffled regime labels are the negative control. Neither validates a causal interpretation.

**Primary estimand.** Held-out improvement in error prediction, measured as mean log-loss reduction per problem. Also report the standardized accuracy difference in percentage points. Repeated offsets belong to the regime measurement, not the sample size; unresolved labels remain unresolved.

Start with a 32-case 0.5B throughput pilot using the full existing classifier. The proposed 2,000-case 7B study is not automatically decisive: even with balanced groups, a worst-case binary accuracy difference has an approximate 95% interval half-width of 4.4 points; imbalance worsens this.

Expand only if labeling throughput fits the remaining budget and held-out predictive value exceeds the cheap baselines. A positive result supports a diagnostic, not causal harm. A null result rejects that diagnostic at the achieved precision, not the dynamics. Excess unresolved labels require a narrower continuous-growth analysis, not forced classifications.

**Resources:** cap the feasibility pilot at two H100 GPU-hours; $0 API. Full 7B labeling may require millions of forward/tangent-equivalent steps and exceed the available allocation. Measure cost before committing.

# Recommended next experiment

## Execute Direction 1: finite-budget paired REPRO

It outranks the alternatives because it tests the missing relevance bridge directly, avoids the task-mix confound, and can produce a meaningful null. Another 70B map or more Feigenbaum digits would refine an already supported existence result; neither resolves this bridge.

### Frozen experimental matrix

Use the existing 0.5B implementation, no residual steering, and 24 new held-out problems after eight development problems.

| Component | Conditions |
|---|---|
| Primary feedback | Existing 16-token-window Soft Thinking implementation |
| Temperature | Two fixed unsteered settings from prior maps: nominally attracting and expanding |
| Arithmetic | Two-digit addition, stratified by carry count |
| Numerical format | FP64; native BF16 path with operation dtypes recorded |
| Execution | Singleton; batch of four with three fixed distractors |
| Budget | 32 soft steps, then the implementation’s fixed answer-extraction procedure |
| Strong baseline | Ordinary greedy inference under the same maximum forward-pass budget |
| Mechanism subset | Six prompts with same-map injected perturbations; longer horizons only here |

Thirty-two steps is an experimental budget, not a claim about a published default. Freeze the answer protocol before held-out execution.

**Independent units:** 24 distinct arithmetic problems. Prompt variants and temperatures stay clustered within problem. Ensure held-out operands/structures are not reused during protocol selection.

### Primary endpoint

For prompt \(i\), let \(d\) indicate that singleton and batched runs return different canonical answers. The primary contrast is

\[
\Delta=
100\,\mathbb E_i\left[
\frac{d_{i,\mathrm{soft},T_1}+d_{i,\mathrm{soft},T_2}}{2}
-d_{i,\mathrm{greedy}}
\right].
\]

This is **excess answer disagreement in percentage points**, not accuracy loss. Use `INVALID` as an explicit parsing outcome and report both-invalid pairs separately so consistently broken inference cannot masquerade as useful reproducibility. Score exact-answer accuracy alongside it.

### Implementation sequence and decisive checks

1. **Benchmark 100 steps.** Estimate matrix cost before launching. Confirm memory headroom: FP64 weights alone are approximately 4 GB at 0.5B and 56 GB at 7B.
2. **Isolate batch layout.** Keep target sequence positions, masks, padding semantics, and model state identical. Verify distractor content cannot influence the target through a batching/cache bug.
3. **Run scoring and dynamical controls.** Known correct/incorrect strings validate scoring. Existing attracting and chaotic cells, with explicit perturbations, validate the divergence measurement.
4. **Complete development runs and freeze.** Do not choose temperatures by held-out answer instability.
5. **Run the held-out matrix.** Save answer, correctness, state distance, logit distance, first divergence step, and per-step runtime.
6. **Probe attribution on six prompts.** Inject equal-norm perturbations into the same FP64 map, including amplitudes comparable to measured execution differences. Track the entire windowed computational state, not just the latest embedding.

FP64 versus BF16 changes the implemented map; it is a robustness comparison, not a clean initial-condition intervention. Same-map perturbations provide the cleaner sensitivity evidence.

### Resource envelope and branches

**Estimate:** approximately 40,000–100,000 forward-equivalent steps including controls, roughly 2–6 H100 GPU-hours after overhead; benchmark-dependent, not guaranteed. Store compact trajectories and summaries rather than all activations, targeting under 5 GB. API cost: $0.

Use 7B only as a bounded replication: eight fresh problems, requiring at most one to two wall-hours on two H200s, and only if measured throughput permits. Do not divert scarce time from completing the 0.5B comparison.

- **Answer discrepancies plus controlled amplification:** expand the task sample and replicate at 7B.
- **Discrepancies without amplification:** investigate numerical-map changes or discrete decision boundaries; retain the reproducibility result but reject the chaos attribution.
- **Amplification without answer discrepancies:** report finite-budget projection robustness; test horizon dependence, not accuracy harm.
- **No measurable batch disturbance:** retain the operational null and use injected perturbations to estimate a sensitivity threshold.
- **Soft inference performs poorly even without perturbation:** do not market this as damage to a useful reasoning method. Narrow the finding to computational reproducibility.

# First 48 hours

**Hours 0–2:** reserve a bounded compute allocation, benchmark throughput, implement paired layouts, validate exact scoring, and run attracting/chaotic controls. Deliverable: executable frozen matrix and verified perturbation exposure.

**Hours 2–7:** complete development and launch the 24-problem held-out matrix. Analyze paired effects as jobs finish without changing the remaining design. Deliverable by morning: answer-disagreement table, accuracy table, divergence-versus-horizon plot, and explicit unresolved cases.

**Hours 7–24:** run the six-prompt mechanism subset. If discrepancies occur, distinguish accumulated perturbation amplification from a single late answer-boundary crossing. If only long-run effects appear, estimate the horizon at which they emerge.

**Hours 24–48:** execute one branch:

- operational effect: expand toward 96 problems within budget;
- dynamics-only effect: run the orbit-preserving crossover pilot;
- implementation failure: repair and rerun the affected frozen cells;
- no sensitivity at tested horizons: retain that null and test only a small, prespecified longer-horizon subset.

# Two-week research roadmap

**Days 3–5 — Establish the boundary.** Finish the chosen replication or horizon study. Deliver a response surface over horizon and disturbance magnitude, with answer disagreement distinguished from internal distance and accuracy.

**Days 6–9 — Identify the mechanism.** Run the four-arm readout/network crossover on held-out prompts. Deliver finite-time growth contrasts and interaction estimates. If replay agreement fails, fix the intervention; do not interpret the failed implementation as mechanistic evidence.

**Days 10–12 — Test an actionable intervention.** If finite-budget answer sensitivity exists, compare fixed and adaptive soft-to-hard feedback. If only long-run sensitivity exists, use the intervention to test local controllability, explicitly withholding task-benefit claims.

**Days 13–14 — Resolve the strongest remaining alternative.** Spend the remaining allocation on one discriminator: a small 7B replication, a growing-context replication, or a second task family. Choose according to the preceding result, not model size alone.

The program’s valuable endpoint is not necessarily “chaos harms inference.” It is a measured boundary identifying **when continuous-feedback instability survives the inference horizon and output projection, which mechanism creates it, and whether a useful intervention changes the computation rather than merely making trajectories look stable.**
