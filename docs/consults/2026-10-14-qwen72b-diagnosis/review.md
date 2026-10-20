# Research opportunity and what the evidence leaves open

**The opportunity is to distinguish genuine contraction from unresolved infinitesimal instability—not simply to search harder for a positive exponent. I recommend one paired implementation-and-tangent experiment on a small, frozen set of 72B states.** It combines a targeted version of D1 with conditional D2, so that the paid run produces a dynamical result rather than only a software audit.

“No detected chaos” is plausible for this loop. The disclosed strongly attracting echo and high-temperature fixed point, together with lower orbit-conditioned gains, support it. But **“Qwen2.5-72B has no chaos” is not established**, and model size provides no persuasive prior: your prompt-blocked results already contradict monotonic size suppression.

Three distinctions matter:

- **Fixed-point wander is not a direct measurement of stochastic computation noise.** It could reflect nondeterministic evaluation, a deterministic numerical microcycle, amplified rounding, cache-state changes, or a genuine small attractor. Compare identical-input evaluations before calling it noise.
- **A return within \(10^{-8}\) is weak evidence of periodicity when computational discrepancies are comparable.** Those returns challenge S, but do not decisively refute it.
- **Large offsets can produce false negatives.** They can cross thin expanding regions, saturate across a switch, or merge nearby branches. Numerical rounding can also create artificial stable cycles or change basin selection. Coarse temperature grids, short horizons and a single initialization introduce separate false-negative mechanisms.

The offset trend strongly suggests numerical contamination, but does not exclude smooth, very narrow expansion.

There is a useful theoretical framing. For your map,
\[
DF_T(x)=\frac1T E^\top[\operatorname{diag}(p)-pp^\top]D z(x).
\]
A concentrated distribution suppresses the softmax covariance **provided network derivatives do not compensate**. At sufficiently high temperature, the explicit \(1/T\) factor favors contraction when derivatives remain bounded on the embedding convex hull. Neither fact implies a monotonic relationship between model size and chaotic breadth. Sharpness can move the switching region, narrow it, or strengthen it locally; entropy alone does not determine the relevant matrix product.

The standard tangent-renormalization foundation is Benettin et al., *Meccanica* (1980). The distinction between transient amplification and asymptotic stability is illustrated by Trefethen et al., “Hydrodynamic Stability without Eigenvalues,” *Science* (1993). Neither supplies a model-size law.

# Ranked experimental directions

The ranking balances discrimination, scientific value, feasibility and cost. Paid estimates below are **alternatives drawing on one shared budget**, not an additive shopping list.

**Common measurement rules:** report exponents in natural-log units per loop step. Temperature cells and tangent restarts are not population replicates; trajectory steps are repeated observations. For these deterministic case studies, report numerical/horizon sensitivity envelopes, not naïve stepwise confidence intervals. Invalid JVPs, OOMs and nonconverged trajectories remain invalid—not nonchaotic. Use no model judges: all endpoints are numerical.

## 1. Separate implementation-dependent instability from resolved tangent dynamics

1. **Question and hypothesis.** Does the jump retain positive tangent growth under an independently controlled execution path? Hypothesis N predicts implementation-sensitive small-offset growth but nonpositive validated tangent growth. S predicts convergent positive tangent growth despite negative large-offset estimates. This adds evidence unavailable from another offset sweep.

2. **Minimum experiment.** Freeze six cases: two jump temperatures showing strong offset disagreement, two steered candidates, one high-temperature fixed point and one low-temperature echo. Compare production execution with an explicit, synchronized, fixed-cache implementation. Full matrix below. These are six selected dynamical cases, not six independent samples of model behavior. Develop on local Qwen2.5-0.5B; freeze tolerances before 72B evaluation.

3. **Baselines and controls.** Production twin trajectories are the strongest existing comparator; larger-offset differences are the cheap baseline. Include a small model run on one versus two GPUs, deliberate float32 rounding as a precision-loss control, and manufactured expanding/contracting maps with known exponents.

4. **Endpoint and decision rule.** Primary: tangent finite-time exponent at each frozen jump case, with paired implementation difference. Expand horizons only if derivative checks pass. Revise execution when discrepancies localize to a correctable component; abandon this JVP implementation if it fails derivative agreement on resolved scales.

5. **Interpretation.** Positive, implementation-invariant results support infinitesimal instability; negative results support contraction in these cases. Mixed signs imply numerical or dynamical sensitivity requiring localization. Invalid derivatives cannot adjudicate N versus S. Positive growth alone does not establish a chaotic attractor.

6. **Resources.** Cheapest pilot: 20–50 local forward/JVP comparisons. Estimated paid envelope: 45–120 minutes after batching and timing, with a hard cap. No API expense. Actual JVP throughput and memory feasibility determine the next decision.

## 2. Resolve the narrow unsteered transition and validate the steered boundary

1. **Question and hypothesis.** Is chaos confined to a narrow, previously undersampled region, or exposed by steering? R predicts reproducible positive cells between existing samples or beyond the steered map boundary. The strongest competitor is a switching periodic attractor contaminated by numerical error.

2. **Minimum experiment.** After Direction 1, test all eight existing steering candidates at \(10^{-4},3\times10^{-5},10^{-5}\), plus validated JVPs where affordable. Add four unsteered transition midpoints and four temperatures extending the candidate steering slices toward \(\log_{10}T=0.4\). Use two independently drawn initial states per case from a preregistered distribution in the embedding convex hull. They sample basin dependence, not new parameter cells. Freeze additional validation temperatures before examining their outcomes.

3. **Controls.** Baselines: \(\alpha=0\) at matching temperatures and nearest previously stable cells. Cheap baseline: a larger-offset run. Positive controls: the numerical expanding-map control and a previously positive small-model cell, verified anew. If feasible, one norm-matched unrelated steering direction tests specificity—not the semantic meaning of “hostility.”

4. **Endpoint and decisions.** Report **confirmed candidates/8**, and confirmed newly tested cells/8 separately—not a new chaotic-area percentage. Expand a slice only when positivity survives implementation, horizon and initialization checks. Abandon the original eight labels if robust larger-offset and validated tangent estimates remain nonpositive.

5. **Interpretation.** Positive unsteered midpoints support missed-band R. Steering-only positives support intervention-accessible instability. Nulls constrain these slices, not all steering vectors. Offset-only positives remain unresolved; invalid runs do not enter the negative numerator.

6. **Resources.** Local harness first; estimated 30–90 paid minutes with candidate batching. Cheapest pilot: strongest candidate, its \(\alpha=0\) match, and one extended-temperature neighbor. Dependency: a defensible numerical resolution from Direction 1.

## 3. Test prompt specificity without repeating a uniform temperature sweep

1. **Question and hypothesis.** Does the original prompt uniquely funnel 72B into the confident echo? P predicts prompt-dependent positive regions or substantially different transition geometry. Competing explanation: prompts merely move the same stable transition.

2. **Minimum experiment.** Use today’s frozen four-prompt result as disclosed evidence, not as proof that its \(10^{-6}\) measurements are resolved. For each prompt, locate the entropy transition using its existing measurements, then freeze eight transition-relative temperatures and two initialization draws. Validate any claimed positive with the calibrated method. These are four fixed prompt blocks; broad prompt-population inference is unwarranted.

3. **Controls.** Strong baseline: the existing 48-temperature test. Cheap baseline: entropy and return structure. Negative controls: clearly stable temperatures on either side of each transition. Positive control: a validated chaotic cell processed through the same batching and exponent code.

4. **Endpoint and decisions.** Primary: positive-exponent cells/8 within each prompt, with initialization agreement reported separately. A secondary endpoint is transition width in log-temperature decades. Expand to new frozen prompts only if prompt variation exceeds numerical uncertainty; revise temperature placement if the transition lies outside the selected interval.

5. **Interpretation.** Positives on alternative prompts refute a prompt-general absence claim. Stable but shifted transitions favor prompt-conditioned contraction over P’s chaos explanation. Mixed basin outcomes reveal initialization dependence. Four null prompt blocks remain four null blocks, not a universal model property.

6. **Resources.** Reuse the running experiment at no additional charge. Cheapest new pilot: two contrasting prompts, four transition-relative temperatures each. Estimated 30–90 paid minutes, conditional on batching. No additional API use.

## 4. Quantify the contraction margin independently of output entropy

1. **Question and hypothesis.** How much stronger must network feedback become before a stable state loses local stability while its token distribution remains unchanged? The hypothesis is that some “sharpness-associated stability” actually reflects weak network feedback. The competitor is an extremely small readout covariance requiring a large feedback increase.

2. **Minimum experiment.** On six local-model fixed points spanning sharp and diffuse distributions, define
   \[
   z_\beta(x)=z(x_*)+\beta[z(x)-z(x_*)].
   \]
   At a genuine fixed point, its distribution is unchanged and \(DF_\beta(x_*)=\beta DF_1(x_*)\). Test \(\beta=0.5,1,2,4\), then launch small perturbed trajectories. Develop on three states; freeze the other three. States are selected cases, not population replicates.

3. **Controls.** Strong baseline: ordinary temperature variation; cheap baseline: entropy versus tangent gain. \(\beta=1\) is a sham intervention, \(\beta=0\) a constant-map control. The predicted local shift \(\lambda_\beta=\lambda_1+\log|\beta|\) is a quantitative positive control.

4. **Endpoint and decisions.** Primary: critical feedback multiplier for loss of fixed-point stability, unitless. Report departures from the predicted shift and finite-horizon sensitivity. Expand to a few stored 72B states only if the local intervention works and paid resources remain; reject the implementation if the identity fails beyond numerical tolerance.

5. **Interpretation.** A small critical multiplier shows a weak stability margin despite fixed entropy. A large one shows robust local contraction. Neither establishes the cause of natural size trends or guarantees chaos after destabilization. Failure to obtain a sufficiently accurate fixed point invalidates the local inference.

6. **Resources.** Estimated 4–12 local GPU-hours, depending on streaming and checkpoint size; no paid compute required for the initial result. Cheapest pilot: one stable Qwen2.5-0.5B state.

# Recommended next experiment

## One experiment: paired execution challenge with tangent follow-through

This outranks immediate D3 or another temperature grid because it can resolve **both false positives and false negatives**, calibrates all subsequent 70B-class work, and still yields substantive dynamical measurements if no software defect is found.

### Frozen matrix

| Component | Cases | Manipulation |
|---|---|---|
| State-level precision challenge | Six cases specified above; four saved phases per orbit | Production versus explicit fixed-cache path |
| Placement challenge | One jump state and one fixed-point state | A second feasible layer split |
| Small-model isolation | Local-tested checkpoint | One GPU versus two GPUs, same backend |
| Dynamical follow-through | Two jump cases, strongest steered candidate, stable fixed point | Validated tangent propagation; production comparison where feasible |

The explicit path should preserve the mathematical map: evaluation mode, fixed prefix K/V and position, no growing cache, autocast disabled, float64 math attention, explicit device transfers, and synchronization at the split. Because this initially bundles changes, **a successful repair identifies implementation dependence, not the culprit**. Isolate the culprit afterward with one-at-a-time toggles.

### Precision and positive-control checks

At each saved state:

1. Evaluate exactly identical inputs repeatedly, both consecutively and interleaved with another input. Check fresh versus reused cache objects.
2. Compare singleton execution with duplicated and permuted batch positions. Batch dependence is diagnostic; a deterministic batch-specific map is not automatically stochastic.
3. Record discrepancies in **both \(L_2\) and \(L_\infty\)**. Do not compare an \(L_\infty\) wander estimate directly with an \(L_2\) perturbation.
4. Compare JVPs with central directional differences across the supplied \(h\) range. Require a resolved region where neighboring scales and JVP agree; a provisional development target is relative error below \(10^{-3}\), with an absolute criterion when the derivative is tiny.
5. Test float64 transfer round-trips bitwise, including low mantissa bits. Ordinary same-dtype copies should not intrinsically lose precision.
6. Capture small per-layer residual snapshots to locate the first divergence. A float64 output dtype does not prove every internal operation was float64.

Use affine controls \(G_a(x)=x_*+a(x-x_*)\), with \(a=1/2,2\), routed through the same transfer and renormalization machinery. Their exponents are \(-\log2,+\log2\). They validate expansion measurement, **not chaos detection**. A deliberate float32 cast tests whether the precision diagnostic notices known degradation.

My leading working suspicion is an actual-path difference involving caches, hooks, casting or backend behavior—not precision loss inherent to GPU-to-GPU copying. The evidence cannot rank those causes reliably. Identical-input repeatability and first-divergence localization are the cheapest discriminators. If everything is repeatable, deterministic rounding amplification remains possible.

### Tangent follow-through

Treat JVP as an **offset-free numerical derivative of the intended smooth map**, not an exact derivative of the literal rounded bit-level computation.

Verify that forward AD crosses device hooks without detachment, that prefix tangents are zero, and that cache mutation is absent. Avoid `inference_mode` during JVP evaluation; test the actual attention implementation locally.

Start with 256 settling/alignment steps and 512 measured steps, using two independently drawn tangent directions. Double the measurement horizon only for unresolved or positive cases. Compare successive halves and restart agreement; require numerical and horizon sensitivity to be comfortably smaller than the distance from the \(10^{-3}\) threshold. These are convergence diagnostics, not confidence intervals.

For a convincingly repeating orbit, the best additional mechanism probe is its **period Jacobian product**. Estimate the dominant multiplier over repeated periods; do not substitute the largest one-period singular value, which can reflect non-normal transient amplification.

### Resources and branches

Benchmark 100 batched steps before committing. Assuming roughly 0.5–2 seconds per batched forward step, JVP overhead and two execution paths suggest **1.5–3 two-GPU pod-hours**, including diagnostics; this is an estimate, not a guarantee. At an illustrative pod rate of [amount redacted]/hour, that is [amount redacted]. If throughput is worse, retain the two jump cases and stable control rather than shortening everything below interpretability.

- **Execution defect found:** repair it and rerun the frozen cases first. Then spend remaining time on Direction 2—not a full map.
- **Validated positive tangent exponent, negative large-offset exponent:** report resolved infinitesimal instability with finite-size contraction. Require longer recurrent, nonperiodic trajectories and basin checks before calling it attracting chaos.
- **Validated nonpositive tangent estimates:** support contraction at the tested jump states; proceed to the narrow-band or prompt alternative.
- **No trustworthy derivative:** report the jump as numerically unresolved. Do not treat the large-offset negative result as decisive against S.

For the paper, keep the definition centered on infinitesimal asymptotic growth, operationalized by validated finite-time estimates; present finite-size exponents separately. “No period up to 64” is a diagnostic, not a definition of aperiodicity.

Until validated, the steering result is **“eight single-offset candidate cells,” not “0.8% chaotic.”** A corrected null supports “no numerically resolved chaos over the specified prompts, temperatures, initializations and horizons.” A corrected positive revises that restricted result. Recheck representative Llama positives at resolved offsets or by validated JVP—not automatically at \(10^{-7}\), which may be below useful resolution.

# First 48 hours

**Hours 0–8, local GPU**

- Build a pure one-step function with immutable prefix state.
- Pass affine controls, deliberate-rounding detection, and small-model JVP/finite-difference agreement.
- Freeze the six cases, direction seeds, convergence tolerances and paid-runtime cap.

**Hours 8–16, one pod session**

- Time the workload; run state-level comparisons first.
- Localize any discrepancy while the checkpoint is loaded.
- Run tangent follow-through on the validated path; save compact state, derivative and timing summaries.

**Hours 16–32**

- Classify outcomes as resolved positive, resolved nonpositive, mixed or invalid.
- Incorporate the completed prompt test with numerical qualification.
- If a repair is established, rerun the frozen cases before exploring new cells.

**Hours 32–48**

- Use remaining paid time for the three-case Direction 2 pilot.
- Start the local feedback-margin experiment.
- Deliver a case-by-case dynamical result table, not merely a dtype audit.

# Two-week research roadmap

**Days 3–5:** Complete the appropriate narrow transition/steering slice. For any positive, extend horizon and initialization checks; for nulls, report the tested resolution and remaining gaps.

**Days 6–9:** Complete the local feedback-margin intervention. Deliver measured stability margins at unchanged anchor distributions, with the predicted tangent-shift control. If paid compute is exhausted, this remains substantive experimental progress.

**Days 10–12:** Follow the discriminating branch:

- **Numerical defect:** test whether its correction also changes one Llama positive and one stable control.
- **Narrow positive band:** test basin dependence and period-product stability near its boundaries.
- **Prompt dependence:** validate transition-relative cells on the contrasting prompts.
- **Resolved null throughout:** quantify contraction margins and transition widths rather than manufacturing a positive.

**Days 13–14:** Produce the scientific synthesis: which implementations agree, where tangent and finite-size behavior diverge, which candidate cells survive, and how prompt, temperature and feedback interventions change stability. The meaningful endpoint is a better-characterized dynamical mechanism—even if 72B remains nonchaotic in every tested case.
