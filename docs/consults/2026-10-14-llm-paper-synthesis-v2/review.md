# Research opportunity and what the evidence leaves open

**The strongest opportunity is to distinguish genuine size-associated contraction from a shrinking or displaced instability window. I recommend a prompt-blocked, entropy-targeted comparison of Qwen2.5-7B and 14B—not another uniform temperature sweep.** This can produce meaningful new knowledge within roughly [amount redacted] conditional on a short throughput benchmark. It cannot establish a law for trillion-parameter models.

The disclosed results already rule out two simple stories: chaos is neither specific to concept steering nor confined to small models. They also suggest three distinguishable explanations for the apparent size trend:

1. **Readout gating:** concentrated probabilities suppress feedback sensitivity.
2. **Network contraction:** larger checkpoints attenuate perturbations before readout.
3. **Search geometry:** instability survives on narrow, shifted ridges that a fixed temperature grid misses.

The gain split favors different explanations in different entropy regimes, but it is a decomposition along an evolving tangent—not a causal attribution. Equal entropy does not imply equal embedding covariance, tangent alignment, or readout gain.

### Concise answers to the brief’s claim questions

- **Theory.** For tied embeddings, the stated derivative has the right structure. With \(h(x)=x+c\), write \(F=\nabla\Phi\), where \(\Phi\) is the appropriate log-sum-exp potential. Then
  \[
  \mathcal E(x)=\tfrac12\|x\|^2-\Phi(x),\qquad
  \mathcal E(F(x))-\mathcal E(x)\leq-\tfrac12\|F(x)-x\|^2.
  \]
  This excludes nontrivial cycles. The broader symmetric-positive-semidefinite argument requires that property throughout a suitable convex domain, not merely at sampled orbit points. Its failure is **necessary under those assumptions, not sufficient for chaos**. For untied checkpoints, use the actual input/output embedding matrices rather than the covariance formula.

- **Temperature.** Sufficiently high temperature gives contraction when the relevant derivatives are uniformly bounded on the feedback state domain. Greedy feedback has a finite image and hence eventually periodic orbits. But neither fact establishes a universal intermediate-temperature band: without a uniform logit gap, small-positive-temperature chaos is not excluded.

- **Repeating text.** The evidence supports a useful low-dimensional return-map description at selected cells. Exponent agreement does not establish conjugacy, scalar sufficiency, or an exact logistic mechanism. Keep return-map exponents and full-map exponents in the same units: divide a \(p\)-step return exponent by \(p\).

- **Interpretability/alignment.** “Local explanations need not compose into long-run predictions” is supported. “Feature dictionaries cannot characterize attractor boundaries” and “loop interventions generally beat behavioral steering” remain hypotheses. The disclosed intervention result supports the particular sampled self-dialogue setting, not general alignment efficacy.

- **Scale.** The defensible present statement is: *size-associated suppression occurs in the sampled operating regimes, while steering can expose instability at 70B*. The frozen predictions are legitimate tests of their specified protocol, but monotonic ordering is too strong to serve as the only test of a scale mechanism. Keep that test intact and add a separately declared search-geometry experiment.

- **Title.** I would use **“Soft-token feedback in language models: bifurcations, hidden chaos, and stability.”** The proposed broader titles obscure the unusual feedback system.

For literature, two exact starting references are Ramsauer et al., **“Hopfield Networks is All You Need,” ICLR 2021**, and Sompolinsky, Crisanti, and Sommers, **“Chaos in Random Neural Networks,” Physical Review Letters 61, 259–262 (1988), doi:10.1103/PhysRevLett.61.259**. These support the energy connection and generic neural-network chaos, respectively—not transformer scale trends or the particular periodic-symbolic reduction. I cannot verify comprehensive primary-record coverage through October 2026 here; novelty in those latter areas remains unresolved.

# Ranked experimental directions

## 1. Is the apparent scale effect contraction, or displacement of instability?

**Highest priority:** directly addresses the central uncertainty, uses existing machinery, and avoids expensive 70B-wide searches.

**Question and prediction.** Within one checkpoint family, does the larger model retain lower feedback growth after a fixed-budget search spanning comparable output-entropy regimes? The contraction hypothesis predicts a persistently negative paired size contrast. Its strongest competitor is a narrow support-transition ridge missed by the original grid. This adds a controlled operating-regime comparison, not another collection of chaotic-cell percentages.

**Minimum experiment.** Use the same checkpoint class and steering setting for Qwen2.5-7B/14B, initially unsteered. Generate eight held-out prefix blocks from a frozen prompt generator; use two separate development prefixes. Within each model/prefix, select three temperatures by a deterministic entropy-targeting procedure, then confirm the strongest candidate. Prefixes—not temperatures, tangent vectors, or time windows—are the independent sampling units. Both models are fixed objects, not independent draws from a population of model sizes.

**Comparisons and controls.** Compare against the original fixed-temperature protocol and a cheap three-temperature fixed grid. Positive controls are a previously reproduced small-model chaotic cell and a synthetic chaotic map. Negative controls are a contracting affine map and the \(J_h=I\) energy-descending map. At a few states, compare checkpoint-storage paths and JVPs before interpreting tiny exponents.

**Endpoint.** Primary: the mean paired difference in **confirmed finite-time Lyapunov exponent, nats per feedback iterate**, larger minus smaller model, across eight prefixes. Confirmation and selection are specified below. Report all prefix-level differences and a prefix-level uncertainty interval; temporal blocks diagnose stationarity but do not increase \(n\).

**Interpretation and decisions.**
- Persistent lower growth supports contraction for this pair and prefix distribution.
- Recovered instability near entropy transitions supports displacement or narrowing.
- Mixed signs imply prefix–scale interaction, not a universal ordering.
- Numerical failures or strongly horizon-dependent estimates leave the comparison unresolved.

Expand if the contrast is reproducible or interactions are structured. Abandon a monotonic law if its frozen test fails; do not abandon the broader scale question. If entropy targets are frequently unattainable, revise “entropy matching” to a support-transition analysis rather than silently dropping those cases.

**Resources.** Estimated 0.1–0.2 million trajectory/JVP updates for the complete pilot and confirmations. Rental feasibility depends on measured throughput; the implementation and budget gate are below. No API or model judge is needed.

## 2. Is the scalar return map dynamically sufficient?

**High scientific value, modest cost:** this converts an attractive descriptive reduction into a testable mechanism.

**Question and prediction.** Does the phase-conditioned leading-token probability predict future returns once its recent history is known? A genuinely useful scalar reduction predicts that additional state coordinates provide little held-out improvement and that small transverse perturbations rapidly lose influence. The competitor is a multidimensional attractor whose projection happens to look unimodal and share its largest exponent.

**Minimum experiment.** Develop on two cells. Evaluate on eight independently chosen parameter neighborhoods, selecting one eligible cell per neighborhood by a frozen rule. Nearby cells within a neighborhood are not independent replicates. Split trajectories into separated training and evaluation blocks, with gaps longer than the measured correlation time.

Fit:
1. cross-validated scalar polynomial and spline maps;
2. two- and three-delay predictors;
3. a low-dimensional state-space predictor using training-only coordinates.

Then perturb states in directions orthogonal to the gradient of the scalar observable, keeping perturbations within the same symbolic band. Because these states can leave the attractor, test whether predictive differences persist **after transverse relaxation**; immediate off-attractor disagreement alone does not refute an on-attractor reduction.

**Controls.** Cheap baseline: phase-conditioned mean or persistence. Positive control: a known scalar chaotic map coupled to contracting hidden coordinates and periodic symbolic output. Negative control: a genuinely two-state synthetic system with the same observable ambiguity.

**Endpoint.** Held-out squared error per return, normalized by held-out observable variance, averaged first within each neighborhood. Secondary: extra predictive value from delays and post-relaxation perturbation effects. Report neighborhood-level uncertainty; numerical escapes count as intervention failures, not evidence of dimensionality.

**Decisions and meanings.** Expand if scalar prediction survives longer horizons and perturbations. If delays consistently improve error by a preregistered practical margin—say 20%—abandon scalar sufficiency and pursue a delay-coordinate model. A null perturbation effect supports attraction toward low-dimensional dynamics, not smooth conjugacy. Exponent agreement alone remains insufficient.

**Resources.** Start with the smallest reproducing model: roughly 50,000–200,000 local trajectory steps, plus inexpensive regressions. A two-cell feasibility test should precede any rental; estimated rental contingency [amount redacted].

## 3. Does directional alignment cause the period-8 loop’s stability?

**Best causal mechanism probe:** stronger than interpreting the enormous gap between a product norm and the product of norms.

**Question and prediction.** Are successive expanding directions misaligned in a way essential to orbital stability? The competitor is that most apparent expansion is confined to irrelevant directions and the norm bound is simply very loose.

**Intervention.** Along the known period-8 orbit, let \(J_j\) be each step’s Jacobian. Apply orthogonal rotations to output deviations:
\[
\widetilde F_j(x)=x^*_{j+1}+R_j\bigl(F(x)-x^*_{j+1}\bigr).
\]
The orbit remains fixed and every step’s singular values are unchanged, while the product becomes \(R_{p-1}J_{p-1}\cdots R_0J_0\). This is explicitly a **phase-indexed artificial intervention**, not a newly discovered autonomous LM attractor.

Begin with one orbit and 16 independently sampled rotation interventions. Those draws estimate intervention effects on that orbit; they do not establish generality across orbits. Extend to separately located cycles only if informative.

**Controls.** Cheap baseline: the unmodified product. Negative control: coordinate changes \(J'_j=Q_{j+1}J_jQ_j^\top\), which preserve cycle multipliers. Positive control: two-dimensional matrices whose expanding axes alternate, giving a stable product, versus aligned axes giving instability.

**Endpoint.** Change in \(\log\rho(M)/p\), **nats per iterate**, using matrix-free Arnoldi with residual checks. Report the intervention distribution within each orbit. Failed eigenvalue convergence is invalid, never “stable.”

**Decisions and meanings.** Frequent destabilization while preserving step singular values causally implicates alignment. A null result rejects the chosen rotation ensemble as an effective probe, not all alignment mechanisms; next use targeted rotations aligning identified expanding directions. Mixed outcomes motivate a geometry-dependent account. Neither result establishes that nonnormality is necessary for chaos.

**Resources.** Existing small-model implementation first; estimated thousands to tens of thousands of JVPs for the pilot, approximately [amount redacted] if rental is needed. Full dense Jacobians are unnecessary.

## 4. Does added feedback memory improve behavior, rather than merely suppress repetition?

**Most useful bridge to ordinary generation:** objective tasks avoid unsupported semantic claims.

**Question and prediction.** Does increasing feedback context reduce *incorrect recurrence* while preserving task accuracy? The competitor is that repetition disappears because the model loses the original attractor or simply becomes less task-consistent.

**Minimum experiment.** Generate 32 development and 200 held-out deterministic tasks from disjoint rule families: one-state transition tasks and two-state recurrences with machine-checkable next outputs. Compare \(K=1,2,8,32\), full context, and a cheap repeat-triggered refresh. Keep instructions fixed and explicitly distinguish memory length from position-index changes.

Task instances are independent units. Use paired sampling seeds if decoding is stochastic; repeated seeds remain nested within task.

**Controls.** Strong baseline: full-context decoding. Cheap baseline: repeat detection plus refresh. Positive control: an oracle whose required memory is known and whose missing state is restored by the intervention. Negative control: valid periodic target sequences, which must not be scored as pathological repetition.

**Endpoint.** Incorrect recurrent output positions per 1,000 generated positions, alongside exact next-output accuracy; average within task before uncertainty estimation. Malformed outputs count as incorrect, while infrastructure failures are reported separately.

**Decisions and meanings.** Expand only if recurrence falls without exceeding a preregistered accuracy-loss margin, such as two percentage points. Lower repetition with lower accuracy rejects “stabilization improves behavior.” No benefit at task-capable baseline accuracy supports a boundary between the soft-loop phenomenon and this task setting. If baseline accuracy is near chance, revise task difficulty on development data and restart the held-out test.

**Resources.** About 100,000–300,000 generated tokens, locally on a small model; estimated rental [amount redacted]. No human or model judge is required. These outcomes measure rule compliance, not hostility or semantic alignment.

# Recommended next experiment

## Run the entropy-targeted 7B/14B comparison

It outranks the alternatives because the present scale interpretation is vulnerable to an identifiable measurement choice: the operating window. It also gives both positive and null outcomes scientific value without requiring a frontier-model extrapolation.

### Experimental matrix

| Component | Frozen specification |
|---|---|
| Models | Qwen2.5-7B and 14B, same checkpoint class |
| Steering | None |
| Development | Two prefixes, excluded from inference |
| Held-out units | Eight independently generated prefix blocks |
| Search | Three candidates targeting entropy 2, 5, and 8 nats |
| Temperature bounds | \(T=0.1\) to \(16\) |
| Initial evaluation | 256 burn-in + 1,024 measured steps per candidate |
| Confirmation | Selected candidate: extend by 2,048 steps; run a second initial state for 256 + 1,024 steps |

Use ten coarse temperature probes followed by bounded interpolation or bisection. Freeze the search rule on development prefixes. Entropy can jump or be nonmonotonic: if a target is unreachable, select the closest observed candidate and record the mismatch. Do not manufacture an entropy match.

Select the candidate with the highest screening exponent. Its confirmation endpoint is the equal-weight mean of:
- the exponent over the final 1,024 steps of the extended trajectory;
- the exponent over the second initial state’s 1,024 measured steps.

The primary estimand is the average paired 14B-minus-7B confirmation difference across prefixes. This is **growth after a specified search procedure**, not the global maximum exponent or prevalence of chaos.

Eight units only resolve large, consistent effects: a conventional paired interval has an approximate half-width of \(0.84s\), where \(s\) is the across-prefix standard deviation. Report every paired result, not just the pooled estimate.

### Implementation sequence and checks

1. **Verify the measurement path locally.** Run affine contraction, synthetic chaos, and the known small-model periodic/chaotic controls. Check exponent units, tangent normalization, and derivative agreement.
2. **Benchmark 100 complete updates.** Include forward propagation, JVP, state transfer, and layerwise casting—not forward-only throughput.
3. **Freeze the search and selection algorithm.** Freeze prefix generation before examining held-out dynamics. Adaptive entropy calibration is allowed; adaptive selection by interesting-looking held-out trajectories is not.
4. **Run paired models in interleaved prefix blocks.** Save state, exponent accumulators, entropy, support statistics, and gain factors. No bulk activation archive is needed.
5. **Confirm all selected candidates, including negative ones.** Otherwise confirmation itself becomes outcome-dependent.
6. **Use twin trajectories diagnostically.** Perturb at relative scales \(10^{-6}\) and \(10^{-9}\); check agreement before saturation. Do not treat their observations as independent samples.

A numerically invalid candidate makes that prefix’s complete search comparison unavailable until rerun. Report missing prefixes explicitly; never substitute zero growth. Different basins under the two initial states are a result, not invalidity.

### Resource envelope

No local hardware specification was supplied. Use the already working 0.5B path locally for controls, and the existing layerwise float64 implementation for larger checkpoints only after memory and throughput checks.

The full design is approximately **125,000 JVP-bearing updates plus 20,000 ordinary forwards**. Estimated rental cost is **[amount redacted]–100** if effective throughput is roughly 1.5–3 complete state updates/second and accelerator pricing is [amount redacted]–3.50/hour. These are assumptions, not guarantees. Batch independent trajectories where it improves throughput.

The cheapest useful two-prefix development pilot is approximately 8,000 JVP-bearing updates, with a **[amount redacted]–15 cap**. If measured costs exceed the remaining budget, do not shorten horizons silently: retain the pilot as feasibility evidence and narrow the planned comparison before opening held-out prefixes.

### Decision branches

- **Larger model remains substantially more contractive:** expand prefix coverage, then test another within-family size pair. Keep network/readout attribution descriptive until intervention.
- **Instability reappears at support transitions:** abandon “larger means nonchaotic”; measure ridge location and width prospectively.
- **Strong prefix interaction:** prioritize context dependence over a universal scale curve.
- **More than 25% of labels change on extension:** prioritize persistence and transient lifetimes, not another prevalence sweep.
- **Entropy matching fails frequently:** analyze the support transition itself; entropy is not an adequate common operating coordinate.

# First 48 hours

**Hours 0–8**
- Implement the frozen entropy-search procedure and prefix generator.
- Run synthetic controls and the known small-model controls.
- Deliverable: passing measurement tests with explicit tolerances and exponent units.

**Hours 8–20**
- Benchmark both target checkpoints.
- Run the two-prefix development pilot.
- Deliverable: measured update throughput, memory use, entropy attainability, and a costed full matrix.

**Hours 20–32**
- Freeze search rules, initializations, invalidity handling, and confirmation criteria.
- If affordable, launch the eight held-out paired blocks.
- If not, revise scope before examining held-out outcomes.

**Hours 32–48**
- Begin confirmations, including negative candidates.
- Deliverable: first paired exponent trajectories and entropy-transition profiles—not a premature size conclusion.

# Two-week research roadmap

**Days 3–5:** Complete the matrix and confirmations. Produce paired size effects, horizon sensitivity, basin differences, and the fraction of entropy targets actually attained.

**Days 6–8, contraction branch:** Apply the alignment intervention to the established period-8 orbit. This tests a concrete loop-stability mechanism without interpreting gain-factor correlations as causes.

**Days 6–8, ridge-recovery branch:** Perform bounded continuation around recovered transitions. Freeze a ridge-width definition and test it on new prefixes; do not retrospectively redefine chaos thresholds.

**Days 9–11:** Start the scalar-sufficiency experiment on two development cells, then evaluate the frozen predictors on independent parameter neighborhoods. A successful delay model is a useful alternative if the scalar mechanism fails.

**Days 12–14:** Expand whichever experiment separated explanations most sharply. If scale contrasts remain inconclusive, add independent prefixes rather than denser cells. If long transients dominate, estimate persistence under the declared horizon ladder. If soft-loop mechanisms are robust, begin the objective finite-memory bridge.

The substantive target is a discriminating result: **size-associated contraction, displaced instability, context dependence, or transient-dominated behavior**. Each advances the science; none requires claiming that trillion-parameter models have already been explained.
