# Research opportunity and what the evidence leaves open

**The strongest next experiment is a causal separation of accumulated feedback error from ordinary numerical answer sensitivity.** Run the arithmetic pipeline with identical initial states, vary batch layout, and compare free feedback against a control that resets the complete recurrent state to a reference trajectory before each soft step. Use a common-layout answer decoder afterward.

This outranks another continuation run because it addresses the largest unresolved inference connecting the geometry to frontier-lab relevance: **does continuous feedback amplify numerical differences into answer changes, or do vulnerable answer decisions change without substantial feedback amplification?** The reported experiments establish layout-dependent answers. They do not yet identify how much of that effect comes from recurrent accumulation, terminal decoding, or genuinely expanding dynamics.

The shrimp work remains scientifically valuable. Its next opportunity is not a more confident name, but a **predictive test of dimensional reduction** and an orbit-level explanation of the cusp pocket.

Four construct-validity boundaries should guide the experiments:

- **Answer disagreement is not accuracy loss.** Arithmetic permits measuring both objectively. Neither should stand in for the other.
- **Low-precision divergence is not itself evidence of chaos.** The reported near-universal bf16 divergence makes the periodic-versus-chaotic distinction insufficient to explain the arithmetic result.
- **A simple critical multiplier does not exclude codimension-two bifurcations.** FM5’s spectral gap excludes additional simultaneously critical multipliers, but not a cusp, which involves nonlinear degeneracy with a simple multiplier \(+1\). Its pending cusp checks remain pending.
- **Comparable leading multipliers do not establish an effectively two-dimensional nonlinear map.** A complex conjugate pair has positive product; clarify whether the 41 negative products occur outside the 91 complex-pair cells. Also distinguish a tracked real multiplier crossing zero from the spectral radius becoming zero.

All experiments below use deterministic numerical or exact-arithmetic outcomes. No human reviewers or model judges are needed.

# Ranked experimental directions

## 1. Identify the causal source of layout-dependent answers

**Question and hypothesis.** Does preserving numerical perturbations across soft steps account for a substantial fraction of layout-dependent arithmetic answers?

The falsifiable hypothesis is that removing accumulated state differences substantially reduces answer disagreement, even while the final soft step retains its native layout-dependent arithmetic. The strongest competitor is direct sensitivity of the final decision to small numerical differences, without substantial recurrent amplification.

**Minimum decisive experiment.** Use one 0.5B model, 24 development additions, and 128 independently generated held-out additions, stratified by operand length. Cross:

- bf16 versus float32;
- batch size 1 versus 8;
- free feedback versus reference-state reset.

Initialize every arm from the same cached prefix and soft state. Decode answers using the same batch-1 decoder. The detailed implementation appears below.

Problems—not steps, layouts, or reruns—are independent sampling units. Freeze prompts, temperature, soft-step count, parsing, and arithmetic strata before generating held-out operands.

**Comparisons and controls.**

- Strong baselines: float32; the state-reset intervention; native-layout forced greedy decoding.
- Cheap baseline: exact same-layout reruns.
- Negative control: reset versus free execution in the reference layout should agree.
- Positive controls: deliberate answer-logit changes must trigger the disagreement detector; known stable and expanding scalar recurrences must produce the expected perturbation-growth measurements.

**Endpoint and decision rule.** The primary estimand is the reduction in bf16 disagreement probability,

\[
\Delta=P(\text{disagreement}\mid\text{free})
-P(\text{disagreement}\mid\text{reset}),
\]

in percentage points per held-out problem. Use paired problem-level uncertainty. Expand this mechanism program if the estimate exceeds 15 percentage points and its 95% interval excludes zero. If the interval’s upper bound is below 15 points, abandon the proposed *large recurrent contribution* under these tested conditions.

**Interpretations.** A positive result establishes accumulated-state mediation, not chaos. A null with substantial free-run disagreement favors immediate numerical sensitivity. A mixed result across precisions suggests an interaction between quantization and feedback. Failure of state-copy or same-layout checks invalidates the causal comparison, not the broader research question.

**Resources—estimates.** Approximately 6–20 H100 hours for the 0.5B matrix and perturbation subset, or $16–64 at the quoted rates, conditional on measured step throughput. The cheapest useful pilot is eight problems with bf16 free/reset arms. No API spending is necessary.

## 2. Test whether the shrimp-like window admits a useful two-coordinate model

**Question and hypothesis.** Can two state coordinates predict finite-amplitude return dynamics around the period-5 window better than one coordinate, while additional coordinates provide little improvement?

The hypothesis is **operational two-dimensional closure over a specified neighborhood**, not a global two-dimensional invariant manifold. The strongest competitor is local spectral coincidence: two multipliers become comparable near the attracting orbit, but nonlinear trajectories retain consequential higher-dimensional information.

**Minimum decisive experiment.** Start with two development parameter locations and four evaluation locations in the period-5 window: body, close leading-pair region, leg, and cusp-pocket neighborhood. Use the return map \(P=F^5\).

At each location:

1. Compute leading invariant subspaces with matrix-free derivatives.
2. Generate finite perturbations at two amplitudes above the numerical floor.
3. Fit one-, two-, and four-coordinate return models and corresponding state-lifting maps.
4. Evaluate autonomous forecasts on 32 fresh perturbation seeds per location.
5. Apply matched transverse kicks to test whether omitted coordinates decay or retain predictive information.

Freeze model class and regularization using development locations. Local training at evaluation locations is allowed, but test perturbations remain untouched; this tests held-out trajectories, not unseen-parameter generalization. Seeds are independent conditional on each location; the four locations are not four independent shrimp windows.

**Baselines and controls.** Compare against the exact linearized return map, the fixed-orbit predictor, and a four-coordinate nonlinear model. Use an embedded two-dimensional Hénon map with strongly contracting extra coordinates as a positive control. Use a constructed three-coordinate system with consequential hidden-state coupling as a negative control. A random two-dimensional projection is a cheap baseline.

**Metric and gates.** Measure forecast squared error over eight returns, divided by the full trajectory’s squared excursion from the orbit, reported per perturbation seed and aggregated by location. Include early returns: eventual convergence to a stable orbit must not create trivial success.

A prospective operational threshold could be median normalized error below 0.05, at least a twofold improvement over one coordinate, and less than 20% additional improvement from four. Bootstrap seeds within locations; report locations separately. Escapes remain outcomes rather than being silently excluded.

**Interpretations.**

- Positive: a useful local reduced model, strengthened by decay of transverse perturbations.
- Null: this coordinate/model combination is inadequate—not proof that no two-dimensional reduction exists.
- Mixed: dimension varies across the window; a one-dimensional fold neighborhood can coexist with two-directional interior transients.
- Invalid: perturbations disappear below the numerical floor, coordinates are ill-conditioned, or trajectories collapse before discrimination is possible.

If spectral coordinates fail through nonnormality or curvature, the viable alternative is a trajectory-trained chart or delay-coordinate predictor—not simply more multiplier scans.

**Resources—estimates.** A two-location pilot should fit within 3–8 H100 hours; the fuller experiment approximately 12–25 hours, depending strongly on derivative cost. CPU fitting and bootstrap analysis can use the existing workstation. Expand to the hostile period-7 window only after the period-5 test is informative.

## 3. Determine the orbit-sheet anatomy of the apparent swallowtail pocket

**Question and hypothesis.** Do the projected crossing and tangent reversals organize distinct period-5 orbit sheets, with the multiplicity changes predicted by ordinary folds and cusps?

The competitor is misleading projection or branch switching: an attractive picture in parameter space need not identify a particular catastrophe or shrimp organization.

**Minimum decisive experiment.** Choose approximately 12–16 parameter offsets around the two reversals and projected crossing. Freeze predicted *local* changes in orbit counts before solving. Continue both stable and saddle branches, use deflated Newton searches, and identify duplicate cycles modulo phase.

Across a generic fold, two local branches appear or disappear. At a cusp, the local unfolding changes between one and three branches. At a projected crossing, first ask whether the two fold points represent different cycles. If they do, the crossing is not a higher-order degeneracy of one orbit.

The independent objects are orbit branches and parameter locations. Newton seeds are numerical recovery attempts, not independent experimental replicates. No global completeness claim follows from unsuccessful searches.

**Baselines and controls.** Forward simulation is the cheap baseline but misses unstable cycles. Full branch continuation is the strong comparison. Test the search and counting machinery on an analytic cusp unfolding and on two independent fold sheets crossing in projection. Offsets outside the predicted pocket supply negative controls.

**Metric and gates.** Report the fraction of frozen offsets at which predicted local branch counts and stabilities are recovered, with residuals, phase-equivalence checks, and sensitivity to starting seeds. This is numerical validation, not a binomial population estimate. Where feasible, certify individual recovered roots or local neighborhoods using existing interval machinery.

**Interpretations.** Agreement supports the local organizing geometry. Missing branches are inconclusive unless their absence is certified. A mismatch with certified roots falsifies the local prediction. Neither outcome alone establishes a global named shrimp class.

There is **no universal replacement for a doubly superstable center in an invertible map**. Fold/flip boundaries, stability regions, and real-to-complex transitions provide an operational skeleton. For an isolated two-dimensional invariant block, \(\operatorname{tr}^2-4\det=0\) locates multiplier coalescence; it does not prove nonlinear closure. A genuine swallowtail degeneracy requires more than two cusps in projection and generically calls for an additional unfolding parameter.

Primary starting points to verify are Hénon, “A two-dimensional mapping with a strange attractor” (1976), and Gallas, “Structure of the parameter space of the Hénon map” (1993). I would not attach the Façanha–Oldeman–Glass terminology without checking their precise definitions against the recovered orbit sheets.

**Resources—estimates.** Approximately 6–15 H100 hours using existing continuation infrastructure. The cheapest useful pilot is the orbit-identity comparison at the projected crossing, followed by four offsets around it.

# Recommended next experiment

## Causal reset experiment, with a finite-perturbation mechanism probe

This is the single next experiment I recommend. It converts the practical result from an association into an intervention-based decomposition, while remaining small enough to finish within two days. Unlike further topology, either a positive or a null directly changes what can be claimed about inference reliability.

### Experimental matrix

Use the existing 0.5B arithmetic pipeline, but freeze its configuration on development problems.

| Factor | Levels |
|---|---|
| Precision | bf16, float32 |
| Batch layout | Target alone; target with seven fixed, length-matched fillers |
| State treatment | Free feedback; reference-state reset |
| Answer readout | Common batch-1 greedy decoder |
| Soft horizon | Fixed, e.g. 64 steps, chosen before held-out execution |

The exact horizon should match the existing pipeline when possible. Do not tune it to maximize disagreement.

For each precision, obtain a batch-1 reference trajectory. Immediately before each reset-arm soft step, copy **all mutable inference state** from the corresponding reference checkpoint: soft embedding, KV cache, positions, and any other recurrent buffers. Execute that step normally in the assigned layout.

Crucially, **retain the final step’s resulting state** and decode from it. Do not copy the reference terminal state before decoding: that would force agreement by construction.

All arms begin from the same reference prefix state. This isolates feedback-stage effects; native-layout prefill and native-layout answer decoding can be secondary diagnostics, not additions to the primary estimand.

Assert identical target-row position IDs and attention semantics across layouts. Process one problem at a time and discard checkpoint traces after extracting measurements.

### Independent units and endpoint

Generate 128 held-out additions independently within four prespecified operand-length strata. Repeated precision/layout observations are paired within a problem.

The primary endpoint is \(\Delta\), the bf16 disagreement reduction defined above. A single binary rate with 128 independent problems has a worst-case approximate 95% half-width of 8.7 percentage points; the paired contrast’s precision depends on discordant pairs and must be calculated separately.

Report secondary outcomes:

- exact numeric accuracy in every arm;
- change in accuracy caused by layout;
- malformed-output frequency;
- float32 disagreement reduction;
- native-layout forced-greedy disagreement.

Use deterministic integer parsing. Invalid output is an explicit outcome category, never an exclusion. Also give a sensitivity analysis treating every invalid pair as disagreement. Rerun infrastructure failures once under the same configuration; persistent failures receive an explicit missingness bound rather than disappearing from the denominator.

### Strong checks and mechanism probe

Before held-out execution:

1. **Same-layout check:** repeated deterministic runs establish the implementation’s numerical floor.
2. **Reference replay check:** batch-1 free and reset trajectories must agree within the declared tolerance.
3. **Parser positive control:** deliberately force two distinct integer outputs and recover disagreement and correctness exactly.
4. **Growth-code positive control:** recover known stable and expanding scalar dynamics using the same distance-recording and fitting routines.

On a prespecified subset of 32 held-out problems, add one controlled perturbation to the soft embedding at step 8 in float32. Use three amplitudes—for example relative RMS \(10^{-5},10^{-4},10^{-3}\)—and both a random direction and the observed one-step layout-difference direction when nonzero.

Record soft-state and KV-cache distances separately, normalized for their changing dimensions. Measure the realized perturbation after assignment; a rounded-away kick is not a null effect.

Fit finite-time log-distance growth over a frozen interval only while distances are above the measured floor and below saturation. Parallel growth across perturbation amplitudes is more informative than one diverging trace.

This probe can show finite-time amplification in growing context. It does **not** establish an autonomous Lyapunov exponent, and float32 probe results do not automatically explain bf16 behavior.

### Decision branches

- **Large reset benefit plus measurable amplification:** recurrent accumulation contributes materially, and amplification is a plausible mechanism. Next test an implementable mitigation.
- **Large reset benefit without sustained growth:** accumulation matters, but exponential growth or chaos remains unsupported.
- **Little reset benefit, substantial disagreement:** prioritize direct numerical decision sensitivity and answer-stage precision.
- **Little disagreement under the controlled initialization/readout:** examine native prefill and decoding as the likely sources of the earlier pipeline effect.
- **Reduced disagreement but worse accuracy:** stabilization is not an accuracy improvement; preserve that distinction.
- **Failed controls or pervasive malformed outputs:** repair the specific measurement or task interface, then use fresh held-out problems. Do not interpret the original comparison.

### Resource envelope

No local GPU specification was supplied. Use the existing workstation for task generation, parsing, toy controls, and statistics; rent one H100 for model execution.

Benchmark eight problems first. An estimated 6–20 GPU hours covers the main 0.5B experiment, with roughly 2–6 additional hours if the perturbation probe is slower than expected. Keep checkpoint storage transient; retaining norms, outputs, metadata, and selected trajectories should require far less space than saving every cache.

Do not begin with 7B or 70B.

# First 48 hours

**Hours 0–6**

- Implement complete-state checkpoint/reset and common-layout readout.
- Generate development tasks and freeze parsing.
- Pass scalar-growth, parser, and reference-replay controls.
- Time eight pilot problems; set a hard GPU-hour cap.

**Hours 6–18**

- Run the development matrix.
- Check invalid-output frequency and whether final-step differences remain observable.
- Freeze configuration, thresholds, perturbation amplitudes, and analysis.
- Generate the untouched held-out set.

**Hours 18–36**

- Execute the 128-problem matrix.
- Run the prespecified perturbation subset.
- Save problem-level outcomes and growth traces; avoid selecting illustrative cases before aggregation.

**Hours 36–48**

- Produce paired disagreement and accuracy estimates with uncertainty.
- Deliver a causal decomposition table and amplitude-aligned growth plots.
- Select the next branch using the prospective rules—not the most attractive individual trajectory.

# Two-week research roadmap

**Days 3–5: test the explanation’s consequence.**

If reset substantially helps, compare free feedback under bf16, float32 softmax/expected-embedding accumulation, and full float32. The mixed-precision arm is a candidate mitigation, not a guaranteed rescue: network and cache errors remain.

If reset does not help, vary only prefill or terminal-decoder precision to localize direct numerical sensitivity. Use fresh arithmetic instances.

**Days 6–9: spend compute on one discriminating extension.**

- With a clear practical mechanism, replicate the key contrast on 7B using approximately 48–64 fresh problems. Treat it as a bounded transfer test, not a precise cross-model prevalence estimate.
- With an informative practical null, run the two-coordinate closure pilot on the period-5 window.
- If closure fails specifically near the cusp pocket, perform the orbit-sheet census rather than enlarging the surrogate indiscriminately.

**Days 10–14: resolve the strongest remaining alternative.**

Test transverse recovery and a second window if reduced closure succeeds. If it fails, compare a trajectory-trained chart before rejecting dimensional reduction. If numerical stabilization improves reproducibility, measure whether it preserves exact arithmetic accuracy on fresh tasks.

Cap the entire program at approximately **65 H100 hours**, costing $169–208 at the stated rates and leaving reserve credit. The deliverable is substantive either way: a causal account of answer instability, a falsifiable reduced dynamics model, or a sharply localized failure of one of those explanations—not another undirected scale-up.
