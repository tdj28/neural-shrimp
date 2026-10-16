# Research opportunity and what the evidence leaves open

**The strongest opportunity is to establish a practical stability horizon for continuous-feedback inference: when additional soft-thinking computation starts amplifying numerical differences into different answers, and whether that boundary can be moved without sacrificing accuracy.**

**My recommendation is one bounded experiment:** a paired replication of **top-15 versus full-distribution soft thinking, at 16 and 64 steps, across two batch layouts**, with matched greedy controls and a small reference-replay intervention. This directly tests the largest unresolved issue: whether the practical instability survives the published method’s truncation, rather than belonging mainly to your full-distribution variant.

Reproducibility is already your strongest practical result. The useful framing is not “GPUs are nondeterministic,” but:

> Numerical differences are the disturbance; feedback dynamics determine their amplification. For a specified inference procedure, precision, and compute budget, can we predict and reduce answer-level sensitivity?

Avoid the stronger claim that regime alone determines reproducibility. Initial disturbance size, finite horizon, clipping, and distance to an answer boundary also matter.

Three scientific distinctions should shape the experiments:

- **Top-\(n\) does not necessarily reduce covariance monotonically.** Renormalization can increase variance in retained directions, while support changes introduce additional boundaries. E1 tests a hypothesis, not a mathematical consequence.
- **Lowering temperature is not a guaranteed gain controller.** Competition may diminish, but \(1/T\) increases; moreover, your covariance proxy omits the network Jacobian.
- **Growing-context inference is not the autonomous one-token map.** Its measured perturbation growth is finite-horizon sensitivity of an evolving computation—not automatically the same Lyapunov exponent or bifurcation mechanism.

The ordinary-sampling null argues against a general “chaotic prompts are worse” headline. It does not rule out numerical reliability problems in actual feedback inference.

# Ranked experimental directions

Across these directions, use executable numerical answers or deterministic string measurements; no model judge is needed. Model-judge agreement would not establish semantic accuracy.

**Common analysis rules:** problems or prompts—not trajectories, seeds, or timesteps—are the sampling units. Cluster uncertainty by execution block when problems share batch interventions, and by template for generated tasks. Canonicalize answers to a number or `FAIL`; report failure rates separately. Retry hardware failures once, then give worst-case bounds for missing paired outcomes rather than silently dropping them. All unsuccessful answers count as incorrect for accuracy.

## 1. Published-method stability and the useful compute horizon

**Value per GPU-hour: highest.**

**Question and hypothesis.** Does batch sensitivity remain substantial with top-15 feedback, and increase between 16 and 64 steps? The falsifiable hypothesis is that truncation attenuates but does not eliminate answer-level instability. The strongest competitor is that the previous result primarily reflects full-distribution feedback or an implementation mismatch.

**Minimum decisive design.** Use 256 previously unexamined arithmetic problems, paired across truncation, horizon, and batch layout. Include top-1 feedback, ordinary greedy decoding, and reference-trajectory replay. The complete matrix appears below. This adds evidence about the published-method regime and finite compute budgets—not another demonstration of divergence in the autonomous map.

**Controls and measurements.** The primary endpoint is the percentage of problems whose canonical answers differ between batch layouts under top-15, 64-step inference. Prespecified secondary endpoints are the top-15/full contrast, horizon interaction, accuracy, and the fraction where exactly one layout is correct. Top-1 is the cheap control; ordinary greedy CoT is the practical baseline. Previously sensitive, excluded sentinel cases test whether the implementation can reproduce an existing effect.

**Interpretation and decision.** Persistent top-15 flips support a relevant reliability problem; full-only flips sharply narrow the implication. Increased flips without increased perturbation growth implicate answer boundaries or support switching rather than stronger dynamical amplification. Failed controls invalidate the attribution, not the project.

**Resources.** RTX first; approximately 2–8 GPU-hours for the full matrix and controls, conditional on actual context-length throughput. A 48-problem development pilot is cheapest. Reserve approximately $10–25 for matched float64 checks, using the running job where it covers the required conditions.

## 2. Stabilization that preserves useful computation

**Value: high, conditional on direction 1 finding consequential instability.**

**Question and hypothesis.** Can an intervention reduce numerical answer flips at comparable accuracy and token budget? Competing explanation: “stabilization” merely turns soft thinking into a less expressive, nearly greedy computation.

**Experiment.** On 48 development problems compare:

- fixed temperatures around the current \(T=0.6\);
- the proposed temperature controller;
- under-relaxation,
  \[
  x_{t+1}=(1-\alpha)x_t+\alpha\,\mathbb E[e],\qquad
  \alpha\in\{0.5,0.75,1\};
  \]
- float32 softmax and embedding accumulation with the transformer still in bf16.

Select at most two interventions before evaluating 256 fresh problems at the unstable horizon. Under-relaxation changes the growing-context computation; do not claim it preserves the original algorithm’s solution.

**Baselines and controls.** Compare against untreated top-15, the best development-selected fixed temperature, top-1, and full float64 on a small matched subset. A controller with its threshold disabled is a negative control. A known sensitive sentinel provides a positive test of whether the intervention can affect measured growth; a periodic sentinel checks that the diagnostic is not manufacturing expansion.

**Endpoint.** Paired reduction in flip rate, in percentage points per problem, subject to a separately reported accuracy difference and runtime cost. A useful target is at least a five-point absolute reduction or halving of flips. A three-point accuracy noninferiority claim will generally require more than 256 problems; size the extension from observed paired discordance.

**Branches.** Reduced flips with maintained accuracy supports engineering utility. Reduced flips plus accuracy loss is a trade-off, not success. No effect despite an operational controller rejects that controller. Failure of the proxy to track directly measured growth rejects the proxy’s control rationale.

**Resources.** Approximately 4–12 local GPU-hours, plus $10–30 for selected precision controls. Do not first scale the proxy controller across all GSM8K: its monotonicity assumption needs this feasibility test.

## 3. Ordinary decoding: numerical sensitivity beyond low margin

**Value: high external relevance; lower expected mechanistic specificity.**

**Question and hypothesis.** Do continuous-feedback sensitivity features predict ordinary-decoding answer flips beyond token margin and entropy? The strongest competitor is simple discrete branching at a near-tied token, without a useful contribution from the continuous-loop theory.

**Experiment.** Start with 128 problems, then expand to 512 only if enough flips occur. Run ordinary greedy CoT and two sampled trajectories per problem in both layouts. For sampling, use per-problem, per-position random streams: merely setting the same global seed does not guarantee common random numbers across batch layouts.

Use a separate development set for predictor selection. Compare a small prespecified margin/entropy/length predictor with the same predictor plus sensitivity features. Features must come from the original trajectory or an independent diagnostic—not from the already-diverged continuation.

**Controls.** Same-layout reruns are the negative control. Teacher-forced replay separates probability changes from subsequent textual branching. A deliberately constructed near-tie logit perturbation is a positive control for the sampler and branch detector. Ordinary greedy margin is the cheap, strong baseline.

**Endpoint.** Primary: answer-flip percentage per problem. Secondary: held-out predictive log-loss improvement, averaged by problem, over the strongest margin baseline. Seeds are repeated observations.

**Branches.** Flips without incremental prediction establish numerical sensitivity, not a chaos-based forecasting advantage. Incremental prediction warrants replication, but still does not establish causality. A null with very few flips is a precision limitation; a null with many flips challenges the proposed bridge. Broken common-random-number checks invalidate the sampling comparison.

**Resources.** Approximately 2–8 local GPU-hours; no API spending. Expand predictive modeling only after roughly 30–50 flip events exist—otherwise use descriptive comparisons rather than an unstable classifier.

## 4. Perturbation ensembles versus ordinary self-consistency

**Value: medium; worthwhile only after distinguishing useful diversity from instability.**

**Question and hypothesis.** Can perturbations create sufficiently uncorrelated, correct alternatives to improve majority-vote accuracy? The main competitor is that they merely diversify wrong answers.

**Experiment.** Develop perturbation amplitudes on 32 problems, distinguishing an amplitude comparable to observed numerical disturbances from the proposed \(10^{-3}\) stress perturbation. Freeze the choice. On 256 fresh problems compare:

1. one soft-thinking run;
2. eight identical reruns;
3. eight perturbed soft-thinking runs;
4. sampled CoT self-consistency.

Match total model-token evaluations, not merely the number of runs. Fix numeric voting, abstention, and tie-breaking rules in advance.

**Controls.** Identical reruns are the negative control and single-run decoding the cheap baseline. Sampled self-consistency is the strongest practical comparator. Synthetic vote sets with known correct majorities and controlled corruptions test the aggregation machinery; they do not demonstrate a model-level ensemble benefit.

**Endpoint.** Majority-vote accuracy, correct problems divided by all problems, with paired problem-level intervals. Also report answer disagreement and error correlation; neither substitutes for accuracy.

**Branches.** A gain over single runs but not matched-cost self-consistency is a limited alternative, not an advantage. Increased diversity without accuracy gain rejects diversity as the operative benefit. Null results with tight bounds can rule out a practically useful gain; broad intervals require restraint. Voting-control failures invalidate the comparison.

**Resources.** Approximately 6–20 local GPU-hours, depending on answer length. The cheapest useful pilot is 32 problems with four candidates. Expand only if the gain appears competitive with sampling; chaos is unnecessary if ordinary sampling buys the same diversity more cheaply.

## 5. Repetition: a narrow bridge to ordinary generation

**Value: lower; objective and inexpensive enough for a bounded test.**

**Question and hypothesis.** Does a prompt’s continuous-loop transition temperature predict repetitive sampled generation beyond entropy and ordinary temperature effects? The competitor is that repetition arises mainly from growing-context token attractors unrelated to the one-token map.

**Experiment.** Use 192 prompts grouped into nonoverlapping development and held-out families. Generate at four fixed temperatures and two seeds, capped at 256 tokens. Predefine repetition as an exact suffix cycle of period 1–8 spanning at least 32 tokens. Estimate loop transition temperatures with a fixed coarse scan; represent unresolved transitions explicitly.

**Controls.** Compare against temperature alone, then temperature plus entropy, initial token margin, and prefix repetition. Synthetic strings containing known cycles, near-cycles, and noncycles validate the detector. EOS termination is a valid nonrepetitive outcome, not missing data.

**Endpoint.** Held-out predictive log-loss improvement per generation, averaged and bootstrapped by prompt. Also report repetition incidence per prompt-temperature pair.

**Branches.** Added predictive value establishes a restricted statistical bridge, not causal identity of the two dynamics. A precise null weakens E5; a mixed family-specific result limits generalization. Detector failures invalidate the measurement. Do not rename exact repetition “poor semantic quality.”

**Resources.** Approximately 3–8 local GPU-hours, including the transition scan; pilot on 32 prompts. Abandon the predictor if held-out bounds exclude a prespecified useful improvement, rather than searching temperatures until a correlation appears.

**Other proposed applications:** agent loops, hallucination, alignment, and training on outputs introduce additional causal pathways and harder validation. They are not the next experiment. If extending to agents later, first use a deterministic arithmetic tool with an exactly specified transcript and perturb only one returned numeric representation. That tests error propagation without conflating it with tool nondeterminism or subjective agent quality.

# Recommended next experiment

## Paired published-method replication with a compute-horizon intervention

This outranks the alternatives because it can either establish a relevant deployment problem or remove the most important practical extrapolation from the existing evidence. It also supplies the conditions needed for a meaningful stabilization experiment.

### Experimental matrix

Use **48 development problems and 256 held-out problems**. Previously explored GSM8K items must not be relabeled fresh; unused training items are acceptable. If no untouched material remains, use executable generated arithmetic with held-out templates for confirmation, retaining GSM8K as descriptive replication.

| Component | Conditions | Problems |
|---|---|---:|
| Main matrix | top-\(n\): 15/full × steps: 16/64 × batch: 8/16 | 256 |
| Matched discrete control | top-1, 64 steps × batch: 8/16 | 256 |
| Practical baseline | ordinary greedy CoT × batch: 8/16 | 256 |
| Mechanism probe | open feedback versus reference replay, 64 steps | 48 prespecified |
| Precision check | matched bf16/float64 top-15 and full | 32–64 prespecified |

Hold \(T=0.6\), answer extraction, stopping conditions, and answer-token cap fixed. Do not add top-5 and top-30 yet: they increase cost without resolving the first decision.

### Batch construction and independence

Construct sixteen blocks of sixteen problems. Compare each block run together against the same block split into two batches of eight.

Literal equal-length, unpadded batches may select an unusual subset of GSM8K. Instead, use the **same padded sequence length, mask, position IDs, and answer position for each problem in both layouts**, with padding behavior validated during development. Do not add semantic filler to equalize lengths.

The endpoint denominator is 256 problems, but the layout intervention is shared within sixteen blocks. Report paired block-cluster intervals and block-specific rates. For reference, an independent-binomial interval near 50% would have approximately ±6 percentage-point precision; clustering can widen it substantially. This is a pilot for large effects, not a two-point accuracy claim.

### Primary endpoint and practical threshold

**Primary:** top-15, 64-step batch-layout answer disagreement, as a percentage of all problems.

Prespecified secondary measurements:

- top-15 minus full-distribution flip rate;
- 64-step minus 16-step flip rate;
- accuracy in each layout and the “exactly one correct” fraction;
- parse failures and ordinary-greedy disagreement;
- directly measured perturbation growth on the precision subset.

Treat a top-15 flip rate above roughly 10% as practically concerning. Evidence for a general reliability problem strengthens if its cluster interval excludes 5%, but expansion should depend on interval width and accuracy consequences—not just a threshold crossing.

### Most useful mechanism probe

Record the soft embeddings from a reference run. Replay those embeddings during the soft phase in both layouts, then let both systems decode answers freely.

This clamps the recurrent concept-input pathway while retaining batch-dependent transformer arithmetic and growing KV caches. Repeat with the opposite layout providing the reference on half the probe cases.

- **Open-feedback flips substantially exceed replay flips:** evidence that the concept-feedback pathway amplifies numerical differences.
- **Replay remains similarly sensitive:** examine KV accumulation, answer decoding, and discrete branch margins.
- **Only embedding distances collapse:** uninformative by itself, because those embeddings were clamped.

Replay is a causal diagnostic, not a proposed inference improvement.

Keep the top-64 covariance statistic explicitly labeled a **conditional proxy**; log retained probability mass and the \(n\)-versus-\(n+1\) logit gap. Support switching and omitted probability mass can defeat its interpretation.

### Implementation and resource envelope

1. Validate masks, top-\(n\) renormalization, numeric parsing, and identical-layout repeatability.
2. Run known sensitive sentinels separately from the evaluation set.
3. Run the 48-problem development matrix; measure actual throughput and memory.
4. Freeze held-out execution settings and analysis.
5. Execute the 256-problem matrix.
6. Run replay and matched float64 diagnostics without selecting cases based on observed flips.

**Estimated workload:** approximately 0.3–0.7 million model-token evaluations, depending on answer lengths and diagnostics. At the disclosed throughput, raw inference is modest; allow **2–8 local GPU-hours** for context-length slowdown and instrumentation. These are estimates, not guarantees. Use a 16-case float64 timing check before spending more than $25; retain most of the $150 for the intervention that the results justify.

# First 48 hours

**Hours 0–8**
- Implement the paired batch-block runner and fixed canonical parser.
- Verify padding/position equivalence and top-1 behavior.
- Run positive and negative controls.
- Deliverable: valid paired trajectories and an executable analysis producing denominators, accuracy, flips, and failure counts.

**Hours 8–20**
- Complete the 48-problem development matrix.
- Test replay on eight development cases.
- Freeze thresholds, held-out allocation, and the matrix.
- If controls fail, repair that specific implementation issue before interpreting instability.

**Hours 20–36**
- Execute the 256-problem matrix.
- Launch the bounded matched float64 subset.
- Deliverable: block-level flip rates, horizon contrasts, accuracy contrasts, and uncertainty.

**Hours 36–48**
- Complete replay probes and direct-growth comparisons.
- Select the substantive branch:
  - **Published top-15 is unstable:** proceed to stabilization.
  - **Only full feedback is unstable:** investigate truncation/support geometry.
  - **Neither reproduces:** test the earlier conditions explicitly.
  - **Mostly answer-boundary effects:** prioritize ordinary-decoding comparisons over a dynamical-gain controller.

# Two-week research roadmap

**Days 3–5: resolve the first mechanism branch.**  
For persistent top-15 instability, compare direct perturbation growth, support changes, replay, and precision rescue. For full-only instability, add top-5 and top-30 on a small fixed subset to distinguish smooth attenuation from support-dependent effects. Deliverable: a discriminating intervention result, not merely a correlation plot.

**Days 6–9: test one remedy or one alternative explanation.**  
If feedback amplification is supported, run direction 2’s development comparison and select at most two interventions. If answer branching dominates, run direction 3’s greedy/common-random-number pilot. Keep float64 spending bounded by the measured per-case cost.

**Days 10–14: fresh evaluation of the selected claim.**  
Expand using untouched problems, with sample size driven by observed paired discordance and the scientifically relevant effect. Replicate a mechanism diagnostic on the smaller local Qwen model, but do not count same-family replication as broad architectural generalization.

The strongest attainable headline is **a measured, intervention-sensitive reliability boundary for continuous-feedback inference**. Whether moving that boundary also improves accuracy is a separate—and now cleanly testable—question.
