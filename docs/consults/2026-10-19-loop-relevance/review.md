# Research opportunity and what the evidence leaves open

**The strongest opportunity is to test whether changing recurrent dynamics improves—or degrades—task performance at a fixed, realistic inference budget.** Your Huginn results establish that recurrent-depth stability can be input-dependent and experimentally manipulated. They do not yet establish when stabilizing the loop helps computation.

**I recommend one next experiment: a paired Huginn relaxation-by-depth study on objectively scored questions, with finite-horizon sensitivity measured on a prespecified subset.** This outranks a larger chaos search or a full Lyapunov-spectrum survey because it directly tests the proposed connection between operating regime and useful inference.

Three distinctions should guide the program:

- **Autonomous versus driven dynamics.** Expected-embedding feedback has your stated factorization under the corresponding map and Jacobian definitions. Actual expected-embedding reasoning may append positions or change conditioning, so it need not be your autonomous one-slot system. Hidden-state and recurrent-depth loops require the derivative of the *complete recurrent update*, including changing auxiliary state where relevant.
- **Discrete versus continuous systems.** Sampled text/tool loops lack the ordinary state-space derivative used here, but admit transition-kernel, coupling, and disagreement analyses. Your zero-temperature limit does not automatically characterize an agentic loop.
- **Asymptotic versus finite-horizon behavior.** Two or three passes need not approach an attractor—but can reach one in special cases. More importantly, asymptotic instability may be irrelevant at that horizon, while a stable, nonnormal Jacobian can transiently amplify perturbations.

“Hopfield-like” does not settle convergence. Particular Hopfield constructions have Lyapunov guarantees under particular update rules; a generic recurrent transformer does not inherit them. Conversely, lacking a demonstrated energy function does not prove that none exists. Under-relaxation is also not universally stabilizing: a multiplier with real part greater than one cannot be brought inside the unit disk by positive scalar under-relaxation.

### Suggested manuscript wording

The following replacement passages total approximately 300 words:

**(a)**  
> This fixed-prefix, fixed-window soft-token map differs from ordinary autoregressive generation, whose context grows, and from recurrent-depth transformers, which repeatedly update hidden states through a shared block. Continuous-state dynamical methods can study both soft-token and recurrent-depth loops, but their update operators—and therefore their derivatives and stability conditions—differ.

**(b0)**  
> Feedback creates a dynamical system, but does not by itself imply convergence or chaos. Energy-based convergence guarantees require specific structure and update rules; they do not follow merely from resemblance to a Hopfield network. The tied readout has the energy property established in Sec. III, whereas no corresponding guarantee is established here for the complete transformer update. Our soft-token system exhibits fixed points, cycles and chaos. Our recurrent-depth experiments exhibit convergence and oscillatory instability, but no observed chaos.

**(b)**  
> In Huginn, stability is governed by the recurrent block’s Jacobian, without the soft-token readout factor. At one GSM8K-style prompt, its multipliers predict a relaxation-induced Neimark–Sacker bifurcation at 1.0777, compared with 1.0778 from the invariant-circle amplitude. A prespecified generation test found accuracy loss at relaxation 1.20 but not 1.05. Six of 30 ARC-Challenge development inputs had unstable native fixed points; under-relaxation restored convergence. These results motivate evaluating update dynamics jointly with iteration count. They do not establish a general relationship between stability and reasoning accuracy, especially at short inference horizons.

**(c)**  
> The repository contains interval-arithmetic certificates, verification code, exact inputs and release records: the period-8 certificate in `codex/certify-llm`, release record `certificates/llm-p8-certificate-final-001.json`, and the coexisting period-2 and period-3 certificates, token loops and basin neighborhoods in `codex/coexistence-task33`.

**(d)**  
> These findings motivate testing whether interventions in continuous-feedback systems alter finite-horizon sensitivity or task-relevant decisions as well as immediate outputs. We tested no safety property: dynamical instability is neither misalignment nor evidence of it.

Keep the press reports out: they add no experimentally supported architectural premise. Based on the supplied summaries—not independent source verification—position the contribution as a bridge between pretrained-model dynamics, prospective predictions, certification, and inference behavior, not as the discovery that looped transformers can bifurcate. Kim and the supplied stabilization literature make that broader novelty claim particularly unsafe. No additional citation can responsibly be declared missing without checking its actual scope.

# Ranked experimental directions

The ranking balances scientific value, discrimination, feasibility, and cost.

**Shared measurement rules.** Use objective answer keys, not model judges. Freeze development choices before held-out evaluation. Questions, deduplicated across related variants, are the independent units; depths, relaxation settings, numerical repeats, and perturbation directions are repeated measurements. Use paired question-level confidence intervals, clustering related items if retained. Count treatment-specific nonfinite outputs as failures in an intention-to-run accuracy endpoint and report their incidence separately; repair infrastructure failures and rerun every affected arm. Never silently discard difficult runs.

Hardware capacity is unspecified. The plans below use the existing Huginn implementation and local hardware first; rental envelopes are **estimates**, contingent on a throughput and memory benchmark. No paid API is necessary.

## 1. Does stabilization help at the inference horizon actually used?

**Question and hypothesis.** At equal recurrent-block evaluation count, does under-relaxation improve answer accuracy or numerical reproducibility? The falsifiable hypothesis is that its benefits concentrate on inputs exhibiting large native finite-horizon amplification.

**Strongest competitor.** Relaxation changes the computational trajectory, not just stability. Any benefit could reflect slower effective progression, regularization, or an advantageous readout time. Conversely, convergence might erase useful intermediate computation.

**Minimum decisive experiment.** Run the relaxation-by-depth matrix specified below on 32 development and 128 untouched ARC-Challenge questions, excluding the previous 30 from held-out evaluation. Use first-answer-position multiple-choice scoring, with a fixed label format whose tokenization has been checked. Measure native and treated finite-horizon sensitivity on 32 randomly selected held-out questions.

**Comparisons and controls.**

- Strong baseline: native inference at matched block-call count and the best native depth selected on development data.
- Cheap baseline: native inference near the relaxation-adjusted effective depth.
- Negative control: intervention machinery with relaxation exactly one.
- Positive controls: the previously demonstrated Huginn instability at its original prompt and horizon, plus an oracle-logit fixture validating answer scoring.

**Primary estimand.** Paired accuracy difference, in percentage points per held-out question, between relaxation 0.8 and native inference at the locally used depth \(K_0\). Secondary endpoints are answer changes and finite-horizon amplification. With 128 independent questions and approximately 20% discordant paired answers, the accuracy-difference interval has roughly ±8 percentage-point precision; this is a substantial-effect pilot, not a test of small gains.

**Interpretation and decisions.** Improved accuracy plus reduced sensitivity supports practical usefulness, but not a uniquely identified stability mechanism. Reduced sensitivity without accuracy gains establishes controllability, not better reasoning. Accuracy gains without reduced sensitivity favor a trajectory/readout explanation. A valid null rules out only effects resolved by the interval. Implementation mismatch makes the comparison invalid.

**Resources.** Cheapest pilot: 16 development questions. Estimated main inference cost: 4–24 GPU-hours, with another 4–24 for selected diagnostics; benchmark before committing. Dependencies are actual memory fit, faithful native-output reproduction, and perturbations measurable above rounding noise.

## 2. Which amplified state directions actually reach the answer?

**Question and hypothesis.** Does large finite-horizon state amplification cause task-relevant output sensitivity, or mostly motion in output-insensitive directions? The hypothesis is that answer-relevant gain predicts perturbation-induced errors better than state norm growth or fixed-point spectral radius alone.

**Strongest competitor.** Most label changes may be explained by small answer margins, without needing a dynamical explanation. Your existing null result for ordinary sampled-answer accuracy makes uncertainty a mandatory baseline.

**Minimum decisive experiment.** On 64 previously untouched Huginn questions, inject paired positive/negative perturbations at a fixed early depth. Compare:

1. estimated high-gain directions of the remaining recurrent computation;
2. random directions matched in norm;
3. directions chosen for immediate answer-margin sensitivity;
4. zero perturbation.

Use three amplitudes established on development inputs. Where feasible, estimate high-gain directions through matrix-free Jacobian-vector and vector-Jacobian products. Otherwise, use finite-difference directional probes and explicitly call the result **sampled directional gain**, not a leading singular value or Lyapunov exponent.

**Controls.** The strong predictive baseline is native answer margin plus entropy; the cheap baseline is random-direction gain. Validate derivative calculations against finite differences on a small smooth fixture and use a known expanding linear recurrence as a measurement positive control. The intervention itself must also move the model’s logits detectably at its largest diagnostic amplitude.

**Metrics.** Primary endpoint: excess error probability under high-gain versus matched random perturbations, averaged within question and then across questions. Secondary metrics are RMS state-amplification ratio and answer-log-odds change per unit relative RMS perturbation. Bootstrap questions, not perturbation runs. At 64 questions, emphasize precision and effect curves rather than fine subgroup claims.

**Interpretation and decisions.** Greater state growth without answer effects identifies an observability gap. Greater answer effects after accounting for margin support task-coupled amplification. This still does not prove naturally occurring numerical perturbations occupy those directions. Mixed signs—some errors corrected, others introduced—argue against equating sensitivity with harm. If no amplitude range separates rounding noise from nonlinear saturation, abandon the local-Jacobian claim and retain a finite-perturbation robustness experiment.

**Resources.** Estimated 8–40 GPU-hours, dominated by differentiation. Start with eight questions and two directions. If backward passes exceed local memory, finite differences preserve a useful experiment without requiring a larger machine or full spectrum.

## 3. Does predictive stability transfer beyond Huginn?

**Question and hypothesis.** Can local multipliers predict relaxation-induced instability in a second released recurrent-depth checkpoint? The falsifiable hypothesis is prompt-level quantitative transfer of the relaxed-multiplier rule, not universal accuracy improvement.

For an actual autonomous update with multiplier \(\mu\),

\[
\mu_\lambda=1+\lambda(\mu-1).
\]

A prediction requires the relevant fixed point, state definition, and update rule—not merely an architecture label.

**Strongest competitor.** The Huginn result may depend on one checkpoint, initialization convention, or long-horizon operating point. Another model may be explicitly nonautonomous, adaptively halted, or far from any relevant fixed point.

**Minimum decisive experiment.** First screen one accessible checkpoint on eight development prompts for faithful recurrence extraction and feasible memory use. On 32 untouched prompts, estimate native fixed points and leading multipliers where convergence permits. Freeze predicted stability boundaries before running a small relaxation grid around them. Evaluate the same interventions at the model’s ordinary inference horizon.

**Controls.** Strong baseline: Jacobian prediction versus observed perturbation growth; cheap baseline: residual norms and oscillation detection alone. Negative control: relaxation-one wrapper parity. Positive control: a linear recurrence with known complex multipliers run through the same prediction code.

**Metrics.** Primary endpoint: absolute predicted-versus-observed threshold error, in relaxation units, per eligible prompt. Report the eligible fraction out of all 32 prompts; failed fixed-point searches must not disappear from the denominator. Task accuracy is secondary and paired by question. Models are not independent replicates merely because many prompts are tested.

**Interpretation and decisions.** Agreement transfers a predictive method, not a general claim about deployed frontier systems. Long-horizon agreement without ordinary-horizon effects limits operational relevance. No accessible fixed points rejects this particular fixed-point analysis for those inputs, not dynamical analysis generally: switch to finite-horizon products. Failure to reproduce native inference invalidates the extraction.

**Resources.** Estimated 8–48 GPU-hours after feasibility screening; checkpoint size may dominate. Use a prospective rental ceiling of roughly [amount redacted]–[amount redacted] assuming suitable GPUs cost [amount redacted]–[amount redacted]/hour. If access, memory, or licensing fails, do not substitute an inaccessible-model speculation: test another Huginn task distribution and label it task transfer, not architecture transfer.

# Recommended next experiment

## A fixed-budget relaxation-by-depth trial

This is the **one experiment to start now**. It tests a practically meaningful proposition with controlled computation, while generating the specific cases needed for the more expensive mechanism probe.

### Experimental matrix

Let \(K_0\) be the configured recurrent depth actually used in your local Huginn inference—not a presumed frontier deployment count.

| Factor | Levels |
|---|---|
| Relaxation \(\lambda\) | 0.8, 1.0, 1.2 |
| Recurrent depth | 2, 3, \(K_0\), \(2K_0\) |
| Additional native comparison | \(\lambda=1,\ K=\operatorname{round}(0.8K_0)\), minimum 1 |
| Questions | 32 development; 128 held out |
| Diagnostic subset | 32 held-out questions, selected before outcomes |

Remove duplicate depth cells. If native inference uses adaptive depth, record its realized depth per question and replay matched counts for the primary comparison.

The primary endpoint is

\[
\Delta A=A(0.8,K_0)-A(1,K_0),
\]

with accuracy defined as correct highest-scoring answer label divided by **all 128 held-out questions**. This measures constrained multiple-choice decisions, not unrestricted reasoning quality.

### Implementation sequence

1. **Preserve the real update.** Instrument the existing recurrent block. Identify initialization, injected randomness, position dependence, cache changes, and any iteration-dependent inputs. If the update is nonautonomous, analyze its finite composition rather than naming an autonomous attractor.
2. **Establish parity.** At \(\lambda=1\), the wrapper must reproduce native logits within a declared numerical tolerance and reproduce answer labels. Use the same dtype and attention implementation across primary arms.
3. **Validate scoring and intervention.** Run the oracle-logit fixture. Reproduce the supplied known instability at its original long horizon; failure at two passes is not a failed control, because the original effect was asymptotic.
4. **Run 16 development questions across the matrix.** Measure memory, throughput, invalid-output rate, and answer-margin distributions. Complete the remaining development questions before freezing implementation choices.
5. **Run the held-out matrix once.** Randomize execution order to avoid conflating treatment with thermal, batching, or software-state effects.
6. **Add diagnostics without retuning treatments.** On the prespecified 32 questions, measure recurrent residuals and paired-trajectory amplification at two perturbation sizes. Require a reasonably consistent small-perturbation range; otherwise report finite-amplitude sensitivity only.

Use fixed initialization seeds for the primary comparison. Any additional seeds or batch layouts are nested repeats, not extra questions.

### Why these comparisons discriminate

Equal block-call counts test practical performance at equal recurrent work. The extra native depth tests the simpler explanation that relaxation merely slows effective progression. The native depth curve tests whether a good readout time is sufficient.

The most useful subsequent causal intervention is **directional perturbation at a fixed early depth**, comparing amplified and matched random directions. Do not begin with mode-selective stabilization: away from a relevant fixed point, suppressing an estimated eigenmode can be difficult to interpret.

### Resource envelope and branches

For illustration, with \(K_0=32\), 160 questions and 256-token prompts, the matrix contains approximately **13.5 million token–recurrent-block evaluations**, excluding diagnostics. This is a workload estimate, not a runtime guarantee. Use the development benchmark to translate it into local time.

- **Accuracy improves and amplification falls:** undertake an independent, larger confirmation; then test directional mediation.
- **Amplification falls but accuracy does not:** expand only if the interval still permits a worthwhile gain. Otherwise prioritize observability, not more damping.
- **Accuracy improves without amplification falling:** investigate effective depth and readout timing; drop the stabilization explanation.
- **Over-relaxation hurts only at long horizons:** report a finite-horizon boundary on the asymptotic result.
- **All effects are small with sufficiently narrow intervals:** abandon scalar relaxation as an accuracy intervention in this setting, while retaining its value as a dynamical probe.
- **Controls fail:** repair the specific implementation or measurement failure before interpreting task outcomes.

A pragmatic expansion threshold is a five-percentage-point potential benefit. Because this pilot cannot reliably resolve that size, any larger accuracy claim should use new held-out questions under frozen settings, not repeated significance-driven expansion of the same sample.

# First 48 hours

**Hours 0–8 — executable setup**

- Add the relaxation hook and recurrence-state logging.
- Record the actual \(K_0\), initialization and cache conventions.
- Prepare the 32/128 split, excluding prior experimental questions.
- Deliverable: a complete matrix manifest and deterministic scoring script.

**Hours 8–20 — embedded controls**

- Run native-wrapper parity and the oracle scoring fixture.
- Reproduce the existing dynamical positive control.
- Benchmark four questions for memory and throughput.
- Deliverable: functioning intervention, measured resource projection, explicit failure flags.

**Hours 20–36 — cheapest useful pilot**

- Run 16 development questions across all matrix cells.
- Test finite-difference sensitivity on four of them.
- Branch: if local differentiation is infeasible, retain forward perturbation pairs; if Huginn does not fit, price one bounded rental session using measured requirements.

**Hours 36–48 — lock and begin**

- Complete development runs, freeze thresholds and numerical settings.
- Launch held-out inference.
- Deliverable: initial paired task outcomes and runtime measurements—not another unbounded parameter scan.

# Two-week research roadmap

**Days 3–5:** Complete the held-out matrix. Produce paired accuracy differences, native depth curves, invalid-rate tables and question-level outcome transitions. Keep first-answer scoring separate from any later free-generation test.

**Days 6–8:** Complete the prespecified sensitivity subset. Compare amplification, answer margin and relaxation response. Determine whether the next uncertainty is *effect size*, *task coupling*, or *implementation validity*.

**Days 9–11: branch substantively.**

- **Promising practical effect:** run a new 256-question confirmation, subject to remaining data and compute.
- **Strong dynamical effect but weak task effect:** begin the eight-question directional-perturbation pilot.
- **Only long-horizon effects:** test a prespecified intermediate-depth grid to locate when sensitivity becomes behaviorally visible.
- **No informative local derivative:** run finite-amplitude robustness curves instead; do not force a Lyapunov interpretation.

**Days 12–14:** Finish the chosen branch and produce one decisive experimental package: an independently estimated intervention benefit, an observed state-to-answer sensitivity relationship, or a bounded null showing where asymptotic dynamics do not explain finite-budget behavior.

That last outcome would still answer the central scientific question more directly than finding additional chaotic windows.
