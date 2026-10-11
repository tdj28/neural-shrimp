# Research opportunity and what the evidence leaves open

**Recommendation: test whether the near-edge transverse response preserves an earlier input when a second input arrives, using small, numerically qualified pulses and an intervention that removes transverse state.** This moves from explaining an isolated readout to testing a concrete neural consequence: usable short-term memory under interference.

**TLDR:** In this fixed three-neuron network, the shrimp core rapidly forgets transverse disturbances, while the selected near-edge cycle retains them. We do not yet know whether that persistence helps recover an earlier input—or mainly causes contamination and readout errors. A two-write experiment can distinguish these possibilities.

The disclosed evidence supports three important conclusions:

- Timing and transverse state can carry distinguishable signals.
- The selected near-edge cycle retains transverse signals much longer than the core cycle.
- A timing-oriented observer can misread those signals, including predicted early sign reversals.

It does **not** establish useful input processing, a shrimp-specific advantage, nonlinear computation, or transfer to trained networks. Long-lived perturbations are compatible with generic weak attraction near a periodic-orbit boundary. Persistence can support memory while simultaneously reducing robustness.

I would **not make the proposed 32-job direct-propagation audit the next standalone study**. Its scientific limitation is that agreement between tangent implementations cannot establish whether retained state is useful. The disclosed physically scaled discrepancies are already tiny for the compatible small-pulse calculations. Instead, embed direct integration and cross-solver checks in a new behavioral experiment, escalating only if they change its endpoint. This does not repair the strict-arithmetic failures or resolve the problematic `.01` trajectories.

Also stop treating geometry-informed waveform control as the presumptive best intervention: its pooled advantage is unestablished. Keep signed-axis and full-linear controls wherever relevant.

# Ranked experimental directions

The ranking weighs discriminating value, executable scope, and cost—not likelihood of obtaining a positive result. All proposed outcomes are new experiments, not consequences already demonstrated by EXP-015.

## 1. Transverse memory under interference, with state-component removal

**Question and hypothesis.** Can the near-edge cycle retain an earlier binary write after another write, and does that information depend on transverse state? The falsifiable hypothesis is that its advantage over the core persists under interference and is substantially reduced by removing the transverse component before the second write.

The strongest competing explanation is simple weak contraction with no additional processing advantage: the near-edge system remembers *everything*, so interference negates its apparent benefit.

**Minimum experiment.** Use two small phase-null writes, one period apart, followed by observation through eight periods. Cross settings, randomized initial phase, first-bit sign, second-bit sign or absence, and three state interventions: identity, retain timing component only, retain transverse component only. The detailed matrix appears below.

Independent sampling units are independently drawn initial phases, paired across settings. Bit combinations, windows, and interventions are repeated measurements within those units—not extra replicates. Separate implementation-development phases from held-out phases.

**Comparisons and controls.**

- Strong baseline: complete time-varying linear response with the same inputs and observation noise.
- Mechanistic baseline: matched one-mode exponential memory.
- Cheap baseline: the existing frozen derivative matched filter.
- Negative controls: zero first write; label permutation; timing-only prediction for phase-null writes.
- Positive controls: immediate write recovery and synthetic injected observation signals; check that component removal actually changes the intended projected state.

**Endpoint and decisions.** Primary endpoint: near-edge minus core balanced accuracy for the **first bit**, at four periods, with the second bit balanced and unknown. Units are percentage points, averaged over held-out phases. Also report retained information without interference and the within-edge removal contrast.

Use analytic Gaussian-noise probabilities, not fictitious noisy-trial counts. Bootstrap phase blocks for phase-average uncertainty; numerical disagreement gets a separate sensitivity bound. Expand only if the effect exceeds a prospectively chosen practically relevant margin and survives controls.

**Interpretations.** Positive results establish a transverse memory contribution in these two cycles. A null with adequate immediate detectability shows that persistence alone does not provide the tested benefit. Memory without resistance to interference indicates a retention–robustness tradeoff. An ineffective or unstable removal operation invalidates the causal interpretation, not necessarily the simpler setting comparison.

**Resources—estimates.** A 16-phase held-out pilot requires hundreds, not millions, of trajectories; approximately 10–100 CPU-hours depending on qualified solver cost. No GPU, rental, API, or human judging is needed.

**TLDR:** We would learn whether the near-edge network preserves a recoverable earlier write, rather than merely retaining disturbances.

## 2. Test whether boundary distance, rather than shrimp geometry, explains behavior

**Question and hypothesis.** Across nearby periodic cycles, does retention follow contraction and observability continuously, or is there a residual association with independently established shrimp organization?

The strongest alternative to a shrimp-specific account is generic near-boundary slowing. Merely comparing another visually selected “outside” point would not distinguish them—especially given the earlier periodic exterior misclassification.

**Minimum experiment.** Continue the selected periodic branch along two short parameter paths. Seek approximately 8–12 numerically qualified cycles spanning transverse contraction, with 16 held-out phase evaluations per cycle. Where feasible, add a different periodic branch with overlapping period and multiplier ranges.

Measure period, both transverse multipliers, finite-horizon observable gain, and the direction-1 memory endpoint. Define boundary events through continuation diagnostics, not image appearance. Develop matching tolerances on one path; test them on the other.

The units supporting parameter-space conclusions are cycles and continuation paths. Their proximity makes them dependent: do not turn dense continuation steps into an enormous sample size.

**Baselines and controls.** Compare a multiplier-only model, full-linear observable predictions, and a cheap period-normalized decay model. A stable synthetic oscillator with adjustable contraction is a positive control for recovering the predicted decay trend. Within-cycle phase variation is a control against mistaking observability variation for contraction.

**Endpoint.** Held-out prediction error in memory accuracy, in percentage points per cycle, and residual gain after matching period, contraction, and immediate detectability. Numerical failures remain unqualified settings, with qualification coverage reported.

Expand only if matching is feasible and a residual exceeds a preset margin, such as five percentage points, with stable direction across paths.

**Interpretations.** Agreement with generic predictors favors a boundary/contraction explanation. Residual differences motivate additional dynamics, but do not identify geometry as causal. No suitable matched branch means the narrower result is a within-branch contraction–behavior relation; shrimp specificity remains unresolved.

**Resources—estimates.** Cheapest pilot: four continuation points and eight phases each. Approximately 5–50 CPU-hours initially, potentially 30–200 for the bounded panel. Dependency: reliable continuation and cycle qualification; a larger atlas is unnecessary.

**TLDR:** We would learn whether these neural effects track generic weak attraction or something not explained by the simplest periodic-orbit descriptors.

## 3. Validate early readout reversal and identify its finite-time cause

**Question and hypothesis.** Do small finite phase-writing pulses produce the predicted early sign reversals, and are those reversals eliminated when their transverse component is removed?

The strongest competing explanation is observer mismatch: the reversal belongs to this derivative filter, not to a general reversal of stored information. Numerical or template errors are additional alternatives.

**Minimum experiment.** At 24 newly drawn held-out phases, compare:

1. the existing phase-writing direction;
2. a pure flow-tangent displacement;
3. its corresponding transverse component;
4. a recombined displacement.

Use both signs at `1e-4`, with a smaller-dose check on selected cases. Construct predictions before finite-pulse integration. Compare equal predicted timing writes where feasible, reporting unequal physical norms; retain an equal-norm analysis separately.

**Baselines and controls.** Full-linear prediction is strongest; finite-shift phase-only prediction is cheap. Zero pulses are negative controls. A deliberately shifted reference trajectory is a positive control for readout sign and clock convention. Recombination is a first-order control, not an assertion of exact nonlinear superposition.

**Endpoint.** Signed readout divided by analytic noise SD, per phase, at one and two periods. Secondary endpoint: fraction of phases with reversed sign. Bootstrap phases, not signs or windows. Use paired dose scaling to distinguish linear transport from finite-amplitude effects.

Expand if reversals reproduce with solver agreement and the intervention changes them as predicted. Abandon “transverse opposition explains the reversal” if validated finite-pulse responses disagree systematically.

**Interpretations.** Positive results establish observer-dependent cancellation in finite trajectories. No reversal despite accurate linear forecasts elsewhere narrows or falsifies the proposed prediction. Reversal only for the old filter implicates observation choice. Solver-dependent signs are invalid evidence.

**Resources—estimates.** A six-phase pilot is roughly 100 traces; the complete panel approximately 400. Expect 5–60 CPU-hours. It is cheaper than direction 1 but mostly validates a predicted consequence rather than testing a new function.

**TLDR:** We would learn when the network retains a write but a timing-based observer reports its opposite.

## 4. Test genuine nonlinear mixing, not just fading memory

**Question and hypothesis.** Do two sequential inputs create a reproducible interaction that is observable beyond additive linear dynamics? Long retention does not itself imply nonlinear computation.

**Minimum experiment.** Apply writes in two distinct fixed directions, using four sign combinations at `1e-4` and `1e-3`, two spacings, both settings, and approximately 32 held-out phases. Compute the interaction waveform

\[
I=\frac{y_{++}-y_{+-}-y_{-+}+y_{--}}{4}.
\]

Additive dynamics give zero. Use development phases to select one observation projection; freeze it before held-out evaluation. Do not optimize a decoder on held-out sign combinations.

**Baselines and controls.** Compare exact nonlinear trajectories with full-linear propagation and a second-order response prediction if implementable. A cheap baseline is the summed pair of isolated responses. Linear simulated dynamics provide a negative control; an explicitly constructed quadratic feature channel provides a positive control for the interaction estimator, not evidence about the network.

**Endpoint.** Interaction magnitude in activation units and in observation-noise SD, per phase; secondary endpoint is held-out accuracy for the sign product under the frozen readout. Dose scaling should approximately follow the product of pulse amplitudes in the perturbative regime.

Expand only if the effect is numerically resolved, detectable at the stated noise level, and not explained by the readout itself introducing nonlinear mixing.

**Interpretations.** A positive result demonstrates nonlinear input mixing, not superiority to a simple nonlinear reservoir. A qualified null bounds mixing at these doses and delays. Mixed phase dependence motivates phase-conditioned processing. Dose-dependent solver failures invalidate those cells; they do not establish rich dynamics.

**Resources—estimates.** Cheapest pilot: eight phases, one spacing, both doses. Approximately 10–100 CPU-hours for the bounded panel. Do not escalate automatically to `.01`; first establish whether second-order signals are measurable in the qualified range.

**TLDR:** We would learn whether the network combines successive inputs nonlinearly, rather than merely retaining their sum.

# Recommended next experiment

## A two-write transverse-memory pilot

This outranks the alternatives because it adds a behavioral question, a strong linear explanation, and a feasible intervention in one small experiment. Boundary attribution becomes worth pursuing only after identifying an effect worth attributing.

### Experimental matrix

Use the two disclosed reference cycles and their existing clocks.

| Factor | Levels |
|---|---|
| Setting | Core; near edge |
| Initial phase | 8 development draws; 16 separate held-out draws |
| First write at \(t=0\) | \(\pm10^{-4}v(\phi)\) |
| Second write at \(t=T\) | Absent; \(\pm10^{-4}v(\phi)\) |
| Intervention at \(T^{-}\) | Identity; timing-only retention; transverse-only retention |
| Observation | Existing 32-sample windows ending at \(2T,4T,8T\) |

Here \(v(\phi)\) is the existing deterministic phase-null unit direction. Writes follow a precomputed schedule; they are not state-feedback corrections. **This is now an externally input-driven experiment**, unlike the prior observation-only clock.

The six write histories and three interventions give **576 held-out trajectories**, plus development and controls. All windows come from the same trajectory.

For a perturbed state \(x\), reference state \(x_r\), and \(d=x-x_r\), implement:

\[
x_{\mathrm{timing}}=x_r+Qd,\qquad
x_{\mathrm{transverse}}=x_r+Pd.
\]

Use the existing phase-specific physical \(Q=fZ^\top\), not an orthogonal projection or a repaired projector. Record actual component leakage and state replacement magnitude. These operations remove components in the **local linear decomposition**; they are not exact nonlinear isochron resets.

### Readout and primary endpoint

Retain the existing scalar filter as the cheap baseline. Add one strong, frozen linear observation rule using the 96 activation coordinates in each window.

From the full-linear response, calculate observation vectors \(A\) and \(B\) for unit-sign first and second writes. Freeze

\[
h\propto(\sigma^2 I+BB^\top)^{-1}A,
\]

with \(\sigma=10^{-4}\). This treats the balanced second write as nuisance covariance. Normalize \(h\) for numerical convenience. Use the **identity-arm readout unchanged across removal arms**, so decoder retuning cannot manufacture an intervention effect. An arm-specific oracle can be secondary.

For each exact finite trajectory, subtract its matched reference observation and compute \(q=h^\top y\). Its observation-noise SD is \(\sigma\|h\|\). Accuracy for first-bit label \(s_1\) is

\[
\Pr(\text{correct})=\Phi\!\left(\frac{s_1q}{\sigma\|h\|}\right).
\]

Average over both first-bit signs and both second-bit signs, then phases.

**Primary estimand:** paired near-edge minus core identity-arm accuracy at \(4T\), with the second write present, in percentage points.

Key secondary estimands are interference loss relative to no second write and the within-edge timing-only-removal contrast. Report signed margins as well as probabilities to expose saturation.

### Checks that must pass

1. **Observation positive control:** inject known synthetic signals into the observation array and verify the analytic probability calculation.
2. **Physical detectability control:** quantify immediate first-write recovery in both settings. If gains differ greatly, add an input-rescaled sensitivity comparison fixed from development data—not held-out optimization.
3. **Negative control:** absent first write must yield chance balanced accuracy.
4. **Intervention control:** verify \(P d\) and \(Q d\) after replacement against their intended values.
5. **Numerics:** cross-check 12 cases selected before outcomes across settings, signs, phases, and interventions using an independently chosen tighter solver configuration. Include continuous score and state disagreement, not only accuracy.

Use the already qualified small-dose CPU configuration as primary. Benchmark tighter DOP853 and Radau checks rather than imposing the expensive proposed maximum step on every trace.

### Sampling, costs, and branches

Sixteen held-out phases constitute a **decision pilot**, not a high-precision characterization. Across bounded phase-level contrasts, standard error can be substantial; report the actual paired interval. If promising, acquire a fresh 64-phase confirmation cohort under the frozen design. Aim for a five-percentage-point interval half-width, but report wider precision honestly.

Missing or unqualified trajectories are not silently dropped. Report affected phase-block coverage and worst/best-case bounds on the primary contrast; if those bounds change the decision, resolve or classify the pilot as inconclusive.

**Estimated envelope:** 15–120 CPU-hours including development and checks, subject to a measured 12-trajectory benchmark. Start with eight owned workers; increase to twelve only if shared headroom permits. Save sampled outputs and diagnostics, not every internal solver step. Expected storage is well below several GiB. Additional API and rental cost: **$0**.

Prospective branches:

- **Expand:** advantage exceeds five points, with a positive paired interval, and component removal supports transverse dependence.
- **Revise:** memory exists without an edge advantage, or interference erases it. Test matched exponential-memory predictions before geometry claims.
- **Reject this utility claim:** validated immediate detectability but no meaningful delayed benefit.
- **Invalid/inconclusive:** numerical or intervention uncertainty spans the decision. Restrict checks to affected conditions rather than reopening a general arithmetic campaign.

# First 48 hours

**Hours 0–8**

- Implement the two-write schedule and three state operations.
- Freeze phase seeds, matrix, endpoint, readouts, controls, and failure rules.
- Run observation and projection unit controls.

**Hours 8–20**

- Benchmark 12 development trajectories.
- Establish achievable runtime and cross-solver score precision.
- Generate linear predictions and immediate-detectability results.

**Hours 20–36**

- Run the complete 16-phase held-out pilot with bounded CPU workers.
- Run prespecified solver checks concurrently at lower priority.

**Hours 36–48**

- Produce three experimental deliverables: phase-paired primary contrasts; interference/removal contrasts; linear-versus-finite residuals with qualification coverage.
- Take the predefined branch. Do not replace an unfavorable primary endpoint with a favorable window.

# Two-week research roadmap

**Days 3–5:** Complete the fresh 64-phase confirmation only if the pilot supports expansion. If retention survives but interference defeats recovery, test whether a one-mode linear memory predicts that failure quantitatively.

**Days 6–9:** Follow the discriminating result:

- **Useful transverse memory:** begin the four-point continuation pilot.
- **Mainly observer cancellation:** execute the six-phase finite-pulse reversal pilot.
- **Full-linear predictions explain everything:** quantify retention–interference tradeoffs before attempting nonlinear tasks.
- **Structured nonlinear residuals:** run the controlled interaction pilot at qualified doses.

**Days 10–14:** Complete one branch, not all branches. Deliver either a confirmed memory intervention result plus a preliminary contraction relation, a validated reversal mechanism, or a bounded nonlinear-interaction result. Preserve the simpler linear comparison throughout.

**Final TLDR:** The most useful next knowledge is whether persistent transverse state in these shrimp-associated cycles functions as recoverable memory under interference. Whatever the outcome, it will sharpen the distinction between retention, robustness, observation artifacts, and computation. It will remain evidence about this fixed three-neuron dynamical system—not trained networks, biological brains, or language models.
