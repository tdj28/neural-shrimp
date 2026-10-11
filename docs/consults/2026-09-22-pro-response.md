# Research opportunity and what the evidence leaves open

**The strongest opportunity is to determine whether this recurrent system itself combines successive inputs nonlinearly, or whether it supplies fading amplitudes that a nonlinear observer interprets. I recommend a small, factorial two-write experiment with a strictly linear readout and first- and second-order dynamical controls. Defer EXP-023.**

This changes the scientific question without discarding the numerical work. It could establish a meaningful connection between attractor dynamics and computation—or show that the current assay’s apparent complexity resides chiefly in the decoder.

## Diagnosis

Taking the supplied results as disclosed evidence, you have learned something real: **failure of one readout does not establish erasure of information.** Interfering inputs can become distinguishable through a different observer. The late agreement of full and one-mode decoders also makes a low-dimensional effective description plausible, although it does not establish scalar dynamics.

Several other findings are expected dynamical-systems behavior: weaker transverse attraction permits longer persistence of small perturbations; stable cycles retain a timing direction; input phase changes observability. Your experiment makes these properties concrete, but neither their existence nor the shrimp’s appearance establishes a general computational principle.

The current bottleneck is primarily **scientific framing**, followed by external validity—not insufficient numerical precision. The completed gates justify moving to a different question, with numerical checks embedded in that experiment. EXP-022’s missing matches were a sampling gap, not a scientific null.

Keep three objects separate:

- **State dynamics:** what inputs do at fixed weights.
- **Parameter-space organization:** how attractors and functionality change with weights.
- **Training or iterative-inference dynamics:** how an optimization or inference procedure evolves.

Your evidence directly concerns the first and partially characterizes the second. It says nothing yet about the third.

Retire these interpretations:

1. “Near-chance linear decoding means the bit was erased.”
2. “A small transverse multiplier implies robustness to weight changes.”
3. “A nonlinear decoder succeeding demonstrates nonlinear computation by the recurrent dynamics.”
4. “Historical continuation nodes are independent networks.”
5. “A shrimp-shaped parameter window explains iterative reasoning.”

The ambitious but reachable question is narrower: **when does a recurrent trajectory create a useful cross-input feature, rather than merely preserve separate input traces?**

# Ranked experimental directions

## 1. Nonlinear input mixing versus nonlinear interpretation

**Highest scientific value per unit cost; recommended next.**

### Question and discriminating hypothesis

Can two signed writes generate a linearly accessible product feature \(s_1s_2\) in the subsequent **state trajectory**?

The hypothesis is that state-dependent recurrent dynamics create a measurable interaction beyond the first-order time-varying response. The strongest competitor is an additive fading-memory system plus a nonlinear decoder—or merely a nonlinear observation function, \(\tanh(x)\).

This adds information absent from EXP-021: it tests where the nonlinearity resides.

### Minimum experiment

At the two established cycles, apply all four sign combinations of two fixed-direction writes, at times \(0\) and \(T\), with amplitudes \(10^{-4},10^{-3},10^{-2}\). Use eight new development phases and 32 independently sampled, sealed test phases, shared across settings.

Observe 32 equally spaced state samples in each of two windows:

- early: \([T,1.25T]\), immediately after the second write;
- later: \([1.75T,2T]\).

The writes use reference-defined directions and externally scheduled times—not directions or event times recomputed from the perturbed trajectory. Otherwise the input controller itself could introduce interactions.

Compare:

1. Full nonlinear ODE.
2. First-order variational response around the unforced cycle.
3. Second-order response, including the cross-write term.
4. Second-order response with that cross term removed.

Use \(x\) as the primary observation; retain \(\tanh(x)\) as a secondary observation. This separates recurrent state mixing from a nonlinear output transformation.

### Controls, metric, and decision

The cheap baseline is the constant parity predictor, with \(R^2=0\). The strongest mechanistic baseline is the full time-varying first-order response, not a scalar exponential.

The primary endpoint is held-out noise-averaged parity-regression \(R^2\), in dimensionless units, using a **fixed linear observer constructed from independently computed second-order reference responses**. Define the primary cell prospectively: near-edge setting, amplitude \(10^{-2}\), early window. Other cells establish scaling and specificity.

The positive control adds a known synthetic parity component to additive reference observations, orthogonal to the two write templates, with analytically known recoverability. The negative control is the first-order system: balanced additive writes have zero covariance with parity, regardless of their persistence or orientation.

Expand if primary \(R^2>0.05\), its phase-sampling interval excludes zero, and direct trajectories exhibit a cross-input state component. Revise the observer—not the hypothesis—if an explicitly labeled oracle linear ceiling is useful but the reference observer fails. Abandon this small-signal mixing route if the oracle ceiling remains below 0.01 throughout the tested range.

A positive result supports finite-amplitude mixing at these settings, not shrimp-specific computation. A null supports an additive-memory account within the tested amplitudes and schedule. Activation-only mixing implicates the observation nonlinearity. Solver-sensitive or control-failing results are invalid, not negative.

**Resources:** approximately 960 full trajectories plus reference sensitivity solves. Estimated minutes to several hours on four CPU workers, with an initial eight-hour wall-time cap; actual timing must come from a small batch. No GPU, API, or rental dependency.

---

## 2. Noise–memory tradeoffs and phase-amplitude reduction

**Best alternative if direction 1 finds little useful mixing.**

### Question and hypothesis

Does weak attraction improve memory once process noise accumulates, or does it preserve signal and noise together?

The hypothesis is that deterministic persistence overstates useful noisy memory. The strongest competitor is that favorable transient transport or observability preserves a genuine signal-to-noise advantage.

Use the same two anchors and a phase-amplitude or full linear time-varying stochastic surrogate:

\[
dx=f(x)\,dt+\sigma_p\,dB_t.
\]

Specify this as additive isotropic Itô noise in state coordinates. It is not the earlier iid measurement-noise model.

### Minimum experiment and controls

Start with one balanced write to isolate retention before returning to interference. At eight development and 24 test phases, simulate paired positive/negative writes at three process-noise levels, with 16 Brownian paths per phase and level. Choose the nonzero levels on development phases to span modest and substantial loss of detectability.

Compare the nonlinear system, full linear stochastic propagation, and a phase-plus-one-transverse-mode reduction. An exact scalar Ornstein–Uhlenbeck memory channel is the cheap baseline and positive control for covariance growth and decay. Zero process noise is the negative control.

The primary metric is change in binary accuracy, in percentage points per phase, at \(2T\), using sealed reference likelihoods; report model likelihood validity separately. Independent Brownian paths estimate stochastic variability, while phases define the conditional phase-average target. The two anchors are not a network population.

Use development timing and variance estimates to decide whether 16 paths suffice; expand only if the confidence interval is too wide to resolve a five-point difference. Failures to retain bounded trajectories are reported separately, not silently discarded.

### Interpretation and feasibility

If the full stochastic linear model explains the result, the general lesson is a controllability–observability/noise tradeoff, not exotic computation. Failure of a phase-plus-one-mode approximation but success of the full linear model would isolate transient multimode transport. Failure of both, with converged simulations, motivates nonlinear noise coupling.

A null anchor difference does not imply equal robustness to every noise direction. Test anisotropic noise only after the isotropic result warrants it.

**Resources:** roughly 4,600 trajectories for three levels and two signs, approximately 4–24 CPU-hours as an estimate. Cheapest pilot: four phases, eight paths, one nonzero noise level. Use coupled Brownian increments for step-halving checks. No accelerator is needed.

---

## 3. Weight-perturbation robustness, rather than more matched-cycle acquisition

**Addresses the shrimp question directly, but costs more and has weaker initial explanatory resolution.**

### Question and hypothesis

Do neighborhoods inside the periodic window preserve functionality under weight perturbations better than near-boundary neighborhoods?

The hypothesis concerns the **volume of functioning parameter space**, not state contraction. Its strongest competitor is ordinary distance to a bifurcation boundary, with no special role for shrimp organization.

### Minimum experiment

Sample 24 independent perturbation directions in the \((\alpha,\gamma)\) plane around each anchor. Use three development-chosen radii, then freeze them. At each perturbed weight pair, test cycle retention and a functional endpoint already established in direction 1 or 2.

The independent sampling units are perturbation vectors. Multiple phases, radii, integration seeds, and corrector attempts are repeated measurements or diagnostics—not new networks.

Primary endpoint: fraction of perturbation directions retaining at least 80% of the anchor’s chosen task score, with all failed attractor/function cases included in the denominator. Report period retention separately.

Compare against:

- local Floquet/bifurcation sensitivity predictions;
- the cheap baseline of Euclidean weight distance;
- anchor label alone.

Zero weight change is a negative control. A numerically qualified, deliberately large perturbation that changes the attractor is a positive control; if no such setting is available, establishing it is a bounded feasibility test.

### Interpretation and feasibility

Better retention would establish local parameter robustness, not complete basin geometry or a unique shrimp mechanism. Equal retention would weaken the core-versus-edge narrative. Mixed topology/function results would distinguish structural stability from utility. Unresolved cycle classification limits the attractor endpoint, but need not invalidate directly measurable task performance.

Expand only if robustness differences exceed ten percentage points or a local sensitivity model makes meaningfully better held-out predictions. Otherwise defer broader mapping.

**Resources:** about 144 perturbed parameter settings, plus bounded reference correction and task trials; estimated 10–40 CPU-hours. Cheapest pilot: eight directions and two radii. Its real dependency is having a task worth preserving—not finishing EXP-023’s full matching ladder.

---

## 4. Transfer to independently initialized or trained recurrent networks

**Highest external-validity ceiling, but premature as the immediate experiment.**

### Question and hypothesis

Does the predictive mechanism from directions 1–2 recur across larger recurrent systems?

A credible bridge is not “larger networks contain shrimp.” It is: **local response geometry and cross-input terms predict memory or nonlinear feature formation beyond spectral radius alone.**

### Minimum experiment

Use 12 development and 24 held-out random seeds for small, 32-unit continuous-time tanh reservoirs with bounded trajectories. First test untrained reservoirs; train only a simple readout. Introduce recurrent training later as a distinct experimental manipulation.

Measure the same factorial interaction and linear parity endpoint, alongside first-order transport and second-order predictions. Network seed is the independent unit; phases or input schedules are nested observations.

Strong baselines are the local linear response, a scalar memory predictor, and spectral radius/contraction summaries. A purely linear recurrent network is the negative control. A hand-constructed recurrent system with an explicit bilinear feature is the positive control.

Primary metric: held-out error in predicting network-level parity \(R^2\), with networks—not trials—as the denominator. Expand if the mechanistic predictor improves prediction error by at least 20% over the strongest simple baseline, with seed-bootstrap uncertainty excluding no improvement.

### Interpretation and feasibility

Success supports transfer of a response-level principle. It does not connect to optimizer dynamics, language-model reasoning, or spontaneous emergence through training. Failure to predict held-out networks despite working controls would directly undermine this proposed bridge.

**Resources:** CPU first; estimated 4–20 hours for a bounded small-network pilot. The RTX can later accelerate batched training, but needs workload-specific validation rather than reuse of the unsuccessful CUDA ODE path. No paid resources are necessary.

# Recommended next experiment

## Replace EXP-023 with a sealed two-write interaction pilot

EXP-023 is not wrong; it answers a lower-priority feasibility question. Its matching gates risk consuming effort before establishing what task-relevant phenomenon deserves matching. Preserve it, but **defer its controller, six-cycle acquisition, and the proposed 972-trace follow-on**.

### Experimental matrix

| Factor | Levels |
|---|---|
| Fixed weights | Core; qualified near-edge |
| New phase contexts | 8 development; 32 sealed test |
| Pulse amplitude | \(10^{-4},10^{-3},10^{-2}\) |
| Write signs | \(++,+-,-+,--\) |
| Write times | \(0,T\) |
| Observation windows | Early and later windows above |
| Dynamical models | Full ODE; first order; second order; cross-term-ablated second order |
| Observation | Primary \(x\); secondary \(\tanh(x)\) |

Both windows come from the same trajectory. Thus the full-ODE matrix contains \(2\times40\times3\times4=960\) trajectories—not 1,920 independent observations.

Initial phase, reference baseline, setting, and write schedule are known to the observer. Signs are unknown. Measurement noise is iid Gaussian with SD \(10^{-4}\) **in state-coordinate units** for the primary assay. This differs from the previous activation-coordinate noise model, so absolute scores should not be compared as identical tasks.

### Why parity regression is decisive

For any additive response

\[
Y=\mu+s_1A+s_2B+\epsilon,
\]

balanced signs imply \(\operatorname{Cov}(Y,s_1s_2)=0\). No affine predictor can beat the constant predictor in population squared error.

Do **not** use linear-classifier XOR accuracy as the decisive test: asymmetric thresholding can produce misleading above-chance accuracy on finite XOR arrangements.

For each phase, construct the reference expansion

\[
Y\approx m+s_1A+s_2B+s_1s_2C,
\]

including second-order sign-independent shifts in \(m\). Obtain these quantities from first- and second-order sensitivity equations, not finite-pulse test class means. Fix the linear minimum-MSE observer implied by this reference model and the specified noise.

Evaluate it on direct trajectories:

\[
R^2
=1-\frac14\sum_{s_1,s_2}
\left[
\big(s_1s_2-w^\top(\mu_{s_1s_2}-m)\big)^2
+\sigma^2\|w\|^2
\right].
\]

This integrates measurement noise analytically. It avoids simulated-noise sampling and avoids a nonlinear decision rule.

Also measure the direct factorial contrast

\[
C_{\rm direct}=\tfrac14(Y_{++}-Y_{+-}-Y_{-+}+Y_{--}).
\]

For a local second-order mechanism, its magnitude should scale approximately as amplitude squared. Contrast alone establishes interaction, not useful recoverability; that is why \(R^2\) is primary.

### Mechanism probe

The most useful intervention is **cross-term ablation in the second-order response model**, paired with the simpler full-versus-first-order comparison.

Agreement with the second-order prediction, amplitude scaling, and loss of predicted parity under ablation would locate the effect in local nonlinear response. This is a mathematical mechanism test, not a claim that the same term can be independently removed from the physical ODE without other consequences.

### Sampling, uncertainty, and stopping

Draw held-out phases uniformly and independently; share them across settings for paired comparisons. Bootstrap phases for the phase-average interval. These intervals describe phase sampling at two fixed networks—not generalization across weights.

Thirty-two phases are a modest pilot, not a guarantee of precision. If the interval straddles the prospective 0.05 usefulness threshold, add at most 32 newly sealed phases. No further expansion is automatic.

Record finite negative \(R^2\) values without clipping. Numerical failures receive no favorable imputation: retry once under the predefined tighter solver configuration, report the denominator, and require conclusions to survive conservative treatment of unresolved cases. Nonperiodic post-write behavior is not automatically invalid; it limits the local-cycle mechanism claim.

### Implementation and resource envelope

1. Generate reference sensitivities, including second derivatives of the flow and observation.
2. Check them against finite differences on development phases only.
3. Run the additive negative and injected-parity positive controls.
4. Time a 32-trajectory batch.
5. Freeze tolerances, observer construction, and primary endpoint.
6. Run the sealed matrix; evaluate noise expectations analytically.

**Estimated envelope:** four single-thread workers, roughly 1–12 CPU-hours, well below available RAM if only sampled outputs and compact diagnostics are retained. Use an eight-hour wall-time pilot cap; investigate unexpectedly slow integration rather than launching a larger batch. No API calls or GPU work.

# First 48 hours

**Hours 0–6:** Implement the factorial design, fixed write directions, state observations, and analytic linear-risk evaluator. Define the primary cell and seal test phases.

**Hours 6–12:** Run development-only sensitivity checks and controls. The synthetic parity control should recover its analytically specified \(R^2\); the additive system should have zero optimal parity covariance. Resolve only failures material to those checks.

**Hours 12–24:** Run development trajectories, benchmark throughput, and freeze the implementation. If second-order reference construction remains unreliable after this bounded effort, retain the direct factorial-contrast experiment and explicitly downgrade the predictive-observer claim; do not substitute fitting test means as “validation.”

**Hours 24–40:** Execute held-out trajectories and compute primary risk, interaction scaling, state-versus-activation contrasts, and model discrepancies.

**Hours 40–48:** Produce one substantive result table and three plots: \(R^2\) versus amplitude, contrast versus amplitude, and predicted versus observed interaction across phases. Select the predefined branch below.

# Two-week research roadmap

### Days 3–5: Resolve the first scientific branch

- **Useful state mixing:** Test whether second-order dynamics predict it and whether cross-term ablation removes the prediction.
- **Interaction without useful readout:** Quantify the noise-normalized ceiling; distinguish weak computation from observer failure.
- **Activation-only mixing:** Treat static observation nonlinearity as the explanation, not recurrent mixing.
- **No useful mixing:** Move to direction 2. Do not launch a larger shrimp atlas.
- **Invalid controls:** Spend one bounded repair cycle, then use direct contrasts and first-order comparisons to retain a narrower valid experiment.

### Days 6–10: One targeted extension

If mixing is useful, add **one new write separation, \(T/2\)**, with the observer construction unchanged. This tests whether the result reflects a general response mechanism or one favorable timing arrangement.

If mixing is negligible, run the cheapest process-noise pilot and test whether full linear stochastic propagation predicts signal-to-noise loss.

### Days 11–14: Test portability, not visual resemblance

For a clear mechanism, run the development portion of the small-reservoir experiment. Proceed to held-out seeds only after controls pass. A failure of response-based predictions across those networks is a substantive negative result for the bridge.

The strongest plausible statement you could earn is:

> Near a recurrent attractor, successive inputs can create linearly accessible interaction features whose strength is predicted by local nonlinear response—not merely by memory duration or a nonlinear decoder.

If negative, the useful update is equally clear: this assay is principally a fading-memory and observation problem, directing effort toward noise, transport, and robustness rather than an unsupported computational interpretation.

**TLDR:** The shrimp tells you where certain trajectories exist, not what they compute. Next ask whether the trajectory itself combines two inputs. A small controlled answer—positive or negative—will teach more than another matched set of cycles.
