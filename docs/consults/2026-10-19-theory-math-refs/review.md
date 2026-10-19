# Verdict

The core derivative factorization, high-temperature contraction argument, and tied network-free energy argument are sound under explicit assumptions. The main problems are the treatment of finite-record Lyapunov diagnostics as bounds, insufficiently defined chaos classification, tied/untied scope, and incompletely specified intervention comparisons.

Numerical results are treated as author-reported summaries, not independently reproduced measurements. The packet contains neither the figure image nor the cited source texts. Most fixes require clarification or narrower claims, not new experiments.

NOT READY FOR READER REVIEW

# Accuracy and claim findings

**B01 — Lyapunov and block bounds: distinguish mathematical bounds from finite-record diagnostics.**  
Affected: “High and low temperature,” especially “the bound tightens as \(k\) grows” and comparisons with the “long-run exponent.”

With Euclidean vector norms and induced spectral matrix norms, the generally valid statement is
\[
\lambda^+(x)=\limsup_{n\to\infty}\frac1n\log\|DF^n(x)\|
\le
\limsup_{n\to\infty}\frac1n\sum_{i=0}^{n-1}\log\|DF(F^i x)\|.
\]
An ordinary long-run average requires its existence. Finite-record block products bound growth over that record, not an independently defined asymptotic exponent. Increasing block length does **not** guarantee monotonic tightening for arbitrary partitions; nested coarsening of the same complete record does.

**Minimum fix:** Define the exponent, norm, block partition, residual-block treatment, and independently estimated reference exponent. State the applicable invariant-measure and integrability assumptions for Kingman, with ergodicity where needed. Separate its asymptotic conclusion from empirical convergence over 32 steps. Preserve the approximately \(1/k\) observation as sample-specific, not a theorem.

**B02 — “Chaos requires…” and “chaotic cells (no period up to 64)” need an operational definition.**  
Affected: opening paragraph, empirical comparisons, figure caption.

Positive largest Lyapunov exponent is not a universal necessary condition for every mathematical definition of chaos. Conversely, failure to detect a period up to 64 does not establish chaos: longer-period, quasiperiodic, and transient trajectories remain possibilities. The existing warning that positive growth alone is insufficient is correct but does not resolve this.

**Minimum fix:** State the article’s operational classification, including exponent criterion, finite observation length, period test, and tolerances. If positive growth was not required, label these cells “no period detected up to 64” rather than “chaotic.” Introduce the theory as addressing positive-exponent dynamics. For the energy argument, say “no nontrivial cycles or sustained nonfixed recurrent dynamics,” retaining the unstable-fixed-point caveat.

**B03 — The covariance and low-temperature statements silently revert to tied embeddings.**  
Affected: Sec. III introduction and low-temperature paragraph.

The displayed coefficient
\[
C=\left\|\sum_{i\in S}(e_i-e_j)(e_i-e_j)^\top\right\|
\]
is correct for tied embeddings at fixed state and steering, with temperature-independent logits and runner-up set \(S\). For untied embeddings it becomes
\[
C=\left\|\sum_{i\in S}(e_i-e_j)(w_i-w_j)^\top\right\|.
\]
Cross-covariance can cancel; the positive-semidefinite covariance interpretation and automatic \(1/T\) readout growth at a tie do not transfer unchanged.

**Minimum fix:** Label the tied derivation explicitly and give the untied replacement. The uniform low-temperature conclusion requires a gap \(\delta>0\) independent of temperature on the invariant sets considered, together with uniformly bounded network Jacobians. Say convergence is “generally non-uniform across changes of maximizing token,” not unconditionally non-uniform. Do not infer small-positive-temperature orbit behavior merely from eventual periodicity of the limiting finite-valued map.

**S01 — Fisher pullback is missing temperature scaling.**  
Affected: Sec. II, “the same covariance becomes the Fisher information.”

For tied embeddings, the input Fisher information is
\[
I_x=T^{-2}J_{\tilde h_\alpha}^{\top}\Sigma_pJ_{\tilde h_\alpha}.
\]
For untied embeddings, its middle factor is \(\operatorname{Cov}_p(w_t)\), not the feedback cross-covariance \(\operatorname{Cov}_p(e_t,w_t)\).

**Minimum fix:** Supply this formula and qualification, or remove the optional Fisher digression. This affects the claimed equivalence between feedback covariance and an information metric.

**S03 — The contraction certificate is not the observed chaos threshold.**  
Affected: “the bound is rigorous but far from sharp” and the numerical \(T_*\) discussion.

The proof establishes a sufficient global-contraction threshold. Its separation from observed chaotic temperatures does not establish how sharp it is for global contraction. Trajectory power-iteration estimates provide estimated lower bounds on \(L(\alpha)\), not a numerical upper certificate establishing contraction at a specified temperature.

**Minimum fix:** Say the certificate “does not locate the measured chaos boundary.” Keep “estimated lower bound” attached to numerical \(T_*\) values; do not imply that \(T=5.6\times10^4\) is certified sufficient.

# Statistical and scope findings

**B04 — Intervention comparisons need a compact specification of matched and unmatched factors.**  
Affected: “Steering makes tokens compete,” particularly the residual/logit-bias comparison and crossover.

The two intervention grids have different sizes and incompletely specified sampling domains. Their cell percentages therefore cannot establish relative efficacy without matching or standardization. A bias along \(Wv\) is also not an equal-dose realization of a residual intervention at layer 12.

**Minimum fix:** Add a compact design table marking factors **matched, varied, fixed, or untested**: model/tiedness, direction identity and norm, residual and bias amplitudes, temperature grid and weighting, layer/position, delivery relative to normalization, repetition/timing, fixed prefix/context, iteration length, burn-in, and precision. Unequal grids may remain descriptive censuses.

For the crossover, define the baseline, four factor combinations, interaction statistic, and cell-selection rule. Mark absent arm-by-readout cells “not measured”; do not infer entropy, Jacobian gain, or chaos incidence across controls. Replace “not derivatives of any actual map” with “assembled tangent products along recorded states, not outcomes of rerun hybrid dynamics.” Preserve the conditional, trajectory-specific readout conclusion.

**S02 — Intervals and mixed results need explicit estimands.**  
Affected: accounting intervals, crossover intervals, and 7B comparison.

“95% interval over 18 temperature rows” does not identify the statistic, weighting, resampling unit, or target of inference. A bootstrap over selected deterministic cells or rows is not automatically a confidence interval for other prompts, steering directions, or models.

**Minimum fix:** Specify the interval construction and intended interpretation. Treat grid percentages as census summaries. Describe the 7B readout estimate \(+0.23[-0.11,0.63]\) as compatible with zero, not established positive growth. One random direction is not a random-direction population. Preserve the network-factor interval spanning zero without converting it into equivalence or invariance.

# Reference audit

No supplied full bibliography entry is orphaned. Four additional keys occur in the supplied Sec. II paragraph without accompanying entries; that is a packet limitation, not proof that the full article omits them.

**R01 — Required live-source check: affine “sharp threshold.”**  
The Wang claim cannot be assessed from its bibliographic entry. Verify the source’s existence, bibliographic details, exact system, assumptions, and whether its threshold concerns local fixed-point stability or global dynamics. If retained, write “a preprint reports…” and name its applicable regime. The minimum alternative is to remove this ancillary sentence and entry.

**R02 — Remove unnecessary citation machinery rather than expanding it.**  
The Fisher paragraph contributes no necessary step to the feedback argument and introduces two unidentified references. Prefer deleting it and those citations; if retained, repair S01, supply complete entries, distinguish each source’s contribution, and attribute provisional reports accordingly.

| Entry or cited key | Exact claim it is assigned to warrant | Disposition |
|---|---|---|
| **Kingman (1973), `kingman1973subadditive`** | “the subadditive ergodic theorem … guarantees convergence as \(k\) grows” | Retain for the appropriately defined asymptotic result. It does not warrant monotonic finite-record tightening, a rate, or sign recovery after twenty steps. Repair B01. |
| **Ramsauer et al. (2021), `ramsauer2020hopfield`** | “the update rule of a modern Hopfield network”; covariance appearing in its Jacobian; the associated energy descent | Retain: this is the essential direct methodological antecedent. Point explicitly to the relevant update/energy result after checking the source. A 2020 preprint key and 2021 proceedings year are not inherently inconsistent. |
| **Wang (2026), `wang2026logitfixed`** | “For affine softmax feedback systems … a sharp stability threshold is known” | R01: verify and narrow, or remove. The supplied metadata alone does not warrant this broad claim. |
| **`wainwright2008graphical` — entry absent here** | “the standard identity for the Hessian of the log-partition function of an exponential family” | Appropriate identifiable mathematical role. Supply the complete entry; check the relevant identity and notation. |
| **`gao2017softmax` — entry absent here** | “the Jacobian of the softmax” | Remove here if it supports only the already displayed derivative identity; it adds no necessary distinct result. Retain only with a specified additional role and complete entry. |
| **`liu2026jacobianscopes` — entry absent here** | Jointly cited for a Fisher metric “used to attribute predictions to tokens and to describe steering geometrically” | Individual attribution is unspecified. Prefer removal with the digression; otherwise source and bibliographic checks are required. |
| **`wang2026fishback` — entry absent here** | The same jointly cited attribution/steering sentence | Same disposition; do not guess which work supports which clause. |

There is no demonstrated need for a longer bibliography. Name the Banach contraction principle and state its hypotheses. The elementary product-norm inequality is already explained. No attention-Lipschitz result is actually invoked, so adding attention-bound citations would be decoration. The packet reveals no specific missing dissenting result that should be guessed into the bibliography. Publication venue alone does not validate an asserted theorem.

# Figure-to-text consistency

**F01 — The caption blurs directional relative gains and matrix factors.**  
Affected: Fig. “Readout and network contributions,” panel (a).

“Mean logarithm of each factor” can be read as a logarithm of an operator norm, whereas the body uses tangent-direction relative gains with state-norm normalization.

**Minimum fix:** Label panel (a) explicitly as mean \(\log g_{\rm read}\) and \(\log g_{\rm net}\), “directional relative gains, not operator norms.” Apply B02’s classification correction to panel (b).

The reported \(101\times53=5353\) count and stated percentages are arithmetically consistent. Caption and body cohort sizes also agree. The actual graphic, axes, plotted intervals, and alt text were not supplied and cannot be checked. Supply the final figure or a compact plotted-value receipt for that check. Retain “23 of 24,” rather than implying universal sign recovery.

# What should remain unchanged

- The derivative factorization is correct; the untied cross-covariance qualification in Sec. II is valuable.
- Under the stated smoothness and compactness assumptions, the diameter bound, its untied Cauchy–Schwarz extension, and the contraction conclusion are valid in the Euclidean operator norm.
- The relative-gain boundary term has the correct sign.
- The tied network-free descent inequality is correct. Compactness, continuity, and vanishing steps justify accumulation only at fixed points; the text appropriately avoids claiming uniqueness.
- Untied network-free cycles really are possible: \(e_1=1,e_2=-1,w_1=-1,w_2=1\) gives \(F(x)=-\tanh(x/T)\), with a nontrivial two-cycle for \(T<1\).
- Preserve the negative findings: one-step norms fail to distinguish the reported classes; overlapping windows are diagnostics; entropy is not established as causal; crossover gains do not establish a counterfactual attractor.
- Keep the derivative check scoped to one state and eight directions—not global instrument validation.

# Required revisions

1. **B01–B03:** Repair exponent definitions, block-bound claims, chaos classification, and tied/untied low-temperature scope.
2. **B04:** Add the compact intervention/control specification and crossover estimand definitions, or narrow comparisons where details are unavailable.
3. **S01–S03:** Correct or remove the Fisher statement; identify interval targets; distinguish contraction certificates from observed chaos boundaries.
4. **R01–R02:** Complete necessary source checks and remove unsupported or redundant citations.
5. **F01:** Correct caption terminology and complete the final visual check.

No raw datasets, trial logs, activation dumps, or source-tree audit are needed for these revisions.

# Publication checklist

- [ ] Mathematical norms, limits, hypotheses, and classifier are explicit.
- [ ] Finite-record diagnostics are not presented as asymptotic certificates.
- [ ] Intervention levels, matching, and missing readouts are disclosed.
- [ ] Census summaries and interval interpretations remain distinct.
- [ ] Untied and 7B conclusions retain their narrower scope.
- [ ] Necessary references are identifiable and claim-matched.
- [ ] Final figure and alt text agree with the corrected definitions.
- [ ] No unsupported semantic, behavioral, consciousness, or surprise claim is introduced.
- [ ] The responsible human approves the revised publication packet.
