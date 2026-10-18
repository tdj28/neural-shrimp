# Verdict

The derivative factorization, high-temperature contraction argument, and Hopfield energy inequality are sound. The must-fix issues are the unconditional low-temperature language, the interpretation of the logit-bias intervention, and the mechanistic conclusion drawn from trajectory-conditioned measurements.

This review covers the supplied excerpts. Numerical measurements are treated as reported, not independently verified. The figure, bibliography, referenced tables, and supplement were not supplied.

**NOT READY FOR READER REVIEW**

# Accuracy and claim findings

**B01 — Blocking: definition of chaos and low-temperature exclusion.**  
Affected: opening, “Too hot or too cold,” and intermediate-temperature conclusion.

- **“Chaos means that two nearby states drift apart…”** is not a sufficient definition; positive exponents can occur at unstable fixed points, and perturbations need not grow every step.

  Replace with:
  > “A positive largest Lyapunov exponent describes average exponential growth of infinitesimal perturbations in a leading tangent direction. To first order, one step maps a perturbation to \(DF\,\delta x\); finite-time growth alone does not establish a chaotic attractor.”

- **“rules out chaos at very high and at very low temperature”** overstates the later, correctly conditional argument.

  Replace with:
  > “proves global contraction at sufficiently high temperature and excludes positive Lyapunov exponents at sufficiently low temperature on invariant sets retaining a temperature-independent positive logit gap, at fixed steering strength.”

  Replace the paragraph heading with:
  > “High-temperature contraction and the low-temperature limit.”

- **“decays like \(Ce^{-\Delta/T}/T\)”** is correct at a fixed state with a unique winning token, with a geometry-dependent prefactor. Make that precise:
  > “At a fixed state with unique winning token \(j\), \(\|\Sigma_p\|/T=e^{-\Delta/T}(C+o(1))/T\), where \(C=\|\sum_{i:z_j-z_i=\Delta}(e_i-e_j)(e_i-e_j)^\top\|\). The prefactor depends on the geometry and multiplicity of the next-logit competitors.”

- **“tends to the … greedy map” / “every orbit … eventually repeats”** needs tie handling:
  > “Away from ties, the limit is the greedy embedding map. A greedy map with a fixed deterministic tie rule has finite image, so every orbit eventually repeats. At ties, however, the softmax limit averages the tied embeddings rather than selecting one token. Convergence is not generally uniform near tie boundaries.”

  The claimed \(1/T\) gain near ties additionally requires distinct tied embeddings.

- Replace **“Chaos therefore lives in between”** with:
  > “The observed chaotic bands lie at intermediate temperatures; the low-temperature argument does not give a global exclusion near ties.”

**S01 — Important: Lyapunov and high-temperature bounds.**  
Affected: first two temperature paragraphs.

The Lyapunov bound is correct with the appropriate limit:
\[
\lambda_{\max}\leq\limsup_{n\to\infty}\frac1n\sum_{j=0}^{n-1}\log\|DF(x_j)\|.
\]
But **“on a substantial part of the attractor”** does not follow. Replace with:
> “A positive largest exponent requires a positive limiting upper average of \(\log\|DF\|\), although that condition is not sufficient.”

The Popoviciu bound, contraction threshold, and untied replacement by \(\operatorname{diam}(E)\operatorname{diam}(W)\) are correct in Euclidean operator norm. The untied bound follows from scalar covariance Cauchy–Schwarz and the two variance bounds. Finiteness of \(L(\alpha)\) uses continuity of the Jacobian as well as compactness.

**“its single fixed point attracts every orbit”** and **“forgets where it started”** are correct above the threshold: every state enters \(K\), then Banach contraction applies.

Replace **“\(\operatorname{diam}(E)=1.2109\) exactly”** with:
> “\(\operatorname{diam}(E)\) is 1.2109 to the displayed precision.”

The fixed-start sweeps do not independently establish global uniqueness. Replace the empirical phrase **“the unsteered map already has a single attracting fixed point”** with:
> “the fixed-start unsteered trajectories already converge to a fixed point.”

Supply the compact temperature comparison underlying “three to four and a half decades”; it cannot be checked here.

**S02 — Important: product norms and Kingman.**  
Affected: block-bound paragraph.

The explanatory idea is right, but the relevant alignment is between successive output and input directions. Replace **“the direction that one step stretches most…”** with:
> “Single-step norms can overestimate product growth because the direction emerging from one step need not align with a maximally stretched input direction of the next.”

Kingman does not provide a \(1/k\) convergence rate. Replace **“roughly as \(1/k\), as the subadditive ergodic theorem allows”** with:
> “approximately as \(1/k\) in this sample. Kingman’s subadditive ergodic theorem supports asymptotic convergence under its hypotheses, but supplies no such finite-block rate.”

Keep the reported numerical overshoots unchanged. Replace **“Blocks of about twenty steps thus locate the chaos”** with:
> “In these 24 selected cells, blocks of about twenty steps largely recover the sign of the long-run estimate along the recorded orbit.”

**S03 — Important: Hopfield conclusion and untied embeddings.**  
Affected: “Without the transformer.”

The gradient representation is correct; explicitly give
\(\phi(x)=T\log\sum_t e^{e_t^\top x/T}\).
The energy inequality, lower bound on compact \(K\), and conclusion that step lengths approach zero are correct.

Step shrinkage alone is not a general convergence theorem. Replace **“such a map has neither cycles nor chaos”** with:
> “Every accumulation point is fixed, and strict energy descent excludes nontrivial periodic or recurrent chaotic dynamics. This does not exclude positive tangent exponents at unstable fixed points.”

The baseline excludes those dynamics for the tied-embedding identity-network map, not for every conceivable non-transformer system.

Replace **“already has a non-symmetric derivative and can cycle”** with:
> “need not have a symmetric positive-semidefinite derivative, so this energy proof no longer applies; cycles are possible even without a network.”

“Can cycle” is valid in principle, not demonstrated for the supplied untied matrices. For example, \(E=(1,-1)^\top\), \(W=-E\) gives \(F(x)=-\tanh(x/T)\), with a nontrivial two-cycle for \(0<T<1\), despite its scalar derivative being symmetric.

**B02 — Blocking: covariance explanation and logit-bias intervention.**  
Affected: opening, second steering measurement, caption (b).

Replace **“The covariance … says how much …”**, **“large when several tokens compete”**, **“vanishes when … almost all”**, and **“\(1/T\) amplifies both”** with:
> “For tied embeddings, \(\Sigma_p/T\) is the Jacobian of the expected embedding with respect to the final normalized hidden state. \(\Sigma_p\) measures directional embedding variance: competing tokens contribute only insofar as their embeddings differ in relevant directions. It tends to zero as one token’s probability tends to one. The scalar \(1/T\) multiplies the composed derivative once.”

Entropy alone does not determine covariance magnitude or directional gain.

**“\(Wv\), the steering vector’s direct effect on the logits”** is incorrect for layer-12 residual steering through subsequent nonlinear layers. Replace everywhere with:
> “\(Wv\), a constructed logit-space direction obtained by applying the output matrix to the residual steering vector.”

Replace **“changes only which tokens compete at the output”** with:
> “At a fixed state, a state-independent logit bias changes output probabilities and the readout covariance while leaving the unsteered network Jacobian unchanged. Its resulting orbit can nevertheless visit different states.”

**S05 — Important: scope of regularity claims.**  
Affected: Section II, “State space and regularity.”

The dense-model argument is sound, but hard top-\(k\) routing can introduce discontinuities. Compactness alone then does not establish the stated Brouwer or invariant-measure conclusions.

After the mixture-of-experts sentence, add:
> “The continuity and differentiability arguments below apply to the dense model described here; hard-routed models require separate checks at routing boundaries.”

For Oseledets, specify the non-invertible-cocycle formulation, allowing exponents equal to \(-\infty\).

# Statistical and scope findings

**B03 — Blocking: accounting and frozen-path evidence do not identify the claimed mechanism.**  
Affected: three-measurement interpretation and conclusion.

Replace **“Their logarithms … add up to the largest Lyapunov exponent up to a boundary term”** with:
> “For a propagated tangent perturbation over \(n\) steps, their mean logarithms sum to its finite-time growth rate plus \((\log\|x_0\|-\log\|x_n\|)/n\), provided the normalization norms are nonzero. Estimating the largest exponent additionally requires appropriate tangent alignment.”

The readout statistic is a **relative directional gain**, not an operator-norm contraction test.

For the crossover, specify the four Jacobian products, reference contrast, and interaction formula. Replace **“steering only the readout raises the growth”** with:
> “using the steered readout factor in the frozen-path Jacobian product raises its finite-time growth”

and make the analogous network replacement. These hybrid products need not be Jacobians of autonomous maps or describe downstream hybrid trajectories.

Replace the concluding **“steering makes the loop chaotic mainly by making next tokens compete…”** with:
> “Readout-side changes dominate these particular growth comparisons along recorded Qwen2.5-0.5B orbits. The accounting and frozen-path crossover do not isolate increased entropy as the mediator or show that a readout-only hybrid sustains the same attractor. The measured readout gain remains contractive in the three highest temperature rows.”

Apply the same limitation to the opening and caption headline. Preserve all reported effects and intervals.

**S04 — Important: grid denominators, intervention inventory, and uncertainty.**  
Affected: measurement descriptions and caption.

**“3.5 % of the same temperature range”** uses the wrong denominator. Replace with:
> “On the logit-bias grid, 3.5 % of cells were classified as chaotic (241 of 6837), compared with 6.2 % on the residual-steering grid.”

These are grid fractions, not fractions of temperature range or matched causal effect sizes. The arithmetic agrees with the reported counts.

Add a compact design/readout table identifying:

- **Fixed:** prefix, token positions, one-token feedback format, start state, and the disclosed layer/direction for the 0.5B experiment.
- **Varied:** temperature, steering amplitude, and residual versus logit delivery.
- **Specify:** amplitude grids and normalization, per-step application, burn-in and measurement horizons, classification rule, crossover directions, and model-specific layers.
- **Untested unless documented:** other prompts, initial states, positions, durations, and delivery schedules.

Record which arms actually contain entropy, factor-growth, crossover, and chaos-classification readouts. Do not imply that the logit-bias arm independently confirms the entropy/accounting results.

For every interval, state its construction, resampling unit, and aggregation weighting. These selected-grid intervals are not automatically population confidence intervals. The supplied text does not establish matching beyond temperature comparisons.

# Reference audit

**R01 — Important: bibliography absent; source checks required.**  
Only four citation keys are visible. This inventories those keys, not an unseen complete bibliography.

| Citation key | Exact claim it supports; disposition |
|---|---|
| `oseledets1968multiplicative` | “its Lyapunov exponents exist at almost every point of every invariant measure.” Retain; source-check applicability to non-invertible derivatives and potentially \(-\infty\) exponents. |
| `eckmann1985ergodic` | Currently supports the same existence sentence. Remove unless it supplies a specifically identified explanation or theorem version not already supported by the primary citation. |
| `kingman1973subadditive` | “the bounds close on the exponent roughly as \(1/k\), as the subadditive ergodic theorem allows.” Retain only for the corrected asymptotic statement, not a rate claim. |
| `ramsauer2020hopfield` | “the update rule of a modern Hopfield network whose stored patterns are the token embeddings.” Retain for methodological attribution; the energy argument should remain established in the text. |

Required source checks: actual entries, author/title/year/version accuracy, theorem passages, and publication status. Do not infer preprint status or an archival venue from citation keys. Attribute preprint-only empirical claims as provisional; venue does not validate the algebra.

Popoviciu’s named inequality needs either an identifiable source or a short derivation. Architectural claims about dimensions and tied/untied matrices need an identifiable model/configuration source. No additional thematic bibliography is warranted merely for breadth.

# Figure-to-text consistency

**F01 — Important: caption overstates comparator and scope.**

- Caption (a), **“Without steering”** → **“In nearly unsteered cells (\(|\alpha|\le0.01\))”**. This comparator is not exactly zero steering.
- Use **“relative directional readout gain”** rather than implying global contraction/expansion.
- Replace **“Why steering makes the loop chaotic”** with **“Readout and network contributions to growth under steering.”**
- Caption (c) says **14 per direction**; the body says **14 chaotic cells** and discusses philosophy only. Reconcile directions, sample units, and the separate 7B scope without inventing counts.
- The actual figure and alt text are absent. Supply the rendered figure or a compact receipt; curve values, intervals, and visual denominators cannot presently be checked.

# What should remain unchanged

- The derivative factorization and cross-covariance generalization.
- The rigorous sufficient high-temperature bound and disclosure that sampled Jacobian gains are lower estimates.
- The energy inequality and shrinking-step argument.
- The negative network-only crossover result, positive interaction, and high-temperature exceptions.
- The distinction between an orbit measurement and a forecast from weights.
- The warning that frozen-path comparisons do not establish persistence of an attractor.

# Required revisions

1. Resolve **B01–B03** before circulation; retain the measured numbers while correcting their interpretation.
2. Implement **S01–S05**, especially the grid denominator and compact protocol/readout inventory.
3. Complete **R01** using actual references and **F01** using the figure or compact receipt.
4. Do not request new raw-data exports or large logs: concise definitions, summaries, and source checks suffice for this review.

# Publication checklist

- [ ] Mathematical exclusions state their hypotheses.
- [ ] Finite-time growth, relative gain, and actual trajectories remain distinct.
- [ ] Comparators, grids, missing readouts, and interval units are explicit.
- [ ] Caption and figure agree with the bounded conclusion.
- [ ] References and architectural attributions pass source checks.
- [ ] Responsible human author approves publication.
