# Verdict

The reported numerical observations merit preservation, but the proposed explanation is not publication-ready. It conflates zero eigenvalues with zeros of the leading multiplier, assumes spectral dominance absent at key reported points, and treats sampled sign changes as established curve geometry. The flip-crossing claim also conflicts with the stated numerical gate.

Only the draft was supplied. I treat its numerical summaries as disclosed results, not independently verified computations. Resolving these issues requires corrected claims and compact method/result summaries—not raw data.

NOT READY FOR READER REVIEW

# Accuracy and claim findings

**B01 — Blocking: the general mechanism is not established.**  
**Affected:** title, “Shrimp-shaped windows,” proposed explanation.

Several premises require correction:

- The scalar identity \(\mu=\prod_i f'(x_i)\) is correct. But two different phases of a *primitive* period-\(p\) orbit cannot both equal the same critical point \(c\): that would imply a shorter period. A doubly superstable crossing needs additional qualifications, such as distinct critical points.
- Higher-dimensional maps retain the identity
  \[
  \det DF^p(x)=\prod_i\det DF(x_i).
  \]
  Thus dimension alone does not establish that determinant-zero crossings must unfold into avoided crossings.
- The usual Hénon map with nonzero \(b\) has nonzero constant Jacobian determinant; it cannot have a zero multiplier. The introductory sentence must not transfer one-dimensional superstable-crossing properties to that case.
- A nondegenerate saddle of a smooth determinant function, with nonzero saddle value, gives a local hyperbolic zero contour. Neither that saddle nor a small coupling parameter has been demonstrated here. Strong transverse contraction alone does not bound coupling or nonnormality.

**Minimum fix:** replace the causal headline and explanation with a conditional geometric hypothesis. Identify the actual one-dimensional comparator. Treat \(\delta=\epsilon s_a s_b-\kappa\) as an unvalidated, basis/model-dependent parametrization—not a measured mechanism. “Fixed sign” must mean *strictly nonzero* when used to exclude zero eigenvalues.

**B02 — Blocking: the two-mode reduction and “leading zero” terminology disagree with the results.**  
**Affected:** proposed explanation; period-5 and DSS6 summaries.

At the DSS6 minimum, the leading-pair modulus is about \(0.0063\), versus \(0.0051\) for the third multiplier: a ratio of roughly 1.24, not ten. The period-5 “near-nilpotent” point likewise has three comparable moduli.

A smoothly isolated, conjugation-closed spectral cluster can define a two-dimensional invariant subspace; a universal tenfold modulus criterion is neither necessary nor sufficient for numerical reliability. The claimed dominance is nevertheless absent at the central examples.

Moreover, \(\delta=0\) means one eigenvalue of the selected block vanishes. It does **not** mean the full map’s leading multiplier vanishes. If “leading” means largest modulus, its vanishing requires every multiplier to vanish.

**Minimum fix:** distinguish spectral radius, tracked eigenvalue branches, and block determinant throughout. Either establish a smooth, numerically resolved block over the relevant neighborhood or retain a three-mode description without claiming a validated two-mode explanation.

**B03 — Blocking: grid proxies are presented as resolved zero curves.**  
**Affected:** hostile period-7 window; DSS6; proposed reading.

Only 529 of 961 DSS6 grid cells have reported solutions. Nonzero sampled multipliers do not establish absence of zeros between cells or in the remaining 432 cells. Likewise, endpoint modulus ratios do not prove that 38 sign-change edges are dominance exchanges rather than intervening zero crossings, complex-pair transitions, or other branch changes. The seven remaining edges are not confirmed zero crossings either.

The “clean” condition \(|\mu_1|>10|\mu_2|\) excludes complex leading pairs and the principal comparable-modulus region, so its per-step factors cannot independently resolve that region.

**Minimum fix:** say “nonzero at all solved grid cells”; report the 7/38 division as an endpoint-based classification rule unless branch tracking establishes the mechanisms. Label interpolated contours and disclose coverage. Reserve “superstable curve” for demonstrated zeros of a clearly defined quantity.

**B04 — Blocking: the claimed flip intersection fails the stated gate.**  
**Affected:** KITE RESULT; projected kite interpretation.

At \(R\), multipliers \(-0.9952\) and \(-1.0005\) miss \(-1\) by \(4.8\times10^{-3}\) and \(5\times10^{-4}\), respectively—not \(10^{-7}\). Small orbit residuals do not establish a flip. Rounded coordinates could explain this, but that reconciliation is absent.

The reported orbit distance also needs a definition invariant to the six possible cyclic phase shifts. Matching token loops neither establishes nor refutes orbit identity.

**Minimum fix:** provide a compact reconciliation using the actual intersection coordinates and separate orbit/bifurcation residuals, or describe two branches *near* the projected crossing. Define the phase-aligned distance. Do not infer two coexisting attractors or a completed kite from these solves; the second nearby solve failed and the fold continuations remain unfinished.

# Statistical and scope findings

**S01 — Important: numerical accounting and extrema need reconciliation.**  
**Affected:** period-6 census and multiplier summaries.

The listed categories sum to \(1946+945+811=3702\), leaving **522 of 4224 member cells** unaccounted for. Also, “most contracting cell” with modulus \(0.0161\) conflicts with the later minimum \(0.0063\) unless these refer to different searches or definitions.

**Minimum fix:** provide exhaustive categories, denominators, and any overlap rule. Label each extremum by scan, branch, criterion, and date. Replace “tonight” with an identifiable run/version date. These are sampled extrema and grid census counts, not uncertainty intervals or population frequencies.

**S02 — Important: the constructed dynamical system is insufficiently specified.**  
**Affected:** “The system”; cross-window comparisons.

The treatment dimensions are incompletely inventoried. This prevents readers from knowing what was fixed or varied and from interpreting comparisons among windows.

**Minimum fix:** add a compact specification covering:

- checkpoint/revision, prefix/chat template, embedding insertion, and context-reset policy;
- steering-vector identity, construction and normalization; addition site; affected positions; timing, order and repetition;
- layer identities, strength ranges, temperature values/ranges, and grid resolution for each plane;
- seeds, continuation direction, and attractor-observation duration.

Mark dimensions **fixed, varied, matched, or untested**. Explain whether “hostile” describes a validated semantic property or is merely a vector/run label. Bind conclusions to this deterministic expected-embedding feedback system, not ordinary sampled generation or language-model behavior generally.

**S03 — Important: numerical validity claims need compact operational definitions.**  
**Affected:** methods; chaos statement; cusp and certification claims.

“Positive finite-time Lyapunov exponents” is appropriately narrower than rigorous chaos, but the horizon, transient removal, normalization, and decision criterion are missing. Dense versus Ritz-reduced spectra also need an accuracy statement. “Three are certified by interval arithmetic” does not identify which orbits or what was certified.

**Minimum fix:** disclose these settings and summarize eigenpair residuals/truncation checks. For cusps, state the nondegeneracy and numerical acceptance conditions. Identify the three certified orbits, certified proposition, bounds, and rounding/enclosure method—or remove the certification claim. No full logs or activation records are needed.

**S04 — Important: comparisons and missing readouts must remain explicit.**  
**Affected:** three-window comparison and causal interpretation.

The windows differ in steering, period, and parameter plane; they are not matched controls. The period-5 singular-value ratio cannot establish period-6 rank structure. Shared small eigenvalue moduli do not by themselves establish a common mechanism.

**Minimum fix:** add a compact window-by-readout table, marking absent measurements “not reported.” If retaining a mechanistic explanation, compare its predictions with the full computation and a simple uncoupled baseline using the *same* readout and locations; report discrepancy. Otherwise retain the descriptive spectral analogy. No audience expectation or frozen surprise benchmark is supplied, so do not recast these observations as broadly surprising.

# Reference audit

**R01 — Important: source identities and claim boundaries are incomplete.**

There is no bibliography. The following inventories all scholarly source attributions present; none has been independently checked here.

| Attribution | Exact claim it is being asked to warrant | Required action |
|---|---|---|
| **Gallas, shrimp/Hénon work — unidentified** | Hénon shrimp morphology: fold/flip boundaries, legs, doubled shells, and the alleged central crossing of superstable curves. | Supply the exact source and relevant figure/passage. **Required source check:** determine which map and which curves actually support each property. Remove the zero-multiplier attribution for nonsingular Hénon dynamics. |
| **Façanha, Oldeman and Glass, Phys. Lett. A 377, 1264 (2013)** | “Use the trace-zero curves as the skeleton,” with closed Type-A geometry and approaching/turning Type-B arcs, under a fixed-sign determinant assumption. | Verify title, complete citation, determinant assumptions, and the exact Type-A/Type-B definitions. A full two-dimensional map’s trace is not automatically equivalent to the selected leading-pair trace of an 896-dimensional map. Retain only the limited analogy the source supports. |
| **Gallas, kite/tail work — unidentified** | Superstable curves of different attractors meet at tail \(T\) as a “pseudo-intersection,” and flip curves meet at \(R\). | Identify the source and figure, potentially distinct from the shrimp citation. Verify the attribution. The authors’ “our check of his cell” is a new computation requiring its own compact method/result disclosure, not evidence supplied by the historical reference. |

Add a primary checkpoint/configuration citation for Qwen and identify any external certification or normal-form method actually relied upon. A focused bifurcation source may support the *conditional* saddle-unfolding argument; broad chaos or language-model references would not repair the missing validation.

No entry is demonstrably orphaned, but the two Gallas roles may be redundant once identified. Merge them if one source does both jobs; remove historical detail that lacks unique scientific value. No preprint is identifiable from the supplied citations. If one is added, attribute its findings as provisional reports rather than authority.

# Figure-to-text consistency

**F01 — Important: visual claims cannot be checked against a supplied figure.**  
**Affected:** “dark line,” “fold side,” head/kite geometry, \(R\), \(T\), and Type-A/Type-B analogies.

No figures, captions, alt text, or figure receipts were supplied. Consequently, figure-to-text agreement cannot be assessed.

**Minimum fix:** provide a compact labeled parameter-plane figure, or replace visual assertions with explicit contour-construction descriptions. Show solved-cell coverage, unresolved regions, orbit-sheet identity, and the distinction among spectral-radius minima, trace-zero contours, determinant-zero contours, and flip curves. Do not connect unsupported regions as established curves. Define `td` and `ad`.

# What should remain unchanged

- The finite-time qualifier on the chaos observation.
- The explicit statement that a failed Newton solve is **not** evidence of orbit absence.
- The caution that dominant-real-multiplier sign changes can arise from dominance exchanges.
- The observation that the same token loop can accompany distinct continuous states.
- The reported nonzero sampled leading multipliers and comparable small multipliers, with corrected sampling scope.
- The distinction between numerically solved orbits and any genuinely interval-certified subset.

These bounded findings remain useful without a universal avoided-crossing explanation.

# Required revisions

Resolve **B01–B04** before reader review. Address **S01–S04, R01, and F01** with compact disclosures and claim corrections, not expanded audit machinery. If the mechanism remains hypothetical, no new experiment is required merely to preserve the descriptive findings.

A defensible replacement explanation, under 180 words:

> At all 529 solved period-6 grid cells, the reported leading multiplier is nonzero. Complex leading pairs occur at nine cells. Changes in the sign of the dominant real multiplier do not alone identify zero crossings.
>
> For a smoothly tracked, isolated two-dimensional invariant subspace, the eigenvalues satisfy \(\lambda^2-\tau\lambda+\delta=0\). Thus \(\delta=0\) marks a zero eigenvalue of that block, not necessarily the leading eigenvalue of the full return map. If \(\delta\) has a nondegenerate saddle with nonzero saddle value, its local zero contour has hyperbolic branches; complex eigenvalues occur where \(\tau^2<4\delta\).
>
> These facts provide a conditional geometric model, not an established explanation of this window. The third multiplier is comparable in magnitude at the reported minimum, and neither the determinant saddle nor a coupling magnitude has been established. “Type-A-like” and “Type-B-like” currently denote geometric analogies, pending source verification and quantitative comparison, rather than validated classifications.

# Publication checklist

- [ ] Mechanistic title and conclusion corrected.
- [ ] Leading multiplier, tracked branches, and determinant zeros distinguished.
- [ ] Grid coverage, classifications, census, and extrema reconciled.
- [ ] Flip gates and phase-aligned orbit distinction documented.
- [ ] Input/intervention dimensions and numerical criteria specified.
- [ ] References identified and attribution checks completed.
- [ ] Figure claims supported or narrowed.
- [ ] Responsible human publication gate completed.
