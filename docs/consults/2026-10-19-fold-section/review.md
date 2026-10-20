# Verdict

The local saddle-node account is mathematically plausible, and the prospective failures are reported unusually well. The principal defect is the conflation of **finite-channel passage fractions** with **effective prefactors inferred from an asymptotic window law**. The local normal form also does not, by itself, determine global reinjection or window locations.

This review covers the supplied Section V, tables, and figure descriptions—not omitted images, supplementary material, or external sources. Numerical results are treated as author-reported computations; arithmetic consistency checks are not independent replication.

Three blocking findings require correction before posting. None requires raw-data submission.

**NOT READY FOR READER REVIEW**

# Accuracy and claim findings

**B01 — Blocking: distinguish the local map, its truncation, and its continuous approximation.**  
**Affected:** “The fold,” “Passages,” “Prospective tests,” and the limits paragraph.

The signs are correct for the reported \(a,b<0\): \(v=-u\) gives
\[
v'=v+A\epsilon+Bv^2,\qquad A=|a|,\ B=|b|.
\]
Near the fold, the stable branch has positive \(u\), negative \(v\), and
\[
\mu_s=1-2\sqrt{ab(\alpha_c-\alpha)}+O(\alpha_c-\alpha).
\]
Consequently, the stated multiplier-limit check is correct.

However, the displayed normal form is a **quadratic truncation**, not the exact projected network map. Add its remainder, ordinarily \(O(u^3,\epsilon u,\epsilon^2)\), and state the simple-\(+1\), nondegeneracy, and remaining-spectrum conditions. Specify eigenvector normalization and coordinate construction: ordinarily
\[
a=\ell^\top G_\alpha,\qquad
b=\tfrac12\ell^\top D^2G[q,q],\qquad \ell^\top q=1,
\]
with a separate normalization fixing \(q\)’s scale.

For \(\epsilon>0\), define \(\kappa=\sqrt{ab\epsilon}\). The missing finite-channel formula is
\[
n_{\rm cont}(v_{\rm in},U)=
\frac{\arctan(BU/\kappa)-\arctan(Bv_{\rm in}/\kappa)}{\kappa}.
\]
Thus \(n_{\rm full}=2\arctan(BU/\kappa)/\kappa\), approaching \(\pi/\kappa\) only when \(U\gg\sqrt{A\epsilon/B}\).

The logarithmic correction has the stated **positive sign**, but is a leading near-identity approximation:
\[
n_{\rm disc}-n_{\rm cont}\approx
\tfrac12\log\!\frac{h(U)}{h(v_{\rm in})},
\quad h(v)=A\epsilon+Bv^2.
\]
It needs slow step variation, higher-order qualifications, and integer exit/overshoot effects; it is not an exact count formula. Avoid reusing \(f\), already the passage fraction.

**Minimum fix:** add these distinctions and replace “the passage formulas integrate the truncated normal form” with wording separating discrete iteration from continuous integration. This protects every “predicted passage” claim.

**B02 — Blocking: the three \(f\) values are not established estimates of one common quantity.**  
**Affected:** window-law derivation, Table `fraction`, Figure `windowlaw`, opening mechanism claim.

The algebra
\[
d_NN^2=f^2\pi^2/(ab),\qquad
\text{slope}=\sqrt{ab}/(\pi f)
\]
is correct **if** \(N\) is the relevant residence time and \(f\) is an approximately constant fraction of the asymptotic full passage. The tabulated \(0.57\) and \(0.58\) follow arithmetically.

But free-trajectory \(f\) uses a **finite-channel** denominator, whereas the window estimates use \(\pi/\sqrt{ab\epsilon}\), outside its asserted regime. These are not interchangeable. Nor are the estimates statistically independent: they share the fold, coefficients, and related dynamics. “One third” is the effective coefficient ratio \(f^2\), not the time fraction.

The normal form controls local slowing; it does not supply the global return or prove that each textual pair corresponds to one geometrically identified channel iterate.

**Minimum fix:** call the window quantities *effective asymptotic prefactors*, distinguish their denominators from the free-passage ratio, and replace “three independent estimates” with “three related comparisons.” Present the window law as an empirical, mechanism-motivated fit rather than a consequence established by the local reduction alone. State how residence counts were identified, or weaken the token-to-residence assertion. “Period-increment-two sequence” is defensible; a broader period-adding classification needs additional argument.

**S02 — Important: spectral separation does not establish one-dimensional dynamics.**  
**Affected:** “the orbits are effectively one-dimensional” and “one direction governs the dynamics near them.”

The reported gaps support dominance of one multiplier in sufficiently late linear relaxation, not a general one-dimensional state-space reduction; transient nonnormal amplification is not excluded. The period-20 next modulus is also very small relative to float64 arithmetic.

**Minimum fix:** describe *estimated asymptotic linear relaxation*, identify the parameter point for the comparison \(-0.271,-0.058\), and avoid precision claims for tiny Ritz values without a compact numerical accuracy qualification. Reserve the center-manifold argument for the fold.

# Statistical and scope findings

**B03 — Blocking: prospective chronology and pooled success need stage-specific wording.**  
**Affected:** Table `prospective` caption and “Prospective tests.”

“Every forecast was fixed before any test trajectory was run” suggests global freezing, but the second primary forecast was selected after the first test. The transparent adaptive sequence is legitimate; the current caption obscures it.

The displayed counts support:

- first test: seven exact short discrete forecasts; 16/17 longer forecasts within one iterate; seven negative nonzero residuals;
- second test: continuous 18/24 and discrete 16/24 within one iterate;
- six continuous misses approximately 3.1–5.2% short.

The first test’s continuous 24/24 result is supplied as a textual summary, not reconstructed here. Combined 42/48 is therefore a descriptive total spanning a **secondary** and a **primary** analysis—not one globally prespecified confirmatory test.

**Minimum fix:** say forecasts were frozen before trajectories **within each test stage**, retain the disclosed adaptation, and label the pooled result accordingly. For the periods-22–30 prediction, specify that coefficients, fit range, target edges, and error metric were fixed before those edges were determined; define whether “3%” concerns \(\alpha\) or distance from the fold. A compact statement suffices.

**S01 — Important: fit ranges, precision denominators, and corrections are underspecified.**  
**Affected:** window-law paragraphs and limits paragraph.

The \(-2.00\) center slope needs its fitted \(N\)-range; it cannot describe all displayed centers, including \(N=0\). The stated width ratio 1.35–3.51 also needs its orbit set: the reported \(N=4\)–7 center locations instead give approximately 2.57–4.50 using \(\sqrt{ab\,d_N}/0.1\).

“Relative precision \(2\times10^{-4}\)” must specify relative to \(\alpha\) or \(d_N\); near a fold this distinction is consequential. Numerical precision alone does not prove that fit residuals are asymptotic corrections.

**Minimum fix:** bind every slope and width range to its subset, define precision and residual denominators, and say residuals are *consistent with* finite-distance corrections. Keep 2.13–2.45 as fit-sensitivity values, not confidence limits.

**S03 — Important: constructed starts require a compact factor inventory.**  
**Affected:** passage tables and generalization to four new folds.

**Minimum fix:** add a small design note distinguishing:

- **Fixed within target:** checkpoint, steering direction/layer, orbit phase, coordinate normalization, channel-width rule, numerical precision, and background context.
- **Varied:** fold/family, continuation parameter, distance, and three start constructions.
- **Matched by construction:** dimensionless channel rule and, in test two, \(\kappa\)—not identical physical states.
- **Untested:** other transverse directions/magnitudes, channel widths, initialization histories, and broader models.

Specify the transverse vector’s selection and norm, and the episode denominator for mean free-passage lengths. “Changes the count by at most one” must refer to the particular displacement tested, not transverse robustness generally.

**S04 — Important: finite-time chaos evidence needs bounded wording.**  
**Affected:** coexistence paragraph and “Chaos writes text too.”

A 300-step exponent is a finite-time diagnostic, not alone proof of an attractor. “Which only reorders floating-point sums” is a runtime-specific assertion not established by the supplied thread comparison.

**Minimum fix:** identify exponent units and finite-time status; describe the thread experiment as sensitivity to numerical execution unless the exclusive mechanism is documented. Restrict “reuses neighboring words” to the observed examples. Preserve the distinction between greedy token readout and soft-state feedback.

# Reference audit

**R01 — Important: source verification and unique roles remain unresolved.**

No bibliography or source excerpts were supplied. The following inventories **all seven identifiable citation keys**, not a verified bibliography. Exact metadata and attribution require source checks.

| Citation key | Exact claim it is being asked to warrant | Recommendation |
|---|---|---|
| `strogatz1989predicted` | “the slow bottleneck that a saddle-node bifurcation leaves behind”; “a full passage takes about \(\pi/\sqrt{ab\epsilon}\) iterates” | Check the actual result, passage convention, and flow-to-map qualification. Retain for the quantitative bottleneck application if supported. |
| `strogatz2018nonlinear` | “the slow bottleneck that a saddle-node bifurcation leaves behind” | Useful as an accessible explanation. If it merely duplicates the preceding citation, retain only one at this sentence. Check edition/year. |
| `kuznetsov2023elements` | “center-manifold reduction restricts the dynamics near a bifurcation to its critical directions” | Retain for the reduction theorem and hypotheses; provide an identifiable section. |
| `kuznetsov2019maps` | “the standard coefficients along the critical eigenvector, computed from the network’s first and second derivatives” | Retain for map-specific coefficient formulas. Check full authorship, normalization, and factor \(1/2\). Its additional center-manifold citation may be redundant. |
| `pomeau1980intermittent` | “This alternation is type-I intermittency” | Retain for mechanism/origin, subject to source confirmation; not a substitute for demonstrating reinjection in this system. |
| `hirsch1982theory` | “This alternation is type-I intermittency” | Give it a distinct role concerning laminar-length scaling or reinjection assumptions, after checking the source; otherwise consolidate the duplicate citation. |
| `kim2026phase` | “The same reduction … predicts relaxation times without fitting … compared with measurements after the fact” | Required source check for every component, particularly chronology. Identify publication status; if a preprint, attribute as a provisional report. Remove “we went one step further” unless the specific comparison is substantiated. |

No visible key is orphaned. Do not add another general textbook merely for authority. Check whether the existing intermittency references support the laminar-length and reinjection qualifications. Add one verified period-adding reference only if the article develops that connection, not merely because periods increase by two. Publication status alone establishes neither correctness nor priority.

# Figure-to-text consistency

**F01 — Important: figures are not visually reviewable; two interpretive labels need tightening.**

The described TikZ picture is appropriate: in \(v\), the negative fixed point is attracting, passage proceeds left-to-right, and the stated 71-step example agrees with the table.

**Minimum fix:** label the burst/reinjection arrow as schematic and empirically motivated—not derived from the quadratic map. Define every time axis as \(F\)-steps or \(G\)-iterates, including the free-trajectory panel. A shaded condition \(|u|<U\) is only a projected-channel criterion unless transverse proximity was checked. Label window-law full/half lines as asymptotic predictions and identify fitted subsets.

The caption’s float32 branch multipliers versus float64 fold calculation is appropriately disclosed. Actual marker placement, axes, images, and alt text remain unchecked.

# What should remain unchanged

- The explicit failures of the one-iterate criteria, negative residuals, and unresolved period-6 shortfall.
- The distinction between discrete and continuous forecasts.
- The admission that the windows are outside the long-channel asymptotic regime and that the exponent is not pinned to two.
- The narrow conclusion that a scalar bottleneck law can work locally in a high-dimensional system.
- The separation of most-probable-token text from the probability-weighted embedding actually fed back.
- Reporting the fixed-reinjection model’s failure, rather than silently omitting that cheaper alternative.

# Required revisions

**Before posting:** resolve B01–B03; bind slopes, width ratios, and error percentages to explicit definitions; complete the source checks needed for mathematical attribution and the Kim comparison. These are claim corrections, not requests for new experiments.

**Can wait for a journal version, if current claims are narrowed:** a global return-map explanation, systematic transverse-start tests, wider channel-width tests, stronger spectral-error analysis, and a substantive period-adding comparison. None is necessary to publish the bounded local result and honest predictive failures.

# Publication checklist

- [ ] Normal-form truncation and continuous approximation distinguished.
- [ ] Finite-channel and asymptotic passage fractions separated.
- [ ] Fit subsets, units, and percentage denominators explicit.
- [ ] Prospective claims frozen and described stage by stage.
- [ ] Start constructions and untested factors disclosed compactly.
- [ ] References identified and necessary source checks completed.
- [ ] Actual figures checked against tables and captions by the authors.
- [ ] Negative findings retained without promotional reframing.
- [ ] Responsible human author makes the final publication decision.
