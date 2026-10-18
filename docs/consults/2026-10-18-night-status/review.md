# Research opportunity and what the evidence leaves open

**The strongest opportunity is to determine what the apparent loss of one-dimensionality at a head actually means. I recommend a small, controlled experiment on the period-11 return map: compare its distance from rank one near and away from the head, then selectively remove longitudinal–transverse coupling in the tangent dynamics.** This distinguishes explanations that additional multiplier grids alone cannot separate.

The supplied results establish substantial numerical organization. They also leave several important distinctions unresolved:

- **Two parameters are not two state dimensions.** Unless a two-dimensional invariant reduction has been established separately, use: “a numerically continued, shrimp-shaped period-five stability region in a two-parameter slice of the 896-dimensional soft-token map.” The fold/flip skeleton supports bifurcation organization; it does not establish a two-dimensional map.
- **Separate the primitive-period-five region from its surrounding cascade footprint.** Period-10/20/40 shells, slowly converging trajectories, and same-text chaos should not share one dynamical membership definition. Token agreement is useful for tracking, not for verifying full-state periodicity.
- **Three comparable eigenvalue moduli do not establish three dynamically necessary directions.** Strongly nonnormal, nearly rank-one matrices can have several comparable tiny eigenvalues. Moreover, imposing the first two characteristic coefficients near zero can itself produce comparable root moduli. This is not independent evidence for dimensionality.
- **The period-11 absolute invariant gate needs scale qualification.** At a multiplier scale around \(4\times10^{-7}\), quadratic coefficients naturally have scale around \(10^{-13}\). An absolute \(10^{-7}\) gate therefore cannot establish that the quadratic coefficient is near zero relative to its natural scale.
- **“Type A” should remain a geometric analogy, not the inherited classification.** A closed trace-zero contour is observable; a fixed-sign, two-dimensional determinant classification is not transferable when its assumptions fail. A genuinely nilpotent three-dimensional block requires all three characteristic coefficients to vanish—generically three conditions, not two.
- **The temperature track shows persistence on the sampled branch, not impossibility elsewhere.** Approaching fourth modes also argue for an adaptive three-/four-dimensional core rather than a permanently fixed \(3\times3\) description.

The boundary gate is useful corroboration, but not sufficient by itself: propagation-dependent selection, incomplete continuation, and four unexplained census discrepancies remain distinct issues. Tail or hub measurements would add anatomy; they would not repair an unsupported dimensional claim.

# Ranked experimental directions

## 1. Resolve the period-11 head: dimensional crossover, nonnormality, or numerical floor

**Question and hypothesis.** Does the return derivative genuinely cease being approximately rank one near the head? The falsifiable hypothesis is that its relative rank-one approximation error increases substantially near the head while a three-dimensional approximation remains accurate. The strongest alternative is a nearly rank-one, nonnormal return operator whose tiny eigenvalues become comparable without a comparable increase in singular directions.

**Minimum experiment.** At fixed temperature, compare paired inner and outer parameter points around the reported head. Estimate singular values of the full-state monodromy operator using JVP/VJP products; independently recover a real Schur representation of its resolved spectral core. Test three predetermined cycle phases. Use development points to fix numerical tolerances and reference bundles, then independently sampled, held-out parameter pairs for evaluation. Details follow below.

**Comparisons and controls.**
- Strong comparison: singular-spectrum diagnostics versus the existing multiplier-ratio criterion.
- Cheap comparison: the existing Ritz spectrum and orbit residual.
- Positive controls: synthetic nonnormal operators with known rank-one versus rank-two structure but identical eigenvalues.
- Application control: the disclosed strongly anisotropic one-block period-eight orbit.
- Negative intervention: changing an orthonormal basis without changing the operator must leave singular values unchanged.

**Endpoint and precision.** Measure
\[
d_1=\sigma_2(M)/\sigma_1(M),\qquad d_3=\sigma_4(M)/\sigma_1(M).
\]
These are dimensionless relative operator-norm approximation errors in the specified embedding metric. The primary endpoint is the paired change in \(\log_{10}d_1\), with parameter pairs—not phases or Arnoldi restarts—as sampling units. Sixteen held-out pairs are a screening experiment for an order-of-magnitude effect, not a precision survey.

**Interpretation and decisions.** Increasing \(d_1\) with small \(d_3\) supports a locally higher-rank tangent return. Persistently small \(d_1\) supports the nonnormal alternative. Large \(d_3\) requires a larger core. Unresolved singular values are an invalid measurement, not evidence for either mechanism. None of these outcomes alone proves an invariant manifold.

**Resources.** Start with four local parameter points. Matrix-free derivatives avoid storing every step Jacobian. Estimated main-run cost: roughly 6–24 GPU-hours, subject to benchmarking; a rented suitable GPU at an assumed \$0.50–\$2/hour would cost approximately \$3–\$48 before overhead. Expand only after repeatability and rank-resolution controls pass.

## 2. Measure existence boundaries separately from basin-selection boundaries

**Question and hypothesis.** Do the period-five footprint’s apparent edges follow actual loss of attraction/existence, or changes in which attractor the initialization reaches? The hypothesis is that completed fold/flip continuation explains the primitive-period-five stability boundary, while many remaining followed-domain edges are selection effects.

**Minimum experiment.** Freeze 24 transects drawn independently by arclength from prespecified boundary strata: well-matched edges, clipped leg continuations, and discrepant edges. Use the completed continuation—not token labels—to define predictions. At five locations per transect:

1. continue the same period-five orbit through the transect;
2. measure residual, primitive period, and stability;
3. run eight independently generated starts from a fixed, declared initial-state distribution.

Develop the protocol on six other transects. Trajectories and neighboring cells within a transect are repeated observations; independent transect draws are the analysis units.

**Baselines and controls.** Compare fixed-start mapping, followed-domain mapping, and full-state continuation. Use known fold and flip segments as positive controls. Use interior transects away from any computed bifurcation as negative controls. A cheap baseline is longer integration from the original fixed start.

**Endpoint.** Primary: signed distance, in grid-step units, between the observed loss-of-stability boundary and its predicted fold/flip location, per transect. Secondary: attraction frequency out of eight starts, explicitly conditional on the chosen start distribution. Slow or chaotic runs remain unresolved; they are not assigned to either basin. Report stratum-specific uncertainty rather than treating boundary pixels as independent.

**Outcomes.** Agreement strengthens numerical boundary organization. Persistent attracting period-five orbits beyond a footprint edge establish selection effects there. Systematic disagreement with qualified continuation falsifies the current boundary account. Incomplete continuation is an invalid test of that segment.

**Resources and next decision.** Pilot six transects; estimated 1–4 GPU-hours. The full experiment is approximately 960 integrations plus continuation, perhaps 4–20 GPU-hours depending on convergence. If discrepancies disappear after completing curves, defer tail tracing. If they persist, prioritize basin geometry. The period-11 skeleton becomes the next application only after its fold/flip curves are available; do not assume it is already a shrimp head.

## 3. Test whether adapted norms predict a useful neighborhood, not merely known stability

**Question and hypothesis.** Can phase-dependent metrics explain contraction near periodic attractors where Euclidean one-step norms expand? The meaningful hypothesis is not “a stable monodromy admits a Lyapunov metric”—that is expected—but that an orbit-derived metric predicts contraction for held-out, finite perturbations.

**Minimum experiment.** Use three development periodic orbits to fix the construction. Evaluate six held-out parameter draws from a predetermined periodic region. Compare Euclidean one-step and composed-return norms with a periodic Lyapunov or Schur-based metric. Freeze that metric before testing independent perturbation directions at three radii around each orbit.

Use complete derivative matrices where practical, or a low-rank construction only if its transverse remainder is measured. Do not silently discard the remainder.

**Controls and baselines.** Cheap baseline: converged \(k\)-step norm estimates. Strong baseline: numerical SVD of the complete return derivative. Positive controls: a stable nonnormal linear cycle with expanding individual factors, and a scalar expanding map. Negative control: the same pipeline with the Euclidean metric.

**Endpoint.** Per orbit, measure the largest tested radius at which the frozen metric predicts contraction and no held-out perturbation contradicts it. Give radius in embedding-norm units and successes over attempted directions; aggregate uncertainty over parameter draws. This is empirical neighborhood validation, not a certified basin.

**Outcomes.** Improvement supports alignment-aware contraction estimates. Improvement only on infinitesimal perturbations adds little beyond Floquet stability. Contradictions require shrinking the claimed neighborhood or rejecting the metric. Failure to bound the transverse remainder invalidates a bound claim but still permits an explicitly empirical comparison.

**Resources.** Start with already available derivative objects; otherwise benchmark their construction first. Estimated 2–12 GPU-hours plus CPU linear algebra.

The existing norm conclusion should be sharpened: algebraic inequalities can be valid while the **power-iteration quantities remain lower estimates of norms**, not certified upper bounds. An observed lower estimate of global \(L\) makes the proposed sufficient threshold large; it does not lower-bound the actual transition temperature. Also check factor dimensions: the diameter-squared formula requires the appropriate embedding-covariance factorization, not an ambiguously defined logits Jacobian.

## 4. Certify coexistence before extending the period ladder

**Question and hypothesis.** Can two distinct primitive attracting orbits be certified at the same parameter value? This adds qualitatively different knowledge from certifying more periods separately: coexistence is then established independently of initialization.

**Minimum experiment.** Target the supplied period-two/period-three candidates. Use the same exact parameter specification for both; establish disjoint orbit enclosures, primitive periods, existence/uniqueness, and attraction. Krawczyk existence alone does not establish attraction.

This is a deterministic theorem test, not a statistical sample. The two orbits are distinct mathematical objects; their five orbit points are not five independent replications. Keep a second, predetermined parameter value as a later transfer test, not something selected after success.

**Baselines and controls.** Cheap baseline: float64 Newton plus Floquet stability. Positive control: reproduce a completed period-eight certificate component. Negative control: an intentionally undersized or displaced enclosure should fail inclusion without being misreported as nonexistence.

**Endpoint and branches.** Primary endpoint: both primitive attracting cycles certified at one parameter value. One success establishes only one orbit. Inclusion failure distinguishes neither nonexistence nor instability; inspect whether center error, enclosure width, derivative bounds, or stability qualification failed. If preparation remains difficult, the viable substantive alternative is numerical coexistence plus targeted local contraction neighborhoods—not a larger unprepared ladder.

**Resources.** At the disclosed 6.6 CPU-hours per orbit point, five points suggest roughly 33 CPU-hours **under a linear-work extrapolation**. Period coupling and preparation may change this substantially. The initial task is one-point preparation and inclusion, then extrapolation. This ranks below the head experiment because the current preparation bottleneck is real and the dimensional question remains more explanatory.

# Recommended next experiment

## A paired rank-and-coupling experiment at the period-11 head

This outranks a wider multiplier grid because it tests the missing premise: **does comparable spectrum mean loss of rank-one dynamics?** It also addresses why the signed-step-factor search failed without trying to rescue that particular coordinate construction.

### Experimental matrix

Use the reported center
\[
c=(-0.0333026,-0.0522112),\qquad \log_{10}T=-0.1.
\]

| Component | Configurations | Purpose |
|---|---:|---|
| Development | Nine fixed points spanning the existing local box | Set solver tolerances, Krylov sizes, phase alignment, and reference bundles |
| Held-out comparison | 16 independently sampled offsets \(\delta\in[-3.5,3.5]^2\,10^{-5}\), paired as \(c+\delta,\ c+4\delta\) | Compare inner and outer neighborhoods without selecting favorable rays |
| Repeated phases | Phases 0, 4, and 8 of every orbit | Detect phase-dependent nonnormality |
| Coupling intervention | Four prespecified held-out pairs; coupling scale \(0,\frac12,1\) | Diagnose longitudinal–transverse feedback |
| Controls | Synthetic operators and one-block period-eight orbit | Verify sensitivity and implementation |

Publish the random seed and freeze the matrix before evaluating held-out points. No replacement sampling for difficult points. If an orbit branch cannot be continued, record that failure explicitly; do not substitute a nearby convenient orbit.

### Baseline and positive-control checks

For small \(\epsilon\), use
\[
A_1=\epsilon I+e_1e_2^\top,\qquad
A_2=\epsilon I+e_1e_2^\top+e_2e_3^\top.
\]
Both have identical eigenvalues, but very different singular-rank structure. The pipeline must distinguish them. This directly tests the inference currently at issue.

Also require:

- JVP/VJP adjoint consistency;
- stability under increased Krylov dimension and independent restarts;
- agreement of paired nonlinear finite differences with derivative actions over a perturbation-size plateau;
- correct recovery of the period-eight anisotropy to the precision the implementation actually supports.

Do not use arbitrarily tiny finite differences at the contracting head: subtraction can erase the signal.

### Primary endpoint

For each parameter point, take the largest \(d_1\) over the three phases. For each independently sampled pair, compute
\[
\Delta=\log_{10}d_{1,\mathrm{inner}}
       -\log_{10}d_{1,\mathrm{outer}}.
\]

Report the mean paired difference in **decades**, with denominator 16 pairs and paired uncertainty intervals. Report the full paired distribution and absolute \(d_1,d_3\) values alongside it. Numerical uncertainty should propagate into intervals for these ratios. Missing spectra cannot be assigned zero; unresolved pairs make the primary comparison incomplete unless their error ranges cannot change the decision.

A prospective screening criterion is a mean increase of at least one decade, with inner \(d_1>10^{-2}\) at most sampled points and small \(d_3\). These are declared practical thresholds, not universal definitions of dimension.

### Mechanism intervention

Freeze two off-head reference bundles before held-out testing. In their longitudinal/transverse coordinates, replace each tangent factor
\[
J_i=\begin{pmatrix}a_i&b_i^\top\\c_i&D_i\end{pmatrix}
\quad\text{by}\quad
J_i(\eta)=
\begin{pmatrix}a_i&\eta b_i^\top\\\eta c_i&D_i\end{pmatrix}.
\]

This preserves longitudinal and transverse self-dynamics while suppressing their feedback. Compare singular structure and resolved characteristic coefficients across \(\eta\).

If the effect agrees across both frozen bundles, it supports a coupling explanation for the **tangent spectrum**. It does not establish that a modified nonlinear map has a superstable-curve crossing. If frame choice changes the conclusion, retain the simpler singular-rank comparison and reject the causal interpretation.

Do not continue “trace zero” or an ambiguously selected leading eigenvalue as a superstable curve. Neither is an adequate zero-multiplier condition near mode switching.

### Implementation and resource envelope

Load and benchmark locally first; no local hardware specification was supplied. Float64 weights alone are approximately 4 GB; working memory may be roughly 8–24 GB depending on differentiation strategy. Use matrix-free products before attempting complete monodromy matrices.

Budget approximately \(5\times10^4\)–\(10^5\) step-level derivative actions, plus orbit refinement. At an assumed 0.05–0.5 seconds/action, derivative work alone spans roughly 0.7–14 hours; slower hardware can exceed the first-day budget. Benchmark rather than promise completion. Cap initial rental spending at \$75. No API calls or model judges are needed.

**Decision branches:** rank loss supports a higher-rank local return model; retained rank-one structure supports a nonnormal scalar-like interpretation; a large fourth singular direction triggers a larger core; unresolved derivatives trigger a focused precision experiment, not a wider search.

# First 48 hours

**Hours 0–4:** Build operator wrappers, phase alignment, synthetic controls, and four local benchmark cases. Deliver timings, memory use, derivative-consistency checks, and resolved singular-value ranges.

**Hours 4–8:** Complete development points. Freeze sampling, tolerances, reference bundles, and practical thresholds. If controls fail, fix the measurement before spending on the main matrix.

**Hours 8–20:** Run held-out paired comparisons. Produce paired rank-error plots, numerical-error intervals, and an explicit failure table. This is the priority for the next 12 hours of compute.

**Hours 20–32:** Run the limited coupling intervention and nonlinear derivative checks. Compare results across both reference bundles.

**Hours 32–48:** Make one branch-specific extension:
- rank loss: resolve the three-/four-dimensional core;
- retained rank one: estimate nonnormal sensitivity and test scalar prediction locally;
- invalid measurement: repeat only the decisive cases at improved precision or with complete derivatives.

Deliver an executable result bundle, not another general survey.

# Two-week research roadmap

**Days 3–5 — Test a local model prospectively.** Fit rank-one and rank-three return approximations on development data. Predict held-out full-state return responses at several perturbation sizes. Compare normalized vector prediction errors, not token agreement. A linear model that works only infinitesimally does not establish a nonlinear reduction.

**Days 6–8 — Add temperature as a discriminating intervention.** Repeat a reduced paired design at \(\log_{10}T=-0.101\) and \(-0.099\), tracking the same primitive orbit. Determine whether the dimensional layer moves with the head or changes character. Evaluate characteristic coefficients with scale-aware errors. Search for a three-coefficient zero only if the core is resolved and the local parameter Jacobian supports such a search.

**Days 9–11 — Follow the informative branch.**
- Robust higher-rank core: test whether coupling predicts the avoided-crossing width.
- Robust scalar-like return: continue phase-specific near-critical factors only where the reference bundle remains qualified; test intersection transversality rather than assuming the parallel curves must meet.
- Head mechanism unresolved: execute the targeted boundary transects, yielding independent knowledge about existence versus selection.

**Days 12–14 — Establish transfer or a clean limitation.** Apply the successful diagnostic to a small, fixed period-five sample. In parallel only if preparation is ready, attempt the period-two/period-three coexistence certificate.

The desired outcome is not necessarily a one-dimensional head. It is a discriminating account of why the observed heads look scalar away from their cores, what replaces that description near them, and which boundaries reflect dynamics rather than initialization.
