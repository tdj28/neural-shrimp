# Research opportunity and what the evidence leaves open

**The opportunity is to prove positive entropy on a small invariant subset—not to enclose the whole observed attractor. I recommend one next experiment: a paired-cell, thin-tube feasibility trial that combines a local covering-graph search with validated propagation of complete return segments.** Its decisive question is whether geometric covering margins survive rigorous treatment of the transverse directions at an affordable cost.

My preferred proof route is **a sparse graph of local covering relations**, initially without cone conditions. For an appropriate covering theorem, a transition matrix \(A\) with spectral radius \(\rho(A)>1\) can establish positive entropy. Cones become necessary only for stronger conclusions such as uniform hyperbolicity or suitably unique symbolic coding. Requiring them immediately could make the proof substantially harder than its scientific objective requires.

Three distinctions matter:

- The reported singular-value ratio is encouraging but is **neither an absolute transverse-contraction bound nor a uniform bound over a neighborhood**. The reported Lyapunov calculations likewise do not certify all 895 transverse directions.
- Failure on a \(2^{-800}\) box is evidence against the existing enclosure representation, not against chaos or every interval-based method. Conversely, increasing precision alone does not cure wrapping.
- A first-order model must include transverse and mixed nonlinear effects. An \(O(h^2)\) remainder along the unstable coordinate alone does not validate a tube of nonzero stable radius.

The faster-expanding cell should receive equal initial search effort, not automatic preference. Its lower required iterate might outweigh worse curvature; the crescent might offer cleaner geometry. **The relevant comparison is certified margin per unit computation, not Lyapunov exponent alone.**

The proposed trial is therefore directionally right, but one rigorous step is insufficient for pricing: composition, coordinate changes, and stable-bound reuse must also be tested.

# Ranked experimental directions

## Common experimental rules

These are deterministic certification experiments, not population studies. Distinct, deduplicated saddle/return chains are the discovery units; successive points, alternative widths, and precision changes on one chain are repeated measurements. Independent search seeds that find the same orbit do not create additional independent evidence.

Use development chains to tune representations, then freeze implementation choices before testing reserved chains or strips. Held-out testing measures engineering transfer; **every strip in the eventual certificate still requires validation**. Numerical searches and sampled perturbations never establish uniform bounds.

Record invalid enclosures separately from geometric failures. Neither means “no chaos.” Report rigorous bounds without statistical confidence intervals; report timing medians and ranges from three repeats, with explicit composition and parallel-efficiency uncertainty.

## 1. Sparse local covering graph — highest expected value

**Question and hypothesis.** Can either cell support a positive-entropy covering graph using short returns, thin tubes, and substantially fewer enclosures than a full unstable-extent covering? The competing explanation is that the visually one-dimensional dynamics conceal folding or transverse leakage that destroys robust covering margins.

A successful experiment would establish something absent from the supplied evidence: a finite, computationally tractable candidate certificate of topological complexity.

**Minimum decisive experiment.** At both cells, use six independently initialized searches with equal step budgets. Search short saddle orbits and recurrent branch segments, initially considering common iterates \(k=1,\ldots,12\). Deduplicate geometrically. Construct local charts along candidate chains and allow sparse, non-full transition matrices.

For each candidate, measure unstable exit margins, stable containment margins, curvature, and intermediate amplification. Reserve boundary and high-curvature strips for validation. Prefer common-iterate graphs initially: for a valid certificate for \(G=F^k\), the corresponding entropy lower bound is \(\log\rho(A)/k\). Unequal return times require a correctly weighted argument.

**Comparisons and controls.**

- Strong baseline: the proposed two-set, full-shift construction at the same cell and iterate.
- Cheap baseline: the sampled scalar log-odds map.
- Positive control: a rotated 896-dimensional product map
  \[
  H(t,y)=(4t^2-2,\;0.1y),
  \]
  using branches \(t\in[-1,-0.4]\) and \(t\in[0.4,1]\). It has explicit strict crossing and stable-containment margins.
- Negative controls: replace the transverse factor by an expanding factor to defeat the intended containment test; test a contracting map to detect spurious branching.

**Endpoint and gate.** Measure rigorous face-clearance bounds in chart units, maximum normalized enclosure error, and CPU-hours per complete-return strip. The proof denominator is **all strips and required boundary faces in the selected graph**, not those that happened to validate.

Expand when held-out complete returns retain a fourfold error-budget reserve and the projected run is below approximately 3,000 CPU-hours. Between 3,000 and 30,000 hours, first change graph, chart, or arithmetic. Above that, compare another route before scaling.

**Interpretation.** Positive results support a local entropy certificate, not chaos throughout the attractor. Numerical geometry without validated transverse containment is a mixed result. Failure to find a graph within \(k\leq12\) rejects that search envelope, not positive entropy.

**Resources.** Cheapest pilot: one candidate chain per cell, three representative strips each. Estimate 1–2 weeks of development and 50–300 CPU-hours for a bounded feasibility study, not a completed proof.

A full-dimensional Conley grid is not competitive. A viable alternative is a tube-adapted isolating-neighborhood computation on the discovered graph—but only with an explicit theorem giving the desired entropy conclusion; a nontrivial index alone is insufficient.

## 2. Saddle–homoclinic certificate — potentially thinner, with an extra theorem burden

**Question and hypothesis.** Does a short saddle orbit have a transverse homoclinic return whose neighborhood is cheaper to validate than the covering graph? The competing explanation is that close numerical returns are tangencies or near misses.

**Minimum experiment.** Search saddle periods 1–8, retaining up to six distinct saddles. Launch both unstable branches, with multiple seed distances, and search returns within 32 iterates. Develop the matching calculation on half the distinct candidates; reserve the remainder.

Construct validated local manifold charts and a multiple-shooting matching problem. A scalar crossing of a validated codimension-one stable chart is preferable to trying to mesh an 895-dimensional manifold.

**Comparisons and controls.** Compare against the direction-1 covering chain along the same excursion. Nearest-neighbor recurrence is the cheap baseline. Test the local-manifold calculation on a linear hyperbolic saddle, and the intersection calculation on manufactured simple-root and tangential-root matching problems. These controls validate components, not a transformer homoclinic orbit.

**Endpoint and gate.** Primary outputs are a rigorous intersection inclusion and a nonzero transversality lower bound, with CPU-hours per certified candidate. Count each distinct saddle/excursion once. Expand only if those margins remain comfortably larger than enclosure uncertainty.

**Interpretation.** A transverse intersection is useful only under a theorem whose hypotheses are actually verified. Standard Smale–Birkhoff reasoning requires an appropriate diffeomorphism setting; near-rank-one behavior makes invertibility certification potentially delicate. Test this explicitly rather than assuming it. If it is inaccessible, use the same excursion to construct covering relations valid for the applicable map class.

A null search does not exclude homoclinic intersections elsewhere. A return with unresolved transversality suggests a covering-chain experiment, not a claim of tangency.

**Resources.** Cheapest pilot: one saddle and its two branches. Estimate 1–2 weeks for numerical discovery and local validation, followed by roughly 100–1,000 CPU-hours if the manifold machinery can reuse the existing multiple-shooting code. This route outranks direction 1 only if it produces a much shorter, better-separated chain.

## 3. Validated contracting-fiber reduction — highest mechanistic payoff, hardest premise

**Question and hypothesis.** Is the scalar hump backed by a uniformly contracting fiber structure, permitting a rigorous one-dimensional factor or useful interval-valued reduction? The strongest alternative is that the observed scalar curve reflects one trajectory and retains hidden-state memory.

**Minimum experiment.** At each cell, select 12 scalar-coordinate locations on each proposed branch. Perturb eight transverse directions, including estimated worst-case singular directions, while correcting back onto the same scalar-coordinate level set. Compare the next scalar output and multi-step transverse separation.

These perturbations are diagnostics only. The decisive follow-up bounds the transformed derivative blocks uniformly:
\[
DG=
\begin{pmatrix}
a&b\\
c&D
\end{pmatrix}.
\]
This separates expansion, stable-to-unstable influence, unstable-to-stable bending, and transverse propagation.

**Comparisons and controls.** Compare against both a fitted scalar map and direct high-dimensional covering. Use the product-map control above, where the scalar quotient is exact; adding stable-to-scalar coupling supplies a negative control for autonomy of that particular coordinate.

**Endpoint and gate.** Measure certified scalar-output width over each fiber, in log-odds units, and uniform transverse operator bounds. Repeated perturbations belong to the same fiber patch. Continue toward a reduction only if a stated invariant-foliation or graph theorem’s inequalities close. Otherwise retain branchwise interval-valued bounds or direct coverings.

**Interpretation.** Numerical fiber contraction supports further work but proves no quotient. Nonzero fixed-coordinate output spread falsifies exact autonomy of that coordinate, not every possible reduction. Near a hump’s critical point, an expanding-cone argument may fail even when entropy can be certified on separated branches.

**Resources.** The diagnostic costs approximately 20–100 CPU-hours; a rigorous screen could require 100–1,000 hours and several additional weeks. Its main dependency is a reusable transverse operator bound. Without that, it is not a shortcut.

# Recommended next experiment

## Paired-cell, complete-return enclosure pilot

This outranks the alternatives because it tests the shared bottleneck—**rigorous transverse control through composition**—while directly producing candidate entropy certificates.

### Experimental matrix

Use the existing exact stored parameters corresponding to both displayed cells.

1. Give each cell six search starts and the same capped trajectory/JVP budget.
2. Select its best local return chain by geometric margin and length, not exponent.
3. Choose three strips per cell: one development strip and two reserved strips covering difficult boundary/high-curvature geometry.
4. Test:
   - plain centered balls versus a centered first-order segment representation;
   - unstable half-widths \(h_0,h_0/2,h_0/4\);
   - transverse radius zero versus the numerically required nonzero radius.

That is up to **72 one-step configurations**, evaluated adaptively: terminate plainly failing configurations rather than completing an expensive Cartesian product. Then propagate at least one reserved strip per cell through its **entire proposed return**.

The nonzero radius must come from approximate tube closure. It cannot be chosen arbitrarily small: center-curve defects and curvature may impose a positive minimum thickness.

### Primary endpoint

For each reserved complete-return strip, report
\[
R=\max_j \frac{\epsilon_j}{m_j},
\]
where \(m_j\) is the saved prediction’s clearance from a relevant forbidden face, and \(\epsilon_j\) is the rigorously bounded departure from that prediction, including transverse input, nonlinear remainder, cache uncertainty, and rounding.

Unstable exit conditions apply to the appropriate boundary strips; stable containment applies throughout. Invalid enclosures receive \(R=\infty\). Report the numerator and denominator explicitly—e.g. “one of two reserved complete returns resolved with \(R<1/4\)”—without extrapolating that result into a proof.

Secondary endpoints are CPU-seconds per strip-return, peak resident memory, and the number of strips needed for the complete graph.

### Implementation sequence

**1. Establish a reusable rigorous cache.** Start at 1024 bits, the precision already reported to work. Test 512 and then 256 bits only after the revised representation succeeds. The existing evidence does not justify budgeting around 128–256 bits.

Cache construction must match each exact map’s steering and prefix semantics. Its uncertainty remains part of every later enclosure.

**2. Implement the smallest useful dependency-preserving representation.** Use
\[
x=c+ut+By,\qquad |t|\le h,\quad \|y\|\le\rho.
\]
Carry the unstable coefficient explicitly. Carry transverse uncertainty through a linear factor or a certified operator bound—not 895 unrelated primal intervals.

Use either a center-Jacobian Taylor form with a proved remainder, or a mean-value derivative enclosure. Include \(h\rho\) and \(\rho^2\) effects; avoid an unspecified extra “remainder” that obscures what has actually been bounded.

**3. Price two transverse strategies.**

- A verified center Jacobian in adapted coordinates, reused across a patch using a rigorous derivative-variation bound.
- A cheaper block/operator bound, accepting it only if it closes containment.

Time 16 signed columns before committing to a full Jacobian. The supplied 56-hour projection is useful context, not a benchmark for these wider domains. Validate chart inverses; floating-point QR factors are not exactly orthogonal.

QR/Lohner-style reconditioning is relevant. Logarithmic norms are not the natural first tool for this discrete map; bound discrete products in adapted coordinates.

**4. Diagnose the bottleneck causally.** Compare \(\rho=0\) with the required \(\rho>0\). If the line succeeds but the tube fails, further unstable subdivision alone is unlikely to solve the problem. Measure \(b,c,D\) and test better charts or a shorter chain.

**5. Test composition before extrapolating cost.** Reconditioning may prevent wrapping, or repeated remainders may destroy every margin. Price complete returns:
\[
H_{\rm total}
=H_{\rm cache}+H_{\rm charts}
+\sum_i N_i\,C_{{\rm return},i},
\]
including amortized derivative work and measured parallel efficiency.

### Layer-specific checks

- **RMSNorm:** its denominator is analytically at least \(\sqrt{\varepsilon}\). Failure to prove positivity is an enclosure failure. Use nonnegative square bounds and tighter center-radius lower bounds; the global epsilon bound alone may give unusably large derivatives.
- **Attention:** retain the rigorous fixed cache and use centered, stabilized softmax calculations. Preserve common-shift cancellation where possible.
- **Large readout:** stream vocabulary blocks. Dropping tokens is valid only with rigorous tail bounds for every needed quantity, including derivative contributions.
- **Memory:** keep original exact weight representations shared or streamed; do not replicate expanded high-precision weights across 32 workers. Determine concurrency from measured resident memory.

Retain Arb and add narrow, transformer-specific primitives. Porting an entire Taylor-model system or building GPU interval arithmetic introduces a larger validation burden than this pilot warrants. GPU floating point remains useful for proposals where precision checks permit; it is not the certificate engine.

### Resource envelope and claim boundary

**Estimates:** 50–300 CPU-hours for the pilot, with a hard staged budget; benchmark the first 100 float64 steps before scheduling discovery. Store selected states and diagnostics rather than complete activations. No API expenditure is needed.

For production, the supplied 2,000–40,000 CPU-hour scenarios remain plausible scenarios, not forecasts. They correspond to ideal 32-core times of about 2.6–52 days before memory and efficiency losses. At an assumed rental rate of [amount redacted]/core-hour, they imply [amount redacted] excluding ancillary charges. Do not rent until complete-return timing resolves the dominant uncertainty. A production-quality engine could take **4–12 researcher-weeks**, potentially longer if transverse bounds require redesign.

The certificate concerns the real-number map of the stored constants. Its trusted base includes exact constant decoding, Arb and elementary-function enclosures, layer implementations, chart bounds, and the covering theorem. Positive entropy would not establish typical-orbit chaos, a physical measure, semantic significance, or every period for \(F\). Graph cycles and periods of \(F^k\) must be translated carefully.

# First 48 hours

**Hours 0–12**

- Benchmark cached float64 forward/JVP execution and rigorous point evaluation.
- Run the analytic positive and negative controls.
- Record timing and peak memory with one worker and a small worker pool.

**Hours 12–30**

- Run equal-budget searches at both cells.
- Extract candidate local graphs for \(k\leq12\).
- Deliver saved charts, branch intervals, transition matrices, and measured margins—not merely return-map plots.

**Hours 30–48**

- Select development and reserved strips.
- Run the zero-radius precision ladder and begin the first-order segment calculation.
- Deliver a first bottleneck classification: missing geometry, rounding floor, unstable curvature, transverse leakage, or composition growth.

If neither cell yields robust geometry, the next branch is targeted saddle-return discovery, not indiscriminate arithmetic optimization.

# Two-week research roadmap

**Days 3–5: obtain a validated segment engine.**  
Demonstrate expected width scaling on the analytic control and one transformer strip. Reuse the proved period-8 implementation as a real-model regression check where compatible. If point uncertainty dominates, improve centering or precision before interpreting width scaling.

**Days 6–8: test nonzero transverse thickness.**  
Time derivative columns, compare adapted-coordinate and inexpensive norm bounds, and measure mixed terms. If only zero-radius segments work, redesign the chart or compare a shorter homoclinic-style chain.

**Days 9–11: propagate complete reserved returns.**  
Freeze development choices first. Deliver rigorous face bounds, composition-growth traces, memory use, and per-return timing. A numerical horseshoe with failed rigorous containment remains a useful, explicitly unresolved candidate.

**Days 12–14: execute the discriminating branch.**

- **Margins survive; projected cost modest:** validate the smallest complete positive-entropy graph.
- **Geometry survives; stable bounds fail:** perform the fixed-scalar fiber perturbation experiment and test verified Jacobian reuse.
- **Long returns dominate cost:** pursue the best saddle excursion and a shorter covering chain.
- **No robust graph appears:** compare targeted homoclinic returns at both cells; do not infer absence of chaos.

The two-week deliverable is a substantive result: either a partially or fully validated entropy construction, or a measured failure of a specific enclosure strategy with the next mathematical bottleneck isolated.
