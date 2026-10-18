# External deep-research review (GPT Pro "astra", run by the human, 18 October 2026): adjudication

The human ran this review outside our consult tooling and pasted its report in the session. The report reviewed the
143-page draft PDF, whose title page still read "15 October 2026"; that date was stale and is now corrected. Advice is
advisory, and each point was checked against the current paper. The human asked us to ignore the code and data
release point: the repository becomes public at publication.

Verdict: major revision with a favourable presumption. The core results are judged credible. The criticism is mainly
calibration: terminology, the boundary between exploratory and prospective evidence, statistical units, numerical
digits, and one missing systems control.

## Already addressed in the current draft (the reviewer may have read an earlier build or missed it)
- Survival of chaos labels with horizon: in the main text and the evidence table. 92 % (stripes map) and 74 %
  (crossing window) of random samples of chaos-labelled cells persist to 10^5 steps (run LT, 18 Oct). We will add the
  survival curve S(t) by exponent tercile as a small figure (accepted).
- Batch-invariant kernels are named, with the systems literature, as removing the layout difference
  (he2025nondeterminism, yuan2025nondeterminism). The intervention itself has not been run (see below).
- Backour and Kim are cited. The comparison will be sharpened (accepted).
- The shrimp checklist exists (primitive orbits, stability, boundary organization), and Type A is called an analogy.

## Accepted and applied now (text)
1. Chaos terminology. "Chaotic" stays the paper's operational term, defined once and prominently as the
   finite-horizon numerical label: positive largest finite-time exponent, no period up to the cutoff after the
   transient, not a proof of an asymptotic chaotic set. The qualifier goes into the abstract, the introduction, the
   Figure 1 caption and the evidence table. "Chaotic attractor" or "chaotic set" is reserved for stronger evidence.
2. "Exact tangent" becomes "automatic-differentiation (AD) tangent" (the derivative of the implemented map in float64).
   "Exact" is reserved for interval and rational proofs.
3. Abstract: the practical consequences split into two mechanisms, dynamical amplification of tiny differences and
   low-precision rounding carried by long feedback to near-tied answers even where float64 loops contract. "The batch
   size changes only the rounding" becomes "changing the batch layout alters the floating-point execution while the
   intended computation is unchanged". "Fresh" becomes "previously unused". "Deciding correctness" becomes "disagreed
   on correctness". "Obeys quantitative laws" becomes "shows quantitative signatures": late finite-order doubling
   ratios approach Feigenbaum's constant numerically, at the float64 floor. The edge-of-chaos null is stated as
   bounded. The tertiary two-loop sentence is cut. "Can hide a chaotic state" becomes "a state with a positive
   finite-time exponent".
4. Introduction: "decide the output" becomes "can change the output". "Irreproducible because of its dynamics, not an
   implementation error" becomes "numerical nondeterminism can become output-level irreproducibility because the
   feedback amplifies or carries it". Add a scope paragraph separating the autonomous map, growing-context Soft
   Thinking, sampled thinking, ordinary generation and Huginn, and a compact claim-to-evidence table early (from the
   evidence ladder).
5. Feigenbaum: name the coordinate dependence and the numerical floor where the ratios are introduced, and give the
   last ratios with their uncertainty, not as a four-digit coincidence.
6. Statistics: a paragraph on units. Map cells are dependent evaluations of a deterministic function, so their shares
   are descriptive area fractions, and p-values from within-row permutations are labelled descriptive. Every interval
   names its resampling unit (rows, prompts, problems, batches). Null results carry their exclusion bounds.
7. Precision provenance in Methods: stored checkpoint weights (bfloat16) → arithmetic (patched float64) → derivative
   (AD) → validation rung. Float64 evaluation does not recover training-time weight precision.
8. Ethics, conflicts of interest and licences: statements drafted. The conflict-of-interest declaration must be
   confirmed by the author.
9. Related work: add CoTS, LaTER and Soft Latent Thinking only after verification against arXiv (in progress), and
   state which conclusions transfer to expected-embedding feedback, which may transfer to hidden-state recurrence, and
   which do not. Sharpen the novelty against Backour (amplification and contraction under mixture feedback) and Kim
   (normal forms in a task-trained looped transformer).
10. Minor wording: "layout decided the answer" becomes "the answer changed between layouts"; failed criteria keep the
    word "failed"; finite-time and asymptotic exponents get distinct wording; residual size is not certification.

## Accepted as experiments (proposed to the human with costs; none run yet)
A. Batch-invariant / fixed-shape control (reviewer: essential). Same float64 loops under fixed-shape execution: the
   layout difference should vanish, and a matched explicit displacement of the size of the measured first difference
   should restore divergence on the Lyapunov schedule in chaos-labelled cells and decay in contracting ones. This
   also rules out data leakage between batch rows. Small: Qwen2.5-0.5B, the 192 + 192 labelled cells or a subsample.
B. Precision-dose scaling: inject displacements from 1e-16 to 1e-2 and test t_eps ≈ log(eps/d0)/lambda
   prospectively. Can share the run with A.
C. Cluster-aware reanalysis of the answer tests (batch clusters), from existing records; no compute.
D. A held-out bifurcation family in an unsteered 7B-8B model (fold continuation, passage-time prediction, a few
   flips). Larger: hours of float64 H100.
E. Survival-curve figure from the LT records (no compute).

## Not adopted now
- Higher-precision (MPFR or interval) recomputation of the period-384/768/1536 flips: a transformer map in arbitrary
  precision over 1536 steps is a large project. We reduce the claimed digits and show the floor instead.
- Restructuring into several papers, or moving whole sections to a companion: the author's decision. The early
  scope paragraph and evidence table address the reading problem now.
- Stochastic-rounding experiment: low value for the paper's float64 claims.
- Reproducibility package: excluded by the human (public at publication).
