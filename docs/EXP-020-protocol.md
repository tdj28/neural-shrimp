# EXP-020 protocol: nonlinear recovery of the first written bit

17 September 2026. **Prospectively specified saved-evidence regression; no new decoder scores at protocol freeze.** This is an outcome-motivated, saved-evidence regression analysis, not a fresh holdout or preregistration. The motivating EXP-019 near-edge phase16 at4T had no-second linear accuracy1.0 and with-second accuracy0.5030952. The analysis includes the entire fixed48-context cohort, not just that example. Execution requires the reviewed executable machine plan, passing synthetic checks and a verified private push before any new target decoder score.


**TLDR:** In the two selected fixed-weight neural cycles, a second input can make the first input hard to read with a linear observer. This study asks whether a nonlinear observer can recover that earlier input from the same saved neural responses. A recovery would locate part of the interference in the readout; a failure would still leave other possible readouts untested. This is a question to test, not a new result.

The original EXP-019 arrays and results are already known. The new classifier definitions and numerical rules are frozen before calculating new decoder scores. All 24 phases in each setting are reused; none supplies fresh validation. The external design record, including superseded drafts and review, remains in `/data2/neural-shrimp-data/audits/EXP-020/`.

## Question and fixed scope

Does a nonlinear first-bit decoder, defined only from the already sealed linear reference responses, recover first-bit information that the covariance-linear observer misses in the presence of the second bit? This is **first-bit classification with an unknown second bit**, not joint two-bit classification. Success would establish recoverability for this declared decoder/noise/observation model. Failure would not establish erasure or absence of information in other readouts.

Use both existing settings, all24 shared ordered regression phase fractions, identity arm only, windows ending2T,4T,8T, and all four nonzero sign pairs (s1,s2) in {−1,+1}². Preserve historical development/heldout labels only as provenance; no new split or selected subset. Sigma and epsilon remain1e−4. Observations contain the same32×3 activation samples and the same noiseless, reference-only zero-write baseline as EXP-019. No new neural integration, GPU use, dose, threshold fitting, finite-class-mean decoder oracle, noisy-trial training, parameter scan or paid call is included.

The main classifier uses positive-write activation vectors A and B from the primary DOP853 reference seal. The actual class observation is Y=mu+sigma*Z, Z~N(0,I96), where mu is the corresponding saved finite activation displacement, not a fitted parameter of the classifier. The classifier is fixed before loading these finite means. Evaluation necessarily uses the held-out-in-code finite mean and its label to calculate correctness, but those values cannot change A, B, orientation, priors, decision regions or any threshold.

The unknown first and second bits each have fixed independent balanced priors. Conditional reference class densities are the two-component mixtures

    p_ref(y | s1) = 0.5*N(y; s1*A+B, sigma²I)
                 + 0.5*N(y; s1*A−B, sigma²I).

Predict +1 when L(y)=log p_ref(y|+1)−log p_ref(y|−1)>0, −1 when negative, and use balanced random tie-breaking for exact ties. The reference likelihood ratio is

    L(y) = 2*A·y/sigma²
           + logcosh[B·(y−A)/sigma²]
           − logcosh[B·(y+A)/sigma²].

This is Bayes-optimal only for that **reference linear Gaussian-mixture model**. Under the finite nonlinear means, it may improve, tie or degrade relative to the original observer. Every degradation stays in the results; no benefit gate or best-decoder selection is allowed. Components of the observed vector orthogonal to span(A,B) do not affect this declared classifier, although the real system could encode useful information there.

## Reduction to a stable one-dimensional probability calculation

Normalize all coordinates by sigma. Because s2 is symmetric, B may be replaced by −B without changing either reference density. Fix this sign from A·B alone so the parallel coefficient a is nonnegative; use the original B sign if the dot is zero. Let

    e = signed B/||B||, b=||B||/sigma,
    a=(A·e)/sigma >=0,
    A_perp=A−e*(A·e), c=||A_perp||/sigma.

For c>0 define v=A_perp/||A_perp|| and observed coordinates t=e·Y/sigma, z=v·Y/sigma. They have independent unit-variance Gaussian noise. The likelihood is

    L(t,z)=2*c*z+g(t),
    g(t)=2*a*t+logcosh[b*(t−a)]−logcosh[b*(t+a)].

For each finite mean compute mt=e·mu/sigma and mz=v·mu/sigma. Its exact correctness probability for first sign s is

    P_correct(s,mu) = E_{T~N(mt,1)} Phi(s*[mz+g(T)/(2*c)]).

This is one real Gaussian integral, not a96-dimensional Monte Carlo problem. The probability is evaluated separately for all four actual finite means; do not assume exact sign antisymmetry or project away finite nonlinear offsets.

Naive subtraction of logcosh terms can catastrophically cancel the linear term. For t>=0 use the cancellation-aware expression, then enforce the analytic odd symmetry g(−t)=−g(t):

    correction = log1p(exp(−2*b*abs(t−a)))
                 − log1p(exp(−2*b*(t+a))),
    g(t) = 2*a*(t−b)+correction                 if t>=a,
           2*(a−b)*t+correction                otherwise.

Do not turn underflow into an exact tie. General positive-coefficient monotonicity follows from

    g'(t)=2*a+b*tanh[b*(t−a)]−b*tanh[b*(t+a)].

It is even and increases with |t|. Its minimum is2*(a−b*tanh(a*b)). Therefore g has at most two stationary points, and its zeros are either only0 or0 plus one symmetric nonzero pair. In the latter case the positive root is strictly between0 and b; a numerical root rounded to b must remain a bracketed probability contribution, not an exact boundary. A proof and synthetic verification of the branch/root classification belong in the implementation review.

Special cases are explicit, with no tolerance-based rank snap:

* A=0 exactly: both reference first-bit densities are identical; balanced random predictions give correctness0.5 for every actual mean.
* B=0 exactly: use the usual Gaussian first-bit linear likelihood along A, with its analytic normal CDF.
* A perpendicular to B exactly: g=0; use the Gaussian CDF along A.
* Exact nonzero collinearity c=0: use the one-dimensional sign of g(t), **including all disjoint intervals**. If b*tanh(a*b)>a, there are three roots; a single threshold is wrong. Sum normal-CDF masses over the correct class's intervals. The exact a=b case has sign(t) with an overlap of reference classes; treat it analytically instead of relying on vanishing exponential corrections.
* Any nonzero c stays rank2, however small. Near-collinearity is a numerical difficulty, not authority to change the classifier. Exhausted precision/subdivision budgets produce unavailable rows.

Rank and reduced inner products are determined from **exact rational representations of the saved binary64 entries** (Python `float.as_integer_ratio()` or equivalent), then ball arithmetic for square roots and transcendental functions. For example D=(A·A)(B·B)−(A·B)² is evaluated exactly as a rational; D=0 means exact collinearity of the stored vectors. This is not a subtractive floating Gram determinant. Compute projected finite means with the same exact dot products before ball conversion. Record a,b,c,Gram determinant sign, reference angle diagnostics and ball radii. Compare with a stable floating QR/SVD reduction on synthetic cases; never substitute a precision-dependent rank cutoff into the definition.

## Numerical qualification before outcomes

Certified backend: local python-flint 0.9.0 with native FLINT 3.6.0 and Arb ball arithmetic, starting128 bits and a fixed ladder256,512 bits. The wheel was checked against official PyPI metadata and installed without changing the base NumPy/SciPy environment. See `requirements-exp020.lock` and `docs/EXP-020-runtime-provenance.json` for hashes and installation evidence. The full synthetic numerical and cost qualification remains a launch prerequisite. No paid service is needed. If the dependency cannot be qualified, revise the design before target scores; do not quietly replace certification with a QUADPACK success flag. Arb supplies outward error bounds, exact binary-input construction and error-function operations. [Official Arb Python documentation](https://python-flint.readthedocs.io/en/stable/arb.html), [ball-arithmetic integration paper](https://arxiv.org/abs/1802.07942).

Use a finite Gaussian interval centered on each actual projected mean with radius10 standard deviations. Certify both tails by normal-CDF bounds and add their full possible contribution to the probability enclosure. Split the interior at certified g stationary points and at the shifted transition roots g(t)+2*c*mz equal to0 and ±2*c, ±4*c, ±8*c; the latter improve efficiency but do not alter the decision rule. Root brackets crossing uncertain stationary points contribute their full probability mass until narrowed. Fixed root-iteration cap256 per bracket; fixed bracket/branch census must be retained.

On each monotone segment transform to probability mass q=Phi(t−mt). The conditional CDF integrand is monotone there, so a subinterval contributes between mass*min(endpoint values) and mass*max(endpoint values). Evaluate endpoints and Gaussian masses as balls. Bisect the contribution with largest upper-minus-lower uncertainty; deterministic ties use increasing segment/interval index. This yields a rigorous enclosure including projection, arithmetic, root-bracket, subdivision and tail uncertainty. Cap65,536 leaves per class score; do not discard uncertain mass. Required final interval width<=1e−4 in probability. These intervals certify this statistical calculation conditional on the saved arrays, not the ODE or measurement-noise assumptions.

Separately run SciPy QUADPACK at absolute tolerances1e−9 and1e−11, relative tolerance0, limit512/1024, using the same declared finite interval and transition splits. Require no warning/nonconvergence status, reported error<=1e−8, nominal results within1e−8 of each other and inside the ball enclosure (allow only their stated quadrature error, recorded). The nominal value is not the certificate. SciPy describes its error output as an estimate. [Official quad documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.quad.html).

Prospective scientific/numerical gates:

1. Reproduce the original linear classifier probability from each saved h, baseline-centered finite mean and sigma at the original rtol1e−12/atol1e−14, with differences reported separately from physical gates.
2. Every required new score has a finite bounded probability interval of width<=1e−4 and passes the nominal cross-check. Rank/root decisions and all unresolved cases remain explicit.
3. For all624 old coarse/fine or fine/independent class pairs, require the **maximum possible** absolute difference between the two new primary-decoder probability intervals to be<=.001. A midpoint match does not rescue overlapping/uncertain gate boundaries. Preserve failure/unresolved status.
4. Mandatory reference robustness: independently reconstruct A/B **and its own zero-write activation baseline** from the already saved alternate RK45 8T references on the unchanged direction and window. Seal this complete alternate reference classifier alongside the primary, before reading finite means. Evaluate both recipes on identical physical finite observations: `mu_alt = mu_primary + baseline_primary − baseline_alt`. Form this translation from the exact saved binary64 entries before ball evaluation, so projection/arithmetic uncertainty remains in the certificate. Evaluate all576 fine class means, with the same probability certification, and require conservative primary-versus-alternate probability difference<=.001. The primary decoder and primary baseline never change. The old linear-observer comparison does not automatically qualify the new nonlinear boundary. Never select the better reference method.
5. Reference-model controls: evaluate all four exact reference means for each declared classifier. The Bayes-dominance control applies only to **balanced accuracy averaged over all four equiprobable reference classes**, never separately to a class-conditional probability. Individual class accuracy may decrease under the Bayes classifier and must remain reported without failing this control. The balanced reference-mixture Bayes accuracy cannot be below the original fixed linear observer's balanced accuracy under that same reference truth beyond the rigorous probability interval. For alternate-reference controls, the original observer remains centered on the primary baseline: its projected mean includes `h·(baseline_alt−baseline_primary)` in addition to `h·(s1*A_alt+s2*B_alt)`. Omitting that offset would test a different, recentered linear observer. Primary and one-mode controls keep the primary baseline. Retain mismatch rather than choose the favorable output. This is a math/control gate, not a finite-trajectory benefit gate.

Synthetic tests must cover rank0, B=0, orthogonal vectors, signed B invariance, exact a=b overlap, collinear four-level discrimination with b>a, all single/three-root branches, near-collinear c down to representable stress scales, highly separated modes and cancellation/underflow, shifted actual means and mean mismatch, label symmetry, invariance when classifier baseline and reference/physical observation means are translated together, probabilities near0/1, all partial-budget failures, exact census, source/packet roundtrip and no target access before decoder seals. Compare closed special cases, a direct four-density likelihood, the scalar/rank2 reduction and the independent quadrature; stress inputs are synthetic, not chosen from new target gains.

## One-mode secondary comparator

Include one declared secondary nonlinear classifier using the already specified one-mode model A_one=rho*B with the same B and signed real canonical Floquet rho. Define the scalar relationship mathematically, rather than detecting collinearity of independently rounded product entries. It is rank1 by construction, so the certified root/CDF calculation is cheap. This tests whether any recovery already follows from generic nonlinear decoding of an overlapping two-write scalar signal. It is not a special shrimp mechanism or a finite-mean oracle.

Use the same actual finite means, priors, baseline and sigma. Do not substitute it for the full reference classifier, choose the better model or adjust rho. Missing/nonreal rho leaves all corresponding secondary cells unavailable and affects only secondary qualification. Preserve its reference-model controls and its own coarse/fine/independent probability comparisons separately. No secondary significance endpoint is proposed.

## Exact complete census and estimands

The controller predeclares144 setting/phase/window cells (48×3). Each has primary full-linear-mixture and alternate-reference-mixture seals plus one secondary one-mode seal,432 fixed classifier definitions including unavailable ones.

Finite evidence:48 contexts×3 windows×4 sign pairs×2 native methods=1,152 rows, plus4 panel contexts×3 windows×4 signs=48 independent rows, for1,200 class/method rows. Every row carries the original linear probability, the primary mixture probability interval and the one-mode interval. This is2,400 new finite-mixture calculations, not2,400 new neural trajectories. The alternate-reference robustness table adds576 fine-mean calculations. Reference-model controls are144 cells×3 recipes×4 signs=1,728 probability calculations. Total planned new mixture probability calculations **4,704**, with every unavailable/failure slot retained. Each of the 1,728 reference-model slots also evaluates an auxiliary certified fixed-linear Gaussian CDF, retained as `linear_reference`. This gives 4,704 mixture calculations plus 1,728 auxiliary linear calculations, or 6,432 new probability evaluations. Replaying the original linear scores is a separate arithmetic check. The 144 cell jobs, 432 definitions and all comparison denominators are unchanged.

There are624 primary method-comparison rows (576 native,48 independent), the same624 secondary comparison rows when available, and576 primary/alternate-reference comparison rows. Aggregation has300 method/context/window groups (144 fine,144 coarse,12 independent); each uses all four sign probabilities with equal weights. Fine is primary; methods are numerical checks, never independent replications. Output all144 fine context/window BA values and gain intervals per decoder; no favorable complete-case subset.

Single primary estimand: mean fine-method **full-reference-mixture minus original covariance-linear first-bit BA at4T**, equally weighted over all24 near-edge regression phases. This expands beyond the motivating phase16 and does not pretend that any phase is newly held out. Core gain, other delays, phase-resolved results, one-mode differences and phase16 itself are descriptive secondary diagnostics. There is no minimum-benefit gate, new bootstrap or population-significance claim. Report a conservative numerical enclosure for the primary mean by summing the class probability bounds and the fixed original linear values.

Each setting's qualified summaries require all its required primary cells, reference robustness controls and numerical method comparisons to qualify. The prespecified complete-study primary requires both settings to qualify, preventing a favorable subset after numerical failures. One-mode-only unavailability cannot invalidate the full-mixture primary but suppresses its own secondary aggregate. If the primary is withheld, retain all24 near-edge gain slots; for an unavailable nonlinear BA use[0,1] minus that phase's original linear BA to form explicit conservative fixed-denominator gain bounds. If the original linear baseline cannot replay, its contribution is also unknown. Do not present failure bounds as a confidence interval or a qualified mean.

## Lifecycle, implementation and interpretation limits

A new metadata-only plan binds the complete EXP-019 receipt, source/plan/summary,48 seals and relevant96 reference packets plus the saved score evidence. No other experiment's finite outcomes enter. Freeze code, complete task/classifier/score census, thresholds and synthetic numerical/cost evidence before target calculations. The controller first loads reference-only inputs and persists/reloads all432 classifier definitions; only then can it load the saved finite displacement vectors. This preserves the reference-only decoder construction although the scientific question is openly post hoc.

Use compact saved displacements where byte/semantic bindings and the completed EXP-019 replay already certify their relation to the original arrays; no newODE solves are permitted. A new verifier reconstructs the classifier definitions, enclosures, every score/control/comparison and aggregation from immutable evidence, without changing original EXP-019 gates. New failures remain in a new immutable attempt. Every original study stays unchanged.

Execution ceilings: 144 cell jobs, at most 4 owned CPU workers with one numerical thread each, 900 seconds producer plus 120 seconds sealing, 30 seconds per cell, 8 GiB combined owned RSS and 512 MiB raw output. Require at least 16 GiB available RAM at launch and preserve at least 8 GiB while running. These are bounded resource allowances informed by the synthetic measurement below, not promises that every target case will qualify. Budget exhaustion retains incomplete cells and suppresses affected summaries; no hidden target retry or tolerance relaxation.

Evidence of recovery would show that the **particular covariance-linear readout loses accessible first-bit information under interference**, not that the nonlinear system gained information. Full-reference Bayes decoding can still fail on the actual nonlinear means because of model mismatch; a negative result would not demonstrate information erasure. The analysis excludes trained-network generalization, universal memory capacity, joint two-bit capacity, a global phase-unknown observer, information-theoretic optimality under the true finite dynamics, and causal shrimp-geometry superiority. Cycle-normalized sampling, phase-informed calibration, independent Gaussian measurement noise and the two selected fixed-weight cycles remain explicit assumptions.

## Preimplementation amendment

The first draft held the baseline fixed in the alternate-reference numerical check. Independent review identified that a nonlinear boundary can also amplify baseline differences. Before any new target scores, root accepted the complete alternate decoder recipe and its fixed-physical-observation translation above. The original draft/contract are preserved as `DESIGN-v1-superseded.md` and `design-contract-v1-superseded.json`. No census, primary recipe or outcome-informed choice changed.

A second preimplementation clarification makes the Bayes-dominance control explicitly a balanced four-class mean, not a per-class requirement. Individual class tradeoffs are valid. The preceding baseline-corrected draft is preserved as `DESIGN-v2-superseded.md`; counts, thresholds and estimands are unchanged.


## Local evidence and execution contract

The input is the complete EXP-019 attempt at `/data2/neural-shrimp-data/raw/EXP-019/attempt-001/`, acquired from source `7eeba42182184a209dbc2f718e8469a5cd97e4a9` under plan SHA-256 `86ae6924a1a89c1d8ae878742f0b012680443fa2cad8746c1853a52d4b6c0543`. Its receipt SHA-256 is `ef9d6020c6c13f3b7ea89fddff4c4eabb112275c6c8b98724f2e56d0735c5001`. The new metadata-only certificate, `docs/EXP-019-reuse-qualification.json`, binds the raw and compact summaries and scientific audits without copying performance outcomes. It explicitly preserves the old alternate-reference strict arithmetic failures; no old gate is changed or reclassified.

Preparation must verify file bytes and hashes from the receipt, input seal, reference packet index and global reference gate. It may parse the metadata-only qualification certificate and reference-only packets. Raw and compact summary bytes can be hashed before sealing but their finite outcomes cannot be parsed for classifier construction. The same restriction covers the finite analysis packet. Persist and reload the full 144-cell, 432-definition census, including unavailable definitions, and an explicit global barrier before reading any finite displacement vectors. A failed global barrier prohibits finite scoring.

The original 96-component displacement vectors and original observer vectors remain binary64 records with exact rational interpretation. Alternate-baseline translation is a sum of exact representations of the three saved vectors, not a newly rounded binary64 vector. Likewise reference class means and the one-mode mathematical scalar relation must preserve their declared exact algebra. Every calculation receives its fixed class identity, saved source binding and probability slot identifier. No ODE acquisition, latent fitting or new finite response enters this analysis.

Use four bounded CPU worker processes and one numerical thread each, supported by the source-bound synthetic benchmark and a contemporaneous shared-machine inventory. The algorithm depends on exact rational arithmetic, arbitrary-precision balls and small adaptive scalar quadratures. There is no validated GPU implementation of these operations in this project. No claimed GPU speedup or paid fallback is assumed. The EXP-012 GPU integration benchmark is a different workload and is not a timing estimate for EXP-020.

Root authors the protocol and an independent synthetic integration test. Separate agents author the numerical core, pure evaluator and controller. Review must disclose these authorships; producer replay alone does not establish independent validation. Synthetic controls may expose design or code problems and justify dated, pre-outcome amendments. Any change after new decoder outcomes exists must be a new visible correction or successor, preserving this attempt.

Before target execution, bind this protocol and the complete executable machine plan in `docs/EXP-020-local-review.json`, run the applicable tests, and verify the private remote freeze commit. The local review binds source-specific audits and synthetic test evidence; it is the final launch gate. New raw output belongs in an exclusively created `/data2/neural-shrimp-data/raw/EXP-020/attempt-001/`. Only compact receipts, source, tests, protocol and selected figures belong in Git.


## Numerical implementation details and root completeness

For positive a and b the derivative can also be written

    g'(t) = 2*a - 2*b*sinh(2*a*b)/(cosh(2*b*t)+cosh(2*a*b)).

It is even and strictly increases on t>0. Thus its value at zero decides between an everywhere increasing likelihood and a likelihood with exactly one positive and one negative extremum. In the latter case, g initially decreases from zero, then increases to infinity. Since g(b)>0, it has exactly one positive decision root below b; odd symmetry supplies the negative root. Positive predictions occupy (-r,0) and (r,infinity). These intervals are all retained by the scalar calculation. When a=b the positive correction has the sign of t even if a floating exponential underflows; the analytic monotone branch prevents false ties.

Root brackets stop at an absolute width of 1e-14 with at most 256 bisections. A scalar bracket may use the certified exterior endpoint b.upper()+1e-14: this avoids subtracting two uncertain copies of an irrational b when g(b) is extremely small. Both endpoint signs still require certification, and the decision root is never snapped to b. This was tested on synthetic irrational norms with high signal separation before target calculations.

The enclosure integrates the centered Gaussian variable u=t-mt on [-10,10]. Each dyadic endpoint is interpreted exactly; Arb bounds the Gaussian mass and the integrand range using endpoints and every possibly intersecting stationary-point bracket. Approximately equal-mass bisection is only an efficiency choice; validity does not depend on the floating inverse-CDF split location. Unresolved optional transition splits cannot discard mass, and stationary uncertainty remains in the integrand bound. Serial interval identifiers break equal-width priority ties deterministically.

The independent nominal integrator may merge split hints separated by at most 1e-10 to avoid QUADPACK failures on intervals a few machine steps wide. This affects only hints, never the integration domain, likelihood or certified partition. Both nominal tolerance runs must still meet their error/agreement gates and agree with the rigorous interval. Closed-form cases use a separate SciPy normal CDF with a recorded allowance of eight binary64 machine epsilons for nominal rounding; their Arb probability interval remains the certificate.

Exhausted subdivisions, roots, precision or nominal checks retain the attempted precision ladder, diagnostics and any available partial probability enclosure. They do not become qualified by agreement with a favored outcome.


For probability display, nominal estimates are clipped to [0,1] only after recording the raw estimate and correction. Qualification uses the raw nominal value, its stated numerical error and the certified interval. A tiny QUADPACK overshoot above one therefore cannot break serialization or silently bypass a gate.


## Measured synthetic cost and execution decision

The final synthetic measurement used the held numerical core SHA-256 `04c586df68f8d06527146243035b1182b6cd7fa75c056b55a33f6794be66e2f3`. All 40 declared 96-dimensional test calculations completed and qualified. The run took 4.159312 seconds total; the median probability took 0.011521 seconds and the maximum 0.788944 seconds. Process peak RSS was 99,811,328 bytes. These are synthetic costs, not target outcomes. The earlier cost record remains retained.

The source-bound receipt is `/data2/neural-shrimp-data/audits/EXP-020/core-synthetic-cost-002/receipt.json`, 1,184 bytes, SHA-256 `76eed8c98f0b2d5a08472bfe005cd50e208ab4f5b38e2e78b422ccf2f886e22e`. It binds the script, source, every measured result and standard output. Six numerical thread limits, package versions, native FLINT version, exact process identity and output sizes are recorded. The earlier `core-synthetic-cost-001` record is preserved because a subsequent display-only nominal guard changed the source.

At the slowest observed synthetic probability time, a 40-slot panel cell's 24 potentially rank-two mixture calculations would take about 18.94 seconds, plus 16 cheaper one-mode calculations, auxiliary linear controls, exact dot products and serialization. The fixed 30-second cell ceiling allows overhead but is not a worst-case guarantee. A 144-cell schedule at that illustrative cost would take about 682 seconds on four workers, before reference sealing and aggregation, within the 900-second producer ceiling. Harder cases remain subject to failure and missingness rules.

A separate integration test passes an analytic nine-maximum synthetic oscillator through the real predecessor acquisition/packet interfaces and the new classifier/evaluator, including all 40 panel slots, three numerical-method labels and saved packet roundtrips. It verifies wiring and evidence handling, not neural solver accuracy. Separate direct four-density likelihood integration checks shifted synthetic means without using the producer's likelihood reduction. The final source-bound local review records the unified suite result and independent source reviews.
