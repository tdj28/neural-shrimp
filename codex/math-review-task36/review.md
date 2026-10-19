# Task36 mathematics review

Status: article pass completed and privately delivered at 2026-10-19T00:45Z; full-report mathematics pass complete at 2026-10-19T00:47Z. Findings are ranked within the article-first and report-only sections. This completes Task36 review of these snapshots; it does not mark any suggested manuscript correction as applied. Reviewed article `/data2/chaos-journal-neural-shrimp/main.tex`, SHA256 `5a37af4762daab697afe41ac13b492d5d9b7ab1cd2e610056d6f64a0b559ed6a`, at 2026-10-19T00:39Z. Line numbers refer to that snapshot. Papers are not edited. This is a source/logic review, not a new numerical experiment or re-audit of closed certificates.

The feedback map can support distinct stable periodic behaviors, and the existing local certificates establish those behaviors at their exact parameters. The core map/derivative algebra and local proof logic checked here are sound. The geometry, symbolic-dynamics and universality summaries need the corrections below. The current blockers concern what the measured spectral diagnostics imply, not the existence of the previously certified periodic orbits.

## Blocking

### A1. A leading-pair sign change does not establish a noninvertible map

- **Location/text:** article `main.tex:1011–1014`, especially “their product D=mu1 mu2 changes sign … a noninvertible setting in the sense of Mira”; report `paper/llm/main.tex:2256–2259` repeats it.
- **Problem:** A product of two eigenvalues selected by largest modulus is not the determinant of the 896-dimensional derivative. The selected pair can change identities without any eigenvalue of the full derivative reaching zero. No continuously isolated two-dimensional invariant subbundle or actual two-dimensional reduction has been established; the article explicitly admits the latter at1042. Orientation of that selection therefore cannot establish singularity/noninvertibility of the actual return map.
- **Evidence/counterexample:** For the nonsingular family diag(0.9,0.5−0.2s,−0.4), 0≤s≤1, the full determinant is strictly negative throughout. The two largest-modulus eigenvalues have positive product for s<0.5 and negative product for s>0.5 solely because the second/third ordering swaps. Append 893 nonzero small diagonal entries to obtain the same issue in 896dimensions. Façanha–Oldeman–Glass2013, p1265, defines D as the determinant of the complete 2×2 monodromy and explicitly assumes fixed sign; p1267 says higher dimensions require additional characteristic coefficients. This is not the same diagnostic as a selected pair product.
- **Replacement:** “The product of the two largest-modulus multipliers changes sign in part of the window. This differs from the fixed-sign two-dimensional Hénon setting, but does not by itself establish noninvertibility or a two-dimensional reduction of the full map.” Remove the asserted inference to Mira’s noninvertible setting unless separately proved.

## Major

### A2. Dominant-real sign-change contours are not proved superstable curves

- **Location/text:** article `main.tex:949–956` and `1032–1034`, “superstable lines” / “superstable curves of the continued orbit”; plotting source `tools/plot_superstable_lines.py:163–185` constructs zero segments by interpolating the modulus-ranked dominant-real field.
- **Problem:** A dominant eigenvalue can change from +a to −a when two nonzero real eigenvalues exchange dominance. Interpolating this discontinuous label produces a spurious zero contour. A genuine zero eigenvalue alone does not establish vanishing spectral radius or all-direction superattraction; the remaining multipliers must still be checked. Source terminology cannot be transferred from a one-dimensional multiplier to a selected higher-dimensional eigenvalue without checking the full spectrum/continuous branch.
- **Evidence:** diag(0.5−0.2s,−0.4) switches the dominant-real sign at s=0.5 while its spectral radius is always≥0.4 and neither eigenvalue vanishes. Façanha2013 pp1265–1267 uses T=trace(M) for the two-dimensional skeleton, with TypeA/B defined by that continuation. The report itself describes nonzero opposite-sign multipliers exchanging dominance at4058–4064.
- **Replacement:** “Contours separating cells with positive and negative dominant real multipliers; they indicate changes in the dominant direction and are not verified superstable curves.” Call the resemblance “Type-B-like avoidance of the plotted contours,” not a verified TypeB trace skeleton, unless a continuous trace contour is independently established. Update caption/figure labels consistently.

### A3. Two-dimensional superstable exclusion needs the nonsingularity condition

- **Location/text:** article `main.tex:911`, “Two-dimensional maps have no superstable curves.”
- **Problem/evidence:** This is false for arbitrary two-dimensional noninvertible maps. For example f_a(x,y)=(x²+a,0) has a fixed point at(0,0) with both multipliers zero when a=0; a redundant second parameter gives a superstable locus. Nonsingular monodromy excludes zero multipliers, not dimension alone. The article’s body at976–977 and the report caption at2369 already contain the appropriate qualification.
- **Replacement:** “In two-dimensional maps with a nonsingular Jacobian along the orbit, such as the nondegenerate Hénon map, no multiplier vanishes; trace-zero curves provide an analogous skeleton.”

### A4. Synchronization threshold omits its parameter range

- **Location/text:** article `main.tex:1068–1070`, synchronized state stable for c>c*=(1−exp(−lambda1))/2.
- **Problem/evidence:** Exact transverse linearization is (1−2c)DF, hence negativity requires abs(1−2c)exp(lambda1)<1. For lambda1>0 and 0≤c≤1, this gives c*<c<1−c*. The displayed lower-threshold-only statement is valid on0≤c≤1/2, not the entire coupling interval.
- **Replacement:** “For 0≤c≤1/2, the transverse Lyapunov exponent is lambda1+log(1−2c), so it becomes negative above c*. More generally the condition is abs(1−2c)<exp(−lambda1).” This is a local asymptotic transverse condition, not a global synchronization guarantee.

### A6. The 65 repeating-text cells do not all have text period seven

- **Location/text:** article `main.tex:833–834`, “In 65 of them … repeats exactly, with period7.”
- **Problem/evidence:** Report `paper/llm/main.tex:2048–2049` says65 repeating-text cells, of which59 have a period divisible by7, most often7 or28. Accepted Task19 stress records also retain periods10 and28. This is a contradiction between manuscript summaries, not a new raw-array recount. The strongest illustrated cell's period must not be assigned to all 65.
- **Replacement:** “In 65 of them the leading token repeats throughout the2000 recorded steps;59 have a text period divisible by seven. The strongest example shown below has period seven.” Preserve the existing finite-horizon qualification.

### A7. A finite token record does not establish a non-generating partition

- **Location/text:** article `main.tex:842–843`, “the partition … is not generating”; report `paper/llm/main.tex:2057–2059` repeats it.
- **Problem/evidence:** Generation is an infinite-itinerary property, or an almost-everywhere property relative to a specified invariant measure. Even a generating partition associates many states with one finite word. A2000-step repeating word and positive finite-time growth do not establish identical infinite itineraries for distinct orbits. The report itself says the sampled groups are not proved invariant. See Eisele1999, sections2–4, https://chaosbook.org/library/Eisele99.pdf, on finite words versus infinite symbolic codes.
- **Replacement:** “The finite leading-token record hides the state variability and positive finite-time growth observed here. This is consistent with non-generating token coding on the putative chaotic set, but does not establish that infinite-itinerary property.” The directly supported behavioral observation remains strong without the formal generating claim.

### A8. Numerical noise has no unconditional cannot-hide-chaos guarantee

- **Location/text:** article `main.tex:439–443`, “can fake a positive exponent but does not hide one”; report `paper/llm/main.tex:768–772` repeats the broad summary.
- **Problem/evidence:** The maximum formula is a simplified noise-floor model. Noise can change the orbit, its sampled distribution or its basin; rounding can erase a twin separation. The report itself records lower noisy estimates at3509–3510, lower float32 values at3517–3527, noise-induced order at3535–3537 and identical rounded rows at3540–3542. Its angular recurrence also needs stronger assumptions than independent noise alone; see the report-only derivation finding below.
- **Replacement:** “In a simplified noise-floor model, additive numerical error can produce a spurious positive twin estimate, approximately max{lambda, log(sqrt(2)sigma/epsilon)}. Rounding can also suppress separation or change the sampled dynamics, so we compare offsets and tangent estimates rather than use this approximation as a one-sided guarantee.”

### A9. Finite spectra and scalar fits do not force a scalar reduction or universality

- **Location/text:** article `main.tex:840–841`, “follows a smooth map with a single quadratic maximum”; `1063–1066`, “which is why one-dimensional maps and Feigenbaum's ratios describe these orbits”; `790–792`, general strongly dissipative universality wording. Report `paper/llm/main.tex:2532–2537` strengthens this to “inevitable.”
- **Problem/evidence:** A fitted return relation need not be an autonomous scalar factor or a coordinate on an invariant curve. Partial finite spectra and Kaplan–Yorke estimates do not establish either. Strong dissipation alone does not force a cascade, and a unimodal family need not traverse the required bifurcations. Collet–Eckmann–Koch, DOI10.1007/BF01008475, treats suitable analytic families, not arbitrary dissipative maps. The report's own two-logistic-map example at2079–2088 states the relevant limitation.
- **Replacement:** “On the sampled range, a degree-eight scalar fit has a single quadratic maximum.” In the discussion: “The finite spectra and singular-value ratios support strong effective contraction in the sampled directions and motivate the scalar approximations. The measured cascade separately provides finite evidence consistent with Feigenbaum scaling; no global one-dimensional invariant reduction is established.” Say scaling also occurs in suitable strongly dissipative higher-dimensional families, rather than follows from dissipation alone.

### A10. The discussion overstates what derivative factorization establishes

- **Location/text:** article `main.tex:1050–1051`, factorization “explains where the chaos lies and how steering creates it.”
- **Problem/evidence:** The identity decomposes growth; it is not a sufficient chaos criterion or a causal autonomous intervention result. Article330–337 explicitly says short-block inequalities do not locate chaos, and409–416 says frozen-path factor swaps do not show that a readout-only change sustains an attractor. The summary drops both restrictions, contrary also to accepted Consult17B03.
- **Replacement:** “The factorization gives sufficient high-temperature contraction and conditional low-temperature bounds, and separates readout-side and network-side contributions to growth along recorded orbits.” No numerical result needs changing.

### A11. The rank-one explanation needs its measured spectral discrepancy

- **Location/text:** article `main.tex:1024–1026`, “eigenvalues small because its leading left and right singular vectors are nearly orthogonal”; report `paper/llm/main.tex:2246–2251` says “not because a second direction matters.”
- **Problem/evidence:** For the rank-one truncation the formula lambda=sigma1(v1 dot u1) is exact, but its eigenvalue is not the full matrix's eigenvalue. The report gives lambda about−0.00027 versus actual leading moduli0.0034–0.0036, about 13 times larger. The discarded spectral-norm remainder is sigma2, approximately 65×0.00018=0.0117 at that phase, larger than the full leading modulus. Small relative singular-value error does not guarantee a small relative eigenvalue error for a strongly nonnormal matrix. This does not refute the near-rank-one observation or the explanatory relevance of near orthogonality.
- **Replacement:** “The leading rank-one truncation is nearly nilpotent because its singular vectors are nearly orthogonal. It illustrates how a large one-return norm can coexist with small eigenvalues, but its eigenvalue is about 13 times smaller than the full leading modulus; the residual terms matter for the full spectrum.” Remove the exclusion of the second direction without an actual error bound.

## Minor

### A5. Fourth-leg caption contradicts the stated period count

- **Location/text:** article `main.tex:935–936`, “every point carries the period-5 family,” versus994–995, “period 5 at127, period 20 at one.”
- **Problem:** “Family” can include doubled descendants, but the caption can be read as all 128points having primitive period 5. The text already provides the exact exception; preserve it in the caption.
- **Replacement:** “All 128sampled points retain the followed family:127 have period 5 and one has period 20.”

### A12. Strongest contraction is not established by the stated root solve

- **Location/text:** article `main.tex:1021`, “This point of strongest contraction”; report `paper/llm/main.tex:2240` repeats it.
- **Problem/evidence:** Solving two characteristic-coefficient conditions finds their common zero, not a minimum of spectral radius or an induced norm. The same paragraph reports a norm near65. No optimization or lower-bound comparison establishing the strongest point is described.
- **Replacement:** “This strongly asymptotically attracting candidate head” or “this point with small leading multiplier moduli.” Keep near-nilpotent as an explicitly descriptive label, not a matrix-norm bound or exact nilpotency claim.

### A13. State the uniform-gap condition in the low-temperature preview

- **Location/text:** article `main.tex:295–297`, “excludes chaos … wherever a margin separates”; `339–347`, “uniform logit gap.”
- **Problem/evidence:** The intended sufficient condition is a positive gap delta0 independent of both state and sufficiently small temperature, at fixed alpha. A positive gap at each temperature can shrink as O(T) and does not imply decay of the derivative. The body may intend the stronger meaning of uniform; this is a quantifier correction, not a failed derivation. With D=diam(E), ||Sigma||<=D²(V−1)exp(−delta0/T), so ||DF||<=L D²(V−1)exp(−delta0/T)/T uniformly.
- **Replacement:** “At fixed steering strength, sufficiently low temperature excludes positive Lyapunov exponents on invariant sets whose winning-logit gaps stay uniformly bounded away from zero as T tends to zero.” This does not establish global exclusion near ties or a unique fixed point on a nonconvex invariant set.

### A14. Keep Kingman's invariant-measure and asymptotic qualifications

- **Location/text:** article `main.tex:332–333`, the subadditive theorem “guarantees convergence as k grows”; report `paper/llm/main.tex:549–575`.
- **Problem/evidence:** Kingman's theorem is an almost-everywhere asymptotic statement for invariant-measure-typical orbits (with its integrability hypotheses), not a bound on an arbitrarily selected later finite segment of the fixed start. The source distinguishes disjoint dyadic bounds from overlapping-window averages; only the former bound growth over the same complete finite record. The empirical comparison with the subsequent2000-step estimate remains valid as an empirical comparison.
- **Replacement:** “Kingman's theorem supplies an asymptotic invariant-measure statement without a rate; the agreement here between finite windows of the recorded orbit and a later exponent estimate is empirical.” See the exact overlapping-window counterexample in the report-only findings.

### A15. Keep the unique-maximum qualification in the zero-temperature summary

- **Location/text:** article `main.tex:1054–1055` and `1104–1105`, greedy token cycles at zero temperature.
- **Problem/evidence:** The body at343–345 correctly says a tie yields the average of the tied embeddings. The limit has finite image and is eventually periodic, but its cycles need not consist of single token embeddings.
- **Replacement:** “Where each visited state has a unique maximizing token, the zero-temperature limit is greedy decoding with one-token memory and its eventual cycles are token cycles.” Elsewhere say finite-valued limit agreeing with greedy decoding away from ties.

## Full-report-only findings

Reviewed `paper/llm/main.tex`, SHA256 `d709c5acc8afb7557fb0c3ff1730d0b2c8bcf41bb143a237084dd6c3d1bd8816`. All report line numbers below refer to that snapshot. Article findings with report locations above apply to both; in particular A1/A2, A7/A8 and A9 are repeated in the report. At2532–2537 the claim that four estimated exponents imply an attractor nearly on a curve and make Feigenbaum ratios “inevitable” needs A9's replacement. Report3046–3051 already contains the coupling-domain and local-versus-global qualifications missing from the article, so it should be preserved.

### Major

### R1. Spectral radius is not the one-period contraction factor

- **Location/text:** report `paper/llm/main.tex:2240`, “contracts about 280-fold per period.”
- **Problem/evidence:** The same paragraph reports leading multiplier moduli about 0.0035 but an induced Euclidean norm sigma1 about 65 at one phase. A perturbation along the leading right singular vector can grow approximately 65-fold over one linearized period. Spectral radius governs asymptotic linear growth rates, with possible transient amplification and norm-dependent prefactors, not a uniform one-step bound. This is also why the actual interval certificates test matrix powers separately.
- **Replacement:** “Its estimated linear asymptotic contraction factor is about 0.0035 per return, despite possible transient amplification: at one phase the one-return operator norm is about 65.” Do not describe this as every perturbation contracting280-fold per period. The rank-one explanation also requires A11.

### R2. A positive sum of two exponents is not a second expanding direction

- **Location/text:** report `paper/llm/main.tex:3686–3687`, a nonnegative sum of the first two estimates “suggests a second expanding direction”; `3681` says none has two positive estimates.
- **Problem/evidence:** lambda1+lambda2 is the leading two-dimensional area-growth rate. It can be positive with lambda2<0. For example exponents(0.2,−0.1,−5) have one positive exponent, positive two-dimensional area growth and Kaplan–Yorke value2.02. The printed Kaplan–Yorke formula itself is correct.
- **Replacement:** “At two cells the sum of the first two finite-time estimates is nonnegative, indicating noncontracting two-dimensional area at this horizon, although only one estimated exponent is positive.” Do not infer a second positive exponent or invariant surface.

### R3. Independent noise does not imply the exact angle recurrence

- **Location/text:** report `paper/llm/main.tex:3478–3487`, “one step gives” c_next²=exp(2lambda)c²/(exp(2lambda)c²+r²), followed by an exact maximum formula.
- **Problem/derivation:** Normalize the difference noise by the reset twin offset and decompose it along a fixed surviving direction. If all other deterministic directions are killed, the actual update is

  `c_next² = (exp(lambda)c + eta_parallel)² / ((exp(lambda)c + eta_parallel)² + ||eta_perp||²)`.

  The printed recurrence additionally sets parallel noise to zero and replaces the fluctuating perpendicular squared norm by r². Independent noise between rows explains an RMS sqrt(2) factor, not those extra assumptions. Gaussian noise has both a parallel component and norm fluctuations; expected logarithmic growth is not the logarithm of an expected norm. A constant leading direction/amplification is another idealization.
- **Replacement:** “In a simplified model, we neglect noise parallel to a fixed leading direction, replace its perpendicular norm by its RMS value, and take the leading amplification to be constant. The resulting angle recursion gives the approximation …”. Mark the maximum formula approximate for the network, retain the measured crossover discrepancy up to0.43 at3507–3508, and apply A8 to its summary surfaces.

### R4. Per-step numerical error does not bound orbit displacement with unit gain

- **Location/text:** report `paper/llm/main.tex:3537–3539`, noise of10^-8 moves a stable orbit “by about that much,” so it changes the attractor only within that distance of a boundary/bifurcation.
- **Problem/evidence:** For the stable scalar map x_next=rho x+delta, a constant perturbation moves the fixed point by delta/(1−rho), arbitrarily larger than delta as rho approaches1. With independent zero-mean noise, the stationary variance carries a factor1/(1−rho²). In higher dimensions the resolvent, transient nonnormal growth and accumulated perturbations matter. Stability alone provides no unit-gain basin margin. This algebraic counterexample does not assert a basin switch in the recorded run.
- **Replacement:** “Small per-step errors can accumulate or be amplified near weak stability and in nonnormal dynamics. Whether they change the attractor depends on the response and basin geometry; the per-step error alone does not bound displacement or exclude a basin switch.”

### R5. The 17/17 sign observation lacks a calibrated sign-test null

- **Location/text:** report `paper/llm/main.tex:1606–1608`,17 negative residuals and “two-sided sign test, p=2^-16.”
- **Problem/evidence:** The arithmetic2/2^17 is correct for independent fair signs. These are related deterministic passage errors at offsets/entrances from four folds, with shared coefficients and truncation errors. No sampling/randomization model establishing that joint null is given. Freezing a later directional prediction does not make these earlier signs iid.
- **Replacement:** “Every one of the 17 nonzero midpoint/kicked-start residuals was negative, a consistent directional discrepancy across these related deterministic cases.” Remove the inferential p-value, or label it explicitly as conditional on an unvalidated iid fair-sign model. Keep the later prospective sign criterion and its actual outcome separate.

### Minor

### R6. Distinguish disjoint dyadic bounds from overlapping-window diagnostics

- **Location/text:** report `paper/llm/main.tex:549–575` and figure caption596–601, block averages “tighten as k grows” and overlapping-window averages called bounds.
- **Problem/evidence:** For a fixed N-step record and k dividing N, define B_k=(1/N)sum_j log||P_{jk,k}|| over disjoint blocks. Nested dyadic partitions satisfy Lambda_N<=B_2k<=B_k. Overlapping averages need not bound the full record or tighten monotonically. Exact scalar log factors(10,0,0,10) give Lambda_4=5 but the mean of the three2-step window rates is10/3. Logs(0,10,10,0) give overlapping rates B_1=5 and B_2=20/3, an increase. Current bound_hierarchy.py distinguishes these two calculations correctly.
- **Replacement:** Define disjoint B_k explicitly as above, then say “Overlapping-window averages for other k are finite-window diagnostics, not generally bounds on the whole record or the later exponent estimate.” Repair terminology rather than data; retain A14's asymptotic qualification.

### R7. The identity-network preview needs the tied-embedding condition

- **Location/text:** report `paper/llm/main.tex:414–418`, “the transformer … makes cycles and chaos possible at all,” after mentioning the untied case.
- **Problem/evidence:** The later body469–473 correctly permits cycles without the transformer for untied embeddings. For example E=(1,−1)^T, W=−E yields F(x)=−tanh(x/T), which has a nonzero 2-cycle for 0<T<1. The tied Hopfield descent proof is correct, but its scope must survive the preview.
- **Replacement:** Add “in the tied-embedding identity-network baseline” to the preview. Do not generalize the no-cycle conclusion to arbitrary untied readout/input weights.

### R8. Attach the parameter unit to a, not b

- **Location/text:** report `paper/llm/main.tex:1602`, “a=−11.900, b=−152.05 per unit log10 T.”
- **Problem/evidence:** a=ell^T partial_p G carries the inverse parameter unit; b=ell^T D²G[r,r]/2 is the quadratic state coefficient. Methods3585–3586 already state this correctly.
- **Replacement:** “a=−11.900 per unit log10 T and b=−152.05.” Retain the eigenvector/state normalization on which individual coefficients depend.

### R9. Router margins control relative logit differences

- **Location/text:** report `paper/llm/main.tex:4621–4623`, a perturbation must move router logits “by about this gap” to switch an expert.
- **Problem/evidence:** A gap g closes when the relative change of the two logits reaches g. If every logit changes by at most epsilon,2epsilon<g is a sufficient no-switch condition. “About” is informal, but this sharp statement avoids a factor-two ambiguity. The finite observed twin no-switch result remains unchanged.
- **Replacement:** “A switch requires closing this pairwise logit gap; a bound below half the gap on every logit's change prevents it.”

## Review scope and sources

This is a derivation, source and interpretation review, not a new model run or a repeat certificate audit. It reads the exact article first and the corresponding mathematical portions of the full report second, checks source formulas and accepted finite evidence where relevant, and preserves all failed or unresolved records. It does not newly certify historical model execution, asymptotic chaos, global invariant curves or an exhaustive bibliography.

Primary source checks used Gallas1994pp204–205; Façanha–Oldeman–Glass2013pp1265–1267 and Fig4; the original Carcassès/Mira quartic models; Collet–Eckmann–Koch's publisher record; and Kingman/Oseledets primary mathematical statements. The local Carcassès/Mira filenames labelled2012 contain the original1991/1992articles, so filename dates are not bibliographic authority. Kingman's publisher DOI is10.1214/aop/1176996798; the invariant-measure hypotheses were also checked in Filip, Theorem3.5.2, https://arxiv.org/html/1710.10694. The noninvertible Oseledets distinction is discussed in González-Tokman–Quas, pp1–2, https://web.uvic.ca/~aquas/papers/paper41.pdf. No private manuscript text was sent to external search.

Accepted Task19 evidence was reused rather than rerun: synthesis SHA256 `66feba6283cd4ac70738d17699a0bee64c7fec8386a9d8ff434da9fac129914f`, stress04 note `3e1cf27f1f0313d156421ed3f6d8117430b0f23a3c229f2ebb1c304a623e9bf2`, stress13 note `6004cce7a60a35ed66263b0ea116b3bd5e8aa94cfa4834811bbecfb9d52719de`. No manuscript, producer tools, shared research log or public artifact was edited.


## Checked and found correct

- Article859–882 distinguishes shooting-root uniqueness in the full product box from uniqueness of all F^p roots in the first box; correct scope.
- Proper-divisor separation at shifts1,2,4 suffices for period8; prime periods2/3 need shift1. Different primitive periods imply different orbits.
- A rigorously enclosed monodromy power with induced norm<1 implies spectral radius<1 and local attraction for a C1map; no whole-original-box trapping follows, and the article does not claim it.
- The explicit tiny basin balls and fixed-start-unproved caveat match the separate released authority; no certificate rerun was required.
- Façanha2013 Fig4,p1267 supports TypeA loop and TypeB avoidance geometry, but as trace skeletons of a two-dimensional map, not automatically dominant-real signs of896dimensions.
- The one-dimensional Gallas quartic, fold/flip multiplier values and head/tail distinction are compatible with Gallas1994pp204–205; a finite observation of doubled shells remains finite evidence, not an infinite-mosaic theorem.

- Exact autonomous state definition: fixed prefix/positions and one fed-back slot; no growing cache. Dense-model positive-epsilon RMSNorm, SiLU and attention yield a smooth/analytic map for T>0; hard MoE routing is correctly excepted.
- The convex hull of embeddings is invariant and absorbing after one step. Brouwer gives a fixed point; compact continuity gives invariant probability measures. Bounded log-plus derivative norm supports the forward/noninvertible Oseledets filtration, with possible minus-infinity exponents and almost-everywhere scope.
- Softmax differentiation gives E^T(diag(p)−pp^T)J_z/T. Tied output weights give Sigma_p J_h/T; untied weights give Cov_p(e,w)J_h/T with the correct orientation. An untied cross-covariance need not be symmetric or positive semidefinite.
- Popoviciu gives ||Sigma||<=diam(E)^2/4; the untied bound is diam(E)diam(W)/4 by Cauchy–Schwarz. On the compact convex hull, a uniform derivative bound<1 yields global contraction after the first iterate. Sampled power-iteration norms are correctly not treated as certified upper bounds on the global L.
- The fixed-state low-temperature leading exponential covariance term is correct. At ties the limiting embedding average and finite-image eventual periodicity are correct; see A13/A15 for summary conditions.
- The tied identity-network Hopfield energy argument is valid: convexity gives energy decrease by at least half the squared step; compactness implies steps tend to zero and accumulation points are fixed. Nontrivial cycles are excluded. Untied embeddings can cycle without the transformer; the body states that exception.
- The directional gain identity telescopes with the printed norm-correction sign. The frozen-path intervention/body interpretation is correctly conditional on the recorded orbit; A10 addresses the broader discussion claim.
- Fold passage prefactors agree with the oriented quadratic normal form: full-channel pi and midpoint pi/2, with arctangent differences for finite boundaries. The one-third reinjection fraction is empirical, not universal. Fold coefficients are applied to the classified +1 crossing, flips to−1; current cascade source has its final critical-reality/test-function/Ritz-residual acceptance guard.
- For z'=−(1+beta)z+c z^3, the leading doubled-cycle amplitude sqrt(beta/c) and doubled multiplier1−4beta+O(beta²) are correct. Complement stability still needs its separate multiplier checks.
- The article reports finite cascade location uncertainty as estimated rather than certified, and numerical chaos/basin labels as finite-horizon. No infinite cascade or asymptotic-chaos proof was inferred.

- The normal-form coordinate orientation is correct: v=sign(b)u and p−p_c=sign(ab)epsilon give v_next=v+|a|epsilon+|b|v². For A=|a|, B=|b|, the finite-boundary continuous passage is [atan(U sqrt(B/(Aepsilon)))−atan(v_in sqrt(B/(Aepsilon)))]/sqrt(ABepsilon). Counts are iterates of G=F^q; conversion to original map steps multiplies by q. This is a truncated normal form, not an exact global native recurrence.
- The original all-row prospective passage failures and secondary/primary predictor distinctions are retained. A successful secondary or amended later test does not erase an earlier failed criterion.
- The report's original49/62 versus amended51/62 scalar-fit results, adverse delay-predictor and stress evidence are disclosed. Those limitations should govern the broader interpretation too.
- The Kaplan–Yorke formula and its simplification under lambda1+lambda2<0 are correct. With a truncated spectrum, the cumulative sum must cross within the retained tail before reporting a definite estimate; finite-time estimates are not proved asymptotic dimensions.
- The local cusp coefficient with its center-manifold correction, nonzero cubic condition and two-parameter unfolding-rank requirement are correctly separated from a mere tangent reversal. The swallowtail quartic fixed-point algebra is correct; the seeded13-point census remains a finite, nonexhaustive test, with mismatches retained.
- The report's certificate appendix correctly discloses the trusted arithmetic/evaluator stack, independent saved scalar/row-norm audit rather than a second transformer execution, exact dyadic parameter interpretation and real exp/log temperature, and exclusions of fixed-start reachability and global geometry. No new certificate was issued by this review.
