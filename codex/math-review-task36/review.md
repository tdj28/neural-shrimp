# Task36 mathematics review

Status: article pass in progress; immediate findings below are ready for triage. Full-report pass follows. Reviewed article `/data2/chaos-journal-neural-shrimp/main.tex`, SHA256 `5a37af4762daab697afe41ac13b492d5d9b7ab1cd2e610056d6f64a0b559ed6a`, at 2026-10-19T00:39Z. Line numbers refer to that snapshot. Papers are not edited. This is a source/logic review, not a new numerical experiment or re-audit of closed certificates.

The exact feedback-map mathematics and local computer-assisted proof statements appear substantially stronger than the geometry interpretations. The current blockers concern what the measured spectral diagnostics imply, not the existence of the previously certified periodic orbits.

## Blocking

### A1. A leading-pair sign change does not establish a noninvertible map

- **Location/text:** article `main.tex:1011–1014`, especially “their product D=mu1 mu2 changes sign … a noninvertible setting in the sense of Mira”; report `paper/llm/main.tex:2256–2259` repeats it.
- **Problem:** A product of two eigenvalues selected by largest modulus is not the determinant of the896-dimensional derivative. The selected pair can change identities without any eigenvalue of the full derivative reaching zero. No continuously isolated two-dimensional invariant subbundle or actual two-dimensional reduction has been established; the article explicitly admits the latter at1042. Orientation of that selection therefore cannot establish singularity/noninvertibility of the actual return map.
- **Evidence/counterexample:** For the nonsingular family diag(0.9,0.5−0.2s,−0.4), 0≤s≤1, the full determinant is strictly negative throughout. The two largest-modulus eigenvalues have positive product for s<0.5 and negative product for s>0.5 solely because the second/third ordering swaps. Append893 nonzero small diagonal entries to obtain the same issue in896dimensions. Façanha–Oldeman–Glass2013, p1265, defines D as the determinant of the complete2×2 monodromy and explicitly assumes fixed sign; p1267 says higher dimensions require additional characteristic coefficients. This is not the same diagnostic as a selected pair product.
- **Replacement:** “The product of the two largest-modulus multipliers changes sign in part of the window. This differs from the fixed-sign two-dimensional Hénon setting, but does not by itself establish noninvertibility or a two-dimensional reduction of the full map.” Remove the asserted inference to Mira’s noninvertible setting unless separately proved.

## Major

### A2. Dominant-real sign-change contours are not proved superstable curves

- **Location/text:** article `main.tex:949–956` and `1032–1034`, “superstable lines” / “superstable curves of the continued orbit”; plotting source `tools/plot_superstable_lines.py:163–185` constructs zero segments by interpolating the modulus-ranked dominant-real field.
- **Problem:** A dominant eigenvalue can change from +a to −a when two nonzero real eigenvalues exchange dominance. Interpolating this discontinuous label produces a spurious zero contour. Even a genuine zero eigenvalue in higher dimension does not make the orbit superstable if other multipliers are nonzero. Source terminology cannot be transferred from a one-dimensional multiplier to a selected higher-dimensional eigenvalue without checking the full spectrum/continuous branch.
- **Evidence:** diag(0.5−0.2s,−0.4) switches the dominant-real sign at s=0.5 while its spectral radius is always≥0.4 and neither eigenvalue vanishes. Façanha2013 pp1265–1267 uses T=trace(M) for the two-dimensional skeleton, with TypeA/B defined by that continuation. The report itself describes nonzero opposite-sign multipliers exchanging dominance at4058–4064.
- **Replacement:** “Contours separating cells with positive and negative dominant real multipliers; they indicate changes in the dominant direction and are not verified superstable curves.” Call the resemblance “Type-B-like avoidance of the plotted contours,” not a verified TypeB trace skeleton, unless a continuous trace contour is independently established. Update caption/figure labels consistently.

### A3. Two-dimensional superstable exclusion needs the nonsingularity condition

- **Location/text:** article `main.tex:911`, “Two-dimensional maps have no superstable curves.”
- **Problem/evidence:** This is false for arbitrary two-dimensional noninvertible maps. For example f_a(x,y)=(x²+a,0) has a fixed point at(0,0) with both multipliers zero when a=0; a redundant second parameter gives a superstable locus. Nonsingular monodromy excludes zero multipliers, not dimension alone. The article’s body at976–977 and the report caption at2369 already contain the appropriate qualification.
- **Replacement:** “In two-dimensional maps with a nonsingular Jacobian along the orbit, such as the nondegenerate Hénon map, no multiplier vanishes; trace-zero curves provide an analogous skeleton.”

### A4. Synchronization threshold omits its parameter range

- **Location/text:** article `main.tex:1068–1070`, synchronized state stable for c>c*=(1−exp(−lambda1))/2.
- **Problem/evidence:** Exact transverse linearization is (1−2c)DF, hence negativity requires abs(1−2c)exp(lambda1)<1. For lambda1>0 and0≤c≤1, this gives c*<c<1−c*. The displayed lower-threshold-only statement is valid on0≤c≤1/2, not the entire coupling interval.
- **Replacement:** “For0≤c≤1/2, the transverse Lyapunov exponent is lambda1+log(1−2c), so it becomes negative above c*. More generally the condition is abs(1−2c)<exp(−lambda1).” This is a local asymptotic transverse condition, not a global synchronization guarantee.

## Minor

### A5. Fourth-leg caption contradicts the stated period count

- **Location/text:** article `main.tex:935–936`, “every point carries the period-5 family,” versus994–995, “period5 at127, period20 at one.”
- **Problem:** “Family” can include doubled descendants, but the caption can be read as all128points having primitive period5. The text already provides the exact exception; preserve it in the caption.
- **Replacement:** “All128sampled points retain the followed family:127have period5 and one has period20.”

## Checked and found correct so far

- Article859–882 distinguishes shooting-root uniqueness in the full product box from uniqueness of all F^p roots in the first box; correct scope.
- Proper-divisor separation at shifts1,2,4 suffices for period8; prime periods2/3 need shift1. Different primitive periods imply different orbits.
- A rigorously enclosed monodromy power with induced norm<1 implies spectral radius<1 and local attraction for a C1map; no whole-original-box trapping follows, and the article does not claim it.
- The explicit tiny basin balls and fixed-start-unproved caveat match the separate released authority; no certificate rerun was required.
- Façanha2013 Fig4,p1267 supports TypeA loop and TypeB avoidance geometry, but as trace skeletons of a two-dimensional map, not automatically dominant-real signs of896dimensions.
- The one-dimensional Gallas quartic, fold/flip multiplier values and head/tail distinction are compatible with Gallas1994pp204–205; a finite observation of doubled shells remains finite evidence, not an infinite-mosaic theorem.

Further source/section findings and full checked-correct list will be added after the article pass and then the full-report pass. Scanned Carcassès/Mira primary PDFs require page inspection; filenames labelled2012 contain the original1991/1992articles, so filename date is not bibliographic authority.
