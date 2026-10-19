# Review of *Chaos and bifurcations in continuous language-model feedback* (arXiv v1 candidate)

Reviewer: Claude Fable 5.1, 18 October 2026.
Reviewed: `/data2/chaos-journal-neural-shrimp` at commit `c8f7fe0` (main.tex 1136 lines, main.pdf 23 pages, 19 figures,
3 tables, 77 references), against the full report `neural-shrimp/paper/llm/main.tex` at `9d322808`, the research
log, the raw scan and cascade files under `neural-shrimp-data`, the physics referee's report of 15 October and the
Cursor audit of 17 October.

## Verdict: GO, after three mechanical fixes (section 1). The science holds.

I recomputed every derivation and every number I could reach by hand or from the raw files, and found no error of
mathematics, theory or data in the article. The must-fix items are a dead URL, one typo and two British spellings.
The should-fix items are wording precision that a careful referee would raise; none changes a claim. The rest of this
document is the evidence.

---

## 1. Must fix before posting

1. **The Data Availability URL is a 404.** `https://github.com/tdj28/neural-shrimp` returns 404 to an anonymous
   request (the user page `github.com/tdj28` exists, so the repository is private or not yet created under that
   name). The sentence says the code and records "are openly available" there. Either make the repository public
   before the arXiv announcement, or change the wording to "will be made available at ..." / "available from the
   author". A reader who clicks this on day one and gets a 404 will not read further. (HUMAN_CHECKLIST item 14,
   "repo rename and public release timing", is still open.)
2. **Typo, Sec. VI, computer-assisted proofs paragraph:** "the maps' fixed start included" → "the map's fixed start
   included".
3. **Two British spellings survived the American-spelling pass** (the figure legend itself was fixed in the last
   commit, so caption and legend now disagree): main.tex line 935 (Fig. 18 caption, "the centre line of the fourth
   (lower-right) leg") and line 993 (Sec. VII body, "Along its centre line"). → "center".

## 2. Should fix (wording precision; no claim changes)

4. **Abstract: "came within one step in 42 of 48 cases".** The body counts iterates of the return map G (= F², F⁶ or
   F⁸), i.e. 2, 6 or 8 map steps. "Within one iterate of the return map" is what was tested. Same sentence in the
   Introduction uses "iterate" correctly.
5. **Fig. 16 caption: "Two-dimensional maps have no superstable curves."** Over-general as written; the body says it
   right: "a two-dimensional map whose Jacobian never vanishes". Add the qualifier to the caption.
6. **Target D shortfall "by 0.4 to 5.7 iterates".** From Table III the largest shortfall is 154.75 − 149 = 5.75.
   By the project's own outward-rounding rule for bounds this should read "5.8" (or quote 5.75). The 3.1 to 5.2 %
   figures in the same paragraph are right (3.7 % and 3.1 % for 149/150, 5.0 % and 3.9 % for 81/82, 5.2 % for 42).
7. **Feigenbaum shrink factor "5.26".** It is correct: from the raw nine flips I get 3.519, 6.539, 3.926, 5.259 (the
   15 October referee's 5.55 came from the 13-digit table values; the raw values give 5.26). But it is not robust:
   the ±3.3×10⁻¹³ uncertainty of the period-384 flip that the paper itself quotes moves this factor between 2.7 and
   52. Since the paper already gives the resulting ranges for δ₇ and δ₈, either drop the last factor or add "the
   last of these is not determined by the flip's location uncertainty". An attentive referee will compute this.
8. **"the exact transverse law u′ = (1−2c)DF u".** Exact is u′ = (1−2c)[F(x_A) − F(x_B)]; the stated form is its
   linearization. "To first order, u′ = (1−2c)DF u" or "linearized transverse law".
9. **The T\* sentence.** The values 2.7×10³ and 5.6×10⁴ are lower bounds on T\*(α) (because the power-iteration
   maxima are lower estimates of L(α)), so the region where contraction is *guaranteed* begins somewhere above
   them. The text does say "lower estimates", but "about three to four and a half decades above the highest chaotic
   temperature" reads as if the threshold were located. "At least three ... decades" avoids the misreading.
10. **"fifteen released models" (lead paragraph and introduction).** Verified: Qwen2.5 at 7 sizes + OLMo-2 at 4 +
    Llama-3.1 at 2 + Llama-3.3-70B + Qwen3.5-35B-A3B = 15, of which 14 dense ("eleven of fourteen" is consistent).
    But the article also uses Qwen3.5-4B and -9B, the reasoning models and Huginn, so "fifteen" is the count of the
    size sweep, not of all models in the article. A reader who counts will notice. Consider "fifteen models in the
    size sweep" or leave as is with that awareness.
11. **README.md of the article repository is stale**: it says 16 pages, 11 figures, 2 tables and names source commit
    `8fe1a7b`, while the PDF is 23 pages with 19 figures and 3 tables, and the last sync went *from* the article
    *to* the full report (`7543b02a`). Not in the arXiv upload, but fix before that repository goes public.
12. **Six bibliography entries are not cited** (bertschinger2004realtime, langton1990computation,
    liu2026learningchaos, mira1991crossroad, papasavvas2015gain, singer1978stable) although the file header says it
    holds only cited keys. Harmless to the build; cosmetic.
13. **arXiv form.** Use the *article's* abstract (1370 characters, under the 1920 limit), not
    `paper/llm/arxiv-abstract.txt`, which is the full report's abstract with numbers (21 %, 14.5/19.5 %, 4.6688
    against 4.6692) that are not all in this PDF. Suggested categories: nlin.CD primary, cs.LG and cs.CL cross-lists.
    Licence CC BY 4.0 as recorded in the log.

## 3. What I checked and found correct

### 3.1 Mathematics and theory (re-derived by hand)

- **Eq. (2)–(3).** The softmax Jacobian of z/T is (1/T)(diag p − ppᵀ); with tied embeddings z = E h̃ this gives
  DF = (1/T) Eᵀ(diag p − ppᵀ)E J = (1/T) Σ_p J with Σ_p = Cov_{t∼p}(e_t). Untied: the cross-covariance
  Cov(e_t, w_t), not symmetric. Correct. The statement that attention enters J only through the fed-back position
  (prefix keys/values fixed by causality) is correct.
- **State space and regularity.** Autonomy from the causal prefix, real-analyticity (RMSNorm with ε > 0, SiLU, fixed
  rotary rotations, softmax), F(ℝ⁸⁹⁶) ⊂ K = conv{e_t}, F(K) ⊂ K, Brouwer, Krylov–Bogolyubov, Oseledets for
  non-invertible maps with log⁺‖DF‖ bounded on K: all correct, and the MoE caveat is the right one.
- **High-temperature contraction.** Popoviciu gives uᵀΣ_p u ≤ diam(E)²/4 for every unit u, hence ‖Σ_p‖ ≤
  diam(E)²/4; with L(α) = sup_K ‖J‖ (finite by continuity and compactness, and independent of T, so the threshold
  is a curve in the plane) F is a contraction of the convex set K for T > diam(E)² L/4 and has one globally
  attracting fixed point. Numbers: diam² /4 = 0.3666; × 1.54×10⁵ = 5.65×10⁴ ("≥ 5.6×10⁴", rounded down, correct for a
  lower bound); log₁₀ of 2.7×10³ and 5.6×10⁴ minus 0.28 = 3.15 and 4.47 decades ("three to four and a half").
  Untied case diam(E)·diam(W) follows from Cauchy–Schwarz on the cross-covariance. Correct.
- **Low-temperature gain.** With a unique leading token j, Σ_p = e^{−Δ/T} Σ_{runner-up}(e_i−e_j)(e_i−e_j)ᵀ (1+o(1));
  the correction from ē − e_j is O(e^{−2Δ/T}). The prefactor C, the pointwise-not-uniform convergence and the
  uniform-gap condition on invariant sets are all stated correctly.
- **Hopfield energy (tied case).** F = ∇φ with φ = T log Σ_t exp(e_tᵀx/T) convex; convexity gives
  φ(F(x)) ≥ φ(x) + F(x)ᵀ(F(x) − x), which rearranges exactly to 𝒱(F(x)) ≤ 𝒱(x) − ½‖F(x) − x‖². Steps shrink,
  accumulation points are fixed points, no cycles. The restriction to tied embeddings (the referee's A1) is now
  correctly stated, with the untied caveat.
- **Lyapunov bound and block norms.** ‖DFⁿ‖ ≤ Π‖DF‖ gives the necessary condition; the subadditive (Kingman)
  convergence without a rate is stated correctly; the gain-split identity (product of per-step relative gains
  telescopes to the finite-time growth plus (log‖x₀‖ − log‖x_n‖)/n) is exact.
- **Fold normal form (Fig. 11).** u\* = ±√(|a|(α_c−α)/|b|), μ = 1 ∓ 2√(ab(α_c−α)), (1−μ)²/(4(α_c−α)) → ab; with
  a = −4.834, b = −25.4285, ab = 122.9. Passage integral ∫du/(aε+bu²) = (2/√(abε)) arctan(U√(b/(aε))) with
  U = 0.1/|b| gives 143.8, 70.4, 33.8, 13.2 iterates at ε = 3×10⁻⁶, 10⁻⁵, 3×10⁻⁵, 10⁻⁴: the measured 144, 71, 34, 14.
- **Table III.** At ε = κ²/(ab) every forecast reduces to (2/κ) arctan(0.1/κ): 22.40, 44.29, 85.29, 154.75 and
  their halves, independent of the fold, as claimed. From the table I reproduce: 7 forecasts ≤ 5 all exact; 16 of
  the 17 longer within one iterate (the single miss 59/61); all seven nonzero first-test residuals negative; second
  test 18 of 24 by the integral, 16 of 24 by the discrete map; at target C every measured count is the forecast
  rounded up; 24 + 18 = 42 of 48 overall; misses 3.1 to 5.2 % short.
- **Window law.** π²/(4ab) = 0.0201; from the Table II centers, d_N N² = 0.0263, 0.0265, 0.0265, 0.0263 for
  N = 4–7; √(ab)/π = 3.53; 6.13/3.53 = 1.74 → passage fraction 0.58, 0.58² = 0.34 ≈ 1/3; C = φ²π²/ab = 0.0266.
  The token counts of all eight stripe orbits give period 2N + 6 with N the number of "is X" pairs.
- **Feigenbaum (Sec. VI), from the raw files** `pod64/cascade/crescent-t-0.156{,-b,-c,-d,-f}/cascade.json`:
  ratios 14.928, 3.1241, 4.4988, 4.6208, 4.6618, 4.6673, 4.6688; deviations −1.545, −0.170, −0.0484, −0.0074,
  −0.00189, −0.00036; Aitken 4.66943; and with α₈ shifted ±3.3×10⁻¹³: δ₇ ∈ [4.66726, 4.66737], δ₈ ∈ [4.66852,
  4.66917], Aitken ∈ [4.66882, 4.67019]. The article's [4.6672, 4.6674], [4.6685, 4.6692], [4.6688, 4.6702] are
  these, rounded outward. The next moduli from flip 5 on (1.8×10⁻¹³ … 3.2×10⁻¹³) are below 4×10⁻¹³ as stated.
- **Other arithmetic.** Kaplan–Yorke 1 + 0.274/1.052 = 1.26; coupled copies u′ = (1−2c)(F(x_A)−F(x_B)) and
  c\* = (1 − e^{−λ})/2; ln(10⁶)/0.88 = 16 and /0.30 = 46 steps ("15 to 50"); 0.647/0.636 = 1.7 % ("within 2 %");
  7 × 0.091 = 0.637; three leading multipliers with e₁ = e₂ = 0 are cube roots of e₃ (equal moduli, as the text
  says the conditions force); 65/3.5×10⁻³ ≈ 1.9×10⁴ ("four orders"); λ = σ₁(v₁·u₁) for a rank-one map.

### 3.2 Numbers against raw data and the full report

- `pod64/cross-512/scan.npz`: 262,144 cells; 78,868 chaotic (30.09 %, "30 %"); 11,661 with neither label
  (4.45 %, "4.4 %"); 73,437 at threshold 0.05; 171,615 periodic. `map-stripes-256`: 35,650 chaotic of 65,536
  (54.4 %, "more than half"). LT run: 143/192 = 74 % and 176/192 = 92 % survive 10⁵ steps.
- 282 numeric literals in the article; all but six occur verbatim in the full report, and those six (9675, 6.17 and
  the four shrink factors) I verified directly (9,675 is in the report with a comma; 6.17 is the fitted slope in
  Fig. 12's legend; the factors above).
- The variance-explained percentages in Sec. II (30/32, 7/33, 21/27, 47/4 %) trace to the NB runs (research log,
  18 Oct 16:50 UTC). The hidden-chaos counts (182 of 1400; 65; 0.18995; 0.091) match run H1 in the log and the full
  report. The earlier "hidden chaos" cell that turned out to be a slow transient (log, 14 Oct 03:26) was a different
  cell (log₁₀T = 0.18945, a fixed point); the article's cell is not that one.
- Model count (15 = 7 + 4 + 2 + 1 + 1), "eleven of fourteen", "9 to 17 against 0 to 7" for the 14B/7B prompt test,
  and the MoE sentence all match the full report.
- The multiplier bounds in Sec. VII (leading modulus ≥ 0.0139 on the grid; multipliers beyond the leading pair
  < 0.0032 on the continuation grid; D < 0 at 41 of 447 cells) are the ones the 15 October audit found to hold
  (0.013904 and 0.003162); the overstated 0.122 of that audit lives only in the full report's appendix.
- All three must-fix findings of the 17 October audit are applied in the article (0.11 to 0.90; the abstract no
  longer attributes slow growth to the wrong reasoning models; the 1/N² law carries its exponent caveat), as are
  the 15 October referee's items A1–A4, B1–B4 and C1–C15 (I checked each against the current text).

### 3.3 Bibliography

75 cited keys, all present; BibTeX runs with no errors. I fetched the arXiv pages of the entries that Crossref
cannot verify: jones2012rossler (1201.4343, "Topological origins of a bi-parameter periodicity hub for the Rössler
attractor", T. D. Jones, nlin.CD), kim2026phase (2608.26556; its abstract has the fold phase, the parameter-free
one-dimensional reduction predicting relaxation times and spectral-gap amplitudes, and the N⁻² tail, exactly as the
Introduction describes it), jaca2026chaos, verazuniga2026selffeeding, johnson2026temperature, ruan2026criticality,
luo2026prediction and ko2026attractor: titles and authors all match. facanha2013endoskeletons (Façanha, Oldeman,
Glass, Phys. Lett. A 377, 1264), he2025nondeterminism and tomihari2025recurrent are confirmed by the referee's
Crossref table. The classical entries (Pomeau–Manneville, Hirsch–Huberman–Scalapino, Collet–Eckmann–Koch, Grossmann–
Thomae, Lorenz 1980, Bier–Bountis, Milnor, Grebogi–Ott–Yorke, Galias, Zgliczyński, Krawczyk, Rump, Govaerts et al.,
Kuznetsov–Meijer, Gallas 1993/1994, Carcassès et al., Mira, Stoop et al., Vitolo et al., Fujisaka–Yamada,
Pecora–Carroll, Platt et al., Ashwin et al., Kingman, Benettin et al., Oseledets, Eckmann–Ruelle) have the right
journals, volumes and pages.

### 3.4 Build, layout and figures

- `build_arxiv_bundle.py`, run on a copy: 25 files, 7.3 MB, compiles in a clean directory with pdflatex only, 23
  pages, text identical to the canonical PDF, no undefined references or citations, zero overfull boxes. The two
  "float stuck" warnings are benign: every figure is placed, the furthest three pages after its first mention
  (Fig. 10); Fig. 19 lands one page before its reference. Pages 21–23 are references only. The TikZ figure (Fig. 4)
  renders cleanly. The PDF carries the right title and author metadata.
- I inspected sixteen of the nineteen figures at full resolution. All are legible, axis-labelled and consistent with
  their captions and with the text (e.g. the 22 % and 21 % in the Fig. 17 and Fig. 9 panel titles are the text's
  21.7 % and 20.8 %; Fig. 16(b)'s title is the b = −1 case of the caption's quartic; Fig. 11(c) shows the four
  passages with their counts). Resolutions are 170–600 dpi.

## 4. Where praise is due

- **The theory section is the paper's spine, and it is right.** The factorization DF = (1/T)Σ_p J is simple, exact
  (verified to 10⁻¹³, Fig. 3) and used three ways with the correct conditions each time: a rigorous contraction
  threshold, a correctly hedged low-temperature exclusion, and the Hopfield energy argument that locates the origin
  of every cycle in the network factor. The consult-17 corrections made it clean. Few papers in this space have a
  derivation a referee can check in ten minutes and find nothing wrong with.
- **The prospective passage tests are exemplary.** Forecasts fixed before the runs, with primary and secondary
  criteria, failures reported with their sign and magnitude, a second test designed after the first's lesson, and a
  mechanism-free statement of what the period-6 miss means. The fact that every forecast reduces to
  (2/κ)arctan(0.1/κ) and the numbers reproduce from the quoted a, b to within a fraction of an iterate is the kind of
  thing that makes a Chaos referee trust the rest.
- **The Feigenbaum cascade to period 1536 in float64, with the precision floor stated and the sensitivity of the
  last two ratios quantified**, is careful numerics; the deviations from δ closing by factors near δ with one sign
  is what universality predicts and what the data show.
- **Honesty apparatus.** Chaos as a finite-horizon label with its measured error; attractors "operational" with the
  Milnor caveat; the 10⁵-step survival rates; the unresolved family switches (4 of 8); the hybrid products "not
  derivatives of any actual map"; the shrimp with explicit numerical criteria, "no one-dimensional head found",
  "Type A an analogy only", "nothing here proved by interval arithmetic". This is unusually disciplined and will
  protect the paper in review.
- **The computer-assisted proofs** of a period-8 orbit and of coexisting attracting period-2 and period-3 orbits of
  the full 896-dimensional transformer map, with certified token loops and basin neighborhoods, are, as far as I can
  tell, new, and the paper claims exactly what the certificates give and nothing more.
- **Fig. 15 (a periodic text on a chaotic orbit)** is the single best figure for a general reader: the return map of
  one probability, a clean single hump, exponent matching the full map within 2 %. It also carries a point that
  matters outside dynamics, that text monitors cannot see this chaos.
- **Writing.** Dense but readable; each paragraph leads with its result; the "One picture" paragraph ties the
  sections together; the Discussion claims only what was shown.

## 5. Suggestions that would strengthen v2 or the journal version (not needed for v1)

- Run the 10⁵-step, two-offset check on the Fig. 15 cell (α = 0.0655, log₁₀T = 0.18995). The LT persistence test
  covered the philosophy maps, and a slow-transient hidden-chaos candidate was found once before on the same line.
  One sentence with the result would close the obvious referee question.
- The Codex Task 36 deep mathematics review was assigned at 00:34 UTC today and has not reported; if it lands before
  posting, triage it against this list.
- For the journal version, Chaos will want the affiliation address (already a TODO) and may ask for CRediT roles.

## 6. Scope of this review

Done: full read of main.tex; hand re-derivation of every equation and bound; recomputation of Table III, the window
law, the Feigenbaum ratios (from raw flip files) and the scan counts (from raw arrays); cross-check of all numeric
literals against the full report; verification that the prior three audit rounds' findings are applied; bibliography
checks (Crossref table plus nine arXiv pages fetched); a clean-room build of the arXiv bundle; page-layout and figure
inspection. Not done: no GPU reruns; the interval-arithmetic certificates were checked for logic as stated, not
re-executed; the behavioral results (GSM8K, Soft Thinking, reasoning models) were checked only for consistency with
the full report and the 17 October audit, which traced them to raw records.
