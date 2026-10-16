# Run E2E status (wrap-up, 16 Oct 2026, about 14:20 PDT)

Exploratory. Qwen2.5-0.5B-Instruct, philosophy direction at layer 12, fixed prefix and opener, float64 on CPU
(SoftLoop/SoftOrbit; Engine applies `keep_float64_norms`, confirmed). Raw outputs: `/data2/neural-shrimp-data/explore/e2e/`
(hash snapshot: `docs/e2e/receipt-snapshot.json`). Code: `tools/e2e_*.py`, tests `tests/test_e2e_numerics.py`
(13 pass). Run commands: `docs/e2e/run-q*.sh`. Draft text: `docs/e2e/draft.tex` exists (partial, values marked
TBD). Figure script `tools/e2e_figure.py` exists; `paper/llm/figures/fig-e2e.png` has NOT been produced yet.

## Q1. Origin of the 2-cycle (done in part)

- Zero-temperature successor map g (greedy, one-token memory; `greedy-scan`, `greedy-cycles`): there is no
  " is it" greedy loop for 0 <= alpha <= 0.12. g(" it") = "'s" for alpha < 0.105; g(" is") = " there" for alpha
  about 0.064-0.077 (" it" at 0.04-0.06). Greedy loops reached from the window tokens: " of course" (alpha 0.05,
  0.0628), " is there" + " of course" (0.065), " is there" + "\n\nThe question" (0.070), ". This is a" (0.08).
  The "\n\nThe existence" loop is not greedy either (g(" existence") = " of"); "\n\nThe question" is greedy on
  0.0665-0.070 (grid of 0.0005; endpoints not refined).
- The 2-cycle at alpha 0.070 continued in log10 T (Newton-Picard, AD multipliers; `p2-tcont-a0.070-{up,down,through}`):
  stable between a fold at log10 T = -0.1741065 ((1-mu)^2 extrapolation; last mu 0.99492, next modulus 0.098) and a
  flip between -0.4325 (mu -0.677) and -0.5325 (mu -1.081) (linear interpolation about -0.513; the refined flip run
  `run-q1g.sh` did NOT run). The fold agrees with the continued fold curve (curve-p2-fold-plus interpolated at 0.070:
  -0.1741065) to about 1e-8.
- Composition: at log10 T -0.19 the soft token after " is" is it 0.26 / there 0.24 / a 0.21; next is " is" 0.74.
  Going down in T the orbit moves onto the " is"/"'s" tie (0.72/0.28 at the flip, 0.50/0.50 below -1.1) and stays
  far from embeddings (relative distances about 0.39 and 0.57); |mu| grows to 27.5 at log10 T -1.127 (run stopped by
  hand there: the --through step control froze; fixed in code since). Provisional conclusion: the 2-cycle is a
  finite-temperature compromise between near-tied continuations, not a continuation of a superstable greedy loop;
  it does not become superstable as T -> 0 at this alpha.
- Not done: alpha 0.0628 continuation (run-q1d), temperature column bifurcation diagram (run-q1e), refined greedy
  endpoints (run-q1c), the crescent P3 in temperature (run-q1f; isola vs connection to the greedy
  "\n\nThe question" loop is UNRESOLVED), the 2-cycle region's low-T alpha boundaries.

## Q2. The period-6 band (done in part)

- Cut alpha = 0.070 (`band-cut-a0.0700`): the band orbit "\n\nThe question\n\nThe existence" is bounded above by the
  flip of the period-3 loop "\n\nThe existence" at log10 T = -0.1392937 (the period-6 orbit merges into it; AD
  multiplier -1.00000007, real) and below by its own flip at -0.1414292 (critical multiplier -0.99999979, real; next
  modulus 0.55). So the band is the first doubling of the crescent's cascade continued to larger alpha.
- Census (provisional, still running; `census/census.json`): at (0.0697, -0.1411) and (0.0700, -0.14036) all five
  starts (fixed start, band state, e_is, the stripe attractor P10 from 0.008 above, the attractor from 0.008 below)
  end on the band's period-6 loop (exponents -0.20 and -0.16). No bistability found at these two crossing cells:
  inside the band the stripe orbit is not an attractor. Third cell (0.0708, -0.1396) running.
- Not done: further cuts (run-q2b: alpha 0.0690, 0.0715, crescent cut at -0.148), continued two-parameter curves,
  overlay figure.

## Q3. One-dimensionality (done in part)

- Sigma_p is NOT near rank one step by step (`rank-spectra`, provisional while running): lambda2/lambda1 of Sigma_p =
  0.87, 0.69 (2-cycle at 0.0628, -0.13); 0.71, 0.40, 0.55 (P3 crescent at 0.0641, -0.156); 0.54-0.77 (P6 band). The
  leading eigenvector is aligned with a competing pair (cosine 0.73-1.00; existence-question 0.94-1.00). DF:
  sigma2/sigma1 0.20-0.65. Composed monodromy sigma2/sigma1: 0.081 (P2), 0.092 (P3; multipliers -0.608, 0.130), 0.261
  (P6; multipliers 0.391, -0.056). Dense AD Jacobians agree with fresh JVPs to <= 6.5e-14; orbit closure <= 1.1e-14.
- Crescent chaotic cell (0.0652, -0.156) (`red-sec-crescent`): twin exponent 0.0950 per step (0.285 per return);
  section every 3 steps; the 1000 section states lie on a curve; pushed through G = F^3, states interpolated along
  the curve land back within median 2.1e-5 (max 1.8e-3) of the curve length (`red-curve1d-crescent`). The return map
  in the noun log-odds s has one minimum, but s folds back near one end of the curve (a short second branch); in
  arclength it is single valued. R1 multipliers and exponent depend strongly on chart smoothing (exponent 0.21-0.39
  per return): no robust R1 number yet.
- Two-token model q' = sigma(h(q)/T) (`red-r2-crescent`, frozen at a chaotic section state): matches the full
  return map's shape over the attractor's s range; first flip of the crescent at alpha about 0.06476 (between 0.0646
  and 0.0648) vs 0.0644767 in the full map; then a stable period-6 orbit up to 0.06525 (no cascade, no chaos) where
  the full map is chaotic.
- 2-cycle with the " is"/"'s" pair (`red-r2scan-p2-is-s-a0.070-T`): reduced flip at log10 T = -0.50813 (bisection)
  vs full about -0.51; reduced multiplier -0.687 at -0.43 vs full -0.677 at -0.4325; -4.42 at -0.82 vs -3.90 at
  -0.827; diverges below -1.0 and fails near the fold (reduced -0.22 at -0.16 vs full +0.34 at -0.19). With the
  " it"/" there" pair the reduced map is superstable (mu about 1e-4) where the full orbit has |mu| > 1 (stopped run,
  log only). Spline artifacts appear as spurious fixed-point pairs near s = -0.9 in that scan.
- Not done: Lyapunov spectra / Kaplan-Yorke dimension (run-q3e), along-curve analysis, rank spectra of the two
  chaotic cells (running).

## Still running (left to finish on their own; no scheduler will start anything else)

- `tools/e2e_rank.py spectra` (PID 194702) -> `explore/e2e/rank-spectra/spectra.json` (4 of 6 cells written).
- `tools/e2e_band.py census` (PID 199913) -> `explore/e2e/census/census.json` (2 of 3 cells written).

## To resume

```
bash docs/e2e/run-q1g.sh    # refined 2-cycle flip at alpha 0.070
bash docs/e2e/run-q3e.sh    # along-curve summary; Lyapunov spectra at 3 chaotic cells (KY dimension)
bash docs/e2e/run-q1f.sh    # crescent P3 in temperature (isola or not)
bash docs/e2e/run-q1e.sh    # temperature column from the 2-cycle state, alpha 0.070
bash docs/e2e/run-q2b.sh    # more band cuts and a crescent cut
bash docs/e2e/run-q1c.sh; bash docs/e2e/run-q1d.sh
.venv/bin/python tools/e2e_figure.py          # writes paper/llm/figures/fig-e2e.png and .json
```
Each script uses 4 threads; run at most two at once. Approximate times on the loaded machine: q1g 15 min, q3e 1.5 h,
q1f 1.5 h, q1e 45 min, q2b 2 h, q1c 15 min, q1d 1 h.

## Failed or abandoned attempts (kept)

- `attempt1-stalled-p2-tcont-a0.070-down`: Newton-Picard stalled at 1e-8 (FD columns not normalized); fixed.
- `aborted1-band-cut-a0.0700`: a period-6 orbit merging with its period-3 orbit was being treated as a fold; fixed by
  the half-period test.
- R1 from a straight segment sorted by s (replaced by curve1d); smoothed-spline curve chart (offline test failed).
- q1b through-continuation stopped by hand at log10 T -1.127 (step control froze).
