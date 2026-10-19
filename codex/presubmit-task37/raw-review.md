# Task 37: independent original-record audit of Huginn and the four P5 cusps

- Three findings: one must-fix numerical error and two should-fix statements (the energy-guarantee wording overlaps the companion citation audit).
- This subtask did not run Task 2 and makes no determination about two distinct period-6 orbits at R.
- The saved Huginn data support accuracy loss at relaxation 1.20, observed accuracy improvement at 1.05, and six natively unstable ARC fixed points recovered by under-relaxation; the four P5 cusp coefficients and unfolding determinants match Table IV.

Audit snapshots: article `/data2/chaos-journal-neural-shrimp`, exact commit `18490e2cbac9a43be22ff0e3e29f383024cc5802`; research exact commit `35eabc6c17ac944564985a7a01ad7f6bc92bf0ce`. The article and full report were extracted with `git show`, rather than taken from the changing producer checkout. No repository, producer file, or original data was edited. No model, GPU, paid resource, API, or process-control action was used.

Article `main.tex` SHA-256: `9c9881359e4622c88202b664f90e1c9340de626450e3f9208175417d984e7d94`.
Research `paper/llm/main.tex` SHA-256: `01099ad004cf130181f3e4275d5e0efa901e6bfe50c052c6bf38095dd239925e`.

The independently written standard-library checker is `/tmp/task37-raw-check.py` (SHA-256 `3ddf24f5325f30f0b61566b0ebf2c5c24d39edd4fc054c8e0c2f120a167f6224`). Its complete row gates and 53 original-file SHA-256 hashes are in `/tmp/task37-raw-check.json` (SHA-256 `ae15d6d52184253aab3140a182a1561a41a62dcafc2ff2f37130cfa9921aef96`). It reads JSON/JSONL records and performs small scalar computations, not model replay. The prescribed `uv run --no-sync python` failed before Python with the snap-confine confinement refusal. With the root's explicit launch fallback, the successful invocation was `HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 10 /usr/bin/python3 /tmp/task37-raw-check.py`. No installation or environment change followed. An initial audit-only extraction mistakenly used the zero-shot parser and the arm string `1.2`; these checker mistakes were corrected against the cot8 producer code before final conclusions. The initial parser output is retained at `/tmp/task37-raw-check-initial-extractor.json`, not used as evidence of a producer failure.

### R1. The claimed analytic Huginn crossing is a coarse grid endpoint [must-fix]

Where: `main.tex`, Discussion, “Beyond the one-token loop,” lines 1184–1186 at `18490e2`. Also full report, “A second system: recurrent depth,” lines 2938–2942 at `35eabc6c`.

Claim: “its multipliers predict a Neimark--Sacker bifurcation of the relaxed loop at relaxation 1.0777, against 1.0778 from the amplitude of the invariant circle beyond it.” The full report explicitly equates the analytic formula to 1.0777.

Evidence: the **original** `/data2/neural-shrimp-data/explore/latent-loop/huginn-fixed-spectrum-gsm-claire/fixed-spectrum.json` (1,817 bytes; SHA-256 `9ce1d136b524919e4337c96ba8980a7b028f4e9437f52cd8d0eca27ea3e76fed`) contains the critical pair `0.5883682635687575 ± 0.7710905065294952 i`, modulus `0.9699266894125702`. Direct substitution gives

`2*(1-Re(mu))/abs(1-mu)**2 = 1.0775400117776854`,

which rounds to **1.0775**, not 1.0777. The JSON's `predicted_stable_lambda_intervals[0][1]` is `1.0777306826706676`. At exact research snapshot `tools/explore_latent_loop.py` lines 961–974, that field is generated from a 4,000-point grid over `[0.001,2.0]` and stores the **first unstable grid sample**, not an analytically solved crossing. Its grid spacing is about `0.000499875`. At the printed 1.0777, the actual saved critical pair already has modulus `1.0000658638387696`.

Crossing classification was checked before interpretation: at the corrected lambda, the complex pair is `0.5564503338778098 ± 0.8308808734874537 i`, modulus 1, argument magnitude `0.15608147670709255` turns; the next distinct mode in the computed partial spectrum has modulus `0.993154078401852`. The radial derivative is `0.4116317364312425 > 0`. Thus this correction preserves the complex-pair crossing and its numerical interpretation.

The **original** `.../latent-loop/huginn-ns-gsm-claire/ns-check.json` (8,787 bytes; SHA-256 `d9f935dd3be558101ff7633e3a2a1cf838c77de4e4102d6826eb57288aadda41`) has all eight requested lambda rows. Fitting `late_distance_median**2` against the four above-threshold lambdas `[1.085,1.09,1.10,1.12]` by ordinary least squares independently gives amplitude intercept `1.0778010471978443`, so **1.0778 is correct**. The difference from the correctly evaluated spectral crossing is `0.0002610354201589`, rather than the apparent `0.0001` difference between printed numbers.

Correct value / wording: “its estimated multipliers predict a complex-pair unit-circle crossing near relaxation **1.0775**, consistent with a Neimark--Sacker bifurcation, against **1.0778** from a fit of squared amplitude.” Correct the explicit equality in the full report too. If retaining 1.0777 as a historical grid output, label it a grid endpoint and give the analytic value separately. The first Lyapunov coefficient was not computed, so do not upgrade this to a nondegeneracy proof.

### R2. The Table IV multiplier-error bound rounds downward [should-fix]

Where: `main.tex`, Appendix A, Table IV caption, lines 1226–1228 at `18490e2`.

Claim: “the critical multiplier equals +1 to within 1.2 × 10^-9.”

Evidence: original `.../llm-soft-loop/pod64/fm5c/locate-curve-a1-up-fold-minus-30/locate.json` records `final.critical_multiplier = [1.0000000012005652, 0.0]`, hence `abs(mu-1) = 1.2005652028079794e-9`, strictly above `1.2e-9`. Its SHA-256 is `da0a1722dd85cdb0fc68a73ba48bf282d6da9b79641c1ebdbb447d263dd6e7c3`. The other three principal rows are below this value. This is a tiny display-bound error, not a failed cusp calculation.

Correct value / wording: “within **1.3 × 10^-9**” if keeping two significant digits and the principal four records. An alternative locator for the same lower-right cusp (`locate-curve-a2-up-fold-plus-23`) has a smaller error, but silently substituting it would not make the printed bound true for all cited original records. The project explicitly requires rounding bounds upward.

### R3. The energy-guarantee sentence needs a specified operator [should-fix]

Where: `main.tex`, Discussion, “Beyond the one-token loop,” lines 1180–1182 at `18490e2`.

Claim: “energy-based guarantees of convergence require specific structure and update rules, as for the tied readout of Sec. ... , and none is established for a complete transformer update.”

Evidence: the data inspected here contain numerical spectra and transients, not a proof of absence of convergence guarantees for every possible complete transformer update. Grammatically “none” refers to energy-based guarantees, so the article's high-temperature contraction argument is not itself a direct contradiction. However, the universal wording exceeds the scope of the Qwen soft-token operator and Huginn recurrent block actually studied. The companion citation reviewer checked Hoover et al.'s deliberately energy-designed transformer, which has an energy descent construction for its modified complete block (continuous-time theorem with discretization); it should not be presented as a guarantee for conventional Qwen/Huginn, but it makes an unrestricted absence statement unsafe. See that review for its independently verified primary-source citation details.

Correct wording: “**The energy-descent argument for the isolated tied readout does not establish convergence of either complete feedback operator studied here.**” This is a precise local scope claim and does not require asserting that no energy-based transformer architecture exists.

## Completed numerical checks and retained limitations

### Huginn threshold and amplitude

- All 24 saved Ritz values (three sets of eight) of the GSM-style prompt were read. The original record's fixed-point residual is `2.3078232516127173e-6`, with explicit float32 dtype. This early record does **not** contain an `arnoldi_converged` flag or Ritz-vector residuals; all three requested sets contain eight values, but convergence diagnostics beyond that are **unverified**, not retrospectively assigned true.
- All eight amplitude rows were gated: lambdas 1.0, 1.05, 1.07 and 1.075 have negative fitted early growth rates; the latter remains above the float32 floor at 3,000 steps and is not labelled fully converged. The four higher lambdas have positive initial growth and late amplitudes `0.0456381887`, `0.0584345423`, `0.0774055943`, `0.1086411960`. The amplitude ratios to `sqrt(lambda - fitted_lambda_c)` are `0.53789033`, `0.52906419`, `0.51952459`, `0.52886363`, supporting 0.52–0.54 at the report's precision.
- All 18 requested rows across the four rotation runs a1/a2/b1/b2 are present; each ns-check has the same admitted saved fixed point. The combined original `huginn-rotation-a1/rotation.json` retains all rows, no locked period up to 60, and deformation of the projected motion in the final b2 group. This is finite-horizon float32 evidence, not an invariant-circle or quasi-periodicity proof. No raw trajectory arrays were read/reprocessed.

### Generation test at 1.20 and 1.05

The three original row files are `.../llm-soft-loop/pod64/huginn/partB-0/rows.jsonl`, `partB-96/rows.jsonl`, and `partB-100/rows.jsonl`, with 384, 16 and 400 rows. All 800 rows pass: no duplicate `(arm,index)`; each arm exactly covers indices 0–199; split test, 32 recurrence steps, cot8, seed equal to index, and arm/lambda agree. Flexible answer extraction was independently recomputed from the **saved text** using the cot8 rule; every saved `pred` and `correct` matches. `core_calls = 32 * new_tokens` on every row. Token-cap outcomes were retained, not dropped.

| relaxation | correct / 200 | accuracy | lost vs native | gained vs native | capped |
|---|---:|---:|---:|---:|---:|
| 1.0 | 66 | 0.330 | 0 | 0 | 9 |
| 1.05 | 79 | 0.395 | 8 | 21 | 10 |
| 1.15 | 51 | 0.255 | 26 | 11 | 4 |
| 1.20 | 19 | 0.095 | 50 | 3 | 15 |

The 1.20 difference is exactly `-0.235`; independent binomial summation for one-sided exact McNemar gives `2.759792394613214e-12`. Thus the claimed loss is supported. The 1.05 observed difference is `+0.065`. The exact empirical paired-bootstrap distribution, computed by standard-library convolution, has fifth percentile `+0.020`, still comfortably above the frozen `-0.04` noninferiority boundary. The producer's specified 10,000-resample seeded Monte Carlo value `+0.025` was read from its analysis record but that exact pseudorandom sequence was **not** replayed here; the small difference from the exact law is not treated as an error.

The precise frozen design was inspected at commit `9805ed32055e7ef790df0cc67316357c1067aef0` (13 October 2026 15:19:13 UTC). It predefines these four arms and admits the capped outcomes. Its B3 hypothesis is explicitly “non-inferiority within 4 points,” and says “It is not a no-loss claim.” The article's “none at 1.05” is supported as an observation in this sample, but for an unambiguous statement of the prespecified conclusion prefer **“the predicted loss at 1.20 and noninferiority within four percentage points at 1.05 (observed difference +6.5 points).”** I did not repeat an older provenance audit of who first read the test items or independently establish the timing of the remote push; the frozen Git object and raw row consistency were checked.

### Six of thirty ARC development questions

Original records for 0–3 are `.../huginn-relax/arc-dev-spectrum-{0,1,2,3}/fixed-spectrum.json`; originals for 4–29 are `.../huginn-relax/arc-dev-spectra/qNNNN/fixed-spectrum.json`. All thirty were read, including residuals, transient lambdas, every saved eigenvalue and every convergence flag. The exact six detected unstable native fixed points are:

| ARC train index | largest saved modulus | final F residual | successful under-relaxation | Arnoldi limitation |
|---:|---:|---:|---:|---|
| 4 | 1.0294215516 | 6.63412e-7 | 0.75 | SR and LR partial; LM converged |
| 6 | 1.0613460487 | 6.46140e-7 | 0.75 | none recorded |
| 7 | 1.0724558122 | 1.10904e-6 | 0.75 | LM and LR partial; SR converged |
| 8 | 1.2726803601 | 1.19891e-6 | 0.75 | none recorded |
| 25 | 1.0092926418 | 6.72798e-7 | 0.75 | none recorded |
| 27 | 1.1238926136 | 6.54209e-7 | 0.75 | none recorded |

The leading eigenvalues in these six records are **real and negative**, beyond -1. A converged spectral set contains the unstable mode in each case. The article's six-of-thirty claim and restoration of convergence are supported. The full report's “to 10^-6” is an order-of-magnitude statement, not a strict `<=1e-6` bound: two of the six residuals exceed 1e-6.

Twelve of thirty questions have at least one incomplete Arnoldi set: **3, 4, 5, 7, 13, 14, 15, 16, 17, 23, 28, 29**. These remain partial. For the other twenty-four cases, say “no unstable mode was detected in the computed spectra”; absence of an uncomputed mode is not established. The relatively high admitted residual of q9 is `6.602636858588085e-5`, below the recorded 1e-4 gate but much larger than the other rows.

The superseded original q4 attempt `.../huginn-relax/arc-dev-spectrum-4/fixed-spectrum.json` is retained: residual `0.021947581321001053`, so it is **not** a fixed-point spectrum eligible for this tally. The successful relaxed fallback record is used explicitly. All 35 traces in the five `arc-dev-conv-{0,...,4}/convergence.json` files were read and retained. The q4 native trace ends around residual 0.0219474–0.0219476 and has no fitted decay window; at 0.55 and 0.75 it reaches about 5.4e-7 and 6.6e-7 with successful fitted decay. Higher-lambda fit-window failures throughout these records remain failures, not fitted rates.

### “No chaos”

The raw records inspected substantiate convergence, instability and an oscillatory branch. They contain no Lyapunov-exponent/twin diagnostic establishing an absence of chaos for every tested Huginn setting. In particular, the high-lambda q4 traces have no fitted decay window, and the rotation-b2 group has a deformed projection; neither is classified as chaos from those facts alone. Therefore **“we observed no chaos” is acceptable only as the authors' finite-sample observational report**, not an exclusion result. Suggested precision: **“In these tests we observed convergence and oscillatory instability; we did not establish a chaotic regime.”** A general “Huginn has no chaos” claim is unverified. No unrelated or new Huginn study was searched or rerun.

### Four P5 cusps

The main four original files (not Task 35 summaries) give:

| cusp / original locator | c | det J | critical multiplier minus 1 | next modulus | located/check gates |
|---|---:|---:|---:|---:|---|
| R upper / fm5c a1-up-fold-minus-19 | +26004.10878991422 | +176.61696691272832 | 9.65827418e-12 | 0.0017703593508768146 | all true |
| R lower / fm5c a1-up-fold-minus-30 | -20452.464962900944 | +88.62608103319272 | 1.200565203e-9 | 0.0011845335648165284 | all true |
| L lower / fm5d a1-down-fold-minus-14 | -9650.631602944686 | +405.9048332622431 | 2.66176636e-10 | 0.007880388437016245 | all true |
| L upper / fm5d a1-down-fold-minus-39 | +47463.30515353251 | -535.3947907774698 | 1.33740352e-10 | 0.0048992933093937 | all true |

Each is explicitly a fold, period 5, located; critical multipliers are real positive with argument zero. The determinant was independently recomputed from each saved 2x2 J. Coefficients, determinant signs/rounding, parameter coordinates, and the at-most-0.0079 next-modulus bound match the article. The original cusp-locator test-function values are `-9.405357211278442e-5`, `-4.374357814374008e-4`, `-3.869106826925872e-5`, `+1.5633047135565903e-4`; all fit `4.4e-4`. Initial bracketing maxima range from `16.31240177782209` to `41.627185338632685`, supporting the printed range at one decimal place.

The record's `final.f` and `final.b` come from **different finite-difference procedures**: `oriented_b` calls `fold_normal_form` with smaller steps while `fold_unfolding` supplies the other b. They must not be silently conflated, but this is not a new article error because the caption names the oriented test function. Cubic plateau spreads are respectively 0.1206%, 0.0526%, 0.0444%, 0.2562%; the saved nonzero-c gates pass. The two duplicated right-cusp locators from the other curve were checked too and agree in c/det J at printed precision. The locate.json records do not save a final full-state closure residual, so **closure was not independently revalidated from these files**. No state arrays, monodromy, or model calls were used; source-commit authority for the historical raw producer is not independently reconstructed here. This check establishes consistency with the original records, not a new proof.

## Check inventory

**Completed:** exact article/research text extraction; all requested Discussion numbers; direct spectral formula versus coarse-grid distinction; eight amplitude rows; eighteen rotation rows; thirty ARC spectra including all convergence flags; original failed q4 attempt; thirty-five original ARC convergence traces; all 800 generation admission and text-scoring rows; high-arm exact McNemar; exact empirical low-arm bootstrap law; four principal and two duplicate cusp-location records; all table c/detJ/multiplier/next-modulus values, scalar detJ recomputation, location/check flags and caption bounds; SHA-256/byte size for all 53 small original JSON/JSONL source files used by the checker.

**Partly completed:** the full numerical validity of the early float32 GSM spectrum (saved Arnoldi convergence flags absent); prespecified-generation provenance (frozen object checked, historical remote push/read chronology not independently repeated); low-arm producer's exact seeded Monte Carlo percentile (exact-law check confirms the gate, RNG replay omitted); “no chaos” (absence not established by these diagnostics); energy-guarantee literature scope (companion citation reviewer checked the relevant primary literature); final cusp closure residuals (not present in the original locate records).

**Skipped by assignment:** model re-execution, GPU/paid computation, original trajectory-array processing, new Lyapunov calculations, new cusp coefficients/orbit closure, Task 2 and its period-6 Newton solves, bibliography/data-availability/whole-article checks assigned to other agents, and unrelated older audits. No off-limits directory or `.env` was opened, listed or searched.
