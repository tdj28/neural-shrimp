# Task40 narrow discussion delta — completed saved reduction

Article snapshot: `bb42a82189da91c37f50804c077832df9e2237ae`. Research reference: `c567545ca93f90324dae75be2c333c8ac452dab5`.

The supplied lines 3187, 3245 and 5153–5154 belong to `paper/llm/main.tex`, the full research report. The corresponding article claims are lines 1655–1658 and 1720–1723. The 57%/38% channel shares appear in the report's appendix, not the article itself.

This is a narrow saved-record check. No models, QR iterations, old controls, old full audits, bootstrap resampling or new trajectories were run. Root checkpointed the independently written `discussion_delta_audit.py` at private commit `5afb77ae13409c0d7ed5ece53c92fecc62a559be` and executed it once as `discussion-001`. Source SHA256 `c219f8b689668b1164271c9b28f10584cfc3f4ca2e828f382c37b96b50aa920d`, 6071 bytes.

The run exited 0, no timeout, child 1637243 reaped; receipt UTC `2026-10-20T03:41:54.881894+00:00`. Launch-through-reap upper time 0.21681141015142202 s; internal reduction time 0.06028466997668147 s. One numerical thread, nice increment 10, offline/CUDA-empty environment, 4 GiB AS, 90 CPU / 120 wall seconds. The result authenticates 11 input files by full byte hashes in `manifest`.

| File | Bytes | SHA256 |
|---|---:|---|
| `discussion-delta-evidence.json` | 62385 | `aaa1c1e750ef1df1240192c6ea95d6c0d3876b033d2cacc71c83ab362365a57d` |
| `discussion-001.receipt.json` | 1128 | `916af7b8791c23568a7549e491e554daf03abb2a367cb8e6e11b7e755d524fe4` |

**Outcome:** the spectrum statement is numerically supported. The article's absolute “vanish for single samples” needs a qualifier; the baseline and means should name their offset/estimator. This is a narrow wording correction and does not reverse the measured suppression by sampling.

## Spectra metadata

The actual three-cell source is `/data2/neural-shrimp-data/explore/e2e/lyap-spectrum/lyapunov-spectrum.json`, not the Task21b KY dataset. Its metadata says complete, k=4, transient=1000 and record=600, CPU float64, with coordinates matching `docs/e2e/lyap-cells.json`. `docs/e2e/run-q3e.sh` declares the same command and parameters.

The saved rows are:

| alpha | log10 T | Four retained exponents |
|---:|---:|---|
| .0652 | -.156 | .10940792690952314, -.8219359395916574, -1.3205223370723547, -1.6896832520736693 |
| .07484 | -.1226 | .42415037775733194, -.9914874613984112, -1.3906901442878306, -1.8064367816976248 |
| .064 | -.115 | .23877119577035952, -1.14555431905787, -1.7955112093942, -2.003065751833011 |

The article's -0.82 to -2.01 is a valid outward covering interval; the attained negative extrema are -0.8219359395916574 and -2.003065751833011. Each saved finite vector is ordered and has exactly one positive component, with negative first-two sum. Independently recomputing `1 + lambda1/abs(lambda2)` gives **1.1331100413541688, 1.4277919734447302, 1.2084328886008044**, agreeing with 1.13, 1.43, 1.21 and the article's 1.13–1.43 summary.

Task21b's closed receipt was inspected only to establish that it covers a different 2000-step population (KY1/KY2/TC), so it must not be silently reused as authority for these 600-step rows. No new runtime/source/loaded-asset qualification is inferred from the small raw JSON. Its exact pin is 1265 bytes, SHA256 `8284fb00e782c9303586b255e25d22c84a6f2029ecfd8cf1f90dd8294f9f1a59`.

## Sampled bridge definition and original scope

The exact inputs are the four `explore/llm-soft-loop/bridge-{exact,m64,m8,m1}-t-0.115/{walk.json,walk.npz}` records. Their original setup uses CPU float32, 32 trajectories per offset, 100 transient and 512 recorded F-steps, at eps=1e-5,3e-5,1e-4 relative to alpha_c=.0627983. This predates the later float64 fold analysis.

The plotted “laminar” criterion is a token-probability mask:
`p(" is") > .5`, or `p(" it") > .1` and larger than both `p(" a")` and `p(" there")`.
The mean includes right-terminated runs of at least six F-steps. The plot retains a qualifying run that touches the left observation boundary, and drops a run still open at the right boundary. These are not the u-coordinate/3U geometric phases of the earlier Sec. V supplement. The new reduction preserves that original convention and records both endpoint cases explicitly.

The independent reduction reproduces the exact plotting convention in `tools/paper_llm_figures.py:74–103`, without importing that module. All 96 coordinate rows per run match their JSON metadata, and all probability entries are finite. Original plotting source SHA256 `84d8d4abce38e7b945c63d39ecb329e15bc61b14d98b98565e68a2444872c164`.

| Map | eps | Retained phases >=6 steps | Mean F-steps per retained phase | Channel-mask steps / 16384 | Channel fraction |
|---|---:|---:|---:|---:|---:|
| exact expectation | 1e-5 | 96 | 120.000000 | 15872 | 96.8750% |
| exact expectation | 3e-5 | 224 | 63.000000 | 14912 | 91.0156% |
| exact expectation | 1e-4 | 0 | undefined | 0 | 0% |
| m=64 | 1e-5 | 649 | 10.519260 | 9496 | 57.9590% |
| m=64 | 3e-5 | 655 | 10.323664 | 9325 | 56.9153% |
| m=64 | 1e-4 | 616 | 10.722403 | 9409 | 57.4280% |
| m=8 | 1e-5 | 248 | 7.814516 | 6350 | 38.7573% |
| m=8 | 3e-5 | 238 | 7.710084 | 6231 | 38.0310% |
| m=8 | 1e-4 | 253 | 7.596838 | 6352 | 38.7695% |
| m=1 | 1e-5 | 1 | 7.000000 | 1750 | 10.6812% |
| m=1 | 3e-5 | 1 | 8.000000 | 1911 | 11.6638% |
| m=1 | 1e-4 | 3 | 6.000000 | 1872 | 11.4258% |

### D1. Qualify the nonzero single-sample phases and the offset-specific baseline [must-fix wording]

**Where:** article Discussion, “Beyond the one-token loop,” lines 1720–1723; related full-report text at lines 3244–3247 and 5152–5157.

**Claim:** phases “shrink from 120 steps to 7.7 and 10.6 steps ... and vanish for single samples.”

**Evidence:** m=1 has **one, one and three** retained phases, with lengths **7; 8; 6,6,6**, across the three offsets. None touches either observation boundary. Thus the actual absence statement is false even under the original six-step filter. The mask remains active in about 11% of recorded single-sample steps. The baseline 120 is specific to eps=1e-5; it is 63 at 3e-5 and has no qualifying phase at 1e-4.

**Replacement:** “At the closest tested offset, the plotted mean laminar length was 120 steps for the deterministic expected-embedding map. Across the three tested offsets, averaging 8 or 64 sampled embeddings gave mean lengths of roughly 7.6–7.8 and 10.3–10.7 steps, respectively; qualifying phases became rare with single samples.”

For a more compact article sentence, retain “about 8 and 11 steps” and refer the exact offset table to the full report. The mechanism remains visibly suppressed; “become rare” is supported, “vanish” is not.

### D2. Treat 10.6/7.7 and 57%/38% as approximate descriptions, not per-offset equalities [note]

The raw **plotting estimator** gives the means above, so 10.6 is not the exact one-decimal mean at each offset. The original log describes a trajectory-bootstrap analysis, while the plotted curve pools retained phases; that historical bootstrap has not been rerun or independently reconstructed. “About 10.6” and “about 7.7” capture the broad scale, but the explicit observed ranges avoid mixing estimators.

The reported channel percentages concern the probability mask at **all** recorded steps, including fragments shorter than six steps. They are not the fraction of steps belonging only to the retained phases. Full-report values should preferably read “about 57–58%” for m=64 and “about 38–39%” for m=8. “Independently of eps” should mean the observed near-flat behavior at these three offsets, not a proved independence or asymptotic law.

Endpoint scope matters for the exact baseline: each deterministic trajectory has a left-boundary run of 90 steps at eps=1e-5 (61 at 3e-5), included by the plot, and a right-boundary run of 134 (17) excluded. Therefore 120 and 63 are the original plotted window statistics, not newly uncensored complete-phase expectations. These details are retained in the evidence; this audit does not silently substitute a different censoring rule.

## Completed / partial / skipped

- **Completed:** exact three-cell spectrum parameters, all 12 finite exponents, sign/order gates and KY arithmetic; raw bridge probability masks, per-offset denominators, all retained run lengths, means and endpoint cases for all four runs.
- **Completed:** exact article/report line distinction; original plotting-method source read, hash-bound without import; identification that Task21b is not the same spectrum dataset.
- **Partial:** original source/loaded-model/runtime provenance and effective numerical precision are declared historical metadata, not newly qualified. The bridge metadata explicitly says CPU float32, while the spectra declare CPU float64.
- **Skipped:** original trajectory-bootstrap confidence intervals, repeat random samples, model/QR/spectral recomputation, full Task21b replay, and asymptotic claims. No such missing check is marked passed.
