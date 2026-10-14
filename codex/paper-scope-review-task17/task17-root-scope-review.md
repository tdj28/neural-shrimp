# Task 17: certificate, Huginn, behavioural and inference scope

Reviewed snapshot: private commit `6fcd6ba78c641c2a4b8dcd9ab8a87f32778f2940`; paper `47452e4083ccb1dec00639ebd0c386588359bb7a3b51453319991d7de2d9189b`; packet manifest `e39408a60b07cc6b353b933c0b3e50465d4704d7667af87e0a3f9c44cdb636bf`. Root checked all 43 exported file sizes and hashes (3,239,041 bytes), with no mismatch. All line numbers below refer to that paper snapshot. This is a source/report review, not a raw-data reproduction or a new numerical audit. Tasks 18/19 are accepted separately; Task 16 follows this review.

## Required scope corrections

### R01 — State precisely what the frozen Huginn tests predicted

Locations: abstract 46–47; contributions 124–128; 1093–1098; 1114–1118; discussion 1583–1585.

The short claims that fixed-point spectra predict “where chain-of-thought accuracy breaks” join two different tests too strongly. Part A tested a gold-conditioned estimated Ritz feature against the gold-row residual entering update 32. Part B tested a decline at the fixed arm lambda=1.20 and noninferiority at 1.05, with a four-percentage-point margin. It did not estimate a prompt-specific accuracy-break threshold or demonstrate spectral instability as a causal mediator of accuracy loss. The body already states the two Part B predictions correctly; the abstract/contributions should use that same finite scope. These are genuine passing frozen tests, not a failure to reproduce them.

Suggested short replacement: “In Huginn, estimated spectral features predict residual rankings on held-out questions; a separate frozen generation test confirms accuracy loss at relaxation 1.20 and noninferiority at 1.05.” Keep the held-out GSM8K 33.0/39.5/25.5/9.5% accuracies, decline −23.5 pp, exact one-sided McNemar p=2.759792394613214e−12, and B3 one-sided bootstrap lower bound +2.5 pp. Do not call primary B3 a superiority test or the bootstrap interval exactly calibrated.

At 1116–1117, change “how far ... from convergence” to “the size of the gold-row residual entering update 32.” A residual is not generally a distance to a fixed point, especially with non-normality or instability. State that the frozen primary gate was the confidence interval's lower endpoint above 0.3, not merely the point estimate above 0.3. The actual mean correlation 0.7010874542 and interval [0.5759852217, 0.7793611492] meet that gate.

Authority: `codex/huginn-independent-replay/part-A-001-report.md` (341bcd48) and `part-B-001-report.md` (8f7fa6a4), both independently accepted. No replay is necessary.

### R02 — Partial Ritz spectra cannot establish the absence of unstable modes

Locations: 1039–1049 and especially 1060–1064 (“For the other 24 the native loop is stable up to ...”).

A detected unstable Ritz mode supports a detected instability; absence of such a mode in computed LM/SR/LR subsets does not establish complete spectral stability. The distinction remains even if every requested subset converges. Replace the development-data classification of the other 24 as stable with “no unstable mode was detected in the computed spectral estimates up to ...”, and distinguish this from observed finite-horizon decay. Use estimated thresholds throughout these paragraphs. The later Part A wording about estimated features and admitted partial Arnoldi sets is appropriate and should be retained.

Authority: accepted Huginn source review and Part A report. This requests narrower wording, not dense-spectrum or model replay.

### R03 — Correct the Part A chronology without erasing its amendment

Locations: 1109–1111.

“Their spectra were computed after the read-out run” is too broad. The first 16 spectra preceded readout generation; spectra 16–63 completed after readouts were generated. The producer attests that outcomes had not yet been opened. The local evidence establishes the recorded amendment/selection and producer attestation, not independent observation of historical blindness. Suggested replacement: “The first 16 spectra preceded readout generation; the remaining spectra were completed afterward under a recorded amendment, with the producer attesting that readout outcomes had not yet been inspected.” Preserve the fixed duplicate-copy rule and the actual independent saved-data reproduction.

Authority: Part A report 341bcd48. The historical ordering deviation is already accepted and is not a new admission failure.

### R04 — Separate loss of trajectory predictability from mixing and impossibility of control

Locations: 1719–1720 (“A chaotic loop forgets where it started and cannot be steered precisely”).

Positive finite-horizon largest exponents do not prove mixing, loss of all information about the initial condition, or impossibility of feedback control. An expanding subsystem can coexist with a conserved component; measured trajectory sensitivity also does not rule out stabilization. Suggested replacement: “Positive finite-horizon exponents limit open-loop trajectory prediction at fixed state precision; our experiments do not establish mixing or limits on feedback stabilization.” The practical statement about operating-point dependence can remain.

### R05 — Do not claim complete historical upload hashes for every run

Locations: 1825–1829.

The sentence says that the manifest records the hashes of uploaded inputs and retrieved results for every run. At the exact reviewed source, the tracked manifest was generated at 02:04:55 UTC and has 165 `runs_with_results` rows; code/data hash fields are nonempty in 57 and null or empty in 108, while result hashes are nonempty in 151 and empty in 14. These are metadata-presence counts, not proof that every empty field means an absent upload or a failed retrieval. They nevertheless contradict the universal metadata claim and the paper's 138-row count. An archive `data_sha256`, where present, also does not by itself attest all historical model/dataset asset bytes. The accepted Huginn authority explicitly preserves missing historical upload-content and remote model-byte hashes, using local receipts, reviewed source and producer attestation as a finite basis.

Suggested replacement: “The run manifest records available code, input-archive and retrieved-result hashes, with absent fields left explicit; detailed per-run receipts and accepted provenance limitations are retained.” Give the manifest's actual cutoff and a clearly defined run count. Update the one-GPU-per-run sentence for the later two-GPU 70B runs. This is a reporting correction, not a request to manufacture hashes or repeat model generation.

Authority: source-only supplement manifest `2ddce5b722cd3defc1f9a30c40ea1378e9777d482cb1f30332e4b1e6f8e34907`, schema excerpt `c8fe89992d74d10eb3beed6ac107e5537275728e0cad8bfb43fcd2771e3dbafd`, original tracked manifest hash `97d6f0fcc0d32b81d93d86863f6ba20ab9dd1e7e2ab2abc67e981f9d6a93ca23`.

## Smaller corrections and retained conclusions

### R06 — Identify the persistence statistic

Locations: 1336–1338 and 1548–1549.

The reported 16 and 6.5 turns are the geometric summaries 1/(1−P_stay), not directly fitted residence-time distributions or independently measured mean visit lengths. Use “geometric persistence summary” or define the expression on first use. The values and their first-versus-second classifier comparison are correct.

Authority: Task 11 report 02d96503, which records 16.070370 and 6.517970 and explicitly distinguishes this statistic from a fitted dwell-time or survival model.

### R07 — Use the oriented passage coordinate consistently

Locations: Methods 336–340 versus the controlled-passage paragraph 480–483 (and the natural-entry coordinate at 511–513).

The Methods correctly defines `v = sign(b) u`, with passage from v=−U (or 0) to v>U. The body switches back to `u_in=−U` and `u>U`, immediately after listing a<0 and b<0. In the original u gauge, the ghost-side increments are negative, so this stated exit orientation is reversed. Replace the body's u with the oriented v, and label the natural-entry coordinate in the same gauge. This is a notation correction; retain all measured and predicted counts and all failed criteria.

### Retain the certificate statement

The rung-3 statement at 1792–1803 agrees with the accepted release 539ead0d: exact stored binary parameters/weights, all 24 layers and the full 151,936-token readout, primitive period 8, uniqueness as an ordered shooting root in the eight-box product, and local attraction. Its explicit warning that this is not uniqueness of all F^8 fixed points in the first box is correct. Do not strengthen this to global window organization, a certified shrimp, native autoregressive generation or training utility. For maximum reproducibility, say “zero-based block 12” and specify that temperature is the real exponential of log(10) times the exact dyadic log-temperature. These are clarifications, not missing proof gates. The trusted Python/FLINT/Arb execution plus independent exact saved-witness audit is the accepted finite proof standard; no new proof-assistant or model replay is needed.

### Retain the adverse behavioural and passage results

The paper correctly preserves both failed all-row passage criteria, the second classifier's generic-kindness benefit, the ramp-versus-fresh 0.05 interval crossing zero, the dose-0.04 L8 interval crossing zero, and the unavailable discarded originals. The pooled second-classifier L8 increment +0.043 [0.024,0.062] correctly rounds the accepted +0.0427424360 [0.0235134605,0.0617517785]. Keep “steering dose” as the efficiency metric: regeneration requires extra generation calls and is not a demonstrated saving in total compute. The finite empirical committor and heuristic extrapolation caveats at 1512–1527 are appropriate; do not restore a Bayes-ceiling or memorylessness claim.

## Review boundaries

Theory/cusp/return-map and large-model/search-scale findings are provided in parallel source-bound notes. This note neither claims an exhaustive independent reconstruction of every paper number nor clears figures that were not rendered in this source review. No producer paper edits, model execution, new tests, external message or pod action occurred.
