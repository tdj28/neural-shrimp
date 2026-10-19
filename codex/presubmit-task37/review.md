# Task 37 — pre-submission review

- **Six findings: one must-fix and five should-fix.**
- **Part 1 is complete at the requested snapshots**, with the numerical and source limitations below.
- **Part 2 has not run: this independent check has not yet established two distinct period-6 orbits at R.**

Article reviewed: `18490e2cbac9a43be22ff0e3e29f383024cc5802` in `/data2/chaos-journal-neural-shrimp`. Research reference: `35eabc6c17ac944564985a7a01ad7f6bc92bf0ce`. Sources were read through Git objects; the moving producer checkout was not edited. The article source SHA-256 is `9c9881359e4622c88202b664f90e1c9340de626450e3f9208175417d984e7d94`; its 26-page PDF is `df7de50340d31368e1c2cb15c5e497563679cf09bc622fd9360a2774073f880b`. Fig. 20 and the changing second-shrimp paragraph are excluded from this textual audit, as requested. Their independent numerical question remains Part 2.

Supporting files in this folder: `raw-review.md`, `raw-check.py`, `raw-check.json` (53 original-file hashes and all row gates); `citation-review.md`, `citation-sources.json`; `check_submission.py`, `submission-checks.json`; and `p6-seed-inventory.json`. This is a fresh submission audit, not a repeat of closed certificate verification. None of these findings invalidates the previously released periodic-orbit certificates.

### F1. Huginn's spectral crossing is 1.0775, not the sampled-grid endpoint 1.0777 [must-fix]

Where: `main.tex`, Discussion, “Beyond the one-token loop,” lines ~1184–1186 at `18490e2`; also `paper/llm/main.tex`, recurrent-depth section, lines ~2938–2942 at research `35eabc6c`.

Claim: “its multipliers predict a Neimark--Sacker bifurcation ... at relaxation 1.0777, against 1.0778 from the amplitude.”

Evidence: original `/data2/neural-shrimp-data/explore/latent-loop/huginn-fixed-spectrum-gsm-claire/fixed-spectrum.json` contains the pair `mu = 0.5883682635687575 ± 0.7710905065294952 i`. Substitution in `2(1-Re(mu))/|1-mu|²` gives **1.0775400117776854**. The stored endpoint `1.0777306826706676` is the first unstable sample of a 4,000-point grid over `[0.001,2]`, as shown by the producer's source at the reference commit; it is not the analytic crossing. At the corrected crossing the pair has modulus one and argument magnitude 0.1560814767 turns; the next distinct mode in the computed partial spectrum has modulus 0.9931540784. The radial derivative is positive. The original eight-row `huginn-ns-gsm-claire/ns-check.json` gives the independently recomputed squared-amplitude intercept **1.0778010471978443**, so 1.0778 is correct. See `raw-check.py` and `raw-check.json`; the full derivation and retained float32/partial-spectrum limitations are in `raw-review.md`.

Correct value / wording: “its estimated multipliers predict a complex-pair unit-circle crossing near relaxation **1.0775**, consistent with a Neimark--Sacker bifurcation, against **1.0778** from a fit of squared amplitude.” Correct the full report's explicit equality too. Their difference is approximately 0.000261, not the apparent 0.0001. No first-Lyapunov-coefficient/nondegeneracy proof is supplied by these records.

### F2. The submission form and archive describe an older 25-page article [should-fix]

Where: `/data2/neural-shrimp-data/arxiv/chaos-v1/arxiv-form.md` and `manifest.json`; the corresponding source archive, rather than an article paragraph.

Claim: “25 pages, 20 figures, 4 tables”; the archive manifest says its source commit is `f987f902a20854213c9bb917f35daeffa1efb6e1` and its test build has 25 pages.

Evidence: the exact requested `18490e2:main.pdf` has **26 pages**, 20 numbered figures and four numbered tables. The form's proposed abstract matches the current abstract after whitespace normalization and has 1,434 characters excluding the trailing newline, within 1,920. The archive SHA-256 matches its manifest (`c316f49f8eee7733a3253bbf28f5efb1cc598dd61036696708e1f668b0038c0d`), and its extracted `main.tex` matches the manifest's old upload hash; this is an authentic older package, not corruption. From `f987f90` to `18490e2`, the article source has 92 changed lines and the bibliography has 68 added lines. `check_submission.py` records the comparisons without extracting into or editing the producer's directory.

Correct value / wording: regenerate and test the upload package from the author-approved final snapshot, then set the form's page count to that build's actual value (**26 at the audited snapshot**). Do not upload the old archive on the assumption that it contains tonight's changes. Recheck after the excluded Fig. 20 edits; no final future page count is asserted here.

### F3. Document how readers obtain the native certificate inputs [should-fix]

Where: `main.tex`, Data Availability, lines ~1263–1270 at `18490e2`.

Claim: “The interval-arithmetic certificates, with their verification code, exact inputs and release records, are there too.”

Evidence: **all three cited paths exist at `35eabc6c`**, and contain verification code, exact map specifications and released certificate records. The release records also point to essential native inputs outside Git. For example, the P8 release points to `/data2/neural-shrimp-data/codex/llm-p8-certificate/final-domain-002/definition.json` (SHA-256 `42959bcd5b1066c5c2ea326dd43e88023c233d6e5f58ab1062005797be230840`, 4,385,478 bytes); the pair release points to `coexistence/p2-final-domain-001.json` (1,055,555 bytes) and `p3-final-domain-001.json` (1,582,618 bytes). These JSONs hold exact centers and radii. Their Git blob IDs are respectively `e86a7a0ff75c19259ef5bf3c34062e376c241910`, `d9ca5440e101ae761384244629f35787ad5026d7`, and `71fe441f4cc0683f126dcf79ba3d974c07facccc`; none occurs in the complete `35eabc6c` tree. Original interval enclosures and power witnesses likewise have external raw-data paths plus hashes. A hash identifies bytes but does not make them available to a reader. See `availability-checks.json`.

The missing blob IDs establish absence of byte-identical standalone files, not absence of every possible equivalent encoding or an embedded archive. Such alternative representations were not exhaustively searched. The inspected release paths do not themselves provide a reader-accessible download of the external inputs, so an explicit retrieval route is needed before treating availability of all exact inputs as verified.

Correct value / wording: if these originals are supplied on request, say “The repository contains the verification code, exact map and input specifications, hashes of the native certificate evidence, and release records. The original center definitions, interval enclosures and other bulky certificate evidence, as well as trajectory arrays, are available from the author upon reasonable request.” Alternatively, identify the existing downloadable representation or archive the actual certificate inputs and cite it. This is an availability qualification, not a failed proof. External anonymous access to the private repository was not verified; publication/visibility remains the author's responsibility.

### F4. Narrow the energy-guarantee statement to the studied operators [should-fix]

Where: `main.tex`, Discussion, “Beyond the one-token loop,” lines ~1180–1182 at `18490e2`.

Claim: “energy-based guarantees of convergence require specific structure and update rules ... and none is established for a complete transformer update.”

Evidence: the limited point about the article's tied readout is sound. The unrestricted wording invites a literature-wide reading. [Hoover et al., Energy Transformer, NeurIPS 2023](https://papers.nips.cc/paper_files/paper/2023/hash/57a9b97477b67936298489e3c1417b0a-Abstract-Conference.html), gives energy descent for a specially engineered recurrent block containing normalization, attention and an MLP analog; [Sec. 2, Eqs. 6–8](https://arxiv.org/html/2302.07253v2#S2) concern continuous-time dynamics, with numerical discretization. That is not a general discrete-time convergence theorem for Qwen or Huginn. It nevertheless makes “a complete transformer update” too broad. See the primary-source distinctions in `citation-review.md`.

Correct wording: “The energy-descent argument for the isolated tied readout in Sec. [theory] does not establish convergence of either complete feedback operator studied here.”

### F5. Attach the safety citation specifically to chain-of-thought monitorability [should-fix]

Where: `main.tex`, Discussion, “Steering and predictability,” lines ~1172–1175 at `18490e2`.

Claim: future testing of steering for safety, “alongside the monitoring of latent computation,” citing Korbak et al.

Evidence: [Korbak et al., arXiv:2507.11473v2](https://arxiv.org/html/2507.11473v2), concerns natural-language chain-of-thought monitorability, including how latent reasoning may reduce it. It does not establish a general method or effectiveness result for monitoring continuous latent states. The article appropriately says “motivate testing whether” and “We tested no safety property”; it does not claim that instability is misalignment or that safety interventions were tested. This is citation precision, not withdrawal of the proposed research question.

Correct wording: keep the proposed test, followed by “Such tests would complement research on how latent-reasoning architectures affect chain-of-thought monitorability.” Retain the explicit disclaimer that no safety property was tested.

### F6. Round the cusp multiplier-error bound upward [should-fix]

Where: `main.tex`, Appendix A, Table IV caption, lines ~1226–1228 at `18490e2`.

Claim: “the critical multiplier equals +1 to within 1.2 × 10^-9.”

Evidence: original `/data2/neural-shrimp-data/explore/llm-soft-loop/pod64/fm5c/locate-curve-a1-up-fold-minus-30/locate.json` has `mu = 1.0000000012005652`, so `|mu-1| = 1.2005652028079794e-9 > 1.2e-9`. All other principal cusp rows lie below it. `raw-check.py` recomputes this and every other requested table value from the originals, preserving duplicated locators separately.

Correct value: **within 1.3 × 10^-9**. This small rounding error does not change the cusp classification or the coefficient/determinant results.

## Checks that passed and limits

- **Architecture boundary — complete.** The fixed-prefix expected-token feedback map differs from an outer autoregressive context-growth loop and an inner recurrent-depth hidden-state update. The cited architectures support the distinction. Shared depth weights alone need not imply an autonomous hidden-state recurrence: Universal Transformers also uses step embeddings/halting. The reviewed text does not claim otherwise.
- **Eight bibliography entries — complete.** Authors, titles, years, venues and DOI/arXiv IDs agree with primary records for Dehghani, Giannou, Bae, Korbak, Zhu, Clark, Wainwright–Jordan and Gao–Pavel. The full per-entry table and links are in `citation-review.md`; retrieval failures and successful alternative primary records are preserved in `citation-sources.json`. No metadata field remains unverified. This does not audit all results in those papers or prove the absence of later versions.
- **Huginn generation — complete against saved rows.** All 800 rows passed admission, coverage, duplicate, answer extraction and scoring checks. Correct counts per 200 at relaxation 1.0/1.05/1.15/1.20 are 66/79/51/19; the high-arm loss is 23.5 percentage points. At 1.05 the observed gain is 6.5 points. The frozen hypothesis was noninferiority within four points; “no loss” should be read as observation, not proof of zero possible loss. Prefer “noninferiority within four percentage points at 1.05” for the formal conclusion. The exact paired-bootstrap law supports that gate; the producer's exact pseudorandom bootstrap sequence was not replayed.
- **ARC — complete against original spectra/traces.** Detected unstable indices are 4, 6, 7, 8, 25 and 27; each recovered a fixed point at relaxation 0.75. All thirty spectra and convergence flags were read. Twelve questions contain at least one partial Arnoldi set; these remain partial, and lack of a detected unstable mode is not a full-spectrum stability proof. The original failed q4 attempt is preserved, not counted as an admitted spectrum.
- **Huginn “no chaos” — partial as an exclusion claim.** The data support convergence and oscillatory instability. They do not exclude chaos with a full diagnostic campaign. “We did not establish a chaotic regime in these tests” expresses the supported limit. The early GSM float32 spectrum lacks saved Arnoldi-convergence and Ritz-vector residual flags; those were not invented. F1 is exact substitution into its saved multiplier estimate, not new spectral certification.
- **Four P5 cusps — complete table/original-record comparison.** All four c values, det J values/signs, coordinates and next-modulus bounds agree; each determinant was recomputed. Four principal plus two duplicate locate records were checked. The maximum next modulus is 0.007880388437016245, consistent with 0.0079. Final full-state closure residuals are not in these locate records, so this review does not independently revalidate closure or rerun cusp coefficients.
- **Gallas anatomy — complete for requested claims.** The original PDF's journal p. 214 gives `(a-b-x²)²-(a+b)`; p. 217, Fig. 7 replaces old `a-b` and `a+b` by new a and b, giving the article's map. The article PDF's Fig. 16 (PDF p. 18) was rendered and inspected alongside Gallas Fig. 7. A small independent 60-digit Decimal Newton solve gives A = (1.019058879422658468…, 1.165290861969399463…) and B with coordinates exchanged. The stored plotted corners (1.019058, 1.165290) agree with Gallas to **six significant digits**, though not six decimal places. At R = (5/4,5/4), the two distinct fixed states `(-1±sqrt(2))/2` both have multiplier -1. The flip-crossing statement passes. The source/formula check is not a rerun of the plot's full grid.
- **Abstract/body and numbering — complete.** The 42/48 passage count, 173/192 chaotic and 7/192 periodic text-change counts, 0.5B–72B range and limited readout-growth claim match the body. Every cited bibliography key is defined; every bibliography entry is cited; no duplicate keys/labels or unresolved source references were found. Built PDF has Figures 1–20 and Tables I–IV. This is consistency checking, not a fresh rerun of all older experiments.
- **Certificate paths — complete as availability inspection.** All specified paths/releases exist at the reference commit; raw certificate bytes are distinguished in F3. Closed certificate computations were not repeated.
- **ArXiv form — complete comparison with the requested built PDF.** Abstract length and figure/table counts pass; page count and package snapshot need F2. No paper, package or public visibility was edited.

## Part 2 — prepared inputs, computation pending

Both original arrays were hash-bound and read as finite float64 matrices. Curve 1 has 29 rows and row/index 6 is at `(0.07290699285213686, -0.15395151820007266)`; curve 2 has 57 rows and row/index 2 is at `(0.07291880394312411, -0.15409386771440967)`. Every `index` is sequential and the row counts agree. Rounded locators in the brief are not substituted for these original metadata. Target R and both body points are pinned as the requested binary64 values in `p6-seed-inventory.json`.

No model was constructed or evaluated, and no new period-6 convergence, multipliers, tokens, distinctness or attraction is claimed. Preparation uses only the independent manual map, not the producer's numerical checker. The first bounded Newton/JVP/full-spectrum execution packet is prepared in this folder and held. It uses bounded Newton–GMRES, retains all 896 monodromy columns and eigenvalues, checks all six phases, and records cyclic alignment and token labels. Six synthetic numerical controls passed; the execution wrapper has separate mocked authority/interface controls. These are preparation checks, not evaluations of the requested model. Conditional body-point execution and an independent saved-output audit remain to be completed after qualified R results exist. Target execution requires its reviewed source, private checkpoint and coordinated capacity after the earlier local memory-floor stops. The automatic approval reviewer currently fails with “Selected model is at capacity” before executing Git operations; no private fetch/checkpoint/delivery is claimed. This review is retained locally for delivery when that blocker clears.

The prescribed `uv run --no-sync python` refused at snap-confine before Python. Small metadata/scalar checks used an explicitly disclosed fallback, `HF_HUB_OFFLINE=1 nice -n 10 /usr/bin/python3`, or the existing project Python for NumPy array inventory, with one numerical thread. No installation or environment change was made. The initial raw-check answer-parser mistake was corrected before conclusions and its output is retained separately; it is not a producer failure.

**Skipped:** rewriting the excluded second-shrimp paragraph/Fig. 20; changing papers or submission artifacts; public release; paid/GPU work; re-auditing closed certificates; new Huginn/model/cusp runs. **Not yet completed:** Part 2 target computation and its independent saved-output audit; Git/private-main delivery. Those omissions are explicit, not positive results.
