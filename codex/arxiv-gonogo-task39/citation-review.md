# Task39 citations, symbolic dynamics and claim strength

Article-only snapshot: `4ca5b5e35bc766581a5e2d326858fcd0edc312bb`. Whole `main.tex` and `references.bib` read. Main SHA256 `30c1f059ee225dbe0c2a3462d50ca501a580d9245e3209974eb17b98f403ceec`; bibliography SHA256 `1f49caa5d53c69aec840f40dcb73ef25f6981cae25553e7f7d626cffbc565f2e`. Line numbers below refer to these exact bytes, not the longer report or former p6-only branch.

**Lane recommendation: GO WITH FIXES.** No must-fix central scientific or citation-identity failure found in this lane. Three should-fix items below, plus one shared nondegeneracy qualification already owned by the raw-evidence reviewer. This is not a whole-submission GO and does not independently validate numerical results or the bundle. No paper/repository edits, Git mutations, target/model calls or numerical science reruns occurred.

## C1 — should-fix: optional pagination does not match the primary print edition

**Location:** main.tex:1250 (`maheswaranathan2019reverse`); references.bib:516.

**Exact field:** `pages = {15696--15705}`.

**Evidence:** The official [NeurIPS citation export](https://proceedings.neurips.cc/paper_files/paper/2019/file/d921c3c762b1522c475ac8fc0811bb0f-Bibtex.bib) leaves `pages = {}`. The publisher's [NeurIPS32 printed table of contents](https://www.proceedings.com/content/053/053719webtoc.pdf), PDF page 49, starts the exact title at 15617, with the next paper at 15627 on page 50. The [official paper](https://proceedings.neurips.cc/paper_files/paper/2019/file/d921c3c762b1522c475ac8fc0811bb0f-Paper.pdf) has 10 pages. Thus the current range is unsupported by the inspected primary edition; this does not prove that no older pagination convention ever used it. Authors, title, venue, volume and year match, and the fixed-point/line-attractor attribution at 1250 is appropriate.

**Proposed change:** remove the optional `pages` field, following the official online export. Alternatively cite the printed edition as `15617--15626`; its ending page is inferred from the starting page and10-page length, consistent with the next start. Do not silently substitute a secondary database range.

**Retained evidence:** `maheswaranathan2019-official.bib`, `neurips2019-publisher-toc.pdf`, `neurips2019-toc-page-49.txt`, `maheswaranathan2019-official-paper.pdf`, `pdf-metadata-receipt.json`. The manifest binds exact bytes. The same publisher contents supports Bai2019's current690–701 range (start 690, next 702, 12-page official paper), resolving that earlier gap without changing it.

## C2 — should-fix: finite-period agreement does not by itself identify asymptotic orbit dependence

**Location:** main.tex:533–534, prompt paragraph.

**Exact quote:** “the reached orbit still depends on the start: the detected periods agree at 86.5 to 93.1\,\%.”

**Evidence:** The manuscript defines finite-window period detection at 449–458 and explicitly acknowledges long transients at 465. Agreement or disagreement of those finite-window labels does not, by itself, establish equality or inequality of asymptotic invariant orbits. The Task38 correction correctly separated the98.9–99.0% chaos-threshold agreement from86.5–93.1% period agreement, but this remaining clause still overinterprets the latter. This is a logical claim-strength check, not a new raw-data recount. Actual coexistence evidence elsewhere is not being rejected.

**Proposed wording:** “For the three random starts, the chaos/non-chaos threshold labels agree with the base run at 98.9 to 99.0% of cells, while the detected finite-horizon periods agree at 86.5 to 93.1%; this comparison does not establish independence of the reached orbit from the start.”

## C3 — should-fix: scope numerical chaos validation to the sampled checks

**Location:** main.tex:1234, “What is and is not claimed.”

**Exact quote:** “the chaotic cells are numerically validated (twin exponents agree with automatic-differentiation tangents, across precisions and horizons)”.

**Evidence:** The same article465 describes random samples of 192 chaos-labeled cells from each main map and says some became periodic after 600–97,500 steps. Other precision/tangent checks are also sampled studies, rather than a validation of every chaos-labeled cell. The first clause “Chaos labels are finite-horizon estimates” is good, but does not make this unrestricted validation clause accurate. [Grebogi–Ott–Yorke](https://doi.org/10.1016/0167-2789(83)90126-4), already cited at 458, supports the distinction between transient and asymptotic chaos. No closed numerical audit was repeated.

**Proposed wording:** “Chaos labels are finite-horizon estimates. Sampled chaos-labeled cells were checked with tangent derivatives, multiple precisions and longer horizons; some labels were long transients, and asymptotic chaos has not been established by a computer-assisted proof.” Continue with the separate three certified periodic orbits, cusp and finite-order qualifications.

## C4 — should-fix, shared with the raw-evidence lane: qualify the Huginn bifurcation classification

**Location:** main.tex:1265–1266.

**Exact quote:** “the relaxed loop's leading complex pair crosses the unit circle near relaxation 1.0775, a Neimark--Sacker bifurcation”.

**Evidence:** [Kuznetsov's primary book/chapter](https://link.springer.com/chapter/10.1007/978-3-031-22007-4_4) concerns a generic Neimark–Sacker bifurcation, which needs more than a complex unit-circle crossing, including the appropriate nondegeneracy/nonresonance conditions. The existing Task37 raw audit, relayed by the assigned raw reviewer, found no computed first Lyapunov coefficient and incomplete convergence diagnostics for the early spectrum; Task39 did not rerun those calculations. The amplitude fit is supporting numerical evidence, not a substitute for all genericity conditions.

**Proposed wording:** “the relaxed loop's leading complex pair crosses the unit circle near relaxation 1.0775, consistent with a Neimark–Sacker bifurcation…” The raw reviewer owns this same finding; count it only once in the integrated review. That reviewer also owns the separate blanket “We did not compute flip coefficients” contradiction at 1301–1302; do not count that as a second finding from this lane.

## Checked-correct coverage and interpretation boundaries

`coverage.md` and `coverage.json` give a separate row for **every 86 cited keys**, their exact article line(s), current entry, primary URLs, reused receipt paths/hashes and new verification notes. There are 88 entries and no missing key; `mira1987chaotic` and `vitolo2011hubs` are uncited. They are harmless unused bibliography records, not dangling article citations. No removed-P5 dangling citation or symbolic claim was found in this snapshot. The full bibliography is byte-identical to the assigned Task38 bibliography; reused checks are explicitly distinguished from fresh requests.

New work saved 28 complete publisher-deposited Crossref responses, three official conference BibTeX exports, the publisher contents and three official paper PDFs. Important earlier gaps closed: Sompolinsky ending 262; Doyon issue 2–3; Krawczyk issue 3; Xu 3082–3095; SoftThinking 187190–187212 and six-author conference byline; PonderLM 91874–91893; Ramsauer ICLR2021 via official conference listing; Bai 690–701 using printed-edition start and paper length. The only detected pagination discrepancy is C1. One narrow unverified field remains: the Fujisaka DOI deposit gives H.Fujisaka/T.Yamada but does not expand the current Hirokazu/Tomoji given names; direct OUP access failed. Use initials if requiring every supplied name expansion to be independently source-bound. This is not evidence that those names are wrong.

- **Symbolic section1128–1193:** the Jones reference is now an interpretation on selected Rössler returns, motivating approximate coding. The text supplies the alignment condition; states pooled 48/58 and single-phase 21.7% separately; explicitly discloses 240 MSS rejections and 1469/46046 total failures; describes 153 C labels as ambiguous coding, not superstable curves; and limits the selected Newton checks to 16 confirmed labels, two complex-pair C cells and five unconfirmed selections. These match the already completed Task38 audit's categories. No new full-state Newton, multiplier or closure authentication is claimed: the original SD Newton artifacts remain summaries-only. The manuscript reports producer calculations; this lane has not independently reproduced them.
- **Singer1143–1146:** the interval-endpoint alternative is restored, and the text explicitly says extra turns establish neither a global scalar reduction nor negative Schwarzian. This addresses the mathematical overreach. Primary metadata: [Singer](https://epubs.siam.org/doi/10.1137/0135020); the theorem formulation was previously checked against Proposition 2.6 in [Ou's original thesis](https://www.math.stonybrook.edu/alumni/2018-Dyi-Shing-Ou.pdf), because Singer's full text was inaccessible. Do not describe that as a new direct proof audit of Singer.
- **MSS1150–1169:** [Metropolis–Stein–Stein](https://doi.org/10.1016/0097-3165(73)90033-2) and [Milnor–Thurston](https://link.springer.com/chapter/10.1007/BFb0082847) are appropriate. The text now distinguishes an orbit label matched to a superstable quadratic itinerary from a measured critical-point itinerary. Explicit filtering prevents membership of retained codes being presented as an independent theorem confirmation.
- **Architecture82–93,1261–1272:** the fixed-prefix soft-token operator is distinguished from growing-context reasoning and shared-block recurrent depth; the energy argument is restricted to the isolated tied readout. The citation group of Universal Transformers, Looped Transformers and Huginn is relevant; it does not imply their update Jacobians equal the soft-token covariance factorization. No broad literature-absence claim about all complete transformer energy functions remains.
- **Safety1251–1256:** Korbak is now attributed specifically to chain-of-thought monitorability, while new intervention tests are proposed and no safety property is asserted. This is consistent with [the primary position paper](https://arxiv.org/abs/2507.11473).
- **Chaos449–469:** operational labels, finite horizons, transient capture and the positive-measure attractor distinction are explicitly stated. C3 targets only the later overbroad summary. The prompt's “Every prompt” at 528 is scoped by the preceding 22 variants; “Every tested prompt” would be a harmless clarity improvement, not a separate scientific failure.
- **Prior-work98–109 and method/theory citations:** the reviewed uses distinguish text-level sensitivity/statistics, autonomous map calculations, equilibrium analyses and the explicitly acknowledged small-transformer fold precedent. The literature records and earlier source reviews support those distinctions. The statement “we know of no earlier computer-assisted proof” is an author-qualified novelty claim; this review is not an exhaustive literature search proving absence. Metadata verification likewise does not independently re-derive 86 cited studies.

## Execution and evidence limits

Only small serial metadata retrieval/parsing ran, using the previously authorized interpreter under `nice -n 10` with OMP/OpenBLAS/MKL set to 1 and HF offline. Public DOI/conference retrieval needed approved network access after sandbox DNS failure. No installation, paid access, external message, target array load, numerical control rerun, model or GPU call occurred. The parent owns Git/checkpointing and whole-submission GO/NO-GO. `failures.md` preserves DNS, browser, missing-library and Poppler-warning outcomes; successful retries do not erase them. Exact response/source/file hashes are in `manifest.json`.
