# Task 27 theory, cascade, bridge and return-map review

Frozen source: commit `66ffa102c56f6d5b22f116a1bb194f503c75562e`; `/tmp/task27-source-001/paper/llm/main.tex`, SHA256 `d78a22090469e1dc32e67e715c0a2f51cc538b6728496c04974d9fc16a99b05f`. All line numbers below refer to this file. Bibliography: SHA256 `33ecc865743c1f9c1bd5bc2d43e081b51b559d7e4de0d885d8d5bbc4f12ba75b`.

This is a report-only review. I read every assigned line in 297–378, 476–538, 1208–1488 and 2141–2325, plus the related-work passages relevant to my citation lane. I reused the accepted Task 13/17/18/19/21 results; I did not rerun models, derivatives, fits, eigenanalysis, statistics or original producer code. New saved values below were read from the completed, hash-bound copies supplied by Inference. Their association with the frozen producer log is preserved; this review does not retrospectively observe their historical execution or manufacture process/source authority. The separate bibliography ledger covers all 36 new keys assigned to this lane and the material older updates.

The factorization, compact-hull high-temperature argument, corrected low-temperature qualification and most of the rewritten scalar-return discussion now have an appropriate finite scope. The main remaining issues are overgeneralizations from local or truncated observations, two concrete numerical/text inconsistencies, and a few source-record or citation-field gaps. “Must” below denotes a false or unsupported statement as written; “should” denotes a material precision or chronology qualification. None requires a new experiment to make the current paper accurate.

## Required corrections

### T01 — Local reduction is not a global scalar reduction, and genericity is not a guarantee

**Must; lines 484–502, especially 488–496.** The local centre-manifold statement near an appropriately classified bifurcation is sound. Small transverse Floquet multipliers of selected periodic orbits do not establish a global one-dimensional invariant manifold, injectivity of a scalar observable, or a scalar reduction of an entire chaotic attractor. They are neither a general necessary condition nor a sufficient global criterion. Likewise a generic two-parameter smooth family with a quadratic extremum need not exhibit every cusp, infinite doubling cascade and saddle-node accumulation listed here. These are conditional local/universality structures, not consequences of “generic two-parameter” alone. A fold curve turning in one plotted coordinate is not by itself a cusp.

Suggested replacement for 488–496: “A scalar description of a whole attractor requires additional global evidence. The small noncritical multipliers of the periodic orbits we computed show strong transverse contraction near those orbits and motivate that description. One-dimensional families can exhibit fold and flip curves, nondegenerate cusps, period-doubling cascades in the quadratic universality class and saddle-node passage scaling. We test the relevant predictions on the specific branches and parameter sequences below.” Keep the actual multiplier sequence as evidence for local contraction. Task 19's observable/conjugacy cautions at 2235–2257 should apply here too.

### T02 — Singer's endpoint alternative has been dropped from the conclusion

**Must; lines 504–514.** The stated theorem permits the immediate attracting basin to contain an interval endpoint. An attracting orbit can lie in the interior while its basin contains an endpoint. Therefore the preceding theorem does not imply “at most one attracting periodic orbit in the interior.” Coexistence alone does not force a second turning point, positive Schwarzian somewhere, or an additional dimension until the endpoint-basin alternatives have been excluded. The negative-Schwarzian and observable/invariant-interval hypotheses are also not independently established for the fitted relations here. [Singer's primary record](https://epubs.siam.org/doi/10.1137/0135020).

Suggested replacement: “For a C³ unimodal interval map with negative Schwarzian derivative, every attracting periodic orbit has an immediate basin containing the turning point or an interval endpoint. If endpoint-supported attracting basins are excluded, at most one attracting periodic orbit remains. The coexistence we observe therefore constrains a scalar interpretation, but does not by itself rule out every unimodal negative-Schwarzian representation.” The stronger alternatives may be stated conditionally, not as an inference already made from coexistence.

### T03 — State the finite KY diagnostic and its actual cohort

**Must scope correction; lines 516–529.** The formula and the condition `λ1+λ2<0` for its one-positive simplification are correct. Task 21b checked all 35 completed rows and 84 checkpoints, retaining KY1's 18/20 and TC's 7/8 rather than treating missing records as observations. Of the 35, one is the previously designated TC periodic control 143; the six TC chaotic members were selected in the earlier H2/TC experiment, not newly selected for KY1/KY2. Native QR column order was not sorted in the producer output; the independent calculation explicitly retained it and sorted for KY sums. Four finite-time estimates over 2000 steps are not an asymptotic dimension proof or a proof excluding hyperchaos at longer horizons.

Suggested replacement for 520–528: “We estimated four leading Lyapunov exponents over 2000 measured steps at 35 completed cells from the KY1/KY2 and prior TC cohorts, including one TC periodic control. At this horizon, 31 cells have a positive leading estimate and none has two positive estimates. The resulting finite-time KY estimates range from 1.008–1.157 for the eight hostile cells, 1.038–1.646 for the fourteen headline cells, and 1.373–2.016 for the nine high-temperature cells. The high-temperature cohort has larger estimates; this descriptive comparison is not a global dimension theorem.” Two nonnegative first-two sums do not imply two positive exponents.

The existing coarse ranges are generally outward except that “one-dimensional to within a sixth” and “the attractor is effectively two-dimensional” should be framed as interpretations of these estimates, not established geometry. Task 21b scientific report SHA256 `491058e5c2222858522bc74db750d1263d0461d87d7f6fd719f52e4131ce8e7f` is the accepted arithmetic authority.

### T04 — The −0.152 search does not establish a bounded doubling bubble

**Must; lines 1258–1259.** Accepted Task 21a retained the half-period collapse and failed forward search. These observations support a bubble-compatible interpretation and “no further flip found with this search”; they do not locate the reverse flip or exclude a narrower cascade. This is the original finite-resolution limitation, not a request to redo the search.

Suggested replacement: “At −0.152 the continued period-12 solution collapsed onto its period-6 parent near α≈0.0657, and the attempted forward search found no next flip. This is compatible with a doubling bubble, but we did not locate its reverse boundary or exclude a narrower cascade on that cut.” The failed crossover prediction remains failed and should stay in the text.

### T05 — Flip 9 contradicts the caption's multiplier tolerance

**Must; line 1340.** The new completed saved period-768 record has critical multiplier `−0.9999995928069132` and Newton residual `8.919160074882895e−12`; its distance from −1 is approximately `4.07e−7`, not at most `5e−8`. Main text line 1264 already acknowledges about `4e−7`. The exact saved record is `saved-evidence-001/pod64/cascade/crescent-t-0.156-f/cascade.json`, SHA256 prefix `edf0b37d`, bound in compact manifest `96c710c0`.

Suggested replacement: “For flips 1–8 the stated 5×10⁻⁸ bound applies; flip 9 has |μ+1|≈4.07×10⁻⁷ (below 4.1×10⁻⁷).” Retain the computed real classification and distinct precision-floor treatment. Do not silently enlarge every earlier tolerance or call best-|g| acceptance a scatter test: source review established that `--g-accept` checks the best residual only.

### T06 — Some longer-memory runs remain on one-token cycles

**Must; lines 1398–1400.** “For every K≥2 the loops leave the one-token cycles” is contradicted by the saved summary. At K=2, seven unsteered runs and one steered run still end on a one-token-map cycle; the unsteered counts remain 6, 5, 5, 5 and 6 at K=4,8,16,32,64. These are a different field from “loop equals full-context loop,” so Table 13 does not make the universal sentence true.

Suggested replacement: “For the tested K≥2, most final loops differ from the one-token cycles: at K=2 only 7/220 unsteered runs and 1/220 steered runs retain one of them.” Keep both comparison definitions distinct. All eight graph/decode/kdecode compact summaries were read; no graph or token sequence was rerun.

### T07 — Leading Ritz values do not establish that the derivative has no zero eigenvalue

**Must; lines 2143–2154, especially 2147–2150.** A finite collection of leading Ritz values cannot exclude a zero eigenvalue in the uncomputed part of an 896-dimensional spectrum. The observed exchange of leading values and complex pair is useful local evidence; “the map's derivative has no zero eigenvalue” is stronger and unsupported by that evidence. A finite grid's absence of an intersection also does not prove global nonintersection.

Suggested replacement: “On this sampled approach the two computed leading multipliers exchange dominance and form a complex pair in 221 cells. The leading-real-multiplier sign no longer defines a scalar superstable curve there. We did not find an intersection of the two tracked arcs at this resolution.” Describe the pair as a two-direction local linearization, not proof that the whole orbit/attractor is globally two-dimensional. Scope “no one-dimensional shrimp centre” and the paragraph's conclusion to the tracked branch and sampled region. No full-spectrum replay is required to remove the unsupported assertion.

### T08 — A generic periodic-window boundary need not contain only folds and flips

**Must; lines 2156–2180.** In a high-dimensional map, attracting periodic windows may also have Neimark–Sacker and other boundary phenomena. A flip need not initiate a cascade to chaos; the paper itself reports interrupted doublings. A plotted fold curve turning back in a coordinate need not be a cusp without the relevant degeneracy and unfolding conditions. “Two independent folds in a two-dimensional [reduction]” is not a general replacement for the one-dimensional double-critical-point condition; for example, an invertible two-dimensional map cannot acquire an exactly zero determinant merely from that analogy.

Suggested replacement: “The continued windows studied here contain classified fold and flip boundaries. Some of their flip branches begin the finite doubling sequences shown above. The located cusp points satisfy the additional degeneracy and unfolding checks in Section …; a curve's visual turn alone is not our criterion.” Restrict the shrimp-centre analogy to the cited map classes and observed branch. Root's separate review covers the newly located nine cusps; this note does not recycle Task 17's superseded statement that only bracket endpoints were available.

## Material qualifications and numerical presentation

### T09 — Make the superstable uncertainty and later P-F2 readout chronology explicit

**Should; lines 1229–1250, 1264–1274, 1281–1288.** The early superstable roots are interpolated estimates between clean opposite-sign endpoint records; `root.record` is a clean endpoint, not an evaluation at the estimated root. The coupled-corner ratio ranges are arithmetic sensitivity envelopes of numerical brackets, not rigorous model enclosures. The original 06:05 P-F2 readout used completed p192 and the pair 4.572/4.643: its point estimates met the two-sided 5% rule, while the earlier ratio's wide bracket envelope did not confirm that rule throughout. The new p384 completion gives a later pair 4.643/4.6646. State this as a later extension, preserving the original readout rather than retroactively substituting its final band.

Suggested addition: “At the original p192 readout the point estimates met the frozen two-sided 5% criterion, but the earlier bracket envelope did not. Extending the computation to p384 gives a later pair of point estimates within that interval, with the numerical sensitivity ranges shown here.” The frozen interval was `[4.43555,4.90245]` using 4.669, not a one-sided lower bound and not a substituted flip ratio.

Use outward rounding when presenting an actual bound: the saved −0.14798 p12 endpoints are `0.06596669617192479` and `0.06597903913953078`, so `1.2e−5` rounds their width inward. The p48 endpoints `0.06611413278446542`/`0.0661141327844687` similarly exceed `3e−15`; p192 endpoints `0.06612447851007368`/`0.06612447851008002` exceed `6e−15`. Label widths approximate or use safe outward values such as 1.24e−5, 3.3e−15 and 6.4e−15. The stated ratio lower endpoint 4.02 cannot be a containing lower bound for the saved point ratio `4.019369236290009`; use the full saved coupled-corner result rounded outward. This review does not substitute a fresh ratio recomputation.

Line 1264 should say “the penultimate flip, of the period-384 orbit.” At 1266, “largest residual divided by slope” is a conditional scatter-based location estimate under the local linear fit, not a certified error bound. At 1288, say “the 768→1536 flip,” since the located return map has period 768; a flip of a period-1536 orbit was not completed.

### T10 — Rename the readout rank quantity and retain its numerical estimation scope

**Must terminology / should evidence scope; lines 530–535 (cross-reference Boundary's caption finding at 443–444).** `trΣ/λmax(Σ)` is not conventional matrix stable rank `||Σ||F²/||Σ||2²`. Call it the trace-to-leading-eigenvalue ratio, or explicitly define a distinct effective-rank statistic without reusing the conventional name. The saved gain records estimate the leading eigenvalue by power/Rayleigh iteration. The new KY-phil/KY-host records use `iters=8`; do not import the older map's four-iteration setting into these rows. Some saved Rayleigh changes are appreciable, so do not describe these values as exact ranks or exact eigenvalues.

The 20 philosophy and 10 hostile gain component records have now been read under supplement SHA256 `74bdd444a83747f1fdb481a9b6870ec0dacd99f3ad553ec99addbde11127edcb`. The producer log identifies the .72/.69 correlations as an inline analysis; no saved join/correlation-summary implementation or output was found. I have not independently recalculated those correlations or the 17/8 joined subset. Retain them as producer-reported descriptive values, or attach the original finite calculation if available. This is a coverage limit, not evidence that the values are wrong. The hostile component entropy/rank rows themselves include values slightly outside rounded [7,9]/[16,19], so range prose should be marked approximate or use actual joined-subset extrema; the subset cannot be reconstructed by silently replacing the KY classifications with the different gain-run classifications.

### T11 — Scope the rotation, memory and growing-context interpretations to the interventions actually made

**Should; lines 373–375, 1390–1394, 1441–1445, 1475–1486.** Sixteen contracting Haar-rotated products support robustness to those sixteen interventions, not general independence from alignment. Say “all sixteen tested rotated products also contracted.” The finite greedy graph was computed with float32 logits; the saved graph records near-tie tokens. Describe it as the computed greedy graph at the stated precision, distinct from an exact-real low-temperature limit near ties.

The ten longer-window summaries contain 240 cells, with both offsets and the period-first classification retained. Their reported counts match the table as saved evidence. They support chaos-like finite diagnostics at several tested K, not all window sizes. The growing-context paragraph correctly calls its growth rates finite and non-autonomous. Keep that distinction in its conclusion: “At these 72B prompts/cells and over 400 steps, the growing-context decoder did not reproduce the sustained sensitivity indicated by the fixed-window tests.” This is preferable to the general causal claim that context “damps even the chaos that steering produces.” The two-of-three threshold was frozen before 72B but applied retrospectively to the earlier 0.5B/7B records; disclose that timing. The held-out 72B file reports `complete:false` with 24 completed coarse cells; the twelve promised finer-line cells are absent and remain excluded. This does not invalidate the stated 24-cell result.

### T12 — The two flip-normal-form tables are consistent with the saved rows; avoid claiming required monotonic errors

**Should; lines 1343–1377.** Both complete FNF2b/FNF4 records, all ten offsets, coefficient ladders and crossing classifications were read from supplement 74bdd444. The listed values match their saved rounding: `c=6492.447473797511` and `12386.105013755889`, real −1 crossings, next moduli .3831891589/.2783078884, smallest amplitude ratios 1.00501306/1.00292049. The p4 largest-offset return multiplier is the complex pair `.1253160765 ± .2066999964 i`, not the scalar near-zero normal-form prediction; the table appropriately says complex. Intermediate omitted offsets remain in the source evidence.

The formulas are leading-order normal-form predictions. Replace “both discrepancies grow as the next order … requires” with “their increasing discrepancies are consistent with higher-order corrections.” A formal remainder expansion does not generally require monotonic observed error at each finite offset. The compact normal-form records do not independently locate the claimed reverse boundary near .0705; the bubble claim there remains supported at producer log/source level rather than a new independently checked reverse-flip receipt in this lane.

### T13 — The rewritten hidden-chaos and return-map claims mostly preserve the accepted limitations

**Accept with scope ledger; lines 2218–2325.** The current wording preserves sampled-cloud separation, the 2000-step text horizon, conditional smooth invertibility, the original 49/62 outcome and outcome-informed 51/62 amendment, the failed delay-sufficiency criterion, and both independent stress cells. These are substantial corrections from Task 17 and should remain. Accepted Task 19 provides the exact phase/fit/finite-record limitations; the product of two independent logistic maps remains a valid counterexample to inferring one-dimensional conjugacy from perfect scalar exponent agreement.

Do not relabel the cross-cell reproduction bit-exact or all fits full-rank: the accepted evidence retained 2070/4140 finite nonzero producer comparisons (maximum about 2.47e−8), 60 fold rank warnings, and excluded rank-deficient selected phases in cell 48 from qualified aggregates. The new PC return/cobweb illustrations are distinct producer analyses, not the independently audited probability-map fits. No new scalar fit or audit was performed here. The concise main-text rounded primary/04/13 discrepancies are consistent with the accepted final Task 19 report.

### T14 — Broaden the description of prior looped-reasoning work instead of narrowing its scope inaccurately

**Should; lines 234–236.** The cited Park–Rusch work explicitly discusses nonstationary cycles/drift and finite-time quotient dynamics, and Viakhirev et al. study finite-depth regimes; “that analysis concerns fixed points and their spectra” is too narrow after listing those sources. [Park–Rusch primary preprint](https://arxiv.org/abs/2610.18174), [Viakhirev et al. primary preprint](https://arxiv.org/abs/2608.18222).

Suggested replacement: “These studies examine fixed-point stability, finite-depth dynamics and nonstationary regimes. Here we continue full-state periodic branches and classify their bifurcations in specified frozen-model feedback maps.” The paper can state its concrete contribution without asserting an exhaustive negative priority claim. Also retain the distinction between the constructed repulsive-attention model and trained networks at 236–237; that distinction matches the cited primary source.

## Coverage ledger

| Assigned lines | Directly reviewed | Accepted reuse | Explicit remaining limit |
|---|---|---|---|
| 297–378 | Chain rule, tied/untied covariance, PSD energy descent, high/low-T assumptions, all prose | Task 18 signed phase-dependent products and factorization; Task 17 analytic scope | New unsteered sweep counts/entropy/slope and 16 rotation outcomes were not independently re-tabulated in this lane; finite producer claims only |
| 476–515 | Local/global distinction, generic-family wording, Singer logic | Task 17 theoretical source reviews; Task 19 nonconjugacy limits | No global scalar invariant manifold or interval endpoint-basin exclusion established |
| 516–538 | KY formula/cohort interpretation, rank definition; all 30 new gain component rows read | All 35 KY rows/84 checkpoints from accepted Task 21b | Inline .72/.69 correlation join/result not available as a saved independent calculation |
| 1208–1342 | All prose/table rows; completed new −.14798/−.156/1.5B compact records, failures and endpoint semantics | Task 21a full saved arithmetic/source/refusal review | New deeper ratios read as saved producer outputs, not independently recomputed; numeric brackets/scatter estimates are not rigorous interval enclosures |
| 1343–1378 | Both FNF JSONs, all ten offsets and coefficient ladders; formula and table comparison | Existing classification/finite derivative conventions | Reverse boundary near .0705 not independently authenticated in this lane; no model or normal-form rerun |
| 1380–1437 | All prose, eight completed graph/decode/kdecode summaries | Earlier sampled-laminar bridge source/log discussion | No new sampling experiment or full graph computation; near-tie precision matters |
| 1438–1488 | All ten WIN summaries (240 cells) and five ST1/ST72 completed summary records, all three-direction crossing/growth/final-distance fields | Fixed-map versus growing-context scope | No replay of every saved time step; unfinished 72B finer line excluded; no asymptotic chaos theorem |
| 2141–2180 | Full conceptual and finite-spectrum scope | Task 17/19 analytic cautions | New detailed superstable grid/PC-return/cobweb arrays not independently audited; leading Ritz spectrum is incomplete |
| 2181–2217 | Every caption and stated interpretation | Existing branch/source reviews where applicable | Plot pixels/layout and full underlying new grid arrays not audited in this lane |
| 2218–2325 | Every paragraph and caption | Accepted Task 19 full primary, 65-cell amendment and fixed stress04/13 evidence | New PC illustrations remain producer-only; no repeated fits or audits |
| Relevant citation/novelty passages | 36 new keys and 14 material old updates, with explicit per-field coverage in companion ledger | Task 13 verified identities/publication candidates; Task 17 source-scope checks | Primary pages absent for some new pagination/venue fields; listed individually, not declared verified |

New completed-original associations use `/tmp/task27-source-001/compact-evidence-manifest.json` (SHA prefix `96c710c0`) and the supplemental exact manifest `74bdd444a83747f1fdb481a9b6870ec0dacd99f3ad553ec99addbde11127edcb`. Pending FBC148d, FBS768n and the finer ST72 line were not read. The reused authorities and exact paths are indexed in `/tmp/task27-reuse-inventory.md`, SHA256 `683718ad95e9ea443f673772e099ccf0b86ab40e0a2e9185f480ac0a6b3892b1`.

Root owns the remaining paper and synthesis; Boundary owns practical/mechanism overlap, including the shared rank-terminology finding. This review does not modify the paper, reopen completed computations or require new proof infrastructure.
