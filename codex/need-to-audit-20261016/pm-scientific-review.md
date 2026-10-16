# PM / PM2: independent saved-row audit, 16 October 2026

The frozen published PM and PM2 summaries are reproduced exactly: all 187 PM and 313 PM2 comparison leaves agree, with no nonzero numerical discrepancy. Both prespecified primary criteria pass on their original stored-answer endpoints. This is independent, unblinded saved-row arithmetic and subsequent review, not a new model experiment. The important corrections are the headline flip-range endpoint, explicit accuracy denominators, outward interval displays, and the interpretation of nonsignificant contrasts. R1-Llama answer strings also expose a small, concrete byte-marker sensitivity that must remain separate from literal reproduction.

The sole admitted audit used independent source `5be03229167e9c1a27009b6e1b588c297df027cb`, ran all 12 manufactured controls once, then the actual records once. The complete saved review covers 19 arms, 8,192 original records, 4,096 within-arm pairs, every cluster and contrast-pairing diagnostic, all six PM2 cells, all published comparisons, and the five complete original orchestration logs. No second scorer, model, bootstrap or statistical replay was run for this final assessment.

## Evidence and scope

The immutable audit is [report.json](/data2/neural-shrimp-data/codex/need-to-audit-20261016/run-001/pm/report.json), SHA256 `6cf7e88ee641bf6240e738745ccfba2bb67fdec4562b5143413df49a4877bff2`. Its [manifest](/data2/neural-shrimp-data/codex/need-to-audit-20261016/run-001/pm/manifest.json) is `a9dbd667025c9c494b2b935327262635c11e1ec5791f366fa3e8b836eb8b93b4`; the original [supervisor receipt](/data2/neural-shrimp-data/codex/need-to-audit-20261016/run-001/pm-supervision/receipt.json) is `c44e01a98abdf715d5f0c70301b65375270fc1e164dd36c4a03679f7624e1858`. The presentation authenticated all seven output files against that receipt and all six manifest members. Full literal selections, every comparison and every row exception are preserved in [saved-review.json](/tmp/need-to-audit-20261016-pm-final/saved-review.json), SHA256 `faa662214aa462f8ab25876fbdf5d59cbc0c5709963ef5f5a53f466ee7704fe4`; bulky original row diagnostics remain unchanged.

The source/publication freeze is `2b2172946cb303b2d4462084df75adee027d11e0`, manifest `3fcbde19749f1cfe53e5309283e0b7a551dff616354e85502ff094abf9c82963`. Line references below refer to these exact copies:

- [Paper](/tmp/need-to-audit-20261016-source-001/paper/llm/main.tex), SHA `bd584258cf739bbb92ae6f9d38ec7c11635e75c528e97798377cd9547206cdfb`.
- [Chaos manuscript](/tmp/need-to-audit-20261016-source-001/local/chaos-main.tex), SHA `a9e4606c75c010649a04a75d35d92dc2e88848f9d874c4616dd44c9f769c2df5`.
- [Blog](/tmp/need-to-audit-20261016-source-001/local/blog/index.md), SHA `cb6c25ac3380b763c99234822ef9ce32d3e7fbab3fb775c0d6e44c2effd6af17`.

Chaos and blog are exact local snapshots, not asserted historical Git sources. Only their PM/PM2 passages are covered here. Current and historical analyzers were read as source, never imported or executed by this audit. Historical PM source is `0f24b34496201e23fbb2decab3a6e802063ff787`; PM2 is `3639819a126169b1679ac135ace0af17dd778388`.

## PM: all arms and contrasts

Each arm contains the same 256 problems, 77 batch clusters and layouts 8/16. Rates and accuracies below are fractions, not percentages; these decimal values are copied from the saved audit.

| Arm | Answer flips | Flip rate | Accuracy, layouts 8 / 16 |
|---|---:|---:|---|
| top15-s64, primary | 54 | .2109375 | .24609375 / .22265625 |
| replay | 15 | .05859375 | .23828125 / .22265625 |
| greedy | 6 | .0234375 | .890625 / .88671875 |
| top1-s64 | 45 | .17578125 | .27734375 / .25390625 |
| full-s64 | 57 | .22265625 | .24609375 / .2265625 |
| top15-s16 | 17 | .06640625 | .29296875 / .296875 |
| full-s16 | 19 | .07421875 | .29296875 / .2890625 |

The primary clustered interval is `[.16260162601626016, .2607003891050584]`. Both the point-rate `>.10` and lower-limit `>.05` criteria pass. Primary exactly-one-correct rate is `.0625`. Across the non-greedy pipeline arms, actual accuracy extrema are **22.265625–29.6875%**, so the paper's enclosing integer range 22–30% is appropriate.

| Primary minus comparison | Difference | Exact clustered 95% interval | Discordant counts | Exact McNemar p |
|---|---:|---|---|---:|
| replay | .15234375 | [.1044776119402985, .2018348623853211] | 44 / 5 | 7.59715845788378e-9 |
| greedy | .1875 | [.13716814159292035, .2362888377445339] | 52 / 4 | 1.1008055578187737e-11 |
| top1-s64 | .03515625 | [-.029411764705882353, .10345171039978014] | 41 / 32 | .3491818381705883 |
| full-s64 | -.01171875 | [-.05303030303030303, .030534351145038167] | 17 / 20 | .7428293587290682 |
| top15-s16 | .14453125 | [.10038574155674526, .19026631224408927] | 46 / 9 | 4.336383137015787e-7 |
| full-s16 | .13671875 | [.09057936833470058, .18439716312056736] | 46 / 11 | 3.3118847623803127e-6 |

The four prespecified secondary tests are replay, top15-s16, greedy and top1-s64. The first three survive the saved four-test Bonferroni diagnostic; top1 does not. The top1 contrast is **not an equivalence result**: its interval includes approximately −2.94 to +10.35 percentage points. The full-vocabulary comparisons remain descriptive, not retroactively confirmatory.

## PM2: all primary cells and the separate A100 replication

A is R1-Distill-Qwen-7B/H100; B is R1-Distill-Llama-8B/H100; C is QwQ-32B/H200; D is the separate R1-Distill-Qwen-7B/A100 replication. GSM uses 256 problems/16 clusters/layouts 8 and 16; MATH uses 64 problems/64 clusters/layouts 1 and 8. Each entry below retains soft and plain separately.

| Cell | Soft flips / n | Plain flips / n | Soft accuracy, smaller / larger layout | Plain accuracy, smaller / larger layout |
|---|---:|---:|---|---|
| A-GSM | 37/256 (.14453125) | 26/256 (.1015625) | .8984375 / .8828125 | .91015625 / .91796875 |
| A-MATH | 39/64 (.609375) | 34/64 (.53125) | .4375 / .484375 | .421875 / .421875 |
| B-GSM | 96/256 (.375) | 103/256 (.40234375) | .52734375 / .54296875 | .50390625 / .4921875 |
| B-MATH | 43/64 (.671875) | 40/64 (.625) | .359375 / .390625 | .421875 / .375 |
| C-GSM | 50/256 (.1953125) | 51/256 (.19921875) | .85546875 / .8671875 | .84375 / .828125 |
| D-GSM, replication | 30/256 (.1171875) | 37/256 (.14453125) | .91015625 / .89453125 | .90625 / .890625 |

The five primary cells, excluding D, pool to **265/896 = .29575892857142855**, clustered interval `[.2734375, .31808035714285715]`. The point-rate `>.05` and lower-limit `>.02` criteria both pass. This is a pooled problem-weighted endpoint, not an equal-cell average. A and D were read from separate preserved roots; the extraction collision noted by the original runner did not overwrite A in this audit.

| Cell | Soft clustered interval | Plain clustered interval | Soft minus plain flips; interval | Flip McNemar p |
|---|---|---|---|---:|
| A-GSM | [.109375,.1796875] | [.07421875,.12890625] | .04296875; [0,.0859375] | .0895310789346695 |
| A-MATH | [.5,.71875] | [.40625,.65625] | .078125; [-.046875,.21875] | .359283447265625 |
| B-GSM | [.328125,.421875] | [.33203125,.4765625] | -.02734375; [-.09375,.03125] | .546713483598813 |
| B-MATH | [.5625,.78125] | [.5,.75] | .046875; [-.09375,.1875] | .6636238098144531 |
| C-GSM | [.1640625,.22265625] | [.16796875,.23046875] | -.00390625; [-.04296875,.0390625] | 1 |
| D-GSM | [.078125,.16015625] | [.1171875,.171875] | -.02734375; [-.0625,.0078125] | .3367836351899314 |

All six soft-minus-plain **flip** McNemar tests are nonsignificant at .05. This is compatible with both modes showing batch dependence; it does not establish equivalence or identify the sole cause. The separate larger-layout accuracy contrasts are retained, not merged with the flip endpoint:

| Cell | Soft minus plain accuracy | Clustered interval | McNemar p |
|---|---:|---|---:|
| A-GSM | -.03515625 | [-.06640625,-.00390625] | .0931396484375 |
| A-MATH | .0625 | [-.046875,.171875] | .3876953125 |
| B-GSM | .05078125 | [.0078125,.09375] | .11747735024505848 |
| B-MATH | .015625 | [-.125,.15625] | 1 |
| C-GSM | .0390625 | [0,.078125] | .12144948495551944 |
| D-GSM | .00390625 | [-.04296875,.046875] | 1 |

Some clustered accuracy intervals exclude zero although the exact McNemar tests do not. Those are different procedures/endpoints; an unqualified assertion that every difference is nonsignificant would erase this distinction.

A minus D soft flip difference is `.02734375`, interval `[-.015625,.06640625]`, discordant 19/12, p `.28104151505976915`. The plain difference is `-.04296875`, interval `[-.06640625,-.01953125]`, discordant 13/24, p `.09887174959294498`. These are finite paired cross-GPU observations, not proof of GPU invariance.

Exactly-one-correct soft fractions, in cell order A-GSM/A-MATH/B-GSM/B-MATH/C-GSM/D-GSM, are `.1171875, .265625, .1328125, .25, .14453125, .0859375`. The saved forced-switch counts refer to the **larger layout**, not all rows pooled: soft counts `22,32,39,45,32,22`; plain `43,33,33,35,35,44`. Thus MATH soft has 32/64 and 45/64 capped larger-layout rows, not uniformly one half. Missing stored answers are retained: one in B-GSM soft/layout8, one in B-MATH plain/layout8, one in C-GSM soft/layout8; none elsewhere. Cold Stop is absent from the reviewed original protocol.

## Row integrity, suffix limits and R1-Llama markers

Across all 19 arms there are no duplicate or unpaired records, negative indices, unexpected layouts, stored correctness-versus-stored-answer/gold disagreements, paired gold/prompt-length mismatches, or cluster row-position/prompt-length failures. All 20 contrast pairing diagnostics also have no unmatched IDs, gold mismatch or cluster mismatch. The initial stored-label result is therefore internally consistent and exactly reproduced.

All PM suffix re-extractions agree, including 465 greedy records whose saved 400-character suffix may be truncated. PM2 has **nine** raw-suffix extraction discrepancies; all nine saved texts are exactly 600 characters. They occur at C-GSM soft index982/layout16; B-GSM soft indices529,939,483,1031/layout8; B-MATH plain index62/layout1; B-MATH soft index19/layout8; D-GSM plain index1055/layout8; A-MATH plain index68/layout1. These saved suffixes cannot establish what the original full-text extractor saw. For example, A-MATH plain index68 has stored answers 46/46, while suffix-only re-extraction yields 1/46 and changes the diagnostic flip count from 34 to 35. This is a limitation of suffix reconstruction, not evidence to overwrite the original endpoint.

All 1,280 R1-Llama records retain byte-level marker characters in the saved representation. The separate diagnostic replaces `Ġ` with space, `Ċ` with newline and `ĉ` with tab before the same numeric extraction; it is not a full tokenizer decode. For B-GSM plain, ten answer strings change and three recover the stored gold number: index377/layout8 (`6Ġ`→`6`), index422/layout16 (`2Ġ`→`2`), and index957/layout16 (`240Ġ`→`240`). Index377 is only 566 saved characters, so at least this mismatch is not explained by the 600-character truncation ceiling. The recorded alternatives are **103→101 flips**, and correct counts **129/126→130/128** for layouts8/16. The two disappearing flips are indices377 and422. Other changed strings remain incorrect. These are disclosed extraction sensitivities, not silently substituted primary results or freshly bootstrapped alternatives.

B-GSM soft also retains two marker-only string changes plus four truncated-suffix discrepancies; its 96 flips remain unchanged. Its normalized-suffix correct-count changes mix truncation with marker handling and must not be attributed solely to marker correction. MATH marker/suffix diagnostics do not change the stored 40 plain or 43 soft B-MATH flips. All nine suffix exceptions, every marker change, stored/alternative correctness, and saved top1-prefix diagnostics remain in the lossless presentation. The top1 first-difference field is explicitly a one-based saved-readout clock; it is not a native-state distance or necessarily the answer-generation phase.

## Frozen design chronology and original failed attempts

Recorded UTC commit times are PM design `b3e0556d` at 11:28:18, PM hardware amendment `0f24b344` at 16:33:49, and PM2 design/source `3639819a` at 17:44:57 on 16 October. The original orchestration logs give:

| Completed original job | Recorded pod start UTC | Device | Seconds after applicable design/amendment | Runner exit |
|---|---|---|---:|---:|
| PM | 16:34:23.526233 | H100 NVL | 34.52623271942139 after hardware amendment | 0 |
| PM2-A | 17:46:33.684096 | H100 NVL | 96.68409609794617 | 0 |
| PM2-B | 18:20:57.572582 | H100 NVL | 2160.572582 | 0 |
| PM2-C | 19:04:31.285042 | H200 | 4774.2850416 | 0 |
| PM2-D | 17:46:41.292270 | A100-SXM4-80GB | 104.29226970672607 | 0 |

The initial PM design precedes PM's recorded start by 18,365.52623271942 seconds. Every recorded start in the retained attempt logs follows the corresponding design commit. Original archive hashes and recorded termination checks are preserved; these are historical metadata observations, not a fresh remote inspection or cryptographic attestation of all loaded weights/clock history.

PM2 A/B/C/D each first encountered a pre-launch cost-ceiling refusal; runtime budgets were reduced from 215→210 minutes for A/B, 205→200 for C and 135→126 for D. B additionally retains 13 short price-rejected pod attempts and C 30, alongside capacity warnings. These attempts have termination records and no successful runner completion; they must not be described as 13 or 30 repeated scientific experiments. The complete logs, including failures, remain in chronology.json. No failed attempt is removed to manufacture a clean first-attempt narrative.

The source-frozen designs predate the recorded successful experiments. This supports the stated confirmatory chronology on the preserved evidence. It does not independently prove the global absence of earlier scoring or all possible undisclosed experiments. The original analyzer CLI was not located: sequential bootstrap RNG uses the disclosed published-summary insertion order, seed20261016, 10,000 draws per bootstrap, without an order search. All 500 leaves matching under this proxy is strong arithmetic compatibility, not retroactive CLI attestation. Both PM2 GSM cluster order and MATH single-problem clusters are retained as defined by source; no new clustering scheme or multiple-testing redesign was substituted.

## Exact publication findings and replacements

1. **Correct the headline lower flip endpoint and state the accuracy scope.** Paper44–47 and141–143; Chaos49–51; blog10,23,81–82,220–223,1430–1433. A-GSM soft is **37/256=14.453125%**, C-GSM soft **50/256=19.53125%**. “15 to20%” is neither the true range nor ordinary integer rounding of the lower endpoint. Suggested: “changed **14.5% and19.5%** of answers, respectively.” These are explicitly approximate point estimates. For a bounded one-decimal range use `[14.4,19.6]%`.

   The A/C soft accuracy extrema are **85.546875–89.84375%**; “86–90%” is conventional nearest-integer approximation, not an enclosing bound. Use “approximately85.5–89.8%” or an enclosing one-decimal range **85.5–89.9%**. Paper141–143's “83–92%” instead spans **both soft and plain modes**, actual **82.8125–91.796875%**; explicitly identify that scope (enclosing one-decimal82.8–91.8%). Including D changes the soft maximum to **91.015625%** and adds the distinct **11.71875%** soft flip rate; do not silently mix that replication into the A/C headline.

2. **Replace equivalence-like wording.** Paper874–878, blog1321–1325: replace “clear null” with “the test did not resolve a difference; it was not an equivalence test.” The top1 contrast interval permits a material positive difference. Paper915–917 and blog1415–1418: “Both continuous and discrete sampled thinking show batch-dependent answers; none of the six flip-rate McNemar contrasts is significant at .05.” Avoid presenting this alone as proof of identical rates or an exclusive causal account. The detailed finite pipeline mechanism evidence belongs to separate experiments, not this saved-answer audit.

3. **Use outward CI displays when brackets are claimed as bounds.** Current source intervals are arithmetically reproduced; the issue is display, not a failed statistic. At one decimal percentage/percentage-point precision, use:

| Passage | Exact interval as fractions | Enclosing percent/pp display |
|---|---|---|
| PM primary, paper864–866 / blog1306–1307 | [.16260162601626016,.2607003891050584] | [16.2,26.1] |
| PM16-step contrast, paper870–871 / blog1317 | [.10038574155674526,.19026631224408927] | [10.0,19.1] |
| PM greedy contrast, paper872–873 / blog1319 | [.13716814159292035,.2362888377445339] | [13.7,23.7] |
| PM top1 contrast, paper874–875 / blog1321 | [-.029411764705882353,.10345171039978014] | [-3.0,10.4] |
| PM2 pooled, paper909–910 / blog1407 | [.2734375,.31808035714285715] | [27.3,31.9] |
| A minus D soft, paper918–919 / blog1420 | [-.015625,.06640625] | [-1.6,6.7] |

   Ordinary nearest-rounding is acceptable when clearly presented as approximate; it must not silently replace an enclosing bound under this project's outward-rounding rule. The replay display `[10.4,20.2]`, A-soft `[10.9,18.0]` and C-soft `[16.4,22.3]` already enclose their saved endpoints.

4. **Disclose extraction sensitivity.** The PM2 summary reproduces literal stored answer strings, including R1-Llama byte markers. Add a method/limitation sentence pointing to the 103→101 B-GSM plain diagnostic and suffix-only limitations. Do not “correct” original frozen counts without a disclosed amendment and a specified extraction endpoint. Paper924–925's source disclosure can be extended; blog PM2 method/detail passages likewise. This issue does not change the current original pooled primary result reported by the audit.

5. **Make cap and accuracy denominators explicit.** Paper920–922 and blog1422–1425: replace “half of the rows reached the thinking cap” with “among the larger-layout soft runs, 32/64 Qwen and45/64 Llama MATH problems reached the cap.” The stored summaries do not establish that exactly half of all layouts/modes did. The Llama GSM **49–54%** range combines soft and plain modes (actual49.21875–54.296875%); soft alone is **52.734375–54.296875%**. Name both-mode scope or report the soft range explicitly.

A compact replacement for the principal claim is: “In the frozen PM test, batch layout changed54/256 stored answers under the published Soft Thinking settings. In PM2, the five primary cells pooled to265/896, meeting the frozen criterion; the Qwen and QwQ GSM cells changed37/256 and50/256 answers, respectively, at soft-mode accuracies spanning85.546875–89.84375%. Discrete sampled thinking also showed batch dependence, with no significant soft-minus-plain flip contrast in these cells. This supports batch sensitivity under the tested protocols, not equivalence of modes or a diagnosis of native chaos.”

## Completion and finite limitations

Inference observed the original audit session1295 reap with exit0, the supervisor257886/start213475255 and worker257996/start213475295 disappear, and the worker group/session close. This review independently authenticated saved receipt/manifest/output bindings; it did not repeat process observation or the original input audit. The supervisor records **5.503576018381864s** external elapsed wall. The worker timer is **5.364549092017114s through input posthash, before final report**, with82,816KiB observed then. These scopes are preserved; no independent whole-worker peak measurement is inferred from the timer. All seven outputs total11,239,999B.

The complete row/cluster/comparison evidence supports faithful reproduction of the stored confirmatory endpoints, while retaining every contrary secondary result, missing answer, suffix discrepancy, marker sensitivity and startup failure. It does not reconstruct discarded full answer text, attest historical loaded-model weights, establish equivalence, prove a unique causal mechanism, or rerun the models. Original summaries and raw files remain immutable. The separate selective-adapter recovery is outside items1–2 and does not alter this completed PM audit.
