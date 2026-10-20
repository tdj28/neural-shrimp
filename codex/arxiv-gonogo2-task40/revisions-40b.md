# Task40b numerical and claim revision check

**GO within this assigned revision scope; one minor caption clarification remains.**
The assigned Task40 must-fix numerical and claim corrections are implemented.
The new stage split is correct: continuous forecast 24/24 then 18/24, with the first-stage discrete primary forecast 23/24.

This is a source-and-saved-evidence comparison, not a fresh numerical audit or a verdict on the separately reviewed build, bibliography, package or public release. Thirteen of the fourteen assigned findings are fully closed; F11 is substantively closed in the body/table, with a small caption alignment recommended below. No new scientific blocker was found.

## Exact scope and identities

- Article before: `bb42a82189da91c37f50804c077832df9e2237ae`.
- Article after: `e300c8eda07361fa5ad7054bf540a3951226c6b4`, read through Git in `/data2/chaos-journal-neural-shrimp`.
- Revised `main.tex` SHA-256: `65fad928b20e812daa973193abba57c64e23e75926ec142902c25a871d2a2824`.
- Adjudication read: `codex/arxiv-gonogo2-task40/adjudication.md`, SHA-256 `8d55415f9d126e48ea5e6351c37558e70498f261f3f5e70568cdf9daf0ed35cb`.
- All article line numbers below are for **e300c8e**, not the full research report or a later edit.

The complete `main.tex`/`references.bib` diff was read. Assigned checks were F1–F7, F9–F14 and F17, plus the new prospective-stage split and breadth/depth sentence. The bibliography change and figure relocation were identified but their primary-source/build verification belongs to Root's separate lanes. Only original Task40 evidence and its already authenticated closed-audit records were reused; the adjudication was treated as a description of the proposed edits, not as numerical proof.

## R1. Align the Fig. 25 caption with the tracked-block description [should-fix, minor]

**Where:** `main.tex`, caption of `fig:delta2d`, lines 1343–1345.

**Claim:** “The two leading directions of the monodromy were followed from cell to cell over 852 continued orbits”.

**Evidence:** `delta-evidence.json`, `/result/selected_pairs`, retains 723 selections of pair `(0,1)`, 63 of `(0,2)` and 66 of `(1,2)`. Thus the selected pair is not the two locally largest-modulus Ritz directions at 129 of the 852 solved cells. The revised body, lines 1437–1439, now correctly says that a two-dimensional block was initialized from leading Ritz directions and tracked by subspace overlap. Table VIII, lines 1688–1689, also correctly limits the trace result to the sampled patch.

**Correct wording:** “A two-dimensional block, initialized from leading Ritz directions and tracked by subspace overlap, was followed over 852 continued orbits.”

This is a small remaining ambiguity, not a numerical contradiction or new scientific blocker: “followed” can be read as referring to the initialization, and the nearby body is explicit. The caption should use the same precise description. The unchanged caption's reference to mixing with a third direction remains appropriate.

## Per-finding disposition

| Finding | Revised location and disposition | Comparison with frozen independent evidence |
|---|---|---|
| F1, natural-passage errors | **Closed.** Sec. V, lines 932–936; Table IV first row, 967–968. | Now states complete phases within `\|u\|<3U` and transverse distance below 0.05, median signed errors, and the single-phase exception. `fold-evidence.json` gives medians −4.878164858%, −4.081632653%, −7.407407407%, −7.692307692% at offsets 3e−6, 1e−5, 3e−5, 1e−4, using `(observed−predicted)/predicted`. These round to the quoted −7.7% to −4.1%. Maximum absolute individual error is 70.588235294%, enclosed by 71%. Mean full-passage fractions are 0.6123335263, 0.6668051516, 0.6106060606, 0.5958677686, matching the descriptive range 0.60–0.67. All 22/29/44/110 complete phases have forecasts. |
| F2, cusp bracket range | **Closed.** Sec. VII, lines 1404–1405. | Names all twelve bracketing points and gives 0.17–39.11. Task40's original-record reduction gives absolute extrema 0.17047709685584778 and 39.1082399960387. The unchanged final coefficient bound is supported by maximum 9.477318099015974e−5, below 9.5e−5. |
| F3, abstract Soft Thinking | **Closed.** Abstract, lines 43–44. | “A smooth full-distribution variant of Soft Thinking” replaces the identification with the published filtered operator. This implements the distinction supported by the frozen primary-source review; no new citation search was performed in this lane. |
| F4, synchronization summary | **Closed.** Introduction, lines 170–173. | Now describes the approximate transverse growth rate and expressly makes locking dependent on starts, duration, tolerance and reached attractor. This matches the checked maximum growth-rate discrepancy 0.0506046891 and the heterogeneous finite-horizon brackets; it no longer claims that observed locking universally occurs near c*. |
| F5, cascade shrink factors | **Closed.** Sec. VI, lines 1112–1113. | After the first ratio, the deviations have one sign. Starting at n=4, the saved reductions are 3.5187907227, 6.5386226332, 3.9263998793 and 5.2590611831. The text correctly quotes 3.52, 6.54, 3.93 and a separately qualified 5.26. The earlier 9.0653424746 reduction is no longer silently included in a 3.5–6.5 claim. The last-location uncertainty remains explicit. |
| F6, fixed-start orbit identity | **Closed.** Sec. VII, lines 1435–1437. | Now says nearest parameters and retains two unresolved cells; it explicitly disclaims a full-state identity check at every continued point. The own saved comparison has 593/595 DITCH2D neighbors labeled period six, two unresolved labels, and 26/26 plus 21/21 line neighbors labeled period six. This is not an identity comparison at the same parameters or an all-start result. |
| F7, location scales and bounds | **Closed.** Sec. V, lines 855, 870 and 1016–1020. | Critical-multiplier discrepancy 3.006483950684924e−13 is enclosed by 3.1e−13; relative b discrepancy 2.2068638969017986e−6 is enclosed by 2.3e−6. Localization is now an estimated scale, with extrapolated P16–20 locations and a uniform-within-cell error model for P22–32, rather than a certified enclosing bound. |
| F9, unsupported “most” | **Closed.** Sec. VIII, lines 1625–1627 and Table VII footnote, 1589. | “Some” and “include pairs” replace the unsupported majority statement. The two saved synchronized full-state period-two examples support existence; no unverified denominator is supplied. The other joint-period claims are unchanged, and Task40's partial full-state classification limits remain. |
| F10, every-start visibility | **Closed.** Sec. VII, lines 1473–1476. | Correctly separates bifurcation curves as properties of the map from the attractors visible in a fixed-start scan. The basin-edge interpretation is tied to the coexistence and saddle evidence, rather than inferred solely from disappearance under another start. |
| F11, tracked block and locality | **Substantively closed; minor caption alignment remains.** Body 1437–1439; Table VIII 1688–1689; caption 1343–1345. | Body/table now name tracking and local numerical scope. Frozen result: 852 solved cells, 38 low-overlap cells, 849 sign agreements, pair counts 723/63/66, trace saddle `(0.07318804087584509, −0.15208331868800173)` with value 0.044884483163678275. The failed pre-data determinant-saddle test and retrospective trace analysis remain disclosed. R1 concerns only the caption wording. |
| F12, Backour assumptions | **Closed.** Sec. II, lines 296–298. | Adds the unbiased two-token reduction, linear logit transport and positive constant coupling required by the primary source checked in Task40. It describes destabilization of the balanced mixture at the threshold, rather than extending the same claim to unrestricted fields or time-varying coupling. |
| F13, descent versus convergence | **Closed.** Sec. III, lines 452–453. | “The descent estimate alone does not establish convergence to a single fixed point when fixed points are not isolated” replaces the stronger model-specific nonconvergence assertion. This does not purport to settle a stronger convergence theorem. |
| F14, coupling attribution | **Closed.** Sec. VIII, lines 1565–1567. | The formula is attributed to the displayed derivation for this symmetric output coupling, with Fujisaka–Yamada supplying the Lyapunov-stability framework. It no longer assigns this exact coupling convention to an unverified original-source passage. |
| F17, sampled passages | **Closed.** Discussion, lines 1732–1734. | Restricts the 120-step baseline to the smallest offset. At eps=1e−5 the saved means are 7.8145161290 for eight samples and 10.5192604006 for 64, correctly rounded to 7.8 and 10.5. Single samples give 1, 1 and 3 retained phases at the three offsets, so “become rare, one to three per offset” is correct. These retain the original probability-mask convention, not F1's geometric channel definition. |

## Additional changed claims

### The 42/48 result by stage: PASS

**Where:** abstract line 51; introduction lines 161–164; supporting unchanged method/result detail lines 1035–1050.

The 48-row `fold-evidence.json` ledger was read and its saved pass flags/error values counted by target group. No trajectories or forecasts were recomputed.

| Stage | Targets | Rows | Continuous within one | Discrete within one | Declared primary |
|---|---|---:|---:|---:|---|
| First | A, B | 24 | 24 | 23 | Discrete |
| Second | C, D | 24 | 18 | 16 | Continuous |

Every row still matches the article's Table V measured count. The introductory split correctly describes **one continuous forecast statistic across both stages**, secondary in the first and primary in the second. It does not turn 42/48 into a pooled primary-endpoint success count. Both stages' all-rows primary criteria failed, as retained in the detailed text. The abstract's compact “24 and 18 of 24” is numerically correct; the introduction supplies the primary/secondary distinction.

The evidence for prospective status is inherited from the authenticated closed Task5 and Task5B audit/receipt records embedded in Task40, not inferred from the new adjudication:

- Task5 audit: `76217ba867aca5e43ef801dce45263a200a5c0854c2e3dd2b84f0d4e9d12eab6`; freeze `78ae193ee3fb4b418c6e29cf92dadde5815c2100`. Continuous 24/24; discrete-primary 23/24, with A midpoint forecast 61 versus measured 59.
- Task5B audit: `e906239572e15002e9811b217b1c87aa2e0018755a6ed29741666bb7a6a15e7d`; freeze `6f1a53b6c41df1b6ccf69944c8c78a339a594bdf`. Continuous-primary 18/24; discrete-secondary 16/24. Six misses are retained, all at target D. The reported 3.1–5.2% uses the continuous forecast denominator, as already checked in Task40.

### Breadth versus depth: PASS as a scope correction

**Where:** introduction lines 158–160.

The new sentence distinguishes scans across model families from the detailed folds/cascades/shrimp/proof work in Qwen2.5-0.5B, with the second cascade in Qwen2.5-1.5B. This narrows the interpretation to the study's stated scope. It adds no new numerical result. Unchanged broad-model scan records were not re-audited.

## Reused evidence pins

All paths below are under `codex/arxiv-gonogo2-task40/`. Their bytes were hashed during this revision check; original raw data, arrays and old workers were not opened or rerun.

| File | SHA-256 |
|---|---|
| `raw-evidence.json` | `ded399f501aa8195e766f12a911af9c2f96f4b8d9f2d7fb7731107976145fbd0` |
| `fold-evidence.json` | `a36547bf00c025e7ee5c3b4bac4c20929775ece476862ff577bf425f816d0b3c` |
| `cascade-evidence.json` | `766c77ea9ad5f7fe0d745712abb39d85230b66b3ff3b44b997168648014b3ea1` |
| `ditches-followup-evidence.json` | `8469fca8046fde17e6ebebd7500b752c1278e5248f4f49b8d054a2a3d50a1524` |
| `delta-evidence.json` | `33c4629b5a64fc02d44d185e58e40ffeb006d4a52da3f282f2e5a77b67c0d135` |
| `discussion-delta-evidence.json` | `aaa1c1e750ef1df1240192c6ea95d6c0d3876b033d2cacc71c83ab362365a57d` |
| `citations-review.md` | `ee04a76eb48f797aa4020e392ff942f3b73d07e9dc77953d8f01efb652cbde71` |
| `review.md` as read | `7063eaa88d923a7982b8c3da699e391769e53bae10649dca277587bf755bf1a8` |

## Coverage and limits

**Completed:** full article-source diff read, exact revised line locations, all fourteen assigned finding dispositions, changed numerical values compared with own frozen evidence, 48-row stage/endpoint-count consistency, breadth/depth scope check, and hashes of reused compact records. Figure-caption source was checked; figure rendering was not repeated here.

**Inherited, not freshly revalidated:** original source/run authenticity, raw numerical reductions and scientific gates from Task40 and its closed Task5/Task5B receipts; primary literature checks from Task40. This includes the retained failed prospective primary criteria and all original Task40 limitations. The nonlinear free-exponent fits and complete fixed-start orbit identity remain unverified at the same scope as before.

**Outside this lane:** F8's actual public-release gate; F15/build/bundle/pagination verification; the new Ko–Geiping metadata check; revised Zenodo files; fresh model/derivative/trajectory/proof computation. Root handles the applicable separate checks. This report does not authorize publication or repository visibility changes.

Read-only inspection errors were limited to discarded selectors using the wrong JSON nesting or a source path from the wrong working directory; corrected explicit paths/keys supplied the results above. Git emitted stream-fd warnings while the requested source reads completed. No failed scientific run, new control run, original evidence change or article edit occurred.
