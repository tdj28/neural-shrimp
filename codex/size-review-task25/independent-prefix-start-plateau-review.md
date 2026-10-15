# Task25 closed saved-evidence review: PS, CP and plateau

I accept these three components as completed, source-bound saved-data reproductions, with the scientific failures and limitations below retained. This is a read-through of the sole completed Task25 analysis, not another scoring run. I read the closed component results, every prefix/model/plateau disposition, all saved comparison leaves and the owner receipts. Small standard-library projections copied saved fields and tallied existing dispositions; no original target files, reducer, correlation calculation, native model, derivative, bootstrap or test was rerun. Boundary reviews scale, gain and depth separately.

The actual source commit is `2497d55f529fb0d9b1673c452792a696ee50db19`. Inference observed execution, session 21323 reaping with exit 0, and original worker/supervisor identity, group and session closure. Its owner closure is `run-001-dispatch/outer-closure.json`, SHA256 `3714d60a82543181f185fb4b321b99e7d9627f7a47c9d840a13687e057ce1645`; original supervisor receipt is `run-001-supervision/receipt.json`, `3e9a3bef9835941bdb23a81471c2464d24be02be0346fe87d56bfadbc0f666fa`. Eight captured source blobs and the output manifest were verified by that owner. External wall time was 7.683673758991063 s. The worker's 7.507569101871923 s and observed 236656 KiB RSS stop before final rehash/report and are not a separately observed full-lifetime peak.

The immutable report's preparation-era sentence that selected arrays “remain unread” is stale: this admitted run has performed the saved-data scoring. It does not describe the completed run. This clarification leaves the original report unchanged.

## Evidence and coverage

All result paths below are under `/data2/neural-shrimp-data/codex/size-review-task25/run-001/`.

| File | SHA256 |
|---|---|
| report.json | `8a2896947f80e7e9208790da8eabbe6a38fb0837041984236287d3997d3e2a41` |
| ps.json | `aa5b6c72a5b5b7d30355a8087c6c11a8e6526addd4222ab33dc999d026dab21f` |
| cp.json | `f0e03bb753246586671bfa0bcfc4905dc3c97dfea855f2ddcf4951aef87d425f` |
| plateau.json | `6c934d6fb74fe5dc668f71376ae77fb227869ab5b3a62488da0705ec6033af57` |
| comparisons.json | `f0c6e61d8aa613f1807a328a048c5f98539f2b20c9176755298186427179e755` |

The executed driver is `196dc27b3a44bc7fe3e1b8a9562107ac67369c1ca17470021acbedf54657dcaf`; prefix reducer `f82cdc47c028874008bddbe1beda37ab2c1bd2dbaeb86596e0e924019f59f97d`; start-state reducer `5e2b3b43acfec280b3045cbe9af6dfc7500c846d37d803484a5d6565a8f86bc9`; plateau reducer `818a0ae44f2bb58afe2868070bb9e22e5059a14ea5ce539b8a8b67360d5e07d3`. The unchanged outer input manifest is `be61223ae317ce665bb3a5b4ef2cfa99d360a2e8d66fd11451320b1ef032ec39`, binding 329 nested pins.

The companion `/tmp/task25-prefix-start-plateau-compact-coverage.json` (333430 B, SHA256 `3292e1316eed4d9d8a4bf28605d06fe380ea2d91664ce0976289ec5743211a99`) retains all 20 PS endpoints, all 14 CP models / 140 profiles, all 251 plateau rows, all 77 finite-difference ladders / 924 probe rows, all 260 tested offset comparisons, full failed/undefined dispositions, and every nonexact shared comparison. Its larger literal projection is `/tmp/task25-prefix-start-plateau-coverage.json` (1438052 B, `2e96b7cdb35b8b97ae1ef2e34a9e6b9f3cf7e243698efcb40c799a9e67381beb`), which also retains all 1440 PS row diagnostics. Original JSONs remain the authority for every full witness and raw saved value; the extract does not replace them.

## PS: both frozen predictions fail

Every endpoint contains 48 coarse and 24 adaptive fine cells. All 1440 rows have finite coordinates/two-offset exponents and valid period fields; all 20 endpoints have complete finite coverage, no refusal, and an exact fine-grid association with zero coordinate differences. All entropy-increment maxima are unique. Prefixes 0/1 remain development;2–9 remain held out.

| Prefix | 7B C1 / C2 / M | 14B C1 / C2 / M |
|---|---|---|
| 0 (dev) | 10 /16 /.9368394870 | 12 /1 /1.0832946699 |
| 1 (dev) | 1 /9 /1.0539014066 | 3 /2 /1.0373801443 |
| 2 | 6 /7 /1.1290296649 | 14 /12 /1.2211260850 |
| 3 | 3 /14 /.8364281320 | 9 /13 /1.0260182333 |
| 4 | 0 /0 /−.0894236440 | 11 /9 /1.1890299721 |
| 5 | 0 /1 /.0756157574 | 15 /10 /1.1183616328 |
| 6 | 7 /13 /1.0723108842 | 12 /6 /1.1721515580 |
| 7 | 3 /10 /1.0873446326 | 13 /15 /1.0872643764 |
| 8 | 4 /16 /1.1410548992 | 17 /7 /1.0904990856 |
| 9 | 5 /3 /1.1517943199 | 11 /11 /.9090029645 |

C1/C2 are source-defined chaotic counts at the two stages; M is the maximum of the smaller twin exponent. Displayed M values are rounded for reading; full values remain in the companion. PS1 has 0/8 declines and all 8 violation indices; PS2 has 3/8 lower maxima (prefixes 7/8/9), below the frozen 6/8 requirement. There are no missing pairs or unqualified held-out endpoints. Stage2 finds chaos for 7/8 and 8/8 held-out prefixes in 7B and 14B respectively. Adaptive-grid samples are additional targeted observations, not an unbiased second estimate of the coarse-grid chaotic fraction.

All 496 common published PS leaves agree exactly. The only additional own key is the empty missing-pair witness; it is not a discrepancy in a published value.

The source-classified 14B chaotic coordinates range from `.36484736355226643` to `.9`. The 7B range is wider than the prose's main band: `−.06808510638297871` (development prefix 0) to `.9` (held-out prefix 6). Thus the paper's word “mostly” for 7B matters. The finite findings support the two-model sampled comparison and a higher-temperature shift of the main observed band; they do not identify a pure translation or explain a general size law.

For frozen `size-section.tex` lines 38–51, recommended wording is: “In this two-model, ten-prompt test, the 14B model has more chaotic coarse-grid samples on every held-out prompt, and its main observed chaotic band lies at higher temperature. A fixed temperature window therefore misses much of the 14B activity.” Keep both failed predictions. The count uses no detected period and both exponents `>.001`, not positivity alone. OLMo prompt extensions in lines 52–55 are outside this PS inventory.

## CP: primary failure, secondary cutoff pass, weaker expanded sensitivity

All 14 models and all 10 prefix IDs per model are present and finite. All 140 profiles have the exact 601-point entropy and 81-point gain grids. All recorded breadth, crossing-width, T=1 entropy, network-range, maximum-gain and maximum-location compatibility checks pass. No entropy cell is ambiguous relative to the 1/10-nat thresholds; every crossing has one candidate pair and no undefined-crossing possibility. Every argmax has a single compatible candidate. All 140 rounded-profile breadth reductions equal the saved scalar exactly; all 140 maximum-gain differences are nonzero and retained, ranging within the original five-decimal compatibility cells (largest absolute displayed saved difference `4.966588267318883e−6`). This is conditional compatibility with rounded profiles, not a reconstruction of unavailable logits, hidden states or directional native gains.

| Saved outcome | Own rho | Own p | Frozen disposition |
|---|---:|---:|---|
| CP1 paper-prompt breadth | .29535687366198754 | .3052709262839073 | fails |
| CP2 median of 10 prompts | .5171349644228008 | .05826423877056072 | holds |
| CP3 network proxy | −.06850933289361891 | .8159856968536429 | descriptive |
| CP3 maximum one-step proxy | −.32486683662458 | .2570904292454167 | descriptive |
| CP3 log parameter count | −.36062299930265146 | .20527030103491833 | descriptive |
| Expanded sensitivity: paper prompt | .17237663770580955 | .555673156174231 | sensitivity only |
| Expanded sensitivity: median | .4437129337903899 | .11199999038564985 | sensitivity only |
| Expanded sensitivity: log parameters | −.3580130167736044 | .20880985934105528 | sensitivity only |

CP2 passes the frozen `rho≥.5` criterion; its p-value does not pass a .05 significance threshold and was not the gate. The fourteen frozen 3dp outcomes remain primary, distinct from the expanded full-precision sweep lengths. The sensitivity result must not retrospectively replace that primary outcome. Ranks, ties and all model-specific fields remain in the extract; reported p-values use the source's approximation, not a newly run randomization test.

The 167 shared published CP leaves contain 158 exact values and these 9 arithmetic differences (own minus producer); the 218 extra own keys are diagnostics/representation fields, not failed published comparisons. There are no missing-own required keys.

| Node/field | Difference |
|---|---:|
| CP1/rho | +5.551115123125783e−17 |
| CP1/p | −1.1102230246251565e−16 |
| CP2/rho | +1.1102230246251565e−16 |
| CP2/p | +1.3877787807814457e−17 |
| CP3/log_g_net_p0/rho | −1.3877787807814457e−17 |
| CP3/log_params/rho | −5.551115123125783e−17 |
| CP3/log_params/p | −1.1102230246251565e−16 |
| sensitivity/breadth_p0/rho | +2.7755575615628914e−17 |
| sensitivity/breadth_median/rho | −5.551115123125783e−17 |

The one-step quantity is a log proxy built from separate medians: `log(median directional g_net)+log(median directional g_read)`. It is not a median of paired products, a leading singular value or an orbit Lyapunov exponent. Its paper-prompt values are all negative, with actual extrema `−3.376931105474913` and `−.5440567183766042` (outward two-decimal display: [−3.38,−.54]). Individual gains/hidden states are not saved, so their native recomputation is outside this result.

Lines 216–224 are numerically consistent if their scope remains finite. “One forward pass” means one **batched** invocation containing the base state and eight perturbed states, not one unperturbed evaluation alone. Replace “it has to be measured on the loop” with “these particular start-state summaries did not establish a reliable predictor.” The claimed `~1e−7` effect of the later normalization fix is not independently measured by this saved-scalar replay.

## Plateau and QF: all failures and all offset ladders remain

All 10 selected run records have their expected coverage: 24 PTa, 6 PTb, 64 B7a, 64 B7b, 16 PTc, 29 QFa and 12 each for QFb prefixes 2–5, totaling 251. Every row has finite expected offsets, a valid saved period field and matching saved/source class. There are no source-refusal indices or missing rows. These flags validate the saved computation's contract, not asymptotic chaos or primitive-period certificates.

| Run | Finite scientific disposition |
|---|---|
| PTa | All 12 originally selected chaotic cells pass all 3 positive-offset tests; all 12 periodic/unresolved controls are negative. Largest saved spread among the 12 positive cells: `.051681925719631816`. |
| PTb | Two of 3 original chaotic selections confirm; the third fails. All 3 originally unresolved controls are negative. |
| B7a history | 49 periodic, 10 irregular, 5 literal hidden-chaos labels remain; the `1e−9` offset is strongly inconsistent with the larger offset. No history is erased. |
| B7b history | 56 irregular and 8 periodic labels; none positive at all 3 offsets. All 64 are negative at `1e−5`; 60 are positive at `1e−7`, including 4 periodic rows. |
| PTc | All 16 negative at all 3 offsets.11/16 satisfy spread<.1 and spread≤.1;5 do not. No equality or undefined case. |
| QFa |6/29 qualify under both variants: 5 visible, 1 hidden. Other 23 are periodic and unqualified as chaos. |
| QFb prefix2 |0/12 qualify; all 12 periodic and all twins/tangent negative. |
| QFb prefix3 |4/12 qualify: 2 visible,2 hidden. Other 8 periodic and negative. |
| QFb prefix4 |0/12 qualify; all 12 periodic and negative. |
| QFb prefix5 |0/12 qualify; all 12 periodic and negative. |

PTb's failed weak candidate at logT `.11574803149606305` has twins `[−.07595480687544003,−.1016744197041453,+.020149432590770254]`. The confirming temperatures `.13622047244094482` and `.19763779527559056` have exponent ranges `[.2261668014927646,.2604735526973197]` and `[.4043820927301901,.4125735874576969]`. Report those as finite three-offset confirmations rather than qualification of all 70B map cells.

The PTc plateau failures are indices 0/1/4/5/9, with saved spreads `.11035462803052365`, `.22427525435326667`, `.11016480661722539`, `.2608093135171319`, `.6297193122012409`. Both declared boundary variants yield11/16, a strict majority. All 16 negative signs should not be conflated with all 16 having a resolved plateau. The historical inline choice of strict versus inclusive boundary remains unavailable, but does not affect these actual outcomes.

All 77 QF probes retain their entire 12-step ladder, all 924 finite nonnegative probe rows, unique `h=1e−4` reference and zero recorded repeat difference. All 260 requested offset-resolution comparisons pass both `<.1` and `≤.1`, with no equality or undefined case. None of the 77 rows has a positive tangent together with a negative twin. This says what the saved local probes and finite trajectories support; it does not replay the native derivative, show an asymptotic rate or prove a global precision bound.

QFa's six positive tangent estimates are `.24481351162738632`, `.35683734091586666`, `.23175337947739455`, `.34862235220483173`, `.3255181799456983`, `.22872683605399097`. Their largest absolute saved tangent/twin discrepancy is `.006926280931835116`, within the stated .01. All 29 QFa rows together have maximum discrepancy `.014964320327290004`, at the negative unsteered logT=.37 row, preserving the .015 claim. QFb per-prefix maximum discrepancies are `.004771796669193695`, `.005609421329595632`, `.0041717342300073845`, `.005349567777506725`.

At the shared `1e−5` offset, PTc and B7b have only 4 exact stored-coordinate matches, with right-minus-left differences `+.013079664793570034`, `+.018060529131387337`, `+.027380751385311736`, `−.0006556556806789349`; 12 PTc coordinates are unmatched. No interpolation or coordinate tolerance was used. QFa's rounded selected coordinates and revised precision path are separate evidence, not replacements for these records.

Concrete wording changes for frozen sec:size:

- Line 120's unqualified “near zero at `1e−6`” is too broad for B7b: the saved middle-offset range is `−1.1202160963330632` to `+.35189647737335106`. Say “mixed negative, near-zero and positive estimates at the middle offset.”
- Lines 120–121 should not turn a positive **finite-time** estimate together with a detected period into a mathematical impossibility for a stable orbit. A stable orbit's asymptotic exponent is nonpositive; finite-time/non-normal growth and numerical detection require the stated qualification. The inconsistent offset history plus the corrected finite results support the precision-warning narrative.
- Line 129 “kept their labels” is not literal for every PTa category. Original periodic selections 15/17 now have period 0 and negative exponents; originally unresolved selection 21 now has period 14. Say “all 12 selected chaotic cells retained positive exponents at all 3 offsets, and all 12 control cells remained negative,” while retaining those period-label changes.
- Lines 124–128 and 232–237 are supported for the selected QFa records, with finite/native-source limitations. Six of 8 steered candidates qualify; two become periodic. Five positive cells have high entropy and one has lead-period 2. The sampled unsteered controls are period 2, and the sampled higher-temperature steered controls are fixed points. These records do not locate a complete boundary, establish a universal 70B rarity rate, or certify all original map labels.

## Limits and disposition

This reproduces the six-component run's PS/CP/plateau arithmetic and reads all saved gates. It inherits exact original source/archive/receipt association and declared precision history; remote loaded weights, historical process births and independent native derivative/trajectory replay are not supplied by it. Selected original PS/CP files completed, while earlier failed, contradictory or precision-limited histories remain in their authority and the plateau output. Prior manuscript/log/producer-summary exposure is disclosed; this review is not a fresh blinded confirmatory experiment.

No gain/plateau/dedicated-depth published machine-summary comparison was invented when unavailable. Plateau prose was checked against the actual closed row ledgers. Other sec:size extensions (OLMo prompt test, longer 1.5B test, clamps, base-model and growing-context studies) are not silently covered here. Subject to the concrete wording qualifications above, these three assigned saved-evidence reviews are complete; no numerical rerun is required by this review.
