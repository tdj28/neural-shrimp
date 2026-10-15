# Task21b — final scientific review of saved Kaplan–Yorke estimates

**Verdict.** The independently reproduced arithmetic supports the reported approximate group dimension ranges, with three material corrections: there are **two**, not three, high-group nonnegative first-two sums; **31 of 35** measured final spectra have one positive retained estimate, while four have none; and the hostile log’s strict “other cells within 0.02” checkpoint bound must be **0.021**. All results remain finite four-leading-estimate diagnostics. They do not prove asymptotic chaos, a global scalar conjugacy, or a causal temperature effect.

This is an **independent unblinded saved-data reproduction**, not a new confirmatory experiment. Producer summaries had already been read. I read all35 final four-exponent rows, all84 checkpoint spectra and their finite/permutation/sign/column-difference/dimension/final-equality fields, every group count/extremum/checkpoint summary, and the final source/output/closure authority. No model, QR iteration, eigensolve, producer analysis or statistical computation was repeated. This document transcribes and interprets the sole accepted arithmetic report; the exact rational numbers remain in that report. Its `computational_status: complete` is not a blanket scientific pass.

**Authority and cutoff.** The scientific result is [run-001/report.json](/data2/neural-shrimp-data/codex/night-results-review-task21/task21b/run-001/report.json), 2,226,009B, SHA256 `f3f7354b6918cfbf889be23ed2896f26ba26c20835054a16dc9371199047ae58`. Actual independent source is `c4c9fa9189d7a0c1aca2bb0eae04103975507986`, reader `2f34384e0957baf3450ffb7352a8d6fa35153a6d4855ebe02169188b2379901c`, card `ecf9debac4578a1b397c9c7164aaaf161ebfcec7c77c6bc7ee255a193de3b321`. The frozen paper/log source cutoff is `b7ffbb7d02a31069d633b62fa1a7bac4711ca93a`, with the exposed log through05:25UTC on2026-10-15. Line references below refer to the exported [paper paragraph](/tmp/task21b-source-metadata-001/paper-dimension-paragraph.tex) (`e5280323…`) and [log/cards](/tmp/task21b-source-metadata-001/ky-log-cards-and-results.md) (`c2944b1a…`), not a later mutable manuscript.

The accepted prior [source/math review](/tmp/task21b-source-math-review.md), SHA `de3f9c98684c32a376f2abef0eaa529da3d33c15e73cac63682bd0ba8ecf57b8`, is reused without another source audit. Input002 is `/tmp/task21b-input-authority-002/inputs.json`, SHA `5406ce14d9273306407ebeab5b8f3ad100cd103c8da62d7d0db03176790e13b8`; its original authority is `fdccec189771d14c365d95fb38143b6427c9b48d3fdfff88381db07ff36fce2e`. All three original tool snapshots share SHA `2615641932edea59f29c6c47a51ff828b7dc13d70c2bdf38b2999017c250721b`; KY1/KY2 came from `f8219106923cd0d5bff07c6702fab731808e3187`, TC from `2ae8f53a416b14645ae6ac739c7da02213bb9fa1`.

Final [metadata/source/output/closure authority](/data2/neural-shrimp-data/codex/night-results-review-task21/task21b/run-001-authority/report.json), SHA `968c86c2e22ef72cda0dc616322b166321052bedcfca27c4de5dc560dd8bc950`, manifest `9dccc33324031fdfbf49f83573b26bf2cf690c86c5ceb27dbbcf558a8ef5fda6`, passed:2output files,10supervision entries,5captured Git blobs and original supervisor/worker/groups/sessions absent. **Inference made the original execution and closure observations**, relayed to Root; I read their sealed authority. Session30657 reaped0; supervisor2843904/start199765009, worker2843916/start199765119. Closure `5b90ee22e5dd4b13ae3d93966ae33e7d68e5ed0f9d217be839d578f4c6f7ce63` records no timeout/failure. The calculation/report-construction timer is0.03160295682027936s, ending before JSON encoding/writing; the separate supervisor window is0.5384629489853978s, metadata authentication0.04358294792473316s. Both processes were nice10/one thread, with1GiB hard address-space bounds. PeakRSS was not captured; the AS bound is not an RSS measurement.

## Coverage and original raw files

All runs used the source-declared Qwen0.5B/layer12, k4,1000transient+2000measured steps, seed0,float64 QR estimates in an896-dimensional single-token state. Original GPU/asset provenance is historical finite authority, not a new model/tokenizer/vector-byte reconstruction.

| Run | Requested/saved | Original disposition | Original raw spectrum file and SHA256 |
|---|---:|---|---|
| KY1 | 20/18 | tool timeout 124; retrieved partial18/20; outer runner0 distinct | [spectrum-ky-philosophy](/data2/neural-shrimp-data/explore/llm-soft-loop/pod64/hidden/spectrum-ky-philosophy/lyapunov-spectrum.json) — `001b592cdcddfe86d5f07f49aa241990584e5f7291b3bb285292f2e331bcc137` |
| KY2 | 10/10 | retrieved selected spectrum tool completed0; other job statuses retained separately | [spectrum-ky-hostility](/data2/neural-shrimp-data/explore/llm-soft-loop/pod64/hidden/spectrum-ky-hostility/lyapunov-spectrum.json) — `68c24277e7b8c9ba16757658a8e191bd9f4d630451647ecb444f3d944d7e32ea` |
| TC | 8/7 | tool timeout 124; retrieved partial7/8, final periodic control406 missing; outer runner0 distinct | [tc-spectrum](/data2/neural-shrimp-data/explore/llm-soft-loop/pod64/hidden/tc-spectrum/lyapunov-spectrum.json) — `4fa09d4cc4fa6f381dcf853c52c685f25ae34634670037bf202d57fe2bb789c1` |

KY1 missing indices18/19 are the last two requested head cells. TC requested six earlier chaotic cells10/47/56/240/293/242 and periodic controls143/406; all six headline cells and control143 are saved, control406 is missing. Thus TC is7/8 and `complete:false`, not6/6 or8/8. Original KY1/TC tool timeouts124 remain failures/incomplete runs even though their archive runners returned0. Original archive hashes/member equality and recorded remote termination are bound by Input002; no historical process births or loaded asset bytes are invented.

The headline14 comprise8 saved KY1 head cells plus6 earlier TC cells; the corresponding requested union has16, with KY1:18/19 absent. The new KY percentile rule is distinct from the earlier TC H2 selection. Selection metadata are retained independently of the final `lambda1>0` scoring; no missing cell is treated as negative or imputed. The prior percentile rankings themselves are not recomputed from upstream map pools.

## All35 final rows and all84 checkpoint dimensions

Values in this table are **nearest-rounded displays**, not uncertainty bounds. `N+` counts positive retained estimates at2000steps; `D250/D500` exist only forTC. Every2000-step checkpoint equals its final saved spectrum exactly. All119 stored spectrum records (35final+84checkpoint, including repeated final records) are finite and have no zero estimates or unsupported/bound-only dimensions. All positive-leading records have exactly one positive estimate. All four-negative records give the finite KY arithmetic value0, which does not independently prove a zero-dimensional invariant attractor. The full four-exponent values, exact rational arithmetic, binary64 comparisons and column/rank differences remain under each `rows[i]` in the linked report.

| Key | log10T | alpha | lambda1 (native) | lambda2 (native) | N+ | D250 | D500 | D1000 | D2000 | D1000−D2000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| KY1:0 | 0.010117647 | 0.071823529 | -0.038310579 | -0.548266647 | 0 | — | — | 0.000000000 | 0.000000000 | 0.000000000 |
| KY1:1 | 0.079764706 | 0.071666667 | 0.210823313 | -0.510466080 | 1 | — | — | 1.410837732 | 1.413001610 | -0.002163877 |
| KY1:2 | 0.092941176 | 0.075588235 | 0.214360998 | -0.574091050 | 1 | — | — | 1.366468872 | 1.373391986 | -0.006923114 |
| KY1:3 | -0.018745098 | 0.077784314 | 0.374511248 | -0.548649798 | 1 | — | — | 1.662454790 | 1.682605279 | -0.020150489 |
| KY1:4 | 0.074745098 | 0.048607843 | 0.365239227 | -0.604043666 | 1 | — | — | 1.612990757 | 1.604656994 | 0.008333764 |
| KY1:5 | -0.000549020 | 0.058176471 | 0.398262843 | -0.391601626 | 1 | — | — | 2.063195772 | 2.008780807 | 0.054414965 |
| KY1:6 | -0.005568627 | 0.059117647 | 0.312117460 | -0.320662926 | 1 | — | — | 1.914014154 | 1.973350630 | -0.059336475 |
| KY1:7 | 0.119294118 | 0.059431373 | 0.437154311 | -0.538155530 | 1 | — | — | 1.823109668 | 1.812319647 | 0.010790021 |
| KY1:8 | 0.129960784 | 0.053156863 | 0.444709094 | -0.518398407 | 1 | — | — | 1.860720319 | 1.857851968 | 0.002868351 |
| KY1:9 | 0.132470588 | 0.066647059 | 0.523512479 | -0.511457317 | 1 | — | — | 2.031036012 | 2.015450550 | 0.015585463 |
| KY1:10 | -0.056313725 | 0.065349020 | 0.040355660 | -1.059991965 | 1 | — | — | 1.034334008 | 1.038071666 | -0.003737658 |
| KY1:11 | -0.131516634 | 0.066110568 | 0.111539976 | -1.177607368 | 1 | — | — | 1.059101071 | 1.094717457 | -0.035616386 |
| KY1:12 | -0.153180039 | 0.075826810 | 0.164814667 | -0.718589292 | 1 | — | — | 1.236122520 | 1.229358646 | 0.006763874 |
| KY1:13 | -0.115960784 | 0.069058824 | 0.287403663 | -0.988385473 | 1 | — | — | 1.286312549 | 1.290780946 | -0.004468397 |
| KY1:14 | -0.149833659 | 0.069809198 | 0.211568868 | -1.054118514 | 1 | — | — | 1.175356804 | 1.200706908 | -0.025350104 |
| KY1:15 | -0.118000000 | 0.070698039 | 0.252595521 | -1.021457938 | 1 | — | — | 1.276795006 | 1.247289205 | 0.029505802 |
| KY1:16 | -0.129315068 | 0.073772016 | 0.336967712 | -1.203679955 | 1 | — | — | 1.276289648 | 1.279947930 | -0.003658282 |
| KY1:17 | -0.056823529 | 0.070525490 | 0.347970811 | -0.944395004 | 1 | — | — | 1.378326478 | 1.368458971 | 0.009867507 |
| KY2:0 | 0.184802716 | 0.065500000 | 0.007951421 | -0.991219500 | 1 | — | — | 1.008805434 | 1.008021856 | 0.000783578 |
| KY2:1 | 0.184845604 | 0.065500000 | 0.012654725 | -0.996909341 | 1 | — | — | 1.012146955 | 1.012693958 | -0.000547002 |
| KY2:2 | 0.190260186 | 0.065500000 | 0.017599326 | -1.313327407 | 1 | — | — | 1.012210385 | 1.013400562 | -0.001190177 |
| KY2:3 | 0.179838456 | 0.065500000 | 0.042376168 | -0.850662712 | 1 | — | — | 1.047689120 | 1.049815477 | -0.002126356 |
| KY2:4 | 0.176922087 | 0.065500000 | 0.054408391 | -1.196483697 | 1 | — | — | 1.045653105 | 1.045473575 | 0.000179530 |
| KY2:5 | 0.179677627 | 0.065500000 | -0.007239854 | -0.349735133 | 0 | — | — | 0.000000000 | 0.000000000 | 0.000000000 |
| KY2:6 | 0.180192280 | 0.065500000 | 0.086136958 | -0.860519320 | 1 | — | — | 1.079800614 | 1.100098808 | -0.020298194 |
| KY2:7 | 0.189842030 | 0.065500000 | 0.047106527 | -0.899732954 | 1 | — | — | 1.111420028 | 1.052356121 | 0.059063907 |
| KY2:8 | 0.180728377 | 0.065500000 | 0.141133411 | -0.902505335 | 1 | — | — | 1.166944636 | 1.156379587 | 0.010565049 |
| KY2:9 | 0.180052895 | 0.065500000 | -0.008817142 | -0.358274725 | 0 | — | — | 0.000000000 | 0.000000000 | 0.000000000 |
| TC:10 | -0.051725490 | 0.071043137 | 0.209825272 | -0.877244554 | 1 | 1.071199038 | 1.062506519 | 1.221810356 | 1.239186748 | -0.017376392 |
| TC:47 | -0.116470588 | 0.073286275 | 0.357682600 | -1.099498188 | 1 | 1.404538666 | 1.435968792 | 1.363681699 | 1.325314406 | 0.038367292 |
| TC:56 | -0.058862745 | 0.077686275 | 0.415343135 | -0.643473560 | 1 | 1.627197165 | 1.672751942 | 1.649617409 | 1.645470398 | 0.004147011 |
| TC:143 | -0.107294118 | 0.077168627 | -0.315431027 | -2.230450378 | 0 | 0.000000000 | 0.000000000 | 0.000000000 | 0.000000000 | 0.000000000 |
| TC:240 | -0.166037182 | 0.076942270 | 0.063227691 | -1.008431503 | 1 | 1.045881969 | 1.056120540 | 1.060460798 | 1.062699043 | -0.002238245 |
| TC:293 | -0.135039139 | 0.067079256 | 0.200012367 | -1.020031541 | 1 | 1.080919965 | 1.139180914 | 1.222094895 | 1.196084493 | 0.026010402 |
| TC:242 | -0.132045010 | 0.071042074 | 0.310268261 | -0.962509143 | 1 | 1.297384365 | 1.312957420 | 1.312369765 | 1.322353572 | -0.009983807 |

Only one distinct final vector is not already descending: KY2:5 (and its identical2000-step checkpoint) has stable permutation `[0,2,1,3]`. Its two middle values are−0.34973513335297063 and−0.34972684458980063. Native first-two sum−0.356974987613740017 becomes sorted−0.356966698850570017; both are negative and D=0 under both conventions. Every other final/checkpoint permutation is identity. All native-order candidate dimensions equal the sorted dimensions in this packet; this agreement does not justify assuming native QR order is always sorted. Decimal-rational versus represented-binary64 versus literal-floating differences are explicitly retained, not used as model uncertainty.

The three nonpositive newly selected cells are KY1:0 (lambda1=−0.03831057852854194), KY2:5 (−0.007239854260769387) and KY2:9 (−0.008817141804073178); prior periodic controlTC:143 has−0.31543102706425385. None changes sign between its retained checkpoints. All35final and84checkpoint vectors meet their finite/shape/permutation contract, and all35final-checkpoint exact-equality fields are true. Earlier checkpoints correctly carry null rather than a false equality requirement.

## Group extrema, counts and ranges

Raw extrema below are displayed decimal prefixes with ellipses; **exact rational extrema** and their full decimal expansions are in the report/companion JSON. Display bounds are the saved outward3-decimal endpoints, not confidence intervals. Positive subsets exclude the four explicitly retained negative-leading rows, not their existence. Group unions overlap other rows and are not additional experiments.

| Group | Saved/requested | Positive-leading | Raw positive D min (attaining cell) | Raw positive D max (attaining cell) | Outward positive range | All-saved D range |
|---|---:|---:|---|---|---|---|
| KY1 high | 10/10 | 9 | 1.37339198631775110… (KY1:2) | 2.01545054973508221… (KY1:9) | [1.373, 2.016] | [0.000, 2.016] |
| KY1 head | 8/10 | 8 | 1.03807166574448812… (KY1:10) | 1.36845897082639088… (KY1:17) | [1.038, 1.369] | [1.038, 1.369] |
| KY2 host | 10/10 | 8 | 1.00802185642293863… (KY2:0) | 1.15637958673219664… (KY2:8) | [1.008, 1.157] | [0.000, 1.157] |
| TC prior headline | 6/6 | 6 | 1.06269904342446108… (TC:240) | 1.64547039802399253… (TC:56) | [1.062, 1.646] | [1.062, 1.646] |
| TC periodic controls | 1/2 | 0 | — | — | — | [0.000, 0.000] |
| Headline union | 14/16 | 14 | 1.03807166574448812… (KY1:10) | 1.64547039802399253… (TC:56) | [1.038, 1.646] | [1.038, 1.646] |
| All requested | 35/38 | 31 | 1.00802185642293863… (KY2:0) | 2.01545054973508221… (KY1:9) | [1.008, 2.016] | [0.000, 2.016] |

The paper’s approximate ranges1.008–1.16(host),1.04–1.65(headline14) and1.37–2.02(high9), and the log’s rounded per-cell lists, agree as **conventional approximate summaries**. Their rounding is not by itself a scientific discrepancy. If they are used as literal bounds, use the outward ranges above: in particular headline min1.038071665744488… is below1.04, and the log’s host max1.156379586732196… is above1.156. “Within0.16” and “within one-sixth” of1 remain true for the eight final positive host estimates. None of those final-only bounds claims to enclose earlier checkpoints.

Exact selected positive temperature extrema are: high KY1:3 `−0.018745098039215688` toKY1:9 `0.13247058823529415`; headline14 TC:240 `−0.1660371819960861` toTC:10 `−0.051725490196078444`; host KY2:4 `0.17692208720514652` toKY2:2 `0.1902601858470336`. Consequently the headline paper’s[−.17,−.04] is a loose covering interval, not an attained or nearest-rounded range. A two-decimal outward headline interval is[−.17,−.05]. High[−.02,.13] is ordinary approximate rounding, but a literal enclosure requires[−.02,.14]. The parameters are reported as original saved values, without rerounding them before membership decisions.

## Checkpoint and assertion verdicts

All84 checkpoints were reviewed: KY1 has36, KY2 has20, TC has28. Each run’s final checkpoint is2000. The exact differences are signed `Dcheckpoint−Dfinal`; bounds below concern absolute differences. No checkpoint resampling, trajectory rerun or confidence calculation was performed.

| Comparison | Maximum absolute D difference (decimal prefix) | Attaining cell | Safe outward6-decimal upper bound |
|---|---|---|---:|
| KY1 high 1000→2000 | 0.05933647514785359806… | KY1:6 | 0.059337 |
| KY1 head 1000→2000 | 0.03561638624936779965… | KY1:11 | 0.035617 |
| KY2 host 1000→2000 | 0.05906390737226027836… | KY2:7 | 0.059064 |
| TC prior headline 250→2000 | 0.16798770958593812246… | TC:10 | 0.167988 |
| TC prior headline 500→2000 | 0.17668022927805945417… | TC:10 | 0.176681 |
| TC prior headline 1000→2000 | 0.03836729227849524626… | TC:47 | 0.038368 |
| TC periodic controls 1000→2000 | 0 | TC:143 | 0.000000 |
| Headline union 1000→2000 | 0.03836729227849524626… | TC:47 | 0.038368 |
| All requested 1000→2000 | 0.05933647514785359806… | KY1:6 | 0.059337 |

Every2000-versus-final difference is exactly0. TC control143 has D=0 atallfour checkpoints. Earlier TC changes as large as0.176680229278059… (TC:10,500steps) remain visible; the new KY claim “within0.06” must not be generalized to every earlier TC checkpoint. These are cumulative same-trajectory sensitivities, not independent replications or a calibrated error envelope. KY2:7’s native four-exponent1000-minus2000 differences are `[0.089531349925958675, −0.3265984285820783, −0.4523119520823026, −0.3255954575891006]`; a modest D change alone does not establish convergence of every exponent.

| Frozen assertion | Verdict and actionable correction |
|---|---|
| Paper12/log29: three high cells have lambda1+lambda2≥0 | **Discrepancy:** exactly2. KY1:5 sum=`0.00666121711897305`, D=`2.0087808072183685436…`; KY1:9 sum=`0.0120551617577222`, D=`2.0154505497350822169…`. KY1:6 remains negative (`−0.0085454650651402`, D=`1.9733506296350492608…`). These are also exactly the two crossings at1000steps; no order convention changes the count. |
| Paper6–7/log33: every measured attractor has exactly one positive exponent | **Discrepancy as universal wording:**31/35finals have one;4/35have none. Restrict the sentence to the positive-leading subset and finite horizon. No sampled vector shows more than one positive estimate; this is evidence, not a universal exclusion of hyperchaos. |
| Log19: all other host cells within0.02 at1000steps | **Strict-bound discrepancy:** KY2:6 has absolute difference`0.020298193993652832979…`; use0.021 after excluding the largest-difference KY2:7. Overall host maximum`0.059063907372260278366…` is below0.06. |
| Log29: high1000-step differences at most0.06 | **Confirmed:** max`0.059336475147853598067…` atKY1:6. |
| Paper9–12/log4/19/28/31: displayed dimension ranges/lists | **Confirmed as approximate summaries.** Use exact/extremal rows and outward ranges above for bounds; do not call nearest rounding a scientific failure. |
| Paper10: fourteen headline cells, logT−.17to−.04 | **Count confirmed for the observed union**,14/16requested. The upper endpoint−.04 is not attained; retain exact extrema or explicitly call it a covering interval. |
| Paper5–6: percentile choice before spectra | **Scoped confirmation:** new KY1/KY2 design and fixed membership are bound. Six prior TC cells had their own earlier selection; do not imply all14were newly chosen under KY. Original pool-ranking reconstruction is outside this saved-spectrum audit. |
| Paper7/log24/34: dimension grows with temperature | **Overbroad inference:** groups change direction, alpha and selection population. Hostile logT values exceed those of the philosophy high group, yet D is lower. Report descriptive group differences; no isolated causal or universal monotone trend is established. |
| Paper10: rising with lambda1 | **Descriptive tendency only, not strict monotonicity over all14.** For example KY1:14 has larger lambda1 (`.21156886784783768`) thanTC:10 (`.20982527202999907`) but smaller D (`1.200706908…` vs`1.239186747…`). D depends on the contracting estimate too. No new association statistic was fitted. |
| Paper12–14/log3/34: effectively2D; scalar reduction holds/fails | **Cannot establish from these spectra.** D>2 here does not mean two positive exponents. Near1 does not imply a globally injective scalar coordinate or smooth conjugacy, and largerD alone does not prove a return-map fit fails. Keep separate return-map and multiplier evidence under its original scope. |

## Suggested paper/log wording

“Four finite leading Lyapunov estimates were computed by QR iteration after1000transient and2000measured steps. Of38requested cells,35returned complete spectra:18/20new philosophy cells,10/10hostility cells and7/8earlier TC cells. Thirty-one had one positive retained estimate and four had none; none of the sampled finite spectra had more than one positive estimate. Using the ordered retained estimates, the eight positive hostile cells gave D_KY in[1.008,1.157], the fourteen observed headline cells in[1.038,1.646], and the nine positive high-group cells in[1.373,2.016] (outward-rounded ranges). The headline set combines eight new cells with six prior TC cells. Exactly two high-group cells had nonnegative first-two sums. The largest1000-to2000-step D changes were below0.06 for the new high and host groups; excluding the largest hostile change, the others were below0.021. These finite estimates describe the sampled regimes and do not establish a global scalar conjugacy or isolate a causal temperature effect.”

If attained temperature ranges are reported, use the exact extrema above and label any shortened range as approximate or outward. Preserve missingKY1:18/19 andTC:406, the original124timeouts, the four negative-leading cells, the separate priorTC selection, and the larger earlyTC checkpoint changes.

The source claims leading estimates, not the full896-spectrum. No retained final or checkpoint sum stays nonnegative through allfourvalues, so this actual packet needs no tail-only lower-bound substitution. Sorting supplies the ordered arithmetic convention, not a proof that the finite columns have converged to the true leading exponents. The QR log uses a1e−300 diagonal floor without a saved hit counter, so absence of floor activation cannot be established here. Historical loaded model/tokenizer/vector bytes and GPU effective-precision behavior were not reconstructed. The original receipts and source/command association are the accepted finite historical basis. These limitations do not request another model run and do not negate the saved arithmetic findings.
