# Task40 raw-evidence review — A1 and A5–A8

Article snapshot: `bb42a82189da91c37f50804c077832df9e2237ae`, read-only `/data2/chaos-journal-neural-shrimp/main.tex`. Research reference: `c567545ca93f90324dae75be2c333c8ac452dab5`. This is a component review, not the combined article verdict.

**Recommendation: fix the cusp bracket range and the universal start/bifurcation sentence before submission.** The replayed answer rates, all 18 synchronization brackets, cited nearby A/B spectra, cusp final tolerances, saddle/exit fits and hook-tip estimates otherwise agree at the stated precision. Several original-evidence limitations remain explicitly partial below; this audit does not establish mathematical certificates or independently replay model derivatives.

## Execution and exact evidence

The independent script was privately checkpointed at `21e628bf28cfb77be54359198de523d4e683b240` before its single authorized run. `raw-001` exited 0, without timeout; child PID 1606434 was reaped. Parent launch-through-reap time was 1.631202329415828 s; the script reported 1.3434875500388443 s. Limits were one numerical thread, nice increment 10, 4 GiB address space, 90 CPU seconds / 120 parent wall seconds, 256 MiB output-file ceiling. Environment had `HF_HUB_OFFLINE=1`, empty `CUDA_VISIBLE_DEVICES`, and all four recorded thread variables equal to 1. No producer imports, model calls, original continuation, or previous control suites ran. Stderr was empty. All five sections completed, with failed or unavailable row gates retained rather than discarded.

| Artifact | Bytes | SHA256 |
|---|---:|---|
| `raw_evidence_audit.py` | 24511 | `42540f529415c739da6623aef77547a6923efcb77b2f03e4b83cd53d262068c6` |
| `raw-evidence.json` | 486420 | `ded399f501aa8195e766f12a911af9c2f96f4b8d9f2d7fb7731107976145fbd0` |
| `raw-001.receipt.json` | 1122 | `2082d0e12d71cc9c00ce8307723f3be383622404d9bc02064cfa591cf978e7fe` |

`raw-evidence.json` contains the exact path, length and SHA256 of all 59 read inputs, every retained curve gate, each raw denominator, all synchronization rows and all 34 saddle-distance rows. The script hashes the same bytes it decodes. Paths below are relative to `/data2/neural-shrimp-data/explore/llm-soft-loop/pod64/` unless stated otherwise. This execution timestamp is 2026-10-20 03:19:38 UTC (19 October local); it must not be backdated to the earlier research snapshot.

## Ranked findings

### F1 — Correct or explicitly redefine the cusp bracketing range [must-fix numerical wording]

Article `main.tex:1379–1380`: “At each of the six, the quadratic coefficient ... falls in modulus from between 2.4 and 39.2 at the bracketing curve points”.

The twelve original endpoint values do not have that range. `kite/locate-la-plus-94/locate.json` begins with `f=-0.17047709685584778` and `2.5663446923657034`; `kite/locate-la-plus-112/locate.json` begins with `0.21386896539308964` and `-2.459650167855898`. Across all twelve endpoint magnitudes, the extremes are 0.17047709685584778 and 39.1082399960387. All six final small-step magnitudes do satisfy the stated bound: maximum `9.477318099015974e-5 < 9.5e-5`.

Replace with: “Across the twelve bracketing curve points, the quadratic coefficient has magnitude 0.17–39.11; at each located cusp it falls to at most 9.5×10⁻⁵.” If the intended statistic is the **larger endpoint magnitude for each bracket**, say so explicitly; those six maxima range from 2.459650167855898 to 39.1082399960387. Do not describe that statistic as covering both endpoint values.

### F2 — A bifurcation need not be visible from every start [should-fix mathematical wording]

Article `main.tex:1463–1465`: “A bifurcation curve belongs to the map and would appear for any start; a boundary between basins of attraction moves or disappears when the start changes.”

The invariant-set bifurcation belongs to the map, but a fixed-start scan can miss it when the start belongs to another basin. This implication is too strong in the multistable setting being discussed. The separately checked adjacent-cell coexistence and saddle evidence still supports the proposed basin-edge interpretation.

Replace with: “Bifurcation curves are properties of the map, whereas a fixed-start scan shows only attractors reached from that start. Together with the coexisting-orbit and saddle checks, the disappearance of this edge supports a basin-boundary interpretation.”

### F3 — State the exit-time measurement convention [note / useful reproducibility edit]

Article `main.tex:1457–1461` gives the correct observed slopes, but they use exits from a normalized saddle-distance radius `10^-3`. The original v2 run also saved radius `3×10^-4`; that gives 25.5551281566 on each side over 15 eligible temperatures, versus 25.7766548377 / 25.7511081034 at radius `10^-3` over all 17 per side. These are finite sampled threshold-crossing fits, not exact asymptotic constants. Suggested addition: “Using the last exit from a relative saddle-distance radius of 10⁻³, ...”. Preserve the invalid first settling-time fit; it was excluded, not overwritten.

## A1 — Answer/reset corrections (main.tex:724–749)

The integer comparator is consistent in the new recording path and historical analyzer in `tools/repro_reset.py`: compare parsed integers; `None` is a value. Text differences are retained separately. Batch-1 exactness intentionally remains a text comparison. The focused comparator tests were read, not executed again.

REPRO-T uses 96 held-out additions for each setting, with zero failed identical-layout checks. At the quoted temperatures: Qwen0.5B bf16 T=1 changes 54/96 = 56.25%, with 23/96 single-layout correct; Qwen7B bf16 T=10^0.2 changes 53/96 = 55.2083%, of which 24 are number-versus-missing. Forced greedy changes 0/96 and 10/96. At T=10^0.4 neither model changes answers, with 90/96 and 96/96 correct. Float64 changes 0/96 and 5/96 = 5.2083% at the respective sensitive settings. The article's rounded values pass.

All six original RR records have 128 paired problems, and every batch-1 exactness check passes. Counts below are parsed-integer changes from the reference, not historical `compared` flags:

| Raw RR run | Free | Reset | Native | Reference rerun |
|---|---:|---:|---:|---:|
| local bf16 | 88 | 7 | 95 | 0 |
| local float32 | 3 | 0 | 2 | 0 |
| H100 Qwen0.5B bf16 | 60 | 1 | 60 | 0 |
| H100 Qwen0.5B float32 | 4 | 0 | 4 | 0 |
| H100 Qwen7B bf16 | 52 | 0 | 74 | 0 |
| H100 Qwen7B float32 | 33 | 0 | 33 | 0 |

Thus the article's H100 reductions 47%→0.8% and 41%→0 pass: 60/128→1/128 and 52/128→0/128. In contrast, historical text comparisons would have produced 85→2 and 82→2. The correction is substantive and now applied to the reported answer rates.

Published Soft Thinking records have 256 paired questions per arm: top15-s64 changes 54 (21.09375%); replay 15 (5.859375%); greedy 6 (2.34375%); top1-s64 45 (17.578125%). Correct counts for layouts8/16 are 63/57, 61/57, 228/227 and 71/65 respectively; top15 correctness changes 16/256=6.25%. The article's 21%, 6%, 89%, 2.3%, 17.6% and 6.2% descriptions match to its chosen precision. “22–30%” is a loose covering accuracy range, not the exact extrema of these four named arms.

PM2 counts, each over 256 paired GSM8K items:

| Model | Soft changes | Sampled changes | Soft finished changes / finished | Sampled finished changes / finished |
|---|---:|---:|---:|---:|
| R1 Qwen7B | 37 (14.4531%) | 26 (10.1563%) | 2/192 | 0/162 |
| R1 Llama8B | 74 (28.9063%) | 55 (21.4844%) | 1/113 | 0/150 |
| QwQ32B | 50 (19.5313%) | 51 (19.9219%) | 0/175 | 0/165 |

Totals match: 158/161 changed soft pairs and 132/132 changed sampled pairs are classified as budget-hit; finished pairs change 3/480 and 0/477. Run metadata records think/answer budgets1536/256. This budget classification follows the disclosed original convention: forced switch, switch at the think limit, or saved answer-token count >=255. Saved tokens exclude EOS and do not always identify the exact stop reason; the audit does not infer unrecorded termination causes.

The separate 4096-budget Qwen run has 128 pairs per arm: 11/128=8.59375% soft and 22/128=17.1875% sampled, matching 8.6%/17.2%. Its run metadata confirms different stop settings and H100 PCIe, so it is not a controlled budget-only contrast.

Limits: parsed answers were audited as saved; tokenizer decoding, exact batch membership reconstruction and dataset authentication were not rerun. Earlier 448-cell reproducibility figures and 4000/1000-item behavior studies in the broader passage are outside this new A1 correction replay.

## A5 — Nearby A/B spectra (main.tex:1376–1378)

All 326 rows in the four original curves were checked against saved residual<=1e-10, |g|<=1e-9 and real critical multiplier within1e-7 of the expected +1/-1. All pass these **saved continuation-diagnostic gates**. This is not independent composed finite-difference qualification or a fresh full-spectrum calculation. The curves save eight listed multipliers and a separately recorded other spectral radius.

| Cited neighborhood / raw curve index | Residual | Critical refined | Other radius | Newton / JVP / unconverged GMRES |
|---|---:|---:|---:|---:|
| A fold, kite-la-curve-minus1 | 5.89205e-11 | +0.999999995331 | .002135078143 | 2 / 15 / 0 |
| B fold, kite-la-curve-plus23 | 1.46389e-11 | +0.999999995576 | .001267912717 | 7 / 40 / 0 |
| A flip, p6s-td-curve-minus1 | 8.13757e-11 | −1.000000004238 | .004168277929 | 7 / 40 / 0 |
| B flip, p6s-ad-curve-plus5 | 3.27340e-11 | −1.000000002936 | .002180134685 | 6 / 35 / 0 |

The article's upper bounds .0022 and .0042 are outward and valid for those nearest appropriate branches; no contrary ±1 is present in the stored spectrum. Nearest points were checked in the declared continuation coordinates `(log10T,10alpha)`. They are nearby points, not exact common-parameter solutions at the quoted A/B coordinates. Evidence supports the local two-sheet interpretation at this sampling resolution.

The full curve records retain unconverged GMRES counts at 24 other rows: fold-plus85–88; fold-minus96–112; flip-A31; flip-B26,28. These rows are not silently called solver-clean: their final saved continuation gates pass but inner solves did not all converge. The four quoted near-crossing rows have zero such counts. Exact counts and every row are in the JSON.

## A6 — Six cusps and scaled unfolding (main.tex:1373–1382)

All six locator records say `located:true`, with all three stored checks true. Independently recomputing the determinant and singular values from each stored 2×2 matrix reproduces its saved values. The source uses coordinates `(log10T,10alpha)`; K is not a derivative matrix in unscaled alpha.

| Locator suffix | Small-step oriented b | Cubic c | det K | Smallest singular value | |mu−1| | Next modulus |
|---|---:|---:|---:|---:|---:|---:|
| la-plus14 (C) | −7.121936e-6 | −13265.0611 | −2624.3051 | 3.904884 | 7.623929e-9 | .0009518272 |
| la-plus94 | +8.026001e-6 | −2246.8742 | +20867.6441 | 1.348088 | 6.306045e-11 | .003227965 |
| la-plus112 | +6.300613e-6 | +2006.1295 | +21959.1277 | 1.016312 | 5.383316e-11 | .003010353 |
| wide2-a64 | +1.445746e-5 | +6097.2160 | +45905.0454 | 3.039811 | 7.676338e-10 | .7390799322 |
| wide2-b82 | +9.477318e-5 | −8005.7537 | +219528.5242 | 2.593069 | 8.161603e-9 | .1437344844 |
| tu-minus34 | −5.651747e-5 | −40211.1853 | +4471.9471 | 12.978405 | 7.068914e-10 | .005358374 |

C is at `(alpha,log10T)=(.07301858033629292,−.15328027306380487)`; singular values are672.0571395060502 and3.904883912438235. Thus C coordinates, cubic coefficient, det, singular values and critical/next values match the article. Across all six, exact |c| extrema2006.129535–40211.185334, |det K|2624.305112–219528.524242, minimum singular values1.016312–12.978405, critical error<=8.161604e-9 and next<=.739079933. The article's ranges are valid rounded descriptions; the two explicit upper bounds8.2e-9 and.74 hold.

Important limits: `final.f` is the small-step oriented quadratic coefficient; `final.b` belongs to a different finite-difference step selection in the unfolding calculation and is not interchangeable. At C they are −7.12e-6 versus+.01059. All twelve source-bracket points pass the saved continuation gate, but the **final locator records do not save final Newton residuals or solver iteration counts**. Those remain unverified here. The six cubic plateau relative spreads range from about9.26e-5 to.0447645; no full model normal-form replay was performed. F1 above corrects the initial bracket range.

## A7 — Synchronization (main.tex:1561–1620, Table VII)

Original complete arrays: near93 pairs×3000steps, refine24×10000, far48×2000; every row was classified by the maximum over its final500distances, using strict `<` at each tolerance. Stored NPZ hashes match the metadata. All18 brackets and all5 nonmonotonicity flags match TableVII:

| Protocol / tolerance | Cell1 | Cell2 | Cell3 |
|---|---|---|---|
| near3000 /1e-8 | (.04,.05] | (.22,.23]* | (.15,.16]* |
| near3000 /1e-6 | (.04,.05] | (.19,.20] | (.13,.14] |
| near10000 /1e-8 | (.045,.05] | none through.19 | (.12,.15] |
| near10000 /1e-6 | (.045,.05] | (.18,.19] | (.12,.15]* |
| far2000 /1e-8 | (.20,.22]* | (.18,.20] | (.14,.16] |
| far2000 /1e-6 | (.20,.22]* | (.18,.20] | (.10,.12] |

A star means an earlier tested coupling also passed. These are finite-grid, finite-horizon terminal-suffix brackets, not unique bifurcation thresholds or a claim of monotonic synchronization.

The3×3000 saved JVP logs give λ1=(.0975014608357,.4370358776552,.1959013166002), hence c*=(.0464494917853,.1770258720570,.0889533211786). The21×2000 twin logs give the predicted range[−.8187892710385,+.4370358776552] and maximum absolute discrepancy.0506046890634, so “+.44 to−.82” and upper bound.051 pass. The largest discrepancy/10-block-standard-error ratio is just below4; the article properly excludes λ1 uncertainty from those error bars.

Refine-run burst counts are13,14,12 at cell2 c=.16,.165,.17 and5,10,6 at cell3 c=.105,.11,.115, using an excursion armed below1e-8 and counted on the next rise above1e-3. These match12–14 and5–10. The original arrays reach about2.4–3.3e-16 in the latter cases, below the article's illustrative “about1e-14” level; that wording is not an asserted floor. The identical-input saved check maximum is8.8932875263e-16, covered by9e-16. Its saved FD table shows the expected quadratic region before a rounding floor; no new derivative run was made.

Joint-periodic metadata retains all18 selected rows:14 have a reported period and a finite return below1e-9;4 do not and remain unqualified. Below-threshold reported joint periods are6,8,10,12; far cell1 c=.14 and.20 have periods6 and4 with nonzero pair distances1.1043 and.8643. Two checked far synchronized examples at c=.12 and.30 report period2; the far c=.24 example is synchronized but has no return below tolerance. This audit **does not establish the article's 'most far first-cell pairs' period2 denominator** from only those selected full-state return diagnostics. Token-pattern evidence alone is insufficient to authenticate full-state period2. Retain that as partial unless a broader raw full-state classification is supplied; a fully supported narrow alternative is “some far-start pairs synchronized onto a coexisting period-2 orbit”.

Original periodic JSONs store the selected primitive-period label and its closure distance, not the full state sequence or all40 tested returns. Consequently minimality and spectra cannot be independently reconstructed here. The transverse producer record explicitly says `tools_dirty:true` at commit `dc17093b8d69f8618fd981dbbb5a5b4fd91516c9`; reducing authenticated saved logs does not authenticate the unrecorded dirty source bytes. Neither caveat invalidates the arithmetic/table replay. The article already correctly limits bubbling to consistency with the mechanism and says candidate transversely unstable orbits and a causal rounding intervention were not computed.

## A8 — Kite edge, saddle and hook tip (main.tex:1307–1310,1452–1470)

The original adjacent-cell basin check has converged orbits on both sides at log10T=−.1535315068493151 and−.15352876712328767, alpha=.07297495107632095. Fixed-start leading moduli are.1840826663 and.6141308276; cross-continued moduli are.1833483079 and.6011826819. Final cross-continuation residuals2.7560e-14 and2.3051e-14 and nonzero gaps.02154/.02174 support local coexistence, matching the caption's rounded.18 and.60–.61. It does not establish coexistence throughout the whole visible edge.

The original saddle record has Newton convergence, residual2.1248471923e-14, saved unstable_count1, mu_u=1.2624643816407222 and next modulus.0007750874082824094. The edge estimate is−.15352996660438276 after34 bisections. Scalar predictions are6/log(mu_u)=25.7438172700 and closest-distance exponent.9684859030, matching25.74 and.97. Only four leading multipliers are retained in this compact record, so the full-spectrum unstable count is relayed as a saved diagnostic, not independently re-eigensolved.

The valid v2 distance array contains1500steps for17temperatures on each side with deltas10^-6…10^-14. All34 saved minima agree with the independently reduced columns. All final saved period6 return distances are small (largest1.31223e-13), and all endpoint labels are side-consistent. Recomputed closest-distance slopes.9857816572 and.9953321214 match.986/.995. At radius10^-3 the17-point exit slopes25.7766548377/25.7511081034 match25.78/25.75. See F3 for the threshold convention and alternate-radius sensitivity. These are parameter offsets, not independently measured distances to the true stable manifold.

All8 hook-tip fold rows pass the saved fold residual/real+1 gate; the largest residual is4.84351e-14. Linear interpolation of the critical-coordinate sign change yields alpha=.07296552523580115. Each of the five finer-strip rows (51alpha values each) has one outcome switch under the declared long-exponent separator−.15. Extrapolations using powers1/2,2/5,1/3 are.0729650689581,.0729653984743,.0729657287503, matching the article's rounded.0729651–.0729657 range. These are local-normal-form and finite-grid extrapolation evidence, not an exact stable-manifold intersection certificate.

The random-start128×128 map's absence of a straight edge remains a finite-grid visual claim; this component did not re-reduce that map. F2 prevents elevating its disappearance to a general diagnostic theorem.

## Coverage and retained limitations

| Item | Status |
|---|---|
| A1 new arithmetic, RR integer rates, PM and PM2/4096 rates | Raw paired records recomputed; all cited rates match |
| A1 tokenizer/batch reconstruction and old448cell/behavior studies | Not replayed; outside these saved reductions |
| A5 every row in4curves and cited A/B neighborhoods | 326saved continuation gates checked;24inner-solver flags retained; no exact common-parameter or full-spectrum replay |
| A6 six locators,12bracket points,2×2 unfolding matrices | Raw coefficient/scalar/matrix checks completed; initial range needs F1; final locator residual/count absent |
| A7 all18brackets,21twins,3Lyapunov records and6burst examples | Independently reduced saved arrays; matches |
| A7 joint primitives / mostfarP2 claim | Selected closure diagnostics checked;4failed/nonperiodic rows retained; primitive minimality and majority denominator partial |
| A8 adjacent-cell coexistence,saddle,34validexitrows,8foldpoints,255stripcells | Saved diagnostics/reductions completed; all quoted rounded values match |
| A8 random-start visual / full stable manifold | Visual claim not re-reduced; no formal proof claimed |
| Model derivative precision, full spectra, certified geometry | Not newly executed or admitted by this saved-data audit |

No numerical section failed, and no automatic retry occurred. The must-fix range and should-fix logical sentence can be corrected without any new model run.
