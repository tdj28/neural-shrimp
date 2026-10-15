## 2026-10-15 01:30 UTC: run (exploratory): FB, the crescent's doubling cascade at two lower temperatures (Planned marker "ratios at other temperatures"), stage 1: flips

- **Question:** Do the finite-order ratios of the crescent's period-doubling cascade behave as the crossover hypothesis of the cascade section predicts? The hypothesis: early levels see the corner of a smoothed argmax, with ratios toward 2, and later levels the smooth quadratic core, with ratio 4.669. It predicts smaller early ratios at lower temperature and later ratios rising toward 4.669. The human approved this review follow-up at about 00:58 UTC.
- **Setup:** The cross-512 map shows the crescent's cascade (period 3 to 6 to 12 to chaos along alpha) only between log10 T of about -0.162 and -0.146, so the paper's cut at -0.14798 lies near its upper edge and the new cuts are lower: -0.152 and -0.156. Stage 1, one pod each (H100-class, job set cascade): tools/llm_feigenbaum.py cascade with the settings of crescent-t-0.14798-v2 (philosophy, layer 12, float64, free alpha, period 3, 8 flips, transient 900, k 6, 24 steps per period, tol 1e-12). Starts and first doubled bands come from the map: T -0.152 alpha 0.0645, next 0.0650; T -0.156 alpha 0.0642, next 0.0647. Stage 2, after stage 1: superstable parameters (tools/llm_feigenbaum.py superstable, as runs C9 and C10), with brackets inside each band from the flips and, past the last converged flip, extrapolated with ratios 4.0-4.6.
- **Prediction, fixed now:**
  - P-F1: the first flip ratio (index 0 of 'ratios', 2.844 at -0.14798) is below 2.844 at both new temperatures and lower at -0.156 than at -0.152.
  - P-F2: the last two superstable ratios computed at each new temperature lie within 5 % of 4.669 (at least 4.436).
  - Each is reported as it comes out, including partial or failed cascades.
- **Output:** `../neural-shrimp-data/explore/llm-soft-loop/pod64/cascade/crescent-t-0.152`, `crescent-t-0.156` (stage 1); `superstable/crescent-t-0.152`, `-0.156` (stage 2)

## 2026-10-15 01:40 UTC: Amendments: CL restarted (JSON bug), FB stage 1 failed at the switch to period 12 and reruns with map-based switching

- **CL.** The local smoke test of cusp_locate on hub7-foldS-plus 13 ran the whole real-model path. Round 1 bracketed b sign(a) from +6.51 to -16.58, and the final unfolding was computed. The tool then crashed writing its JSON, because numpy booleans sat in the checks. The four CL pods ran the same code, so they would have lost their results at the end. Fixed: plain bools and a numpy-aware JSON fallback, with a test that a result serializes. The pods are terminated after about 20 minutes and relaunched at the fix commit, with unchanged run card and targets.
- **FB stage 1.** Pods mpjn8uzrkptamm and h7yv0po5o7lc1n: runner rc 0, tool exit 1, results retrieved.
  - At log10 T -0.152: flips 3->6 at alpha 0.0648738 and 6->12 at 0.0652532.
  - At -0.156: flips 3->6 at 0.0644767 and 6->12 at 0.0650494.
  - Both then failed switching to period 12 (SearchError: the switch returned to the period-6 orbit). Until three flips give a measured ratio, the tool places the switch at half of last interval / 4.0. The cross-512 map shows the period-12 band at about 0.85 of the period-6 band at -0.152 and about 1/15 of it at -0.156, so both switches fell outside the band.
  - New cascade options (default behaviour unchanged): --ratio-guess, --switch-fraction, --max-substeps. The reruns use ratio guesses read off the map, 1.2 at -0.152 and 15 at -0.156. These are search aids for the switch only; the flips are located as before.
  - The predictions P-F1 and P-F2 stand as frozen. The map's rough band ratios already suggest that the first ratio at -0.156 is far above 2.844, against P-F1.

## 2026-10-15 02:02 UTC: run (exploratory): FB stage 1c, the deeper cascade levels on the local GPU (human: "Do it")

- **Question:** Do the later ratios of the crescent's cascade at log10 T -0.156 approach 4.669, now that the first ratio there is 14.93 and the frozen prediction P-F1 has failed (scored in the next entry)? And what is the first ratio at -0.152?
- **Setup:** Local RTX 2000 Ada, float64, tools/llm_feigenbaum.py cascade with the settings of the pod runs (philosophy, layer 12, free alpha, transient 900, k 6, tol 1e-12), but starting from the last level that ran cleanly, so that the first switch uses a known point:
  - (A) T -0.156: period 12 at alpha 0.06507 (inside the measured period-12 band 0.0650494-0.0650877), next 0.0650907 (inside the period-24 band), 5 flips, ratio guess 4.669, 96 steps per period after a switch.
  - (B) T -0.152: period 6 at alpha 0.06505, next 0.06535 (start of the period-12 band on the cross-512 map), 5 flips, ratio guess 4.0, 96 steps per period.
- **Reading:** The ratios are scored against P-F2 (later ratios within 5 % of 4.669) and reported as they come out. A failed switch is reported with its message.
- **Output:** `../neural-shrimp-data/explore/llm-soft-loop/pod64/cascade/crescent-t-0.156-c`, `crescent-t-0.152-c`

## 2026-10-15 02:03 UTC: FB stage 1 result: frozen prediction P-F1 fails; the first ratio at log10 T -0.156 is 14.9, and the cut at -0.152 crosses a doubling bubble

Pods mpjn8uzrkptamm and h7yv0po5o7lc1n (first attempt), x03brom1shhb3t and pt50aoqu7xj1bo (map-based switching). All runner rc 0, hashes verified, terminated and verified; commits bb5bd418 and 5ad048f2. Outputs cascade/crescent-t-0.152(-b) and crescent-t-0.156(-b).

- **log10 T -0.156:** flips 3->6 at alpha 0.06447665673849548, 6->12 at 0.06504935553077586 and 12->24 at 0.06508771941580027, all with critical multiplier -1 to 6e-11. The first ratio is 14.93. The search for the 24->48 flip then walked to 0.0650993 without finding it (band estimate from the measured ratio, 16 substeps). So the period-24 band is wider than 1.2e-5, and the second ratio is below about 3.2.
- **log10 T -0.152:** flips 3->6 at 0.06487375301996827 and 6->12 at 0.06525318227183999. Continued forward from the switch, the period-12 solution collapses onto the period-6 orbit near alpha 0.06568-0.06573. The doubled orbit merges back into period 6, a doubling bubble. On the map this cut shows period 12, a sliver of chaos, then period 6 again. This cut has no complete cascade from period 3.
- **P-F1 (first ratio below 2.844 at both temperatures, lower at -0.156):** fails. The ratio is 14.9 at -0.156, and there is none at -0.152.
- **Reading:** Along nearby cuts the early ratios depend strongly on where the cut crosses the crescent: 2.84 at -0.148 and 14.9 at -0.156, with a bubble at -0.152 in between. The simple crossover hypothesis (a smoothed-argmax corner, ratios toward 2 that fall at lower temperature) is contradicted. Universality concerns the limit and is tested by the later ratios (P-F2): stage 1c on the local GPU, from period 12 at -0.156.

## 2026-10-15 03:03 UTC: FL15 normal form (1.5B fold point), the scoring rule for the FL lines (fixed before their results), FB stage 1c interim: the third flip ratio at log10 T -0.156 is 4.499

**FL15 normal form.** Pod gzlvd9ydhgfd0o (SECURE H100 NVL; runner rc 0, hash verified, terminated and verified; 267 s; commit e7ea0b09). The clashing upload copies of branch-p4 are byte-identical to the local ones. Outputs fl15/curve-fold-p4 and fl15/normal-form-p4.
- At alpha 0.05858823529411764 the fold point of the period-4 orbit lies at log10 T_c = -0.09120288061238553. Orbit residual 2.3e-14; crossing multiplier +1.0000000000000244 (real +1, a fold); next-largest modulus 0.0076. The gate is met: fold coefficients for a real +1.
- Normal form in log10 T: a = -7.1590, b = -21.0947, so ab = 151.02 (pi / sqrt(ab) = 0.2556). b converges with the step: -21.279, -21.111, -21.0965, -21.0948, -21.0946 for relative steps 1e-2 to 1e-4.
- For comparison, the 0.5B fold ('is it' at log10 T -0.115) has ab = 122.93 per unit alpha.

**Scoring rule for the FL lines (tools/foldlaw_score.py, tests/test_foldlaw_score.py), fixed before the line results.** The card leaves several choices open; they are fixed now:
- Windows are maximal runs (at least two cells) of one period beyond the fold. A run whose neighbouring run has half its period is a doubling and is not counted. The run nearest the fold counts per period.
- The fan is the windows from the fold outward, as long as their periods step by q.
- d is measured at the window edge nearest the fold, from the fold point above. Centres are reported as a check.
- Spread is (max - min) / mean of d (x + delta)^2 with x = p / q, minimized over delta.
- Refinement of the card: with a free offset, a different power also passes over a short range. d = 0.004 / x passes over x = 3-8 with delta = 4.9 (spread 6 %).
  - Since N = x - k with k >= 0 and the law's lap offset is below one lap, x + delta can exceed the laps by at most one, so delta <= 1.
  - Both the free fit (the card's criterion) and the bounded fit are reported, and a free fit needing delta > 1 is flagged.
- The 0.5B reference values: f = sqrt(C ab) / pi = 0.576 from C = 0.0266, ab = 122.93.

**FB stage 1c interim (local GPU, log10 T -0.156).** The flips, each with critical multiplier -1 to 3.5e-10 and next modulus 7e-5, 4e-7 and 2e-13:
- 12->24 at alpha 0.0650877194157999 (the pod value was 0.06508771941580027);
- 24->48 at 0.06509999933181365;
- 48->96 at 0.06510272895244879.

With the pod's 3->6 and 6->12 flips, the interval ratios are 14.928, 3.1241 and 4.4988. The deviations from 4.669 are +10.3, -1.55 and -0.17. These are flip ratios; P-F2 concerns superstable ratios and stays unscored. The run continues to the 96->192 and 192->384 flips (5 flips from period 12).

**Amendment (human, about 03:00 UTC: "And double that maybe?").** After the 192->384 flip, a follow-on cascade run starts from the period-384 orbit inside its band and locates the 384->768 flip. The start and next points come from the measured ratio, the transient covers at least 50 periods, and the settings are otherwise those of stage 1c. It runs on the local GPU, before stage 1c's queued cut at -0.152.

## 2026-10-15 03:29 UTC: run (exploratory): FB stage 2 (superstable ratios for P-F2) and the -0.152 cut on H100 pods (human: "H100", about 03:25 UTC)

- **Why:** The human chose an H100 pod for the superstable stage that scores the frozen prediction P-F2, and wants the paper ready for editing by the morning, Pacific time. The local GPU keeps stage 1c's -0.156 flips (to 192->384) and the 384->768 continuation. The queued local run at -0.152 moves to a pod, so the two run in parallel.
- **FB152d (pod, job set cascade):** the -0.152 cut from the period-6 orbit at alpha 0.06505, settings of stage 1c otherwise. It uses --next 0.0652832 (3e-5 past the measured 6->12 flip at 0.0652532), so the walk along the period-12 band takes substeps of 1.5e-5, against 4.8e-5 in stage 1. It allows up to 64 substeps, enough to reach beyond the merge near 0.0657 found in stage 1. Question: does the period-12 orbit flip anywhere on this cut, or does it merge back into period 6 without a cascade, a bubble as stage 1 found? A narrow flip interval could have been stepped over in stage 1.
- **FBS156 (pod, job set superstable, after the local 96->192 flip):** superstable parameters at log10 T -0.156 for the bands of periods 6, 12, 24, 48 and 96, from the measured flips. Each bracket runs from 5 % to 98 % of its band, as at -0.148, with transient 20000 and settle factor 20. Bands 192 and 384 follow in later pods, once the local run has located the 192->384 and 384->768 flips.
- **Reading:** P-F2 (frozen at 01:30 UTC) holds at a temperature if the last two superstable ratios computed there lie within 5 % of 4.669 (at least 4.436). If the -0.152 cut has no cascade, P-F2 cannot be scored there, and this is reported as such.
- **Output:** `../neural-shrimp-data/explore/llm-soft-loop/pod64/cascade/crescent-t-0.152-d`, `superstable/crescent-t-0.156` (and `-192`, `-384` later).

## 2026-10-15 03:35 UTC: Amendment to FB152d: the start jumped too far

Pod gqn3rlkugdknn2 (COMMUNITY H100 NVL; runner rc 0, tool exit 1, hash verified, terminated; 136 s; commit 65aeba3f). The period-6 orbit at alpha 0.06505 (mu -0.005) did not follow to --next 0.0652832: "period-6 Newton did not converge". The jump of 2.3e-4 is 61 % of the period-6 band. The rerun FB152e starts at alpha 0.06520 (86 % of the band, a jump of 8.3e-5), with all else as in FB152d. The queued local run of stage 1c (start 0.06505, next 0.06535) would have failed the same way.

## 2026-10-15 03:47 UTC: FB results so far: -0.152 has no cascade (a doubling bubble, confirmed at finer steps); at -0.156 the fourth flip ratio is 4.6208; FBS156 launched

- **FB152e.** Pod mdleh5grbwo78y (COMMUNITY H100 NVL; runner rc 0, tool exit 1, hash verified, terminated; 507 s; commit 2d04c0a4).
  - The 6->12 flip is at alpha 0.0652531823, which reproduces stage 1's 0.06525318227184.
  - Walking the period-12 orbit in substeps of 1.5e-5 (stage 1 used 4.8e-5), no flip was found before the orbit collapsed onto the period-6 orbit at alpha 0.06567343 (Collapsed: |F^6 x - x| / |x| = 1.6e-11).
  - The cut at -0.152 therefore crosses a bubble 6->12->6 without a cascade, at that search step, and P-F2 cannot be scored there. (The walk does not log mu along the period-12 orbit, so its closest approach to -1 is not recorded.)
- **Stage 1c at -0.156 (local).** The 96->192 flip is at alpha 0.06510331968162615 (critical multiplier -1 to 4.8e-9, next modulus 2.9e-14). The flip ratios are now 14.928, 3.1241, 4.4988 and 4.6208, with deviations from 4.669 of +10.3, -1.55, -0.17 and -0.048. The run continues to 192->384, then 384->768.
- **FBS156** (superstable stage at -0.156) launches now, with brackets at 5-98 % of the measured bands:
  - 6:-0.156:0.064505291678110:0.065037901554930 12:-0.156:0.065051273725027:0.065086952138099 24:-0.156:0.065088333411601:0.065099753733493 48:-0.156:0.065100135812845:0.065102674360036 96:-0.156:0.065102758488908:0.065103307867043

## 2026-10-15 04:35 UTC: FNF4 result: the flip normal form predicts the doubled orbit where the map is far from one-dimensional; FBS156 interim; MB2 and FNF2 relaunched

- **FNF4.** Pod gtul0qkyyp2p6x (SECURE H100 80GB HBM3; runner rc 0, hash verified, terminated; 253 s; commit 2acbc8b8).
  - Flip of the period-4 orbit at alpha 0.07369205994966807, log10 T 0.0319: multiplier -1 to 3.1e-12, real; next modulus 0.278, so the crossing is classified as a flip.
  - c = 12386 (plateau 1e-3/3e-4, spread 9e-5): supercritical.
  - Offsets 2.5e-5, 5e-5, 1e-4, 2e-4 and 4e-4 (alpha below the flip):
    - beta = 0.0175, 0.0348, 0.0685, 0.1325, 0.2470.
    - Amplitude ratio (measured / sqrt(beta/c)): 1.0029, 1.0059, 1.0122, 1.0254, 1.0558. The error doubles with the offset (O(beta)).
    - Doubled-orbit multiplier against 1 - 4 beta: 0.929870 vs 0.929829 (4e-5), 0.860871 vs 0.860777 (9e-5), 0.725657 vs 0.726034 (4e-4), 0.4538 vs 0.4700 (0.016, beta^2 = 0.018). At the largest offset the multiplier is complex (0.125 +- 0.207i), beyond the normal form's range.
  - By the card: confirmed (ratio 1.003 at the smallest offset, multipliers agree to O(beta^2)). The local reduction holds at a next modulus of 0.28, where the global scalar reduction is a rough guide.
- **FNF2.** Pod 16bu5ertdbsrtl (rc 0, tool exit 1, 132 s).
  - The continuation from alpha 0.0775 toward the flip stopped at 0.07678: lfb.evaluate refused a complex leading pair -0.512 +- 0.261i (next modulus 0.575). That guard belongs to flip location, not to continuation.
  - Fixed: the continuation now uses Newton alone, and the guard applies only at the flip. Tests still pass. Rerun as FNF2b.
- **MB2.** Pod mh8mf3kqiqbrco never became SSH-ready (15-min ready limit): terminated and verified, no results. Relaunched as MB2b. MB1 (1214 s) and MB3 (1874 s): rc 0, hashes verified, terminated.
- **FBS156 interim.** Pod 4cz4ixtug3n9u6 (SECURE H100 NVL; rc 0; hash verified; 2661 s). Superstable parameters at log10 T -0.156:

  | p | s_p | how | clean bracket |
  |---|---|---|---|
  | 6 | 0.0647848 | stopped at the transverse spectrum | [0.0645053, 0.0650379] |
  | 12 | 0.0650664 | stopped | [0.06505127, 0.06506819] |
  | 24 | 0.06509398 | stopped | [0.065093875, 0.065099754] |
  | 48 | 0.065101410848 | stopped | width 2.4e-9 |
  | 96 | 0.065103035545868 | converged | width 2e-15 |

  - The ratios are 10.21, 3.713 and 4.572.
  - The brackets of s_12 and s_24 are wide (the second clean point stayed at the bracket end), so the ratios that involve them are poorly determined; worst-case ranges follow.
  - P-F2 is scored on the last two ratios once bands 192 and 384 are in. Those bands were part of the FBS card before this result.

## 2026-10-15 05:11 UTC: FB stage 1c: the 192->384 flip and the fifth ratio 4.6618; run FBD (deeper flips on an H100) and FBS156-192

- **Flip 192->384** (local, log10 T -0.156): bracketed between alpha 0.06510344639876085 (mu -0.9999998) and 0.06510344639876837 (mu -1.0000001), so alpha = 0.0651034463987655 +- 5e-15; next modulus 6e-14.
  - The interval ratios are now 14.928, 3.1241, 4.4988, 4.6208 and 4.6618. The deviations from 4.669 are +10.3, -1.55, -0.17, -0.048 and -0.0074.
  - The local run is still refining the last digits; that cannot change the ratio at the quoted precision. The local follow-on (384->768) starts when it ends.
- **FBD (pod, job set cascade):** the same cut, starting from the period-384 orbit at alpha 0.06510345998979336 (middle of the estimated 384 band), next 0.06510347503852487, ratio guess 4.661793821985329, 2 flips (384->768, 768->1536), transient 40000, 96 steps per period, run limit 150 min. The local follow-on computes 384->768 independently, as a cross-check across devices.
- **FBS156-192 (pod, job set superstable):** the period-192 band at -0.156, bracket 0.0651033260174831 to 0.0651034438644227 (5-98 % of the band), transient 40000, settle factor 20.

## 2026-10-15 06:05 UTC: FBS156-192 result: P-F2 holds at log10 T -0.156 (last ratios 4.572 and 4.643); DSS3 refused at its seeds (the period-6 hook is not effectively one-dimensional)

- **FBS156-192.** Pod 9emvmdzf4axl8t (SECURE H100 80GB HBM3; runner rc 0; commit b5af43a7). s_192 = 0.06510338547902846 (converged, clean bracket 5e-15).
  - Superstable ratios at -0.156: 10.21, 3.713, 4.572, 4.643 (r3 4.5716 (worst case 1.02-4.64) r4 4.6429 (+- 0.0070 from the s48 bracket)).
  - P-F2 (frozen at 01:30 UTC: the last two superstable ratios computed at a new temperature within 5 % of 4.669, at least 4.436) **holds at -0.156**: 4.572 and 4.643. The first depends on the imprecise s_24 (worst-case range above); the second only on s_48, s_96 and s_192, which are precise.
  - Band 384, after FBD's 384->768 flip, will make the last two ratios both precise.
  - At -0.152 P-F2 cannot be scored: no cascade (FB152e).
- **DSS3a/b** (period-6 hook of the philosophy map; pods ran commit d89d5bc9). Both refused at the seed, as the tool requires a clean real leading multiplier:
  - seed (0.080471, -0.071235): leading multiplier complex, -0.0195 + 0.264i (next modulus 0.264);
  - seed (0.080471, -0.068588): mu = -0.261 with next modulus 0.099, below the cleanliness factor 10.
  - So the hook's period-6 orbits are not effectively one-dimensional: two multipliers of similar size, sometimes a complex pair. Superstable curves (leading multiplier 0) are not defined there, and Singer's theorem does not apply. This is the regime where two-dimensional shrimp organization (as in Figure fig:zooms, the period-5 x period-6 crossing) can occur. A two-dimensional test is needed instead.

## 2026-10-15 06:26 UTC: FBD result: the 384->768 flip at log10 T -0.156, ratio 4.6673; the locator's acceptance threshold made explicit

- **FBD** (pod 5kcjngot3vja4p, H100; runner rc 0, hash verified, terminated; 3329 s; commit b5af43a7; output cascade/crescent-t-0.156-e/progress.jsonl). The locator bracketed the flip of the period-384 orbit in 12 evaluations and then refused, as designed: the best evaluation had |mu + 1| = 7.9e-6, above its hard-coded acceptance 1e-6. The crossing itself is clean at every evaluation: a real critical multiplier, Ritz residual below 1e-14, next modulus about 1e-13.
- **Precision floor.** The 11 evaluations within 1e-12 of the root scatter by 1.8e-5 in mu + 1 about a straight line. This is the measured precision floor of the period-384 multiplier: a product of 384 step Jacobians, with Newton residuals 1e-14 to 4e-11. The second planned flip (768->1536) was not attempted.
- **Root from the logged evaluations** (stated method, CPU, inline script; output kept in the session scratchpad and reproduced here):
  - least-squares line through all 12 evaluations, slope -7.956e7 per unit lambda;
  - root lambda = 0.06510347354857382, standard error 7e-14, band +-2.3e-13 (scatter divided by slope);
  - all near-root evaluations lie in [0.065103473548189, 0.065103473549129].
- **Fifth flip ratio** (f192 - f96)/(f384 - f192) = **4.6673**: 4.6672-4.6675 over that whole span, and 4.66716-4.66750 for a root moved by +-1e-12.
  - The flip ratios at -0.156 are now 14.93, 3.124, 4.4988, 4.6208, 4.6618, 4.6673 (Feigenbaum's delta = 4.6692).
  - Distance from delta: -0.170, -0.048, -0.0074, -0.0019.
- **Tool change** (tools/llm_feigenbaum.py, tests 8/8 pass): the acceptance |mu + 1| <= g_accept is now an argument (--g-accept, default 1e-6 as before) and --lam-rel-tol is exposed. A deeper run must set g_accept above the measured floor of its period, stated in its run card, never below it.
- The local cross-check of the same flip (RTX 2000 Ada, started 05:14, older code) is still running and will hit the same refusal. Its evaluations are an independent location on other hardware.

## 2026-10-15 06:27 UTC: run (exploratory): FBD2 (the 768->1536 flip) and FBS384 (the band-384 superstable point) at log10 T -0.156

Human (14-15 Oct): "And double that maybe?", "What about 768?". Both runs are on H100 pods (cap 5.5/h) and use commit HEAD (with --g-accept).
- **FBD2**: cascade crescent-t-0.156-f, period 768, --flips 1. The start (mid band) and next points follow the rule of fb768b.sh:
  - start = f384 + 0.5 w = 0.0651034764570688; next = f384 + w (1 + 0.25/d) = 0.0651034796771438, with w = (f384 - f192)/d and d = 4.66733.
  - transient 80000. --g-accept 3e-4: the floor measured at period 384 was 1.8e-5, and I expect at most about 1e-4 at period 768. If the scatter of mu + 1 exceeds 3e-4, the locator refuses and the root is read from the logged evaluations as in FBD.
  - --lam-rel-tol 2e-12, so the bracket stops near the expected noise band instead of spending evaluations below it.
  - Gives the sixth flip ratio (f384 - f192)/(f768 - f384).
- **FBS384**: superstable crescent-t-0.156-384, bracket [0.0651034470, 0.0651034730]. This is inside (f192, f384) with insets of about 2 % of the band. Transient 60000.
  - Gives the superstable ratio (s192 - s96)/(s384 - s192). With s384, the last two superstable ratios both come from precisely located points.

