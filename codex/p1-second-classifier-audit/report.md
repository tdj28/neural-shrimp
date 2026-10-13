# Task11: second-classifier robustness audit — 2026-10-13

Completed independent CPU-only replay of the44 producer-scored arms. This is exploratory robustness on consumed dialogues, not a new confirmation, a replacement for the original frozen L8 test, or a human-label validation. The two instruments may share errors. Source29e2bffdd45efb630e8ed9967cf720da21086f92 was privately frozen before the actual runs (admission6bb87a3c82e4f10b27c37429a559163a94e82192); no analysis source or threshold changed after outcomes. Both instruments use stored float32 toxic scores strictly greater than0.5. Producer-summary exposure before analysis is recorded in plan.md.

## Findings and limits

- Main-dose capture and persistence remain increasing from0.04 to0.07. Their magnitudes shrink: the geometric stay summary at0.07 changes from16.070370 to6.517970 turns. This is1/(1−Pstay), not a fitted dwell-time or survival model.
- Ramp minus fresh0.05 remains positive but loses a positive95% bootstrap lower bound:0.0578125 [−0.00078125,0.11798828125], versus0.090625 [0.021875,0.16171875]. Ramp–fresh0.04 and both fixed-dose hysteresis contrasts retain positive intervals.
- Generic-kindness cannot support an unqualified no-benefit or direction-specific-necessity claim under both instruments. Over turns5–39, the second classifier gives kindness0.169420 versus no-control0.372768 (first:0.571429 versus0.583705). These are actual separate-arm shares, not a new matched-dose effect estimate. Kindness mean counter-dose0.056107 exceeds the observed constant grid[0,0.04]. The legacy clipped-interpolation efficiency is preserved with its flag, but is not a valid same-dose comparison; it is excluded from any robustness-success tally. No aggregate“13/14 robust” claim is made.
- The three L10 regeneration arms retain lower shown-reply hostility than the interpolated same-run constant-dose response. This audit cannot classify discarded original replies and therefore cannot establish second-classifier would-be hostility or basin prevention.
- Exploratory pooled L8 state-feature increment remains positive:0.042742 [0.023513,0.061752], versus first-classifier0.033280 [0.017960,0.048166]. The dose0.04 secondary row changes to0.020497 [−0.009025,0.049447]; dose0.045 remains positive. These refit L7 models and alter score-derived baseline features as well as labels. They do not isolate a label-only change in fixed fitted models.
- Random/cooking control magnitudes are instrument-sensitive: r1-080 share0.118945→0.001563; ck-0800.071680→0.001172; r2-0800.006445→0. Undefined onset/stay statistics for no-event rows are retained.

## All44 label-agreement rows

Fractions use all saved turns0–39. “First only” and “second only” are the two discordant cells; agreement is not classifier accuracy. All four confusion counts, denominators, actual sample arguments and unrounded values are in descriptive.json. Agreement ranges0.618945–1.000000.

| Arm | First share | Second share | Agreement | First only | Second only |
|---|---:|---:|---:|---:|---:|
| p1-free | 0.000000 | 0.000000 | 1.000000 | 0.000000 | 0.000000 |
| p1-h040 | 0.101953 | 0.077930 | 0.953320 | 0.035352 | 0.011328 |
| p1-h050 | 0.295508 | 0.197070 | 0.885938 | 0.106250 | 0.007812 |
| p1-h050s2 | 0.282031 | 0.156055 | 0.860352 | 0.132812 | 0.006836 |
| p1-h060 | 0.569922 | 0.346484 | 0.761719 | 0.230859 | 0.007422 |
| p1-h060s2 | 0.545898 | 0.352148 | 0.801172 | 0.196289 | 0.002539 |
| p1-h070 | 0.871484 | 0.619336 | 0.741211 | 0.255469 | 0.003320 |
| p1-r1-060 | 0.000391 | 0.000195 | 0.999414 | 0.000391 | 0.000195 |
| p1-r1-080 | 0.118945 | 0.001563 | 0.881836 | 0.117773 | 0.000391 |
| p1-r2-080 | 0.006445 | 0.000000 | 0.993555 | 0.006445 | 0.000000 |
| p1-ck-060 | 0.025586 | 0.000000 | 0.974414 | 0.025586 | 0.000000 |
| p1-ck-080 | 0.071680 | 0.001172 | 0.929492 | 0.070508 | 0.000000 |
| p1-ramp-s1 | 0.325195 | 0.186719 | 0.850586 | 0.143945 | 0.005469 |
| p1-ramp-s2 | 0.321875 | 0.208789 | 0.870117 | 0.121484 | 0.008398 |
| p1-c040-s2 | 0.126953 | 0.093359 | 0.942969 | 0.045312 | 0.011719 |
| p1-exit030 | 0.177539 | 0.103906 | 0.912695 | 0.080469 | 0.006836 |
| p1-exit020 | 0.180469 | 0.106836 | 0.918164 | 0.077734 | 0.004102 |
| p1-exit000 | 0.141602 | 0.077734 | 0.929492 | 0.067187 | 0.003320 |
| p1-exitm020 | 0.115039 | 0.069531 | 0.948242 | 0.048633 | 0.003125 |
| p1-c030-s1 | 0.032227 | 0.030664 | 0.982422 | 0.009570 | 0.008008 |
| p1-c030-s2 | 0.041992 | 0.032227 | 0.980469 | 0.014648 | 0.004883 |
| p1-exit040-s1 | 0.212500 | 0.124414 | 0.902930 | 0.092578 | 0.004492 |
| p1-exit040-s2 | 0.270313 | 0.179688 | 0.891797 | 0.099414 | 0.008789 |
| p1-exit030-s2 | 0.230664 | 0.158398 | 0.910937 | 0.080664 | 0.008398 |
| p1-exit000-s2 | 0.159375 | 0.098242 | 0.929102 | 0.066016 | 0.004883 |
| p1-exitm020-s2 | 0.147266 | 0.096289 | 0.939648 | 0.055664 | 0.004687 |
| p1-ctl-none | 0.560156 | 0.368164 | 0.796680 | 0.197656 | 0.005664 |
| p1-ctl-trig-cancel | 0.153711 | 0.111133 | 0.950000 | 0.046289 | 0.003711 |
| p1-ctl-trig-over | 0.149609 | 0.113867 | 0.952539 | 0.041602 | 0.005859 |
| p1-ctl-trig-long | 0.092773 | 0.070117 | 0.969531 | 0.026562 | 0.003906 |
| p1-ctl-trig-kind | 0.563477 | 0.200781 | 0.618945 | 0.371875 | 0.009180 |
| p1-ctl-const-02 | 0.085352 | 0.058594 | 0.952930 | 0.036914 | 0.010156 |
| p1-ctl-const-04 | 0.011719 | 0.008398 | 0.994336 | 0.004492 | 0.001172 |
| p1-ctl2-none | 0.569922 | 0.346484 | 0.761719 | 0.230859 | 0.007422 |
| p1-ctl2-const-02 | 0.101953 | 0.077930 | 0.953320 | 0.035352 | 0.011328 |
| p1-ctl2-const-04 | 0.010742 | 0.009961 | 0.996484 | 0.002148 | 0.001367 |
| p1-ctl2-regen-06x3 | 0.000586 | 0.006836 | 0.992578 | 0.000586 | 0.006836 |
| p1-ctl2-regen-04x0 | 0.008984 | 0.009570 | 0.988867 | 0.005273 | 0.005859 |
| p1-ctl2-regen-04x3 | 0.005273 | 0.006445 | 0.992578 | 0.003125 | 0.004297 |
| p1-ctl2-trig-06x3 | 0.138281 | 0.104883 | 0.954883 | 0.039258 | 0.005859 |
| p1-det040 | 0.106738 | 0.080176 | 0.949219 | 0.038672 | 0.012109 |
| p1-det045 | 0.182422 | 0.107813 | 0.907813 | 0.083398 | 0.008789 |
| p1-conf040 | 0.102637 | 0.076709 | 0.948096 | 0.038916 | 0.012988 |
| p1-conf045 | 0.173291 | 0.102979 | 0.912988 | 0.078662 | 0.008350 |

## Capture and persistence

Conditional median onset is zero-based among dialogues with an event. The separately retained censored median assigns40 to never-hostile dialogues. Stay uses adjacent origins0–38; transition numerators/denominators, never counts, per-turn shares and late-window measures remain in the raw output.

| Arm | Share first→second | Stay first→second | Conditional onset first→second |
|---|---:|---:|---:|
| p1-free | 0.000000→0.000000 | undefined→undefined | undefined→undefined |
| p1-h040 | 0.101953→0.077930 | 0.355030→0.361538 | 9.000000→6.000000 |
| p1-h050 | 0.295508→0.197070 | 0.667801→0.653455 | 3.000000→3.000000 |
| p1-h050s2 | 0.282031→0.156055 | 0.625802→0.570694 | 3.000000→3.000000 |
| p1-h060 | 0.569922→0.346484 | 0.787986→0.729902 | 2.000000→2.000000 |
| p1-h060s2 | 0.545898→0.352148 | 0.798821→0.767178 | 2.000000→1.000000 |
| p1-h070 | 0.871484→0.619336 | 0.937774→0.846578 | 1.000000→1.000000 |
| p1-r1-060 | 0.000391→0.000195 | 0.000000→0.000000 | 23.500000→11.000000 |
| p1-r1-080 | 0.118945→0.001563 | 0.854701→0.375000 | 15.500000→20.500000 |
| p1-r2-080 | 0.006445→0.000000 | 0.454545→undefined | 4.000000→undefined |
| p1-ck-060 | 0.025586→0.000000 | 0.409449→undefined | 25.000000→undefined |
| p1-ck-080 | 0.071680→0.001172 | 0.592068→0.000000 | 18.000000→33.000000 |

## Hysteresis contrasts

Late means use turns35–39. Each contrast independently resamples whole dialogues4000 times, seed0. All14 seed-level rows and11 pooled histories, including seed2, are retained in the full output.

| Contrast | First difference [95% CI] | Second difference [95% CI] |
|---|---:|---:|
| ramp-minus-fresh040 | 0.290625 [0.233594, 0.348438] | 0.156250 [0.105449, 0.208613] |
| ramp-minus-fresh050 | 0.090625 [0.021875, 0.161719] | 0.057812 [-0.000781, 0.117988] |
| hold040-minus-fresh040 | 0.094531 [0.044531, 0.146094] | 0.054688 [0.010918, 0.099219] |
| hold030-minus-fresh030 | 0.077344 [0.039844, 0.115625] | 0.042187 [0.011719, 0.075781] |

## L6 controllers

Hostility uses shown replies5–39; bracketed intervals are4000-draw dialogue bootstraps. Dose is the recorded absolute mean counter-dose. All late shares and lexical gratitude values remain in the raw output. Kindness is explicitly outside the constant-dose grid.

| Arm | First share [95% CI] | Second share [95% CI] | Mean dose | Outside grid |
|---|---:|---:|---:|---|
| p1-ctl-none | 0.583705 [0.524554, 0.643979] | 0.372768 [0.310937, 0.436607] | 0.000000 | False |
| p1-ctl-trig-cancel | 0.151562 [0.136830, 0.167193] | 0.105580 [0.090179, 0.122545] | 0.025714 | False |
| p1-ctl-trig-over | 0.144420 [0.132812, 0.155357] | 0.106027 [0.092634, 0.119196] | 0.034607 | False |
| p1-ctl-trig-long | 0.085045 [0.077232, 0.093086] | 0.061384 [0.053795, 0.068973] | 0.030013 | False |
| p1-ctl-trig-kind | 0.571429 [0.520977, 0.619420] | 0.169420 [0.147098, 0.192634] | 0.056107 | True |
| p1-ctl-const-02 | 0.084821 [0.066964, 0.105580] | 0.054241 [0.041735, 0.068080] | 0.020000 | False |
| p1-ctl-const-04 | 0.013170 [0.003571, 0.025675] | 0.009375 [0.002009, 0.020312] | 0.040000 | False |

Within-grid cancellation/overshoot/long-pulse efficiencies remain positive under both instruments, meaning worse shown-response suppression than the interpolated constant-dose comparator by that declared metric. Kindness’s clipped value is not interpreted as a same-dose effect.

## L10 shown-reply controllers

No second-label value is imputed for discarded originals. The0.04 single-regeneration arm regenerates12.1875% of turns5–39, about one in8.2; its first-window fraction is14.53125% and final-window fraction13.359375%, so“one in seven/eight” depends on the window.

| Arm | First shown share [95% CI] | Second shown share [95% CI] | Mean dose | Regenerated5–39 |
|---|---:|---:|---:|---:|
| p1-ctl2-none | 0.593973 [0.534821, 0.654247] | 0.344420 [0.282137, 0.408929] | 0.000000 | 0.000000 |
| p1-ctl2-const-02 | 0.106920 [0.080804, 0.136384] | 0.079018 [0.055352, 0.107366] | 0.020000 | 0.000000 |
| p1-ctl2-const-04 | 0.010937 [0.004241, 0.019866] | 0.010045 [0.003795, 0.018527] | 0.040000 | 0.000000 |
| p1-ctl2-regen-06x3 | 0.000670 [0.000000, 0.001562] | 0.006250 [0.002673, 0.012277] | 0.023170 | 0.093750 |
| p1-ctl2-regen-04x0 | 0.009821 [0.005580, 0.014732] | 0.008482 [0.004911, 0.012723] | 0.004875 | 0.121875 |
| p1-ctl2-regen-04x3 | 0.005357 [0.002679, 0.008705] | 0.005134 [0.002455, 0.008705] | 0.013339 | 0.082366 |
| p1-ctl2-trig-06x3 | 0.130580 [0.116964, 0.143304] | 0.094420 [0.080134, 0.108036] | 0.023692 | 0.000000 |

## Exploratory L8 detection

Event: first hostile, horizon3. L7 train arms only; L8 test arms never fit PCA/scaling/logistic or choose thresholds. Each instrument replaces event labels, risk sets and score-derived baseline values, retaining state/dose/axis features and the frozen method. “Positive windows” counts risk-window labels, not distinct event dialogues. All six rows are estimable; each has2000/2000 valid dialogue-bootstrap draws, with no undefined draws.

| Instrument / test | Risk rows | Positive windows | Dialogues | Baseline AUC | State AUC | Increment [95% CI] |
|---|---:|---:|---:|---:|---:|---:|
| first / pooled | 14573 | 1982 | 937 | 0.656015 | 0.689295 | 0.033280 [0.017960, 0.048166] |
| first / p1-conf040 | 8376 | 966 | 468 | 0.621919 | 0.659617 | 0.037698 [0.013051, 0.062126] |
| first / p1-conf045 | 6197 | 1016 | 469 | 0.678689 | 0.708411 | 0.029722 [0.009150, 0.050205] |
| second / pooled | 17003 | 1684 | 927 | 0.653963 | 0.696705 | 0.042742 [0.023513, 0.061752] |
| second / p1-conf040 | 9246 | 842 | 464 | 0.632835 | 0.653332 | 0.020497 [-0.009025, 0.049447] |
| second / p1-conf045 | 7757 | 842 | 463 | 0.674277 | 0.735750 | 0.061473 [0.038304, 0.085762] |

Train rows/positive windows: first7679/965; second8680/874. All four logistic fits converged: iterations11/13 and9/14, respectively; zero convergence or other captured fit warnings. Both actual PCA fits used randomized solver with the fixed random seed; the larger synthetic profile used covariance_eigh, as selected by the unchanged sklearn auto rule. This timing-profile difference changes no target method. Thresholds are trained on L7 negatives; realized test FPR/sensitivity and all saved scores/IDs are retained. There were no degenerate training/test rows, model retries, or omitted dose rows.

## Evidence and execution

All198 required files (621,125,151bytes) were hash-verified before and after each run against inputs-001.json SHA93facfe72743e25dcd4025b2ae22db09d387e2edd32818b7069469f091c30ab9. All44 second files match the completed producer manifestc90331389be2b2a6496c1b02925d99766252d945ea5a7b1c3698a5547af044ad at sourcefeac2a796e4aa673e2d7f44654a9dc06b2070416. Its writer-closure statement is producer-attested; independent local admission checks the actual bytes, not a rerun of the scoring model.

Runs were CPU1 plus owned supervisor, nice10,4GiB hard AS, sampled4GiB aggregate RSS,300s/64MiB per target,20GiB startup/16GiB runtime available-memory floor. Descriptive completed5.855759s with162,471,936B peak RSS; L8 completed8.140423s with583,569,408B. Both supervisors reaped their workers, all four recorded identities were absent, and all25 target artifacts were made read-only. No target failure or retry occurred. Synthetic profile and20 tiny controls are retained.

Absolute raw outputs:

- /data2/neural-shrimp-data/codex/p1-second-classifier-audit/descriptive-001/descriptive.json — SHAb3cc3add6efbe7c9120529e2afdce95f0270cced89397662253ae05d5927eef2.
- /data2/neural-shrimp-data/codex/p1-second-classifier-audit/l8-primary-001/comparison.json — SHA20275b1712570364bf65ae7f1b8f4f8f8b83e95fb6dab80fe1042b5a85bc413a.
- Same L8 directory: first-l8.json, second-l8.json and both *-scores.npz files retain fit diagnostics, exact risk IDs/labels/scores and operating points.
- Compact profile-001-receipt.json, descriptive-001-receipt.json and l8-primary-001-receipt.json bind the full output inventories, source/card ancestry and process closure.

Paper references are the frozen1491f379 paper/llm/main.tex capture803–831, history834–853, L6871–883, L10885–897 and L8926–933. This audit does not edit the paper or retune any original headline. Root independently reviews these saved values and receipts before manuscript/comms use.
