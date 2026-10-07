# Paper outline (working draft, 7 October 2026)

Status: planning document. Every result below carries its evidence label from
the [adaptive research protocol](adaptive-research-protocol.md). Nothing here
is a public claim; public release, authorship and posting are human decisions.

## Working title

**Beyond the edge of stability: quasi-periodicity, phase locking and shrimps
in the learning-rate plane of gradient descent**

## Thesis (to be confirmed)

Beyond the edge of stability, deterministic gradient descent with per-layer
learning rates follows the quasi-periodic route to chaos. The period-2
oscillation of the edge of stability undergoes a Neimark-Sacker bifurcation of
the two-step map, giving quasi-periodic training on an invariant circle.
Rational rotation numbers produce phase-locked periodic windows (Arnold
tongues) whose periods are twice the rotation denominator and whose order
along learning-rate slices follows the Farey tree (a devil's staircase).
Breakdown of the invariant circle gives chaos; inside tongues, orbits lose
stability through period doubling and secondary Neimark-Sacker bifurcations,
and some domains show shrimp anatomy. Binary trainability maps label almost
all of this structure "trainable".

## Evidence so far

| Claim | Evidence | Label |
|---|---|---|
| Verified attracting periodic windows embedded in chaos exist in per-layer learning-rate planes (two networks, two seeds each) | EXP-028: 61 windows confirmed by producer and independent audit | Prospectively frozen, audit-confirmed |
| The published binary trainability score labels mostly non-convergent training "convergent" | EXP-028 cross-tabulation: 87-100% of score-convergent cells are not fixed points | Prospectively frozen (descriptive) |
| Ridge-free published setting has no finite periodic attractors; outcomes fragile | EXP-028 S1 maps and audit re-simulation | Prospectively frozen (descriptive) |
| A window with shrimp anatomy (fold body, flip edge into period 2p, legs) | A2 local map of S2-28011-zoom:587; 46 fold, 15 flip rays; 8/15 flips open onto period 36; ray/map edges agree on 57/64 rays | Exploratory |
| Window boundaries mix fold, flip and Neimark-Sacker | A2 rays over all 61 windows | Exploratory |
| Quasi-periodic training on invariant circles of the two-step map | Return maps (98% variance in two axes), rotation 0.2194 near the 2/9 window | Exploratory |
| Locked windows are Arnold tongues: period 2q, rotation m/q, Birkhoff-ordered | 2/9 window exact; 65/115 orbits well-ordered vs 0.7% random null; S1R plateau periods all 2q | Exploratory |
| Devil's staircase with Farey mediants along slices | S2: 2/9, 3/13, 1/4 plateaus; S1R: 1/9, 2/17, 1/8, 3/23, 2/15 | Exploratory |
| Odd-period windows follow a different (fold-flip) route | Odd-period S2 windows mostly not well-ordered | Exploratory, unresolved |
| Mechanism yields a training-relevant consequence | Not yet tested | Open |

## Figures (planned)

1. Period-resolved trainability map versus the published binary map
   (same cells), with the chaotic tongue and windows (S2 coarse and zoom).
2. The route: bifurcation sketch along a slice (fixed point, period 2,
   invariant circle, tongues, chaos) with return maps at each stage.
3. Devil's staircase along slices for both networks, Farey tree overlay.
4. Shrimp anatomy of a tongue window: local map with typed boundaries and the
   period-2p cascade.
5. Census over confirmed windows: boundary types, periods versus rotation
   denominators, well-ordering.
6. Consequence (to be chosen): rotation number as an order parameter for
   reproducibility or loss behavior.

## Needed before any headline claim

1. **A2 survey completion:** local maps for all 61 windows; frozen version of
   the shrimp-consistency criteria.
2. **Generality (exploratory):** staircases on more slices, seeds and the
   width-16 network; the learning-rate x momentum plane; a real-data network.
3. **Mechanism details:** locate the Neimark-Sacker curve of the period-2
   orbit (two-step map multipliers crossing the unit circle as a complex pair);
   continue tongue edges as fold curves; explain odd-period windows.
4. **Consequence (A3b):** test whether rotation number or locking predicts
   reproducibility (uncertainty exponents, function-space disagreement) or
   loss behavior, against generic controls (FTLE, sharpness).
5. **Confirmatory test on fresh data (frozen and pushed before running, with
   independent audit):** new seeds and at least one new network. Frozen
   predictions: (a) a quasi-periodic band with well-defined rotation numbers
   between the period-2 region and chaos; (b) locked windows with periods 2q
   on rotation plateaus m/q; (c) Farey-mediant ordering of plateaus along
   frozen slices; (d) Neimark-Sacker of the period-2 orbit at the band's onset.
6. **Writing and release:** human authorship; AI-use disclosure; primary-source
   verification of every citation (see [novelty check](novelty-check-2026-10-06.md));
   clean reproduction package; human decision on public release.

## Related work map

Trainability fractals (Sohl-Dickstein 2024; Torkamandi 2025; Liu 2024; Liang and
Montufar 2026), edge of stability and period doubling (Cohen et al. 2021;
Kalra, He and Barkeshli 2025; Chen et al. 2023; Song and Yun 2023; Hofmann
2026), chaos in training (Kong and Tao 2020; Herrmann et al. 2022;
Jimenez-Gonzalez et al. 2026), reproducibility (Ly and Gong 2025), shrimps and
hubs (Gallas 1993, 1994; Stoop et al. 2010; Jones 2012), circle maps, tongues
and torus breakdown (Arnold; Aronson et al. 1982; Jensen, Bak and Bohr 1983-84;
Kaneko 1983-86; Curry and Yorke 1978), learning-in-games tongues (Metaxas and
Sapsis 2026), heavy-ball cycles (Goujaud et al. 2025).

## Amendment, 7 October 2026 (after the roadmap review)

Per the human-approved Pro roadmap review (`docs/consults/2026-10-07-roadmap/adjudication.md`,
verdict "not ready to freeze"), the plan changes as follows. Outcomes seen
before this amendment: everything in the evidence table above plus the
consequence pilot (`docs/explore-consequence-2026-10-07.md`); no reserved
seed or held-out architecture has been opened.

1. **Primary claim (bounded, existential):** in specified finite tanh networks
   under deterministic full-batch gradient descent in float64, a period-2
   branch undergoes an attracting Neimark-Sacker bifurcation, and a
   rule-selected periodic window is connected to the corresponding resonance
   by continuation. Window fractions stay descriptive; the staircase is
   exploratory; "route to chaos" only with a prespecified connected path
   through converged chaos. The binary-score point is phrased as "conflates
   dynamical regimes", not "misclassifies its target".
2. **Order of work (dependency gates):** (a) controls and exploratory ancestry
   of the 1/5 window (running); (b) acceptance rules frozen (onset, circle,
   rotation at prespecified detunings, ancestry through both folds, outcome
   categories); (c) bounded CIFAR profiling on the first subset; (d) RunPod
   Phase 0 only (at most $5: port, parity, timing); (e) freeze: claims,
   held-out configurations, gates, single primary endpoint, costed workload;
   (f) held-out existence-and-replication test: two prespecified independent
   losses across two architectures, one unused, no substitution; (g) real-data
   confirmation on the frozen second subset; (h) Phase 2 as a separate decision
   (no CNN for this paper).
3. **Diagnostic:** loss-spectrum readout and AR precursor remain exploratory and
   secondary until sampling, harmonic, abstention, AR(2)-baseline and
   specificity rules are frozen; described as local system identification
   unless prospective warning is shown.
4. **Deferred:** odd-window mechanism (kept as disclosed exceptions),
   nested-bifurcation expansion, production-resolution figures. The four-weight
   model stream already running finishes as exploratory side evidence.
5. **Added:** a small matched float32/float64 check at four regime points.
