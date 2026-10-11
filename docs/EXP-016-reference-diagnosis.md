# EXP-016 reference diagnosis — post-outcome saved evidence

**Human-friendly TLDR:** This attempt does not yet establish whether lingering activity in the near-edge neural cycle improves recovery of an earlier input after interference. The acquired histories agreed numerically where they ran, but 11 phase contexts were blocked because the phase coordinate did not normalize accurately enough at the intervention time. This is a reference-coordinate reliability problem, not evidence that the network lost memory or that the proposed intervention has no effect. Both selected settings remain unqualified for their full-cohort mean.

The audit used saved evidence only: no producer imports, new ODE, parameter fitting, repaired adjoint, altered threshold, or replacement outcome. It independently decoded numeric packets, checked 296 input-file bindings and all 43 frozen source bindings, and reconstructed physical RHS, reference differences, and normalization. It passed 14,981 checks in 3.065 seconds. Scope is reference diagnosis, not an independent replay of all finite-pulse scores or the entire raw inventory.

## Exact failure census

There are **11 failed contexts out of 48**, not 12: four core and seven near-edge. All 96 reference integrations completed. Every context completed its saved reference evaluation; the only failed gate is `normalization`. Core phases 0, 3, 9, 10 fail; near-edge phases 4, 7, 11, 14, 15, 16, 22 fail. All 48 initial checks pass. Every failure occurs in pairing the nominal periodic adjoint with the physical flow at the independently integrated one-period event state.

| Context | Initial signed z·f−1 | DOP853 event | RK45 event | Maximum / 1e−7 | Blocked traces |
|---|---:|---:|---:|---:|---:|
| core--p00 | 2.34991097692e-08 | 1.58216382662e-07 | 1.5947333365e-07 | 1.5947333 | 81 |
| core--p03 | 1.28282784395e-08 | 1.22992946494e-07 | 1.24013073588e-07 | 1.2401307 | 54 |
| core--p09 | 3.66813348585e-08 | 2.88948648119e-07 | 2.90272308634e-07 | 2.9027231 | 54 |
| core--p10 | 2.6128035957e-08 | 2.44046097775e-07 | 2.45381164499e-07 | 2.4538116 | 54 |
| failed-near-edge--p04 | 6.00056893063e-08 | 1.03898037951e-07 | 1.04862286632e-07 | 1.0486229 | 54 |
| failed-near-edge--p07 | 6.40326778267e-08 | 1.33332021424e-07 | 1.34279669606e-07 | 1.3427967 | 54 |
| failed-near-edge--p11 | 5.79764030118e-08 | 5.93087990941e-07 | 5.94432278511e-07 | 5.9443228 | 54 |
| failed-near-edge--p14 | 6.31737118173e-08 | 1.07278315564e-07 | 1.08129719845e-07 | 1.0812972 | 54 |
| failed-near-edge--p15 | 6.78832661194e-08 | 1.13027502469e-07 | 1.13629305076e-07 | 1.1362931 | 54 |
| failed-near-edge--p16 | 7.70536030537e-08 | 2.57580395235e-07 | 2.58900296757e-07 | 2.589003 | 54 |
| failed-near-edge--p22 | 6.32736563144e-08 | 1.07337409183e-07 | 1.08236151375e-07 | 1.0823615 | 54 |

The 621 blocked traces equal 11×54 coarse/fine histories plus the 27 independent traces at blocked core phase 0. Thus 2,079 of 2,700 traces completed: 675 development and 1,404 held out. There are 1,080 completed/qualified trajectory comparisons out of 1,404 in the saved summary. All 324 unavailable comparison rows remain. Twenty core and 17 near-edge phase contexts qualify; neither setting qualifies over its full 24-phase prerequisite census. The primary mean and bootstrap interval are correctly withheld. The retained primary contrast failure bound is −19.2052432114 to +24.5447567886 percentage points, including all 16 held-out phase blocks.

## What the arithmetic localizes

The gate is a direct three-component inner product, not a readout quadrature. Its declared statistic is the maximum of the initial residual and the two event residuals, compared with 1e−7. The current source implements that rule and the saved evidence reproduces it. No index mismatch, repaired vector, threshold mismatch, or implementation defect was found in this bounded review.

Both full reference algorithms agree well over all eight periods. Global maxima across both segments and all contexts are:

| Difference | Maximum | Frozen relevance limit |
|---|---:|---:|
| state | 3.65349173137e-09 | 1e−6 |
| activation | 2.25996871217e-09 | 1e−7 |
| pulse_scaled_phi_all_nine | 7.03774399868e-11 | 1e−6 |
| pulse_scaled_activation_phi_all_nine | 8.82084960629e-11 | 1e−7 |

Every state, activation, all-nine tangent, composed prediction, primary observer, margin and analytic accuracy gate passes. Raw full-tangent disagreement reaches 7.0377427619e−7, but its declared epsilon=1e−4 relevance-scaled difference is only 7.0377439987e−11. These are empirical method agreements, not certified global error bounds.

The largest DOP853-to-RK45 change in event normalization is 3.0455424849e−9. In contrast, the event-minus-initial normalization change reaches 2.5226731326e−7 in core and 5.3511158793e−7 near the edge. Therefore simply changing between the existing two reference integrators would not fix any of the 11 failures: both fail at the same contexts.

The inherited 4,096-interval orbit/adjoint packet passes its own saved-knot checks: maximum knot normalization about 1.39608645e−8 (core) and 6.04833588e−8 (edge), relative orbit closure 8.49391065e−11 and 5.61015382e−10, and relative adjoint periodicity 3.70560324e−9 and 2.41210925e−9. Independent SciPy cubic-Hermite evaluation at the 24 sampled phases reproduces the actual initial values and their passing normalization. New event states differ from their initial periodic-template states by up to 2.66019704e−8 (core) and 3.64533246e−8 (edge); the two solvers disagree at that event by only 2.50e−10 and 2.29e−10 respectively.

This localizes the issue to consistency of the inherited approximate periodic state/adjoint, an arbitrary-phase cubic-Hermite initial state, and the freshly integrated event state. Small state differences can matter in the oblique phase coordinate. It does **not yet identify how much comes from interpolation error versus inherited periodic-orbit/adjoint approximation**; the same saved evidence cannot supply an independent exact curve. It also does not establish intrinsically difficult neural dynamics or a scientific memory failure. Normalization at the old grid knots and at the new initial phases alone was insufficient to guarantee normalization at the later intervention state.

## Recommended next action

Freeze the small all-48-context reference reconstruction described in [the EXP-017 protocol](EXP-017-protocol.md) before any new solves. It compares the original interpolation against independently reconstructed dense state/adjoint samples, then tests one-period event consistency with two tighter method paths. Do not launch another 2,700-history pulse study until the fixed reference census qualifies; preserve this attempt and its 621 blocked histories permanently. A future reconstructed-reference attempt is a new exploratory attempt, never a retroactive pass of this one.

## Provenance

Original source: `9904368b7b13721292efdc84dcf31efd7add30c7`. Plan SHA256: `f9b9b2e5052bea79a4a4eae5273e7496468c7211fbf42dd5fcbf535e12ac2e71`. Original receipt SHA256: `5bb1ce4aa6e3934af04049853b621628ab9cf86e36a6b4091db75b01211f47ef`. Full per-context values, source/input bindings, and checks are retained at `/data2/neural-shrimp-data/audits/EXP-016/reference-diagnosis-001/`. This repository copy adapts only those location references. All original files were read-only.
