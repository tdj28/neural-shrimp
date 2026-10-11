# EXP-023: fixed intermediate parameters for cycle-reference matching

21 September 2026. Exploratory reference-only pilot. Draft pending complete executable plan, tests, local review and private source freeze. No new target integration may precede that freeze.

**Human-friendly TLDR:** The near-edge network retains some inputs longer than the core network. To test whether ordinary contraction and input visibility can explain that difference, we need comparable cycles. The historical samples skipped the required core contraction levels. This pilot tests two fixed intermediate parameter settings, then checks whether they actually provide qualified, comparable references. It does not yet measure new memory performance.

## Why this is a separate experiment

[EXP-022](EXP-022-results.md) closed at input preparation: its historical selector supplied four of six slots. Both core targets were absent from the sampled metadata. Its original selection, missing slots and all unlaunched jobs remain unchanged. Adjacent nodes in its initial attracting prefix bracket both targets, providing a concrete prospectively bounded successor. This design was chosen after learning that availability result, using old unforced metadata only; it is not an independent confirmation of historical coverage.

Keep the two EXP-018 canonical anchors and the two selected EXP-006 near-edge seeds unchanged. Introduce exactly two new core parameter settings at alpha=1.064. For target r, adjacent historical multipliers rL/rR and gammas gL/gR, compute `q=(r-rL)/(rR-rL)` and `gamma=gL+q*(gR-gL)` in the recorded binary64 evaluation order. Bind both original node records and their first-maximum states. Choose the nearest endpoint in q, with a tie resolved by lower original node index, as the provisional starting state and period.

| Core target | Historical bracket | Fixed gamma | Initial endpoint |
|---|---|---:|---:|
| 0.25 | nodes 14–15 | 10.314407775231995 | 14 |
| 0.50 | nodes 15–16 | 10.319667049937767 | 16 |

These are newly proposed parameter values, not historical certified cycles. Neither interpolation nor a successful seed correction proves continuous branch membership or a new qualified attractor. No target-dependent parameter adjustment, fallback endpoint, bin widening, additional search, selective retry or memory-guided replacement is permitted. A missed bin remains an outcome.

## Two bounded seed bridges

Changing gamma moves the cycle, so the old 1e-6 same-parameter refinement neighborhood cannot honestly serve as a parameter-change bound. Each new core setting first receives one fixed DOP853 seed-bridge job. Reuse the full-state closure plus third-neuron maximum-section four-variable Newton equations and tight forward integrator: rtol=1e-13, atol=1e-15, maximum step T/16384. Use at most six Newton updates, fixed damping factors 1, 1/2, 1/4, 1/8 and Armijo coefficient 1e-4, retaining all 25 possible trial slots including failed, rejected and unneeded rows. No extra event integration is added.

Center the bridge neighborhood on its fixed nearest endpoint x0,T0. Its state scales are max(1,abs(x0)) coordinatewise. The frozen state radius is `2*max(abs(xR-xL)/scales)+1e-6`; the frozen relative period radius is `2*abs(TR-TL)/T0+1e-6`. These engineering envelopes derive only from old adjacent endpoints. They limit correction displacement; they are not rigorous branch certificates. Reject out-of-neighborhood trial states before integration. Preserve the condition-number limit 1e10, relative closure limit 1e-12 and absolute section residual limit 1e-12.

A completed, converged bridge remains an unqualified provisional seed. Seal and reopen its exact state, period, fixed parameters, source trial and source hashes before dependent work. A failed or nonconverged bridge blocks that cycle's subsequent jobs without changing its planned slot. The subsequent two-method tight refinement uses the original 1e-6 state and period neighborhoods around this new same-parameter provisional seed. Historical seeds and bridge seeds must retain truthful, distinct provenance; neither enters as a newly qualified memory reference.

## Full fixed acquisition and evaluation

Use six cycle slots in the same order as EXP-022: two anchors, core-rho025, core-rho050, failed-near-edge-rho025, failed-near-edge-rho050. Reconstruct both anchors as well. Use exactly eight fresh common phase fractions from `Generator(PCG64(2026091702)).random(8)`, recorded in the machine plan with split `fresh`. Preserve the inherited third-neuron maximum convention and nine-maximum primitive identity. Equal fractions across cycles define a measurement convention, not functional phase identity or independent networks.

After the two bridges, the panel requires 12 tight shooting jobs, six paired reference jobs and 96 direct tangent jobs: **116 top-level jobs** total. Their maximum nested census is **50 bridge trials + 300 tight shooting trials + 24 cross-verification forward integrations + 12 adjoints + 192 direct tangent segments = 578 IVP slots**. Preserve all 48 phase contexts, both planned matching pairs, all 15 cycle-pair similarity checks, every failed or blocked slot and every unneeded trial. Never replace the full panel with its successful subset.

Retain the tight DOP853/RK45 methods and thresholds used by EXP-018/019. Verify full-state closure, maximum section, primitive nine-peak identity, four candidate/method forward comparisons, neutral mode and transverse stability. Each cycle has one globally normalized adjoint; individual phases are evaluated from it without renormalization. For each qualified phase start, integrate state and all nine tangent entries directly over 0–T and T–8T with an identity tangent reset at T, under both methods. Do not substitute repeated-period composition for direct propagation.

The inherited reference gates remain separate from saved-arithmetic replay tolerance: adjoint normalization 1e-7, state 1e-6, activation 1e-7, standardized margin 0.01 and accuracy 0.001. Reference Gaussian linear accuracies may be calculated for those compatibility checks. There are no finite input trajectories or new mixture-decoder runs. Dose and observation noise retain numeric value 1e-4 in their different state and activation-observation units.

Save both signed/complex transverse multipliers and log-modulus rates per physical model time only when resolved. Match the two methods' two transverse modes by minimum total displacement with a lexicographic permutation tie. The radius of each paired mode is the maximum of its method displacement and the two arithmetic floors `64*eps*max(1, spectral matrix norm)*max(1, eigenvector condition)`. A nonfinite floor is unresolved. A resolved mode has both moduli greater than its radius. A positive real mode must have zero imaginary part and real part greater than the radius in both methods. Unique dominance requires separation of the two modulus discs. This sensitivity diagnostic is heuristic, not a rigorous interval or replacement stability gate. An unresolved tiny secondary mode remains recorded without an invented logarithm.

Save phase-dependent immediate detectability, first/second observable response vectors A/B, Gram matrices at immediate/2T/4T/8T, state and activation operator gains, and descriptive mode/write/read orientation. Modal coordinates are basis decompositions, not orthogonal energies or separate attractors; preserve their normalization, conditioning, reconstruction residuals and any failure. They add no qualification gate.

The two primary pairs are the cross-path candidates in the same target bin. Both refined dominant modes must remain uniquely resolved and positive. For each mode, both observed method distances from the target plus its descriptive radius must fit within the exact caliper 1/40, interpreting saved binary64 values as exact rationals for the containment comparison. No secondary-mode matching gate is added. Require relative period difference <=0.02 using the larger period denominator, and the difference of eight-phase RMS immediate detectabilities divided by max(left,right,1e-12) <=0.10. If both detectabilities are below 1e-6, matching fails as insufficient signal. These are fixed feasibility calipers, not practical effect thresholds.

For all 15 cycle pairs, retain normalized-time loop similarity using the 8192-node full-state loop, FFT cyclic alignment and exactly 64 golden-section steps within one grid interval using cubic Hermite state/physical-flow interpolation. Retain a 4096-grid diagnostic. No reversal or time warping is allowed. Scale each coordinate by max(1,maxabs of both loops). Relative period difference <=1e-8 and aligned scaled-state RMS <=1e-8 define numerical near-identity. Same exact alpha/gamma plus near-identity fails distinctness; an unresolved same-parameter comparison also cannot certify distinctness. Cross-parameter near-identity is reported without alone rejecting the pair or proving orbit identity. Bind the outer descriptor parameters to the selected cycle to prevent misclassification.

Report ledger completion, numerical reference qualification and full matching feasibility separately. Full feasibility requires all six new references, all 48 two-method contexts, both fixed pairs and required distinctness checks to pass. A reference may fail a bin even if its underlying orbit is numerically stable; retain both facts.

## Resources, lifecycle and audit

Use four owned CPU workers with all six numerical thread variables set to one. Current reference kernels have no qualified GPU replacement. The EXP-018/019 historical proxy for the 114-job panel is 669.75 worker CPU seconds. For the two bridges, a deliberately conservative planning proxy charges all 50 maximum trial slots at the 8.2092-second maximum historical shooting-job wall time, adding 410.46 seconds; this is a cost bound for planning, not a measured bridge throughput or guarantee. The combined ideal four-worker proxy is about 270 seconds before sealing and analysis.

Freeze producer lifetime at 1800 seconds; bridge/shoot/reference per-job time at 180 seconds; tangent per-job time at 90 seconds; sealing at 180 seconds; owned RSS at 12 GiB; raw output at 8 GiB; launch available RAM at least 16 GiB, maintaining an 8 GiB reserve. Recheck CPU/RAM/GPU/disk availability at launch. Leave unrelated processes untouched and record exact ownership and cleanup. No paid compute or further Pro call is part of this study.

Bind exact historical bytes, prepared selection, gap-bracket metadata, all source files, packages, protocol and machine plan. Require a clean committed checkout and verified matching private branch HEAD before any bridge or neural reference integration. Seal and reopen each prerequisite before dependent work. Save immutable raw evidence exclusively under `/data2/neural-shrimp-data/raw/EXP-023/attempt-001/`, with complete worker and nested ledgers and source-validity postflight. Never overwrite earlier raw evidence. Verify all owned processes terminated.

Saved replay recomputes packet bindings, physical RHS/tangent identities, correction arithmetic, scientific gates, descriptors, comparisons and summary arithmetic without new neural solves or filling absent outcomes. Separate local audit checks physical associations and summary arithmetic; disclose overlapping authorship and shared inputs. Preserve all failed tests and attempts. Push compact results and receipts after review; Git does not back up the external raw arrays.

## Interpretation and next boundary

This can establish comparators and numerical feasibility within two related gamma paths. It cannot establish a new memory effect, independent network replication, complete shrimp boundaries, a geometry-specific benefit or trained-network transfer. Matching full A/B Gram geometry and noise fixes the reference mixture prediction, so geometry cannot add information to that same mathematical quantity. Later finite responses would test model mismatch, while a geometry robustness question needs independently measured boundaries and a fixed parameter-perturbation design.

The proposed later 972-trace identity-only finite-write study requires a separate complete prospective freeze. Failure here remains a result and does not authorize an expanding search for successful memory.
