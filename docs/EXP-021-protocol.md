# EXP-021: arithmetic correction for tiny-signal nominal quadrature

17 September 2026. Exploratory correction and saved-evidence regression. No fresh holdout, formal preregistration or new neural simulation. Complete code, source-bound synthetic numerical evidence, independent review and the private push must precede any new target calculation.

**TLDR:** The preceding analysis suggests that a nonlinear observer can recover an earlier input that a linear observer misses after a second input. Its complete-study result remains withheld because the numerical cross-check failed when the core cycle's remaining signal was very small. This successor fixes that arithmetic check while preserving the neural data, readout definitions, full cohort and scientific requirements. It asks the same behavioral question and does not assume the answer.

## Preserved failed attempt and reason for correction

EXP-020 ran from source `69ccc6e2c43236950dec246cfd3d66874bc89278` under plan SHA-256 `bf5c6a13973d6449ef3be4f992b66b8d14e940b4db5c453c08f69007f04b5712`. Its immutable receipt SHA-256 is `f54d3e16c528867bdca88131b53a650e0d6efbeca03b7d761bddf6949006e2e5`. All 144 cells and 4,704 mixture calculation slots returned records; 304 mixture calculations failed the unchanged nominal-quadrature requirements, leaving 25 cells incomplete. The complete-study primary remains withheld. All 1,728 auxiliary linear controls completed.

The failures occurred in tiny-signal core cases at 4T and 8T. Their Arb certificates were available, but the separate binary64 likelihood subtracted two nearly equal softplus values and divided their noisy difference by a small perpendicular response. The saved precision ladder cannot remedy this because that nominal callback rounded its coefficients and arithmetic to binary64 at each precision. No root, subdivision or probability-width failure was recorded in these unavailable slots.

This is an outcome-informed numerical correction. Known EXP-019 and EXP-020 outcomes are disclosed, every original artifact remains immutable, and no previous study is requalified. A new ID and exclusive raw attempt distinguish the correction. The metadata-only `docs/EXP-020-correction-history.json` binds the failure, source, plan, raw receipt/summary and audits without carrying accuracies, gains or finite vectors into classifier construction.

## Exact statistical scope, unchanged from EXP-020

The complete mathematical definition is the frozen [EXP-020 protocol](EXP-020-protocol.md), SHA-256 `91510a49ee7319c853ced785ebbbb6da9a49278fe079994506d065aa20a04d3d`. The new machine plan inherits its frozen scientific source files. The preserved choices are:

- Both selected fixed-weight three-neuron cycles, 24 reused paired write phases per setting, the identity arm, windows ending at 2T, 4T and 8T, and all four nonzero first/second input sign combinations.
- Input amplitude and observation-noise standard deviation both 1e-4. Observations comprise the same 32 by 3 activation samples, a known phase and a noiseless calibrated baseline. This is Gaussian measurement noise on deterministic saved means, not a noisy dynamical-system integration.
- First-bit classification with the second bit unknown and balanced. Neither the second-bit label nor the finite class mean determines the classifier. The template densities are equal-weight mixtures with means s1*A+B and s1*A-B.
- The primary DOP853 A/B/baseline recipe, a separately sealed RK45 A/B recipe with its own baseline and exact fixed-physical-observation translation, and the secondary mathematical one-mode relation A=rho*B. No selection of the better reference or decoder is allowed.
- Exact rational binary64 geometry and mean sums; no tolerance-based rank snap. All scalar branches, disjoint decision regions, roots, priors, ties, reference controls and observation vectors remain unchanged.
- The original fixed-linear score replay at rtol 1e-12 and atol 1e-14; conservative solver/reference probability differences at most .001; certified new probability widths at most 1e-4; and reference Bayes dominance checked only on the balanced four-class mean.
- The primary estimand is the equally weighted mean fine-method full-reference-mixture minus original linear first-bit accuracy at 4T across all 24 reused near-edge phases. Both settings must satisfy all required primary cells and numerical controls. No favorable complete-case average, new bootstrap or significance threshold is introduced.

The exact census remains 144 cell jobs, 432 classifier definitions, 1,200 finite class/method rows, 4,704 mixture probability slots, 1,728 auxiliary reference-linear CDFs and 6,432 new probability evaluations. There are 624 primary method comparisons, 624 secondary method comparisons, 576 alternate-reference comparisons and 300 method/context/window groups. Methods are numerical checks, not replications. Every failure or unavailable secondary remains present; secondary-only failures cannot rescue or invalidate an otherwise qualified primary by choosing another model.

## Changed numerical callback

The frozen exact geometry, coefficient balls, transition-root census, scalar implementation, Gaussian-tail treatment and adaptive Darboux enclosure are reused without mutation. The new pure module delegates classifier definitions and all scalar/linear probabilities to the frozen EXP-020 implementation. Its explicit adapter injects only the successor probability function into the unchanged evaluator. No monkeypatch of a frozen module is permitted.

For a rank-two calculation, SciPy still performs two QUADPACK integrations at epsabs 1e-9 and 1e-11, epsrel zero, and subdivision limits 512 and 1024. The difference is the conditional Gaussian CDF supplied at each quadrature node. The binary64 node itself has an exact rational meaning. The original Arb coefficient balls remain in the point calculation, rather than being rounded into double constants.

At t=u+mt, evaluate the direct likelihood as a ball:

    g(t) = 2*a*t + log(cosh(b*(t-a))) - log(cosh(b*(t+a))).

Its ball must overlap the independently written, cancellation-aware frozen likelihood expression. Then enclose Phi(s1*(mz+g(t)/(2*c))). Convert its midpoint to a bounded binary64 probability only when the outward bound on that float's absolute difference from the entire CDF ball is at most 1e-15. Record every run's point count and maximum error. A failed point check advances only through the existing 128, 256 and 512-bit precision ladder; exhaustion remains unavailable.

This bound concerns the conditional CDF at the evaluated nodes. The floating Gaussian weight and adaptive QUADPACK integration remain an independent nominal estimate, not a second certificate. The primary rigorous probability interval is still the unchanged Arb mass/enclosure calculation. Sharing the ball-arithmetic library is disclosed; the two integration algorithms differ, while synthetic direct-density checks and source review provide additional validation of the likelihood itself.

The nominal no-warning, convergence, reported-error and between-run agreement gates remain unchanged. Both nominal runs must have reported error at most 1e-8 and agree within 1e-8. The final raw nominal value must pass the original certified-interval containment check using only its stated QUADPACK error allowance. No additional point-error allowance is added to rescue a failed score. The point bound is four orders below the tighter requested absolute quadrature tolerance. Display-only clipping to [0,1] continues to record and gate the raw nominal value.

### Nominal partition amendment from synthetic testing

Before any EXP-021 target calculation, the point-arithmetic-only prototype failed all four sign cases of a deliberately narrow synthetic transition at c/a=.01. Its two nominal tolerances disagreed by about 2.6e-8 despite the unchanged certificate being available. This synthetic failure and a clearly labeled reconstruction of the earlier prototype are retained in the external design packet. The unchanged gates correctly rejected it.

The final nominal integration therefore adds conditional-CDF transition hints at levels -8 and +8, found on the same monotone likelihood segments, and fixed standard-normal coordinate splits at every integer from -9 through +9. Endpoints remain -10 and +10. Root brackets, unresolved extra hints and merged nominal split values are retained; hint merging within 1e-10 affects only quadrature subdivision. No probability mass or tail is discarded. The certificate's original partition and algorithm are unchanged. A successful successor can stop at 128 bits where the earlier failure exhausted 512 bits; therefore certificate endpoints need not be bit-identical across different ladder stopping points. Comparisons of the unchanged algorithm at equal precision remain valid.

## Synthetic validation and measured limits before new scores

Synthetic tests cover tiny signal scales across orders of magnitude, b near one with leading-term cancellation, extremely separated responses with tiny transverse components, ordinary smooth cases, exact scalar/linear delegation, both signs, physically translated mean sums, partial precision failure, point-error exhaustion, nominal gate failure and preservation of the frozen module. Direct four-density likelihood integration supplies a separate moderate-scale probability comparison. A full synthetic analytic-oscillator packet pipeline verifies the explicit adapter, all 40 panel calculation slots, the alternate reference, original linear replay and serialization.

The final source-bound synthetic benchmark qualified all 48 calculations in 3.8441 seconds, with maximum 0.81001 seconds, median 0.03080 seconds and peak RSS 100,552,704 bytes. The benchmark used all six numerical thread controls set to one, retained its input/output and package versions, and performed no target reads. It supports retaining the inherited limits without promising target completion. The limits are four owned single-thread CPU workers, 30 seconds per cell, 900 seconds producer time plus 120 seconds sealing, 8 GiB owned RSS and 512 MiB raw output. Launch requires 16 GiB available RAM; retain an 8 GiB reserve. These limits remain unchanged in the machine plan. The narrow, adaptive arbitrary-precision scalar workload uses four CPU workers; no validated GPU implementation exists for this decoder. This decision is distinct from the earlier ODE GPU benchmark. No target scores may be used for runtime tuning or sample selection.

## Lifecycle and interpretation

EXP-019 remains the qualified source of all neural evidence. Its complete receipt and all 299 original historical bindings are retained. Add the EXP-020 raw receipt, raw summary hashed without decoding its performance outcomes, and metadata-only correction certificate for 302 historical file bindings. Certificate-linked audits are also hash checked. Preparation and classifier construction cannot parse old finite outcomes. Persist and reload all 432 definitions and a global barrier before loading the finite displacement vectors, as in the original run.

Use the same ownership-aware worker lifecycle, complete ledger, partial calculation journals, exact source/private-remote guard and source-validity postflight. Preserve every old source file. Verification reconstructs saved definitions and existing calculation records and does not fill missing outcomes. Any failure of this correction remains an additional immutable attempt rather than permission to relax gates or rerun selected phases.

Recovery by this reference-defined nonlinear observer would show accessible first-bit information missed by the original linear readout under the specified measurement model. It would not show that information was created, prove optimality under the actual nonlinear dynamics, establish joint two-bit capacity, or demonstrate that shrimp geometry causes the benefit. Agreement with the scalar one-mode comparator is a mechanistic control consistent with generic overlapping amplitude signals. These two fixed-weight cycles do not establish a result for trained neural networks.

The final numerical core SHA-256 is `85b6a514c8fc57775d8a1f72dc70102e4c0920609b3cd4485121283220f10301`. The source-bound synthetic cost receipt SHA-256 is `a208c403f9861e82b3ec23baa3f5d81d671664a5b9e470f90b7f10a6df65b39c`; the independent numerical review SHA-256 is `0ea0e4f6e344fe78e349f6c7f777da798dbd6d049cf24d78d6eccb6acfd881fc`. The review passed 20 synthetic tests, including 126 separately derived density-ratio point comparisons and three separately partitioned whole-integral comparisons. Its author did not write the new core, but wrote the inherited EXP-020 evaluator; this is disclosed source-level independence, not independent replication of the neural data.

The source-bound local review records the final unified tests, literal plan audit, shared-resource inventory and authorship. Commit and verify the private push of all 75 source bindings, machine plan and review before execution. The separate design packet, failed-prototype reproduction and cost/review evidence remain under `/data2/neural-shrimp-data/audits/EXP-021/`.
