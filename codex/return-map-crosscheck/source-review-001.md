# Task 19 independent reducer: narrow source review

Reviewed the prepared files under `/tmp/task19-prepared/codex/return-map-crosscheck/` and the corresponding synthetic test source. No target data, producer analysis, model, or tests were executed in this review.

Accepted scientific source at this check:

- reduce.py: `d87ec765cde3c0cc1145e6225b9b98b9a9c85468df85368c938c571aebda098b`
- trajectory.py: `8da808f5413d4f7caca49b6c0cba4c8182ff4322c91e123513e9ccb3e9737524`
- runtime.py: `8913d1adc576ecc5df3ef9cd4dccea67c5627bb3b99ce82b736afedf04403b06`

The column-scaled SVD solve, power/Chebyshev derivative scaling, historical degree from the phase-0 observation count, amended degree cap, seeded random fivefold split, equal weighting of fold MSEs, adjacent-pair purging, and per-phase aggregation follow the declared formulas. Selection does not depend on the full twin exponent. Historical residual qualification retains rank warnings; the stronger full-rank qualification is separate for the own purged method. The trajectory retains absolute-offset twins and pairs each output state with the probability read from its own input, matching the producer clock. Fixed phase 6 is explicitly qualified only when the observed symbolic period is seven. Sample-band distances use actual cross-phase state differences.

One concrete producer-contract discrepancy was found jointly with Root: producer Chebyshev roots are converted to physical probability coordinates before its imaginary-part tolerance, whereas original power-fit roots use normalized coordinates. The corrected source now retains a separate `producer_stationary_count` with the literal corresponding rule and uses it for amended producer comparisons. The own normalized-root and derivative-sign/curvature maximum diagnostics remain distinct. A focused synthetic case with complex normalized roots inside the physical tolerance is present in the test source.

No additional mathematical blocker was found in this finite source check. This is source acceptance, not execution acceptance, scientific reproduction, or resource/launcher admission. Root handles those independently. The broader interpretation limits remain in the accepted source/claim review: fitted scalar prediction and exponent agreement do not establish smooth conjugacy, an invariant curve, or infinite-time chaos.
