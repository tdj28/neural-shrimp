# Independent Task 5 outcome review

2026-10-13T02:25:15.513579+00:00. **Behavioral TLDR:** the independently computed fold geometry predicted all 24 observed escape times closely, including a new period-8 case, but the frozen strongest claim failed: one midpoint passage escaped two complete G steps sooner than the iterated quadratic predictor. The separate, predeclared unrounded arctangent comparison was within one step for all 24 cases. Keep both results: the latter does not rescue or replace the failed primary all-row criterion.

This is an independent audit of saved full-state trajectories and execution provenance, not a new model replay, a rigorous orbit certificate, a test of sampled text, or generalization to another checkpoint/prompt. No model was loaded or called; no test design, outcome or acceptance criterion was changed. The reviewer also previously reviewed the design and authored its offline audit, so the rerun below is not claimed as wholly independent verification software. A separate ad hoc reconstruction used NumPy einsum projections and freshly recomputed scalar recurrences without importing runtime or model code.

## Frozen design and provenance

- Freeze: `78ae193ee3fb4b418c6e29cf92dadde5815c2100`.
- Machine plan SHA-256: `3ac03b7510060f52fda73d6e832a98b8c6ef5600095197050e9b745d96c59810`, 15,338 bytes.
- Raw attempts: `/data2/neural-shrimp-data/codex/task5-fresh-prediction/passage-A-01/` and `passage-B-01/`.
- Reviewed saved audit: `/data2/neural-shrimp-data/codex/task5-fresh-prediction/passage-audit-v1.json`, SHA-256 `76217ba867aca5e43ef801dce45263a200a5c0854c2e3dd2b84f0d4e9d12eab6`.

Both launcher manifests bind the exact freeze, four threads, 18 GiB address space and a 21,600-second watchdog. Both child runs completed with exit zero, without timeout. Their retained launcher elapsed times were 154.79 seconds (A) and 127.22 seconds (B). All 22 receipt-listed artifacts per attempt, including source snapshots and all four trace groups, matched their hashes and byte sizes. Snapshot and Git-tree plan bytes match the reviewed machine plan; all eight bound Python source files match the freeze. A and B used the same frozen design.

Plan creation was 2026-10-13T02:08:31.955937+00:00. A's private-ancestry verification, runtime start and first-target-evaluation timestamps were respectively 02:12:22.178419, 02:12:23.072434 and 02:12:24.213298 UTC. B's were 02:12:38.280833, 02:12:39.455388 and 02:12:41.057102 UTC. Thus retained timestamps order plan creation, private-push verification, start and first evaluation correctly. This is source/receipt provenance, not an independently timestamped public registration.

The offline audit was rerun against the exact freeze: all 256 target checks for A, all 256 for B, and all 36 study checks passed. Integrity is true; primary all-row success is false; separate arctangent all-row success is true. Missing/censored/failed rows: zero.

## Complete results

G=F^2 for A and G=F^8 for B. Every tuple below is (full, midpoint, full+transverse); each row therefore retains three planned trials. Errors are measured minus predicted. No prediction was rounded or refitted.

| Target | Epsilon | Measured G | Frozen quadratic G | Quadratic signed errors | Arctangent signed errors |
|---|---:|---|---|---|---|
| A | 3e-6 | (118,59,117) | (118,61,118) | (0,-2,-1) | (0.884529,0.442264,-0.115471) |
| A | 1e-5 | (56,28,56) | (56,29,56) | (0,-1,0) | (0.023155,0.011578,0.023155) |
| A | 3e-5 | (26,13,26) | (26,14,26) | (0,-1,0) | (0.045362,0.022681,0.045362) |
| A | 1e-4 | (10,5,10) | (10,6,10) | (0,-1,0) | (0.255732,0.127866,0.255732) |
| B | 3e-6 | (26,13,25) | (26,14,26) | (0,-1,-1) | (0.596591,0.298296,-0.403409) |
| B | 1e-5 | (10,5,10) | (10,5,10) | (0,0,0) | (0.495108,0.247554,0.495108) |
| B | 3e-5 | (4,2,4) | (4,2,4) | (0,0,0) | (0.519691,0.259846,0.519691) |
| B | 1e-4 | (2,1,2) | (2,1,2) | (0,0,0) | (0.914383,0.457192,0.914383) |

The primary rule was within one G iterate for **every** row. A passes 11/12, B passes 12/12, total 23/24; the maximum absolute discrepancy is two. Report the conjunction as failed, not as a pass rate substituted for the original claim. The separate raw-arctangent criterion passes 24/24 with maximum absolute error 0.9143833486313906. Finite deterministic selected cases provide no population success probability or confidence interval.

## First-exit and single-failure checks

Every retained full state was reprojected into the frozen u coordinate and transverse norm. Every first strict crossing was reconstructed; the prior state was checked against the saved pre-crossing state, and the crossing state against its saved counterpart. Finite prefixes, masked NaN tails and G-to-F clocks agree. Across all rows the smallest bracketing margin was 8.289480355859339e-6. The largest discrepancy between this separate saved-state projection and the runtime coordinate was 9.107298248878237e-18. The audit's conservative projection comparison tolerances were 1.2903067555213144e-12 (A) and 3.119777319725607e-13 (B); every bracketing margin exceeded the appropriate tolerance. These comparisons concern projection of saved states, not a bound on accumulated model evaluation error.

For the only primary failure, **A-0-midpoint**, the first exit occurs at G=59, corresponding to 118 F steps. The frozen predictions are quadratic 61 and raw arctangent 58.55773561712685. Signed errors are -2 and +0.44226438287314807. At the last pre-exit iterate, U-u_58=2.4554333052021664e-4; at exit, u_59-U=1.588596061899237e-4. The crossing is therefore not ambiguous at the saved-state projection precision, and no off-by-one or integer-rounding change resolves the primary failure under the frozen definition.

The design review and `arctangent-scope.md`, committed in the freeze before outcomes, already showed that A-0-midpoint could not satisfy both within-one criteria: the quadratic acceptance set is {60,61,62}, while the arctangent set is {58,59}. Its observed 59 passes only the secondary. The first-order Euler correction in that earlier note compared the scalar quadratic recurrence with its integral; it did not predict or prove the full-map error and is not a post-outcome replacement predictor.

## Interpretation and next use

The result supports accurate finite-range forecasts across these selected fold/entrance cases, including P8, while falsifying the uniform within-one claim for the iterated quadratic predictor over the complete planned design. The transverse kick changed the smallest-epsilon exit by one G step for each target and left the other six paired counts unchanged; this concerns one frozen transverse direction per target, not all perturbations. B's largest offsets have only one to four G steps, so within-one accuracy there is a weak relative-error condition and supplies no asymptotic scaling result.

Do not reclassify the arctangent as the primary endpoint, exclude the midpoint, shrink the kick, or alter epsilon after seeing these data. A subsequent higher-order or coordinate-correction explanation would be exploratory on these trajectories and would need its own fresh predictive test for a new headline claim. The present saved-data audit establishes no specific cause for the quadratic discrepancy beyond excluding the checked counting, projection and provenance errors.
