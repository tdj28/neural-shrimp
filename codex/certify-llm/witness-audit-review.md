# Independent saved-witness rational checker review

13 October 2026 UTC. Reviewer: Codex `literature_cascades`.

**Verdict:** no unresolved blocker for the stated role: independently checking the exact rational recombination and strict comparisons of already authenticated conditional-gate witnesses. This is a useful additional arithmetic audit, **not** an independent reconstruction of every proof computation or a certificate finalizer. In particular, a passing return does not establish the correctness of supplied matrix-product or preconditioned-residual enclosures.

Reviewed source `witness_audit.py` SHA-256:
`48098fec18005f5d521dfe6685cf311532814263ceaa982c763aae864b505d90`.

Reviewed tests `tests/test_codex_llm_witness_audit.py` SHA-256:
`4a32ed7980e442c9f090cd17edafa5f29a206b630f0b887690c22f18f0a7a128`.

The full checker and tests were read against unchanged `shooting_gates.py`, `proof_gates.py` and `shooting_certificate.py`. Only tiny synthetic computations were executed, CPU1/nice10. No actual target, candidate, derivative-shard or gate-budget report was read or evaluated. No existing implementation, raw evidence or Git state was modified by this reviewer.

## Mathematical coverage

Original midpoint/radius pairs are decoded directly to standard-library `Fraction` endpoints. Integer-only pair validation rejects booleans and floating aliases, and negative radii are refused. No Arb decoder widening, float underflow or rounded display number enters these comparisons. Point-valued saved upper/lower witnesses must have exactly zero radius.

For the factored shooting root bound, every phase pair `(i,j)` is present exactly once. The checker re-evaluates the scalar upper-bound inequalities

`R_ij_bound >= P_i_bound * B_bound * Q_j_bound + segment_ij_bound`,

`perturbation_ij_bound >= R_ij_bound * deltaJ_j_bound`,

`K_ij_bound >= inverse_defect_i_bound + perturbation_ij_bound`.

The inverse-defect contribution must vanish outside column zero. The causal segment vanishes for `j >= i`; an adjacent empty segment has norm at least one. The identity prefix at `i=0` and identity suffix at `j=p-1` also have norms at least one. These last two checks were added in response to this review. They avoid accepting an obvious identity-norm understatement without performing a matrix multiplication.

Radius weighting has the correct direction and indexing: each saved block contribution is at least `K_ij_bound * r_j`. The complete phase sum bounds the sum of those contributions, and the phase contraction bound is at least that sum divided by **`r_i`**. Thus unequal radii are handled correctly. The saved correction's complete coordinate intervals supply an independently recomputed infinity norm; its saved norm cannot understate it. The residual intervals must contain the full externally admitted image-minus-next-center intervals, with cyclic phase order.

Each inclusion margin must be a lower bound on `r_i - image_delta_bound`, and each contraction margin a lower bound on `1-q_i`. Strict positivity is required; zero is failure. Allowing an upper bound to be looser, or a lower margin to be smaller, is mathematically conservative. The checker does not demand equality where the producer may have rounded outward. Flags are reconciled with these exact inequalities and with the full conjunction across phase rows. A correctly recorded failed sufficient condition remains a consistent failed result.

Primitive-period checking recomputes every proper divisor of the supplied period. For each divisor and coordinate, the saved gap cannot exceed the exact center separation minus the two nominal radii. The complete positive-gap witness list must match, rather than merely containing one claimed witness. This proves phase-box separation conditional on a shooting root; it does not assert that an entire phase box is forward invariant or that `F^d(X_0)` was enclosed.

For attraction, all phase Jacobian hashes are matched in the declared order `J_(p-1) ... J_0`. The full saved monodromy matrix is hashed using the producer's independent row/entry convention and tied to the final chain digest. The saved first-power intervals must contain the original monodromy intervals. Both powers 1 and 2, all rows and all coordinates are required, with unweighted norm weights exactly one. For each saved matrix row, the checker sums `max(abs(lower),abs(upper))` over every entry and requires the recorded row bound to dominate that exact sum. The aggregate norm and lower margin have the correct directions. Attraction uses **any** successful declared power, selecting the first, while the overall root/primitive/attraction combination uses **all** three gates.

## External bindings and remaining premises

The exact centers and positive radii are supplied externally, their shooting-domain digest is recomputed, and both the root and overall report must match it. The contract, its period, the enclosure/proposal identities, external image intervals and ordered external full-Jacobian hashes are compared against the saved report. Every partial-product hash now has valid lowercase SHA-256 syntax, including intermediate records. Full intermediate products are not present and are not reconstructed.

The caller must obtain **all** external arguments from one already authenticated admission chain, including the linkage between its combined enclosure digest, image records and individual Jacobian hashes. This function is not an artifact reader: it cannot independently establish that unrelated caller arguments describe one model or one previously executed calculation merely because some supplied strings agree. Receipt/source/plan/config/assets/backend/prefix authority, successful whole-box evaluator execution and original-domain coverage remain outside its scope.

Likewise the following are deliberately not recomputed:

- matrix-derived base factor bounds for signed prefixes, suffixes, inverse defect and `J-A`;
- nonempty segment products whose individual matrix norms are not retained as separate inputs;
- the preconditioned residual recurrence and its matrix-vector products;
- the ordered monodromy multiplication and the squared-monodromy multiplication.

The original gate-source review and authenticated producer execution must still support these values. A saved hash authenticates bytes, not the mathematical truth of a multiplication. The new checker catches arithmetic recombination errors **after** those source/execution premises; it does not remove them. It correctly returns `dense_matrix_products_recomputed:false`, `preconditioned_residual_products_recomputed:false`, `source_enclosure_hypotheses_established:false` and `certificate_passed:false`.

An explicit synthetic limitation control replaced a genuinely nonzero saved second-power matrix by zero, then made its scalar row/norm/margin fields coherently agree. This still passes the scalar checker, as expected from its documented non-replay scope. It demonstrates why neither this function's success nor self-consistent metadata may finalize a certificate. Actual use must authenticate the original complete report and retain the independently reviewed matrix-computation authority; otherwise this would be a false positive for a stronger claim the function does not make.

## Tests and review changes

I independently reran the final focused suite: **27 passed in 0.32 seconds**. Its fixtures exercise the actual unchanged tiny conditional-gate producer, including nonzero interval-Jacobian perturbations, strict boundary failure, missing/duplicate blocks, missing proper divisors, norm and margin understatement, corrupted bindings and inconsistent aggregate flags.

Two requested consistency additions are resolved in the reviewed source: known identity prefix/suffix norm bounds and syntactic validation of all partial-product digests. Root also added the explicit external-contract period check. The previous producer test's mistaken corruption of an already-zero block was corrected before this review; the current test changes a genuinely positive contribution. No gate criterion was weakened.

I additionally ran **16 independent tiny controls**, separately from the test module:

- periods 2, 3, 4 and 8 with unequal phase radii, noncommuting signed interval matrices and nonzero uncertain image residuals;
- a genuinely nonnormal case where the first-power infinity norm fails and the second-power norm succeeds;
- valid primitive-separation failure and valid root-inclusion failure, retained without inventing success;
- eight separate factor, weighting, norm or lower-margin corruptions, all refused;
- the explicit matrix-product non-replay limitation above.

All passed. The controls never used target data. They provide coverage beyond the original equal-radius diagonal fixture, without claiming a general theorem from examples.

Retained controls:

| File | SHA-256 |
| --- | --- |
| `/tmp/codex-witness-independent-controls.py` | `9ef96d6d2bf96da6aee4f6e55af1c68780eedb8e1346030da77d82fadacd46ad` |
| `/tmp/codex-witness-independent-controls.json` | `35714c88123531e42cce8e087080ed08dce79cb113ad26e2a896416b0953e2db` |

The final control JSON binds the reviewed checker hash. Archive these files if they are used as part of a durable final audit. No actual-size runtime, memory limit or successful target witness audit has been established here.
