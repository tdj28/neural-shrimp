# Task21e — source and queue scope, preparation only

The exact queue requests: “21e Cusps (CL, CL5, CL6): codim2/*-cusp-*/locate.json. Check each located cusp’s conditions and the claim that all nine reversals of the fold curves are located cusps.” The frozen excerpt is `/tmp/task21c-source-001/assignment-and-queue.md`, line 20, SHA256 `d2d33c2bf75ec9bea18ea79c3203c7439abf7af1c380d8a3d2f502f45562c959`. Inference also identifies the unchanged request at the owned worktree’s `comms/claude-to-codex.md:787`.

This note prepares a finite independent saved-scalar reconstruction. It does not authorize a run, read new cusp outcomes, or establish an interval certificate. The team is already exposed to the producer’s results and the accepted Task27 review; this is unblinded reproduction. Task21d’s completed-result review takes priority when released.

## Reusable accepted work

- Task14’s independent endpoint review is `/home/d7082791602/.codex/worktrees/llm-hostile-fold-crosscheck/neural-shrimp/codex/cusp-crosscheck/endpoints-001-independent-review.md`, SHA256 `7d71c93d809bb26914964b693c45bb3b01b2d55a22d5c4fbe5d20cf9108570e4`. Its full-period, full-state endpoint checks support a candidate cusp segment. Opposite endpoint quadratic signs and nonzero endpoint cubic coefficients do not locate a cusp or establish rank-two unfolding. Reuse that accepted evidence without repeating its model or saved-array audit.
- The accepted Task27 source/claim review is `/tmp/task27-root-review.md`, SHA256 `fa898c7adfc1d392a80ef5d4ad1f91a502f7d0966b610c721180efdc0818be85`, with `/tmp/task27-theory-review.md`, SHA256 `cb9280cd8b4bc0cbdb6db262af14c842585c5741244f063adf35f7e38ddb44b9`. It distinguishes nine located cusp records from six other codimension-two/reversal dispositions. The later located records must not be dismissed using Task14’s earlier endpoint-only limitation. Conversely, fifteen total dispositions are not fifteen cusps.

## Frozen source contract to associate with the actual attempts

The already frozen Task27 source root is `/data2/neural-shrimp-data/codex/paper-review-task27/source-001`. `tools/cusp_locate.py` has SHA256 `a8e711ad9bf4530633b6c41b341b43680326ff6d869f859634e6ec2fae3957de`; `tools/bifurcation_curves.py` has SHA256 `d7b665f874aa9183e5dc62be79a9e0ba75051a30d20b9b7e926781e5e541fcf6`. These are source-review references, not a substitute for Inference’s original-attempt source association.

`cusp_locate.py:37–40,92–185` locates a sign change of `f = sign(a) * b` along a fold curve, retains successive brackets, and takes a final traced secant step. Its literal final checks are:

1. `abs(final.f) <= btol * initial_f_max`, using the actual recorded tolerance and original endpoint maximum;
2. `final.c != 0` and cubic plateau spread either absent (`None`) or strictly below 0.1;
3. `abs(det_J) > 0` and finite reported condition number;
4. `located` is the conjunction of those checks.

The location’s default small state-step ladder is `(1e-3, 3e-4, 1e-4)`. The final `fold_unfolding` call uses a different default ladder `(1e-2, 3e-3, 1e-3, 3e-4)` and parameter steps `(1e-4, 3e-5, 1e-5)`. Its returned `b` is the last cubic-table row’s quadratic coefficient. It is not the same estimate as the location’s `final.f` or necessarily the same small-step `b`. Keep their identities and any discrepancy explicit.

`bifurcation_curves.py:247–302` first selects the multiplier nearest +1 and refuses if its imaginary part or real displacement exceeds `unit_tol` (default `1e-6`). Its complementary-modulus refusal is the literal compound `nxt > 1 - separation*1e-3 and nxt >= 1 - 1e-3`, with default `separation=0.5`; do not replace this with the docstring’s informal description. It normalizes the critical vectors, forms the two-parameter unfolding matrix, chooses adjacent derivative rows by the first minimum absolute component-difference sum, and reports determinant, singular values and condition. The locator saves the cubic plateau spread but drops the full cubic and parameter-step tables.

## Proposed finite saved-data checks

After the exact immutable input packet and bounded source/card are accepted, independently reconstruct every available scalar check and retain every original row, refusal and mismatch. Check bracket sign/round ordering, the initial maximum and relative final `f`, literal check/located conjunction, final multiplier and complementary-modulus source criteria, 2×2 determinant arithmetic and saved singular-value/condition consistency. Reconstruct coefficient plateau selection only where its complete saved ladder exists; otherwise retain that field as inherited. Preserve actual parameter units and scaling from the associated source/arguments. Record signed differences as well as absolute differences, including tiny nonzero values.

The “all nine” claim needs an explicit one-to-one ledger from the nine selected fold-curve reversals to their located records, original curve/point or bracket identifiers, and final dispositions. A glob count or nine true booleans alone does not establish coverage. Other crossing/reversal tests remain separate. Report confirmed / discrepancy / cannot-check per claim, exact raw path, and suggested finite-numerical wording; do not invent new tolerances or a theorem-level pass.

## Metadata still needed from Inference

The genuinely outstanding items are the exact nine-record inventory and original curve/reversal selection; per-attempt source/arguments and dependency association; recorded schemas and availability of ladders; original receipt/completion/failure and termination basis; and current immutable input hashes. Preserve missing historical provenance explicitly. Full states, monodromies, eigenvectors or derivative tensors are not assumed to exist and are not prerequisites for completing this narrower scalar audit: their absence limits what can be independently checked. No new model, AD, dense eigenanalysis, source retuning or raw-array audit is proposed.
