# Independent source review of the curvature helper

**Pass for the supplied-array helper and synthetic scope.** Reviewed source
`5ae73c76180f9cbb901812d9ec27a959389eeb04` implements the corrected design
`403aff7e76e747b7968c402aeaac4e425f21e59b` without a blocking mathematical or
accounting defect. This does not authorize empirical acquisition or establish
that the six-reference approximation succeeds. No empirical arrays, target
outputs, model, acquisition or numerical profile were opened or run.

The reviewer wrote the earlier empirical design/input inventory but did not
write this helper. Eighteen separately written analytic/adversarial cases plus
the thirteen author cases passed: **31 tests in 0.17s**, CPU/BLAS 1, nice 10.
This test duration is not a workload resource profile. Producer files were
not modified. The source and test identities are bound in the companion
[curvature-mechanism-helper-independent-review-receipt.json](curvature-mechanism-helper-independent-review-receipt.json).

The full model spectrum preserves the selected-plane roots and every bulk
multiplicity. Independent characteristic-moment checks include dimensions 2,
5, 6 and 7, negative overlap, orthogonal and aligned limits. The top curvature
is the largest algebraic mode, rather than largest magnitude; the trace bulk
convention is fixed. Mixed, nonconjugate, nearly real and second/third cutoff
ties cannot yield positive attribution. The aligned tolerance deliberately
uses the stated one-dimensional limiting convention and abstains from complex
attribution; it is not exact collinearity of all supplied floating vectors.
`leading_pair.qualified` denotes the real/conjugate-pair class; the separate
cutoff-tie flag is also checked by `analyze_curvature`. A future caller must
retain both decisions.

The product-error expression follows directly by expanding
`(S_b^0+E_b)(S_a^0+E_a)` and using each rank-one factor's exact real-arithmetic
operator-norm formula plus Frobenius upper bounds for residuals. Tests check
that expression against a separately perturbed full product. Nonuniform-rate
whitening is `P^(-1/2) M P^(1/2)`. When external native matrices are supplied,
the first native matrix is the target and its formation mismatch is included
in the error diagnostic; the second mismatch independently prevents a positive
result. The tested small mismatch inside the agreement tolerance still enters
the radius. Without native matrices, scope is expressly the supplied symmetric
Hessian-pair product, not an actual closed-orbit monodromy.

The residual-aware expression
`rho_num = ||M V − V Lambda||_F/sigma_min(V) + kappa_2(V)*delta_op_total`
is consistent with the diagonal-basis identity in the design. An independent
analytic nonnormal 2D basis checks the singular values, residual contribution
and radius. The left residual correctly uses the adjoint and conjugate
multiplier. The helper explicitly performs two Hermitian eigensystems, one
full left/right general eigensystem and one complete right-basis SVD; a test
counts those calls while forbidding hidden norm/condition/inverse/solve APIs.
Returned decomposition evidence survives subsequent rejection, and an
injected SVD failure remains an unresolved attempted operation.

The helper does not upgrade floating singular values to validated invertibility,
spectral residuals to eigenvalue-count theorems or its numerical radius to an
interval enclosure. Orbit closure, full model/control/data identity, original
and opposite native derivative acquisition, the six-slot aggregate decision,
file durability and measured resource gates remain a future caller's work.
Controls hold strengths fixed while changing orientation; their better error
is correctly labeled an oracle comparator. Complex modulus depends on the
strength determinant, so orientation receives no modulus or NS-crossing credit.
No novelty, training, inference, LLM utility or empirical accuracy claim is
supported by this source-only pass.
