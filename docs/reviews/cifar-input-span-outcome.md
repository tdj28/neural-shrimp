# CIFAR input-span diagnostic — reduction valid, residual hypothesis rejected

8 October 2026 UTC. Independent saved-outcome audit pending.

**TLDR:** A large set of weights has analytically simple decay on this fixed
training batch, but it accounts for almost none of the failed orbit correction's
residual. The proposed explanation of the solver stall therefore fails this
check. We have no new training, prediction or inference benefit and no accepted
CIFAR orbit.

For first-layer W=QB+N with Q orthonormal, range(X^T) contained in range(Q) and
Q^T N=0, plain full-batch GD with isotropic ridge lambda/2||theta||² splits as
(z,N) -> (G(z),qN), q=1-eta*lambda. Period-p residuals and their derivatives in
the complementary directions are (q^p-1)N and (q^p-1)v_perp. If q^p!=1, a
periodic orbit has N=0; its primitive period is preserved and its monodromy is
the active monodromy plus q^p I. Attraction also requires |q^p|<1. This is an
elementary exact-coordinate derivation, not a novelty or interval claim.

Retaining all 512 thin-QR columns, including any rank-deficient completion,
gives 16,746 active coordinates and 81,920 complementary directions in the
98,666-parameter model. The numerical basis, packing and all saved-vector
identities pass. The original complementary squared-residual fraction is
1.7489397669e-8; the true linear-residual complementary fraction is
6.3553641512e-9. The preset >=0.5 priority criterion is false. These directions
do not account for most of the saved residual. Other weak directions inside
the active map and other numerical causes remain untested.

The fixed decomposition used five saved state/residual pairs and p/Ap/e only
at their original common basepoint. It evaluated no new model map, gradient,
Jacobian/Hessian product, correction or spectrum. Collection took 1.107 seconds
internally, 1.410 including the wrapper; profile, collection and both immutable
launcher inventories total 40,829,832 bytes. All source/input/manifest hashes
passed. No GPU, paid compute or protected data was used.

The coordinate reduction could simplify a future implementation, but speed
and convergence were not measured here and no follow-on solver is authorized
by this result. Deleting N may change predictions outside the fixed input span;
this is not proposed as a training or inference optimization. Independent
outcome review is being prepared before numerical target reads.
