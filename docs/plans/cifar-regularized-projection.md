# Same-current022 regularized projected correction — preparation

The independently saved current512 direction has accurate forcing .419 but
large physical norm. Actual gradient remainder scales almost quadratically
along that direction, and the accepted alpha1/64 realizes only32.27% of the
predicted merit improvement. The global trust cap permits alpha up to .18187;
nonlinearity, rather than trust alone, explains rejection at1/8,1/16 and1/32.

Propose a different direction in the SAME authenticated512-dimensional action
span at originalcurrent022. Let V be all actual basis vectors, W=HV their exact
saved actions, and g that original gradient. Numerically factor VᵀV=RᵀR and
SVD W R^-1, so the regularizer measures physical direction norm, not assumed
coefficient norm. Choose the largest positive ridge mu whose projected residual
norm is at most .49||g||, by at most64 fixed doublings and exactly64 bisections.
This solves the numerical minimum-physical-norm discrepancy proposal; no exact
optimization or interval theorem is claimed. Recompute Wc and Vc explicitly,
retain every bracket and bisection scalar, and verify shorter physical norm
than the unregularized projected least-squares direction. Full numerical action
rank and a well-conditioned actual basis metric are required.

The future caller must bind full unchanged current512002 and its independent
basis/event/acceptance audits, all512 same-state actions, original trust origin,
and native source/library identity. Profile the actual98666×512 whitening/SVD
before research vectors. A separately admitted candidate will require fresh
physical Hp with true forcing<=.5, negative gradient-merit slope, and original
5% global trust/1e-4 Armijo trial decisions. Retain all rejected trials and compare
first acceptance with the original alpha1/64 outcome at that SAME starting point.
Do not reuse current022 Hessian actions at the accepted final point. No new basis,
post-outcome ridge selection, or silent forcing relaxation is part of this plan.

This commit contains only the pure mathematical helper and three synthetic
controls (diagonal discrepancy/shorter norm, invariance under nonorthogonal basis
rescaling/mixing, unsupported span/rank/nonfinite rejection). No actual arrays,
profile, candidate or native call is admitted yet. Complete measured resource
plan, source-bound caller and independent review precede any target work.
