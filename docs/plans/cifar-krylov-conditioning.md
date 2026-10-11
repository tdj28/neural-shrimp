# Saved CIFAR MINRES conditioning diagnosis

11 October2026. Root authorizes a saved-only diagnostic after the four accepted
stationary-merit steps. No new HVP/gradient or correction, no automatic expanded
iteration budget, no old-state action transfer to the final point. The exact
historical operators are at states000/010/014/018; finalstate022 has no HVP.

For each, use all256 consecutive recorded unpreconditioned MINRES vectors/actions
from that same state, its exact initialgradient and separately acquired truep/Hp.
Reconstruct the original recorded SciPy three-term recurrence, retain allalphas,
betas and residuals. Check V'V at fixed16/32/64/128/256 prefixes before displaying
Ritz values. A qualified numerical projection requires Gram infinityerror<=1e-6
and reconstructed-vector error<=1e-8. These are diagnostic tolerances, not rigorous
full-Hessian spectrum/multiplicity/gap claims; unqualified spectra stay absent.

Solve fixed saved-action least residual min||g+Wc|| with SVD/rcond1e-12, never
normal equations. Separately fit actualp inV and compare its predictedWc with
fresh actualHp; both relativeerrors<=1e-6 qualify the representation. Preserve
singularvalues,coefficients,cancellation scales and full small Gram/projected
matrices. The action-span improvement is only a numerical prediction and would
need a new actualphysical HVP before any merittrial. Firstv=-g/||g||, if matched,
gives Hg=-||g||W0 for saved gradient-merit slope; it provides no currentstateHg.

Helper synthetic controls cover exact symmetricLanczos, lostorthogonality,
changed actualaction, suboptimalscaledcorrection and malformedinputs. Caller
must bind original complete source/inventories/state/action events and root's
independent accepted-step audit. No target caller/run card yet. Planned single
CPU1nice10 saved-only profile at full98666x256 before four historical operators,
with fixedwall/RSS/raw caps and all attempts retained. Freeze thresholds/scopes
above before examining actual projected outcomes; allfour cases are consumed
and correlated development evidence.
