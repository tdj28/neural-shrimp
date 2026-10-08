# S1R lower-gap independent saved-outcome audit

2026-10-08 UTC. **Pass for saved arithmetic and provenance.** The one authorized
execution used frozen auditor `80972cb9e0311ddd07d54cdd95d3cd0dfced38d1`, reviewed
at `0e24ae8cc1f3ec4bb94e4d81e0a60716067b3e7b`, under authorization
`c7be8b65445bd6de0091ec7107210556907cd514`. The prospective source and synthetic
controls were frozen before the producer outcomes. Full identities and retained
summaries are in [the companion receipt](s1r-lower-gap-independent-outcome-receipt.json).

The audit validated all six fixed states, 24 retained full decompositions and
161 journal events, with exact 12 map/12 Jacobian/24 SVD attempted and returned
producer counts. It independently assembled the map derivative and parameter
chain rule from retained gradients/Hessians, checked saved bases by residuals
and orthogonality, and reproduced scaling, native phases, secants, tangent
orientation, saved linear residual and all 13 historical trial decisions.
There were no partial roles or counter/checkpoint discrepancies.

The retained Newton direction has relative true forcing `3.1382113e-9` and
Frobenius normwise backward error `1.6312269e-16`, passing the declared linear
equation gates. Yet every saved nonlinear trial fails strict infinity-norm
residual decrease: the smallest step, alpha `2^-12`, still has residual ratio
`1.3377828`. Thus this fixed failure is not explained by a materially inaccurate
solution of its saved linear equation. It does not establish that a smaller
step, different continuation method or different arithmetic would succeed.

Every checked square and augmented matrix is numerically full row rank under
the fixed thresholds. At the failed state, sigma_min(A) is `9.7972314e-8` raw
and `3.3126612e-8` scaled, with condition numbers `1.4598176e7` and `7.5129847e7`.
The augmented matrix is better conditioned in each corresponding metric.
This supports diagnosing a poorly conditioned local fixed-rate correction;
it does not establish a fold or absence of a branch.

The raw and scaled secant conventions orient the failed-state null direction
oppositely: its converted h component is approximately `+2.1483283e-6` raw
versus `-2.1483283e-6` scaled. The corresponding secant cosines are `0.90865`
and `0.022083`. Consequently a sign-turn narrative at this nonclosed state is
not robust to the declared metric. Its tangent has only residual-level-set
scope. The two qualified endpoint pairs pass their saved numerical agreement
checks; phase-transport cosines are one to floating-point precision. None of
these checks proves interval continuation or ancestry.

The audit took `0.721185s` internally and `0.958419s` in the launcher, exit0,
using CPU/BLAS1 and nice10. Its seven raw and six launcher files total
`1,858,251` bytes, below8MiB. Both full manifests and read-only modes were checked;
launcher/worker/timeout PIDs `277738`, `277825`, `277824` are absent. No model,
derivative, SVD, solve, correction or trajectory was rerun, and there was no retry.

Raw audit manifest: `b26b314eec9134f4746fdd85d20dc8cbe3f34ef9b116c9e5e08c6725a88c6ce4`.
Launcher manifest: `eb18510443935eb51f7f642120327fc1ecf82bc71c7e2e82eaa706a030ad88ff`.

The analytic derivative primitive and returned arrays remain shared inputs,
not independently regenerated derivatives. The source reviewer authored the
producer, an explicitly disclosed overlap; the saved-arithmetic implementation
is separate and root independently reviewed the producer. This is a numerical
arithmetic/provenance audit, not a theorem, utility result or authorization for
further target work.
