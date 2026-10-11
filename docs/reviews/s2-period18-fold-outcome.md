# Selected period-18 window: local fold test

**What we learned about training:** at one boundary of the strongest numerical
shrimp-anatomy candidate, the stable 18-update training cycle appears to end in
a simple saddle-node (fold). This is exploratory local numerical evidence;
there is no rigorous boundary or whole-shrimp proof and no utility measurement.

The target was fixed before new outcomes: S2-28011-zoom:587, EXP-028 row111,
the lowest-index historical +1 boundary ray (ray0). The new numerical code and
short run card were committed at ca439bebe3abe9ed197b180b017dcb93b20e699f.
The old classification served only to select a location. The new computation
solves the full system R=F^18(theta)-theta=0, (M-I)v=0 and ||v||=1.

All 25 fixed corrector points from the saved interior state qualified as
primitive attracting period-18 orbits numerically. One augmented root solve
converged in 42 function evaluations; its final infinity residual was 1.25e-10.
The point is at log10 learning rates (1.879464569373347,-0.08562528015073712).
The two checked adjacent native phases have closure residual below 4e-11, one simple
multiplier near +1 and all remaining multipliers of modulus below 0.806.
Proper divisors 1, 2, 3, 6, 9 are separated by at least 0.206 in the stated normalized
infinity metric.

For R(theta,s), with s along the saved scaled ray and ||v||2=1, normalize the
left null vector by w^T v=1. The phase 0 transversality a=w^T R_s is approximately
-0.50262214 and quadratic coefficient c=w^T R_theta_theta[v,v] approximately
-8.000116; the local parameter curvature -c/a is approximately -15.91676.
Five finite-difference steps decreasing by factors of four retain nonzero signs
and converging values. The next phase also passes, with different coefficient
magnitudes because its right null vector is normalized separately. This is a
local maximum in the selected ray coordinate under the fold interpretation,
not a global description of the window or proof of a connecting branch.

The raw outcome preserves every augmented evaluation and each corrector result.
Four synthetic controls check a simple fold, vanishing quadratic coefficient,
defective +1 mode and an inappropriate -1 mode. Runtime was 0.73s on one CPU
thread, with no GPU, paid service or protected holdout. All owned processes
exited and both immutable manifests pass. The compact source-bound numerical
receipt is `docs/reviews/s2-period18-fold-outcome.json`.

This fixed-step path and local proximity support numerical association with
the selected window; they do not enclose a continuous branch. Interval
certification of the augmented root and nondegeneracy, flip boundaries, cusps,
resonance ancestry and training benefits remain separate work. Independent
review is recorded in a companion rather than rewriting these raw outcomes.

The original execution records three core source hashes and the complete Git
commit. Additional transitive source hashes and post-execution runtime versions
are recorded separately in `s2-period18-fold-environment.json`; these are
explicitly later observations, not retroactively added raw runtime records.
