# Lower S1R gap: the stored equation is accurate, finite steps fail

8 October 2026 UTC. **Independent saved-data audit pending at creation.** The
72-parameter neural-network branch still has an unresolved lower connection.
This diagnostic points toward a continuation/globalization question: the saved
Newton direction solves its local equation accurately, yet every retained trial
increases the original infinity-norm residual. It establishes neither a fold nor
absence of a connecting branch, and supplies no training advantage.

The single source-reviewed run completed all six fixed roles, twelve analytic
map/Jacobian pairs and twenty-four full SVDs. The last stalled state's true
relative forcing is3.1382112783e-9 and Frobenius normwise backward error is
1.6312269079e-16. Reconstructing the stored residual differs by1.3435318531e-15
in Euclidean norm, versus residual norm9.8311661468e-7. Thus both fixed equation
gates pass without nearzero or arithmetic-resolution abstention. This is an
equation check using the separately written analytic primitive, not proof of
forward accuracy of a direction in an ill-conditioned system.

All thirteen historical trials, from step1 through2^-12, fail strict decrease
in the original unscaled infinity norm. The smallest tested step still has
residual ratio1.3377827666. No trial was reevaluated, no smaller step was added,
and no correction was applied. Accurate local linearization does not guarantee
these finite steps succeed.

Every raw/scaled A and augmented [A,b] resolves full row rank under the fixed
numerical band. At qualified native phase0 endpoints, raw A's condition estimate
rises from343.835 at the predecessor to831.314 at the last accepted pair. Their
oriented original-coordinate h components are -0.0504643 and -0.0293377; both
remain nonzero. Phase1 gives -0.0512994 and -0.0299475. Saved phase transport
cosines agree with unity to floating-point precision. These are numerical local
branch tangents at qualified endpoints, not a fold enclosure.

At the terminal **nonclosed** failed state, raw A's condition estimate is
1.4598176373e7 and its smallest singular value9.7972313675e-8. Raw [A,b] has
condition1131.293; under the predeclared scaling these conditions become
7.5129846984e7 and1.8022439442e6. Therefore the conditioning improvement is
coordinate dependent. The original-coordinate null direction is nearly
horizontal in h, with |h component| about2.148328291e-6. Its secant-selected
orientation is positive in raw coordinates and negative in scaled coordinates:
the orientation metric changes. This is **not** evidence of a signed branch
turn, and a local residual-level-set direction at this failed state must not be
called an orbit-branch tangent. No rank-deficiency or fold gate passed because
there was no such scientific acceptance gate in this diagnostic.

Scientific time1.1480s, launcher1.6431s, observed child-group peak RSS91983872B.
All266 scientific files (4987618B) and five launcher files (65469B) are immutable
and full-hash verified; all three owned PIDs are absent. No GPU, paid compute,
holdout or new trajectory was used. Execution8143a886, source5cecd5f0 and
independent producer review38d7141a are bound in the companion receipt.

Stop for independent saved-data audit and a separate next-method decision.
Pseudo-arclength from the last qualified endpoint is a plausible candidate to
evaluate, but has not been designed, implemented or launched by this result.
No wider parameter step, tolerance relaxation, failed-state restart or extra
line-search trial is authorized here.
