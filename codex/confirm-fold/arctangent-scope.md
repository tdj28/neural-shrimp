# Pre-outcome distinction: discrete passage and continuous arctangent

13 October 2026 UTC. Independent scalar-only check requested by the coordinating agent before any Task 5 full-model passage outcome. This note changes neither the runtime nor the planned primary/secondary predictors. It uses the supplied qualified target-A coefficients, not producer trajectories:

- A = 7.054924948575865;
- B = 24.92734030789243;
- U = 0.1/B = 0.004011659437583009;
- epsilon = 3e-6.

The primary prediction counts the first strict exit above U of the discrete return map

    u_(n+1) = u_n + f(u_n),   f(u) = A epsilon + B u^2.

The separately reported arctangent is the continuous passage integral

    T_arc = integral_(u_in)^U du / f(u).

They need not agree to one iterate, even before comparison with the full language-model map. Independent direct scalar iteration and evaluation of the integral give:

| Entrance | Discrete first-exit N_quad | Unrounded T_arc | Integer counts satisfying discrete ±1 | Integer counts satisfying raw arctangent ±1 |
|---|---:|---:|---|---|
| Full, -U | 118 | 117.1154712342537 | 117, 118, 119 | 117, 118 |
| Midpoint, 0 | 61 | 58.55773561712685 | 60, 61, 62 | 58, 59 |

The midpoint acceptance sets are disjoint. Consequently no eventual integer full-model count can make *both* tests succeed on this row. This is a pre-outcome incompatibility between two accuracy criteria, not a failed or successful full-model experiment. The discrete primary remains a meaningful prospective test. Both original predictions, signed errors and separate verdicts must remain in the eventual report; joint success is already impossible for the full trial table. Do not silently round the arctangent or replace it with the discrete prediction.

## Leading discretization correction

The sign and form of the proposed correction are correct. To see it, temporarily write an explicit-Euler step as u_next=u+h f(u). Its leading modified differential equation is

    du/dt = f(u) - (h/2) f(u) f'(u) + O(h^2).

Expanding reciprocal speed, integrating and dividing elapsed time by h gives the formal step-count approximation

    N_cont,corr = (1/h) integral du/f(u)
                  + (1/2) log[f(U)/f(u_in)] + O(h).

At h=1 this supplies the suggested leading correction to T_arc. Equivalently, in a local Taylor expansion the extra derivative term cancels the leading defect of the continuous time coordinate under one discrete step. It is an asymptotic approximation where the map increment varies slowly; setting h=1 does not turn the remainder into a rigorous small-error bound. It also does not remove the integer first-exit convention and endpoint overshoot.

For the midpoint row the correction is

    0.5 log[(A epsilon + B U^2)/(A epsilon)]
    = 1.4967252323071203,

so the corrected continuous estimate is 60.05446084943397, about 0.94554 below the discrete first-exit count 61. For the symmetric full channel, f(-U)=f(U), so the leading logarithmic correction cancels exactly. This explains why the bare arctangent can agree closely over the full channel yet miss the midpoint count by more than one step. It does not establish a uniform error guarantee at these parameters or for target B.

## Reporting boundary

This is a mathematical/numerical audit of the two planned scalar predictors, conditional on the supplied coefficients. No new fitted coefficient, full-model observation or threshold selection was used. The corrected formula is an explanatory diagnostic only; it is not substituted for the planned raw-arctangent secondary endpoint. Later use as a tested predictor would require its own explicitly recorded prospective status.
