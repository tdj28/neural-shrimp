# Utility design: equal pulse size and equal update length are different questions

8 October 2026 UTC. Data-free algebra and design decision after the four-gradient feasibility failure; no new run, arithmetic on target arrays, dose, control acquisition or outcome is authorized. Historical cards/receipts remain unchanged. This is elementary geometry, not a novelty claim.

The fixed R=0.05||Pg|| candidate is infeasible under its25percent block cap at all four inspected forks (`docs/reviews/utility-dose-independent-outcome.md`). Reducing R in a disclosed future design could restore algebraic feasibility, but would not supply the missing controls or solve a second interpretive issue. Equal treatment-minus-sham displacement does not equate the total GD update length. The current paper already states this distinction.

## Why both quantities cannot match a uniform reduction

At a fixed state let s=Pg be the sham descent increment: theta_sham=theta−s. Write a rate reduction's displacement relative to sham as delta, so theta_arm=theta−s+delta. The uniform fractional reduction kappa in(0,1) gives delta_u=kappa*s. Suppose another arm is required to satisfy both

    ||delta|| = kappa ||s||,
    ||s−delta|| = (1−kappa)||s||.

Expanding the second square gives s dot delta=kappa||s||². This attains equality in Cauchy–Schwarz, hence delta=kappa*s when s is nonzero. The uniform arm is the unique such vector. This holds for arbitrary parameter-space directions, not just two learning-rate groups. For orthogonal hidden/output components with positive norms H,O and delta=(a*s_h,b*s_o), the equivalent identity is

    H²(a−kappa)²+O²(b−kappa)²=0,

so a=b=kappa. If s=0 the intervention is degenerate; if a group norm vanishes, its numerical rate fraction is unidentifiable from displacement. Thus it is impossible to add a distinct direction arm that matches both quantities to uniform backoff. A failure to match both is not necessarily a coding error.

## A possible symmetric contrast, not selected for execution

Choose a nonzero v perpendicular to s and let delta_±=kappa*s±v. These two arms have equal pulse norms and equal total update lengths:

    ||delta_±||² = kappa²||s||²+||v||²,
    ||s−delta_±||² = (1−kappa)²||s||²+||v||².

For two nonzero blocks one option is

    v_h = tau*(O/H)*s_h,   v_o = −tau*(H/O)*s_o,

so ||v||=tau||s||. The two arms' block fractions are kappa±tau*O/H and kappa∓tau*H/O. To keep both arms between zero and a cap c, require kappa in[0,c] and

    0 <= tau <= min(kappa,c−kappa)*min(H/O,O/H).

Strong group imbalance therefore limits the admissible angular contrast. Any later choice of kappa/tau must be disclosed before pulse outcomes, retain every fork, and pass its own exact arithmetic/finite-value checks. No values are chosen here.

A uniform comparator with the same total update length would use kappa_match=1−sqrt((1−kappa)²+tau²), when this lies within the allowed nonnegative fraction cap. It has a different pulse norm unless tau=0. That comparator can test practical loss reduction at equal initial update length, while the paired ± contrast isolates two directions with both norms matched. Sham remains useful. These are different estimands and should not be conflated.

## Decision before spending on tails

Do not launch the old candidate with a smaller dose solely to obtain feasible arms. First choose whether the target is practical rate-control performance or a conditional directional-response mechanism, and specify the appropriate comparator. For a paired contrast, generic predictors include the gradient, block imbalance, curvature, phase and local transient gain; unequal response alone is not evidence for shrimp geometry. A sensitivity asymmetry can occur in ordinary periodic or chaotic dynamics too.

The independent ordinary-periodic/chaotic controls, initial-state support, recovery-map qualification and all historical accessibility failures remain required for a geometry-specific claim. The existing zero-qualified-control result is unchanged. No parameter scan, alternative-control search, training pulse, protected evaluation, or LM controller is justified by this note. The next useful preparation is a concrete comparator and common-support design, not a larger training run.
