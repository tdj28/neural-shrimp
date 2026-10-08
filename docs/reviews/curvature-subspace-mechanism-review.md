# Independent review: curvature-subspace mechanism note

8 October 2026 UTC. Source reviewed: root commit
02ebdc87bf9805070467927d51147402d6b1d8e3,
docs/plans/curvature-subspace-mechanism.md (line references below).
**Pass as a data-free mathematical design note. No algebraic blocker, empirical
validation, novelty verdict or authorization for larger-model work.** This
review derives the stated formulas; it does not evaluate a saved or new model.
No trajectory, derivative, solver, profile or literature consultation was run.

## Algebra and scope checked

For fixed symmetric positive-definite P, J_j=I−P H_j and
P^(-1/2) J_j P^(1/2)=S_j, so J_b J_a is similar to S_b S_a (lines25–30).
Use the same P at both phases. Its variation with a continuation parameter
belongs in a total branch derivative. With s=sqrt(1−c²)>0 and basis
(u,(v−cu)/s), the restricted first factor is diag(a−alpha,a), and the
second is [[b−beta c²,−beta cs],[−beta cs,b−beta s²]]. Their product yields
exactly T=2ab−b alpha−a beta+alpha beta c² and
D=ab(a−alpha)(b−beta). The complement has scalar multiplier ab, with
multiplicity n−2. No division by a or b was used, so zero-factor limiting cases
remain covered. At c=±1 the selected line multiplier is
(a−alpha)(b−beta), and the n−1 dimensional complement multiplier is ab.

The roots are (T±sqrt(T²−4D))/2. Thus T²<4D is exactly the complex-pair
condition, automatically implies D>0, and gives modulus sqrt(D) and the stated
principal angle. At c=0 the two plane multipliers are b(a−alpha) and
a(b−beta), both real. For a=b=1,alpha=beta=2, the product of reflections has
D=1,T=4c²−2: c=±1 gives +1 and c=0 gives the repeated −1. Generic intermediate
orientations give a complex pair. This constructs tangent-factor algebra,
not a realized attracting training orbit or a transverse NS crossing.

The product-error inequality (lines67–70) follows by expanding exactly three
error terms and using a single consistent submultiplicative norm. It is an
operator-error statement in the whitened coordinates. When quoting ordinary
Euclidean error for the original-coordinate product, include the similarity
conditioning factor ||P^(1/2)|| ||P^(-1/2)||, or keep all claims explicitly in
the whitened norm. A small residual gives backward spectral evidence; forward
pair error and pair identity require separation/conditioning or a suitable
pseudospectral enclosure. The note correctly avoids treating the product as
symmetric or using an incorrectly paired complex left eigenvector.

D=1,|T|<2 locates the modeled nonreal unit-circle pair. Actual crossing needs
a nonzero total branch derivative D'; at D=1 the modeled modulus slope is
D'/2. The |ab|<1 complement condition is relevant when n>2 and vacuous in a
strictly two-dimensional model. Actual full-state primitive closure,
transverse modes, low-order nonresonance and a nonzero cubic coefficient remain
separate. The note already keeps these distinctions. A frozen Hessian path or
a drifting two-step tangent product cannot supply the missing orbit branch.

## Suggested additions before an empirical gate is written

1. **Separate orientation from radial prediction (lines43–48,61–65,94–97).**
   Add: “At fixed a,b,alpha,beta, orientation c² changes T but not D. It can
   change real-versus-complex character and angle, but cannot change the
   modulus sqrt(D) while the modeled pair remains complex. A transverse
   unit-circle crossing therefore requires changing phase strengths along
   the branch.” This prevents a good modulus fit from being credited to
   rotating directions. The reflection family has D identically1 and is
   consequently not a transversality example.

2. **Include a necessary phase-inertia check (after line51).** Add: “A complex
   pair requires both symmetric phase factors to be indefinite on the active
   plane. If either factor is positive or negative semidefinite, all nonzero
   eigenvalues of the product are real.” For a definite factor, the product
   is similar, up to sign, to a symmetric congruence; the semidefinite version
   follows by reducing to its range. In the rank-one model this requires
   a(a−alpha)<0 and b(b−beta)<0. This is necessary, not sufficient; rotating
   eigenvectors or a nonzero commutator alone remains insufficient.

3. **Make the success rule test incremental geometric information
   (lines86–97).** Predeclare target-pair selection, abstention for ambiguous
   pair identity or clustered sharp eigendirections, and the bulk fitting
   rule using phase matrices only. Compare the rotating model with a
   same-strength commuting model for complex-pair classification and angle;
   compare modulus separately with the strength-only sqrt(D) baseline where
   a complex pair exists. Require a declared error margin or a credible
   cost/accuracy advantage, including curvature-estimation and qualification
   costs. Do not count a fit using the target pair as prediction. A vacuous
   full-operator spectral bound is an unresolved qualification, not evidence
   that a visually matching projected pair represents the full product.

## Decision

This is a useful falsifiable screening proposal if it asks whether a fixed,
cheap representation explains the actual complex pair better than matched
strength-only information and remains reliable on the explicitly selected
real/complex/crossing references. Failure then rejects that approximation.
The elementary rank-two/product-of-reflections algebra itself is not a new
research contribution; this review does not certify prior-work completeness.
Any novelty claim still needs the stated Hofmann/edge-of-stability comparison.

Even a passing small closed-orbit test would justify only a separately costed
matrix-free estimator test. It establishes no extrapolation to drifting LM
states, controller benefit, geometry-specific utility or NS mechanism in a
large model. The no-launch boundary and requirement to inventory already
saved phase matrices before choosing thresholds are appropriate. No paper or
frozen CIFAR source file was edited by this review.
