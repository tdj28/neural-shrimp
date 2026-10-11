# Independent review: utility dose geometry decision

8 October 2026 UTC. Reviewed root source49fd84ae02773b5fa83cec83b89e8531a12afe27,
docs/plans/utility-dose-geometry-decision.md. Line references refer to that
version. **Pass as a data-free algebra/design note; no mathematical blocker.**
No target arrays, model, derivative, norm, solver, profile or scientific run was
evaluated. No dose or comparator was selected. This is not an independent
re-audit of the earlier four-fork feasibility outcome cited in the note.

The uniqueness argument (lines9–18) is correct for the stated Euclidean norms.
For s nonzero, squaring both constraints yields s·delta=κ||s||² while
||s||||delta||=κ||s||². Equality in Cauchy–Schwarz fixes delta=κs. Equivalently,
||delta−κs||²=0; disjoint hidden/output blocks give exactly
H²(a−κ)²+O²(b−κ)²=0. Both group norms must be positive to identify both rate
fractions. If s=0, delta=0 remains the unique displacement but rate fractions
and a directional intervention are unidentifiable. These statements concern
vectors at the same state, not equality of subsequent training trajectories.

For v_h=τ(O/H)s_h and v_o=−τ(H/O)s_o (lines22–35), the inner product with s is
τOH−τHO=0 and ||v||²=τ²(H²+O²). With τ>=0, ||v||=τ||s||. Orthogonality gives
both displayed ± norm identities. Requiring both signs' hidden fractions in
[0,c] is τ O/H<=min(κ,c−κ); requiring both output fractions is
τ H/O<=min(κ,c−κ). Combining them gives the stated necessary and sufficient
bound on τ. Strong imbalance restricts this contrast. The zero-group case
cannot be handled by these ratios and is already excluded by the hypothesis.

The equal-total-update-length uniform comparator (line37) also follows:
κ_match=1−sqrt((1−κ)²+τ²). The stated feasibility check is safe. For the usual
reduction cap0<=c<=1 and jointly feasible paired fractions, it lies in[0,c]
automatically. For τ>0 and κ<1, κ_match<κ, whereas the paired pulse norm is
sqrt(κ²+τ²)||s||>κ||s||, so that comparator necessarily has a different pulse
norm. It tests a different matched resource than equal treatment-minus-sham
size. At τ=0 the two paired arms collapse to the same uniform intervention.

Two concise additions would strengthen the note without changing its design:

- After line33: “A nonzero directional contrast requires0<κ<c andτ>0;
  κ=0 orκ=c permits onlyτ=0.” The existing closed bounds include this
  degenerate case, whereas line22 begins with nonzero v.
- After line41: “Orthogonality to s=Pg does not generally imply orthogonality
  to the loss gradient g. A matched ± response can therefore reflect an
  ordinary first-order gradient effect.” More precisely, writing the paired
  states as z±v, z=theta−(1−κ)s, a smooth loss contrast begins with
  2 grad L(z)·v. Even when P is scalar and g(theta)·v=0, the gradient at z
  need not have that property. This reinforces the already required gradient,
  curvature and transient-gain baselines; it introduces no proposed dose.

The estimand separation (lines37–43) is appropriate. Equal initial update norm
permits a practical comparison at that matched initial quantity; equal pulse
and update norms between the ± arms isolate their displacement directions at
that state. Neither comparison removes generic directional sensitivity,
proves shrimp causality, equates longer-run dose or establishes useful
control. Common initial-state support, independently qualified ordinary/
chaotic controls, recovery-map qualifications and the retained accessibility
failures remain relevant before a geometry-specific claim. The note correctly
authorizes no pulse, scan, new control search or larger training study. No
historical card, receipt, source or manuscript was edited by this review.
