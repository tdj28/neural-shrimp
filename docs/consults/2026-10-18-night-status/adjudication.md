# Consult 15 adjudication (18 October 2026)

Review: `review.md` (gpt-6-astra, Pro, xhigh; 69,333 tokens; cost in the private ledger). Raw bundle:
`../neural-shrimp-data/consults/2026-10-18-night-status/execute-001`. Advice is advisory; each point was checked
against our data and code before acting.

## Accepted and applied now (wording in the paper, the Chaos article and the blog)

1. **"Of a two-dimensional map" overclaims.** Two parameters are not two state dimensions, and no two-dimensional
   reduction of the dynamics is established. Accepted. The claim becomes "a numerically verified shrimp" (in a
   two-parameter plane of the full map). The head is described separately: on the continuation grid the linearization
   has at most two non-negligible directions and a determinant that changes sign; its closed trace-zero curve is the
   geometry of Type A heads, an analogy only.
2. **The three comparable moduli are forced by the defining conditions.** If the sum and the sum of pairwise products of
   three roots vanish, the roots are the cube roots of their product: equal moduli at 120 degrees. So "the head is a
   point where three directions contract together" and "no temperature makes the head one-dimensional" are not supported
   by the moduli. Accepted. The text now reports what the track measures: the common modulus, the cube root of the
   leading block's determinant, at 2.4-5.6e-3, a fourth multiplier approaching it twice, and the point's drift. Whether
   the head's return map is locally of higher rank or nearly rank one and strongly nonnormal is stated as open (singular
   values would decide it).
3. **"Type A" is a geometric analogy.** Accepted; already partly stated (the classification assumes a determinant of
   fixed sign). Made explicit.
4. **Period-11 gate scale.** At multiplier scale 4e-7, the absolute gate |c1| < 1e-7 is not meaningful: the NT-P11
   solution has |c1| = 5.3e-8, i.e. 0.14 of the root scale. Accepted. The period-11 "near-nilpotent point" is recorded as
   not established. It was never in the paper.
5. **Norm bounds framing.** The power-iteration norms are lower estimates of norms, so the computed bounds are not
   certified upper bounds. The paper already says this; the T* sentence states that T* (computed with the true L) is at
   least the value obtained from the observed lower estimate of L, and it does not bound the actual transition
   temperature. The factor-dimension check was done: J is the Jacobian of the final normalized hidden state
   (896 x 896) and the readout factor is the embedding covariance E^T (diag p - p p^T) E (tools/norm_bounds.py, class
   Probe), which is the factorization the formula needs. The text now names J explicitly.
6. **Membership.** Periods in the followed domain come from full-state recurrence; the token word only decides family
   membership and propagation. Already stated; made explicit where counts are given.

## Accepted as next work (ranked by the review; our decision)

- **(Review rank 1) The rank question at a head.** Measure singular-value ratios of the full monodromy, d1 = s2/s1 and
  d3 = s4/s1, near and away from a head, with synthetic positive controls (identical eigenvalues, different rank
  structure) and the strongly anisotropic period-8 orbit as an application control. We will run a reduced version first
  at the period-5 head (whose near-nilpotent point is established, with |c1| about 1e-5 of the root scale) and at the
  period-11 candidate, before any larger design. Pending a tool (block power iteration on M^T M with JVP/VJP products).
- **(Review rank 2) Existence against selection at the shrimp's edges.** Partly answered by the boundary gate and FM5X
  (curves to the leg tips). Transects with continuation and multiple starts are a good follow-up; not started tonight.
- **(Review rank 3) Adapted norms.** Interesting; deferred.
- **(Review rank 4) Coexistence certificate.** Already the other agent's next target (Task 33 pair).

## Not adopted

- Freezing a 16-pair randomized held-out design before any measurement is more apparatus than a first look needs; we
  will report a small fixed set of points with all failures, as exploratory, and design a confirmatory run only if the
  result is to carry a headline claim.
