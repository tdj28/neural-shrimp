# Consult 14 adjudication: a computer-assisted proof of chaos

Review: `review.md` (gpt-6-astra, Pro mode, xhigh effort; response resp_06627f0120187580016ad4471219e487d09ac9893a6ecb3f25,
review SHA-256 75ae03c2...; raw bundle in ../neural-shrimp-data/consults/2026-10-18-chaos-certification/execute-001).
The advice is advisory. Each point was checked against our own records before adoption.

## Adopted

1. **Target a small invariant subset, not the whole attractor.**
   - Prove positive topological entropy for an iterate G = F^k with a sparse graph of covering relations (h-sets
     along short local returns, non-full transition matrix A, entropy bound log rho(A)/k).
   - Cone conditions are needed only for stronger statements (uniform hyperbolicity, a conjugacy). Requiring them
     at once would make the proof harder than its goal. Covering relations also do not need the map to be
     invertible, which matters here, since F maps R^896 into the convex hull of the embeddings.
   - This replaces our draft of a full two-set horseshoe over the attractor's unstable extent.
2. **Corrections to our evidence statements.**
   - sigma_2/sigma_1 and the Lyapunov spectra are not uniform bounds on transverse contraction over a neighborhood.
   - The failed tiny boxes are evidence against that enclosure representation, not against chaos, and precision
     alone does not cure wrapping.
   - A first-order segment must also bound transverse and mixed terms (h*rho and rho^2). An O(h^2) remainder along
     the unstable direction alone validates no tube of nonzero stable radius.
3. **One rigorous step cannot price the proof.** The pilot must propagate complete returns, including coordinate
   changes and the reuse of transverse bounds, because composition may either stay controlled or destroy every
   margin.
4. **Precision.** Start at 1024 bits, the precision already shown to work, and test 512 and 256 only after the new
   representation succeeds. Our draft's 128-256 bits was not supported by evidence.
5. **Representation.** x = c + u t + B y with |t| <= h and ||y|| <= rho. The unstable coefficient is carried
   explicitly. The transverse part is carried through a linear factor or a certified operator bound, never as 895
   unrelated intervals. Two transverse strategies are to be priced: a verified centre Jacobian in adapted
   coordinates, reused over a patch with a derivative-variation bound; and a cheaper block or operator bound. Time 16
   signed columns before committing to full Jacobians. QR (Lohner-type) reconditioning is relevant.
6. **Choice of cell.** Search both candidate cells with equal budgets and compare certified margin per unit of
   computation, not Lyapunov exponents.
7. **Controls.**
   - Positive: the product map H(t, y) = (4t^2 - 2, 0.1 y) rotated into 896 dimensions, with branches
     [-1, -0.4] and [0.4, 1].
   - Negative: an expanding transverse factor, to defeat containment; a contracting map, to detect spurious
     branching.
8. **Endpoint.** For each reserved complete-return strip, R = max_j eps_j / m_j, where m_j is the predicted clearance
   from a forbidden face and eps_j the rigorously bounded departure; invalid enclosures count as infinity. The
   numerator and denominator are reported, and no extrapolation to a proof is made. Secondary endpoints: CPU-seconds
   per strip-return, peak memory, and the number of strips needed.
9. **Layer checks.**
   - The RMSNorm denominator is analytically at least sqrt(eps), so a failure to prove positivity is an enclosure
     failure; use tighter centre-radius lower bounds.
   - Attention: centred, stabilized softmax against the fixed rigorous cache.
   - The readout is streamed in vocabulary blocks; dropping tokens needs rigorous tail bounds.
   - Exact weights are shared or streamed, and concurrency is set from measured memory.
10. **GPUs.** Keep Arb and add narrow, transformer-specific primitives. GPU interval arithmetic or a ported
    Taylor-model system adds a validation burden the pilot does not warrant. GPUs remain useful for float64
    proposals and searches. We revisit this only if measured production costs are large.
11. **Scope of a result.** Positive entropy of F^k on a compact invariant set of the real-number map of the stored
    constants. It is not chaos of typical orbits, a physical measure, or every period for F; graph cycles of F^k must
    be translated carefully into periods of F.

## Adopted with a note

- **Resource envelope.** The pilot needs 50-300 CPU-hours and 1-2 weeks of development; a production engine
  4-12 researcher-weeks; production compute remains the scenario range of 2,000-40,000 CPU-hours (not a forecast).
  Locally, memory limits concurrency: the period-8 proof capped each process at 18 GiB, and about 48 GB are free, so
  the pilot's wall time depends on the measured resident memory of the new engine.
- **Alternatives.** The saddle-homoclinic route (2) needs a theorem whose hypotheses include invertibility, which
  is delicate for this near-rank-one map; it stays second. The contracting-fiber reduction (3) stays a diagnostic.
  A full-dimensional Conley grid is not competitive.

## Not adopted

- Nothing was rejected. The review's three-repeat timing medians are kept, and its statistical caution
  (deterministic certification; distinct chains as units) matches our protocol.

## Next step (for the human to approve)

The paired-cell, complete-return pilot of the review, in its stated order, on local resources only:
- the float64 search for local covering graphs at both cells (GPU or CPU);
- the analytic controls;
- the Arb segment engine at 1024 bits with the transverse representation, and one reserved strip per cell propagated
  through its complete return.

The Arb engine extends the period-8 proof's code. Assigning it to Codex needs the human's approval.
