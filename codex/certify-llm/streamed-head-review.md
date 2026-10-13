# Independent all-vocabulary streamed-head review

Reviewed 2026-10-13, without a checkpoint load or target-model evaluation.

Final reviewed source: `streamed_head.py`, SHA-256 `f6096528f5748392386e97e9a4722f5967aeb0ae24897aac1465d205fca4f24a`. Tests: `tests/test_codex_llm_streamed_head.py`, SHA-256 `f36f42d3e1980985767385278a381a120b54121ea8b952cafcd793a9fecdf1d5`.

No unresolved mathematical blocker was found. This is a separately reviewed arithmetic implementation; it does not turn a partial derivative pilot into an orbit certificate or authorize a target run.

For constant positive temperature T, exact stored embedding matrix E, normalized hidden value h and signed hidden derivative matrix U, the code implements

    z = E h/T, p = softmax(z), F = E^T p,
    DF = E^T diag(p) (E U/T) - F [p^T(E U/T)].

Every vocabulary row contributes to both probability normalization and the signed derivative sums. The first pass retains only scalar logits/probabilities. The second pass computes each bounded vocabulary chunk's logit derivatives, weighted embedding contribution and derivative mean, accumulating with Arb before subtracting the outer product. There is no top-k approximation, nonnegative replacement of signed derivatives, omitted tail, floating matrix product or inferred floating error bound. The formula remains an enclosure when h, U and a fixed direction-independent T have nonzero radii; interval dependency can widen the result and no cancellation-based narrowing is assumed.

The common logit shift is an exact selected midpoint, held constant over the entire supplied box. Differentiating the softmax is invariant to that constant shift. Positive temperature and positive normalization denominator must be established, and nonfinite results fail. Input shape, stored binary weight dtype and direction-count checks preserve the full stated dimension and all supplied signed derivative columns.

`StreamedHeadQwenJet` repeats only the short forward orchestration through the final RMSNorm. Decoder blocks, attention, rotary constants, steering, prefix handling and normalization remain inherited. The body loops and soft-map readout agree with the previously reviewed map. State and direction balls retain their radii. Existing prefix precision/identity guards remain active; the added guard rejects embedding rebinding. Immutable arrays and parameters throughout the two passes and cache lifetime remain an explicit caller contract: attribute checks do not authenticate externally mutable aliases.

The optional label diagnostic chooses a midpoint candidate but checks its rigorous lower logit bound against every other token's rigorous upper bound. Only a strict positive enclosed gap establishes that candidate's label on the supplied domain. Ties and overlapping intervals do not pass. The observation is local to this input domain and does not certify an orbit, an independently selected phase box or ordinary autoregressive generation. The per-call diagnostic is cleared before evaluation, preventing a failed later call from exposing an old successful label.

## Memory and implementation refinement

The original full-matrix `np.isfinite(embedding).all()` admission temporarily allocated a V-by-d boolean array, contradicting the initial literal auxiliary-storage claim. Independent review requested bounded chunk validation. The final source checks finite weights by vocabulary chunks and explicitly includes the bounded weight validation/conversion term. Extra head storage is O(V+dK+bK+bd), beyond the already stored model and unchanged transformer-body storage. It avoids V-by-K tangent materialization, including in the final transpose product. This is an allocation-structure argument, not a measured target peak-RSS claim.

The same arithmetic work can still be expensive. Repeated affine calls may copy the right-hand-side derivative matrix and weight chunks; too-small vocabulary chunks can add substantial overhead. A separately frozen pilot can choose a larger bounded chunk such as 1024, while retaining every row, but actual timing and enclosure widths must be measured. The streamed regrouping can produce different interval widths from the original head, so bitwise equality is neither expected nor required.

## Independent checks

The original streamed/Jet/linear/map suite passed 141 tests independently; the only warning was an existing Torch deprecation notice. In addition, an independently written exact-rational covariance calculation checked 1161 output endpoints: a 17-token, three-state, two-direction example at three chunk sizes, including all 64 corners of the interval direction box. These checks use the exact uniform-probability covariance at h=0, rather than reusing the producer's differentiation formula implementation or floating comparisons. They cover signed directions and final partial chunks.

After the finite-check delta, all 55 streamed-head tests passed again. No existing evaluator, cache, correction or refinement source was modified by this implementation. No target speed, memory, full-Jacobian enclosure or new rigorous orbit result is inferred from these synthetic checks.
