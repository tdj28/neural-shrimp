# All-token streamed readout: engineering assessment

2026-10-13. Code inspection and algebra only; no target call or timing run.
Inspected unchanged `ball_qwen.py` (`fcc90849...e98a80`), `ball_jet.py`
(`46b14d8b...9719`), and `ball_linear.py` (`5d54200f...45ff2`).

The existing final head materializes vocabulary-sized logit Jets,
temperature-scaled Jets, and probability Jets (`BallQwen.forward`, lines
283–285). `linear_sequence` then packs all probabilities and K derivatives
into another V-by-(K+1) Arb matrix; `affine` additionally copies input blocks.
The weight conversion cache is bounded, but these activation arrays are not.
For V=151936 and K=896, one array of full 1024-bit midpoint payloads alone
would occupy approximately 16.2 GiB, before radii, containers, and model
weights. This is a material obstacle under the 18 GiB attempt cap.

For exact stored tied embeddings E, normalized hidden state h, its signed
K-column derivative U, and positive constant temperature T, let

\[
 z=Eh/T,\quad p=\operatorname{softmax}(z),\quad F=E^T p.
\]

An exact identity is

\[
 DF=\sum_v p_v e_v (e_v^T U/T)
       -F\sum_v p_v(e_v^T U/T).
\]

A two-pass implementation can retain only V primal logits/probabilities,
then stream vocabulary chunks. First compute all scalar logits, choose an
exact common shift from their midpoints, and enclose the positive denominator
and every probability using Arb. In the second pass, form only a b-by-K chunk
of signed logit derivatives. Accumulate the signed first moment, weighted
Jacobian term, and derivative mean; subtract the final outer product only
after these signed sums. Every token contributes. The final transpose
multiplication must also be chunked, not call the existing affine routine on
a full V-by-K input. Storage becomes O(V+dK+bK+bd), including bounded weight-validation and
conversion buffers, beyond model/body storage.
All arithmetic, including sums, remains outward enclosed; a nonpositive
softmax-denominator enclosure fails rather than being patched numerically.
Rearrangement may change interval widths, so equivalence is algebraic and
containment-based, not bitwise equality to the old head.

A separate subclass can reuse the inherited prefix cache, decoder blocks,
normalization, and direction arithmetic. It must override the short forward
orchestration to stop at normalized hidden Jets and invoke the streamed head;
the existing hooks do not provide a clean final-head interception. Copying
that short loop is reviewable and requires no copied attention/decoder
formulas or changes to the original evaluator. Readmit every direction's
shape/count/finiteness/radius, preserve temperature and cache guards, and pin
the new source independently. Existing refinement ancestry remains unchanged.

This saves memory, not the O(V*d*K) head arithmetic. Two exact embedding
products per batch remain. Larger native batches could amortize conversion
and repeated body work; actual gains depend on FLINT kernels and allocation.
A covariance-first alternative forms E^T diag(p) E and multiplies by U, but
costs O(V*d^2) even for a small K. It is unattractive for 64/128 columns and
only plausibly competitive near a full hidden Jacobian; no timing claim is
justified without a separate experiment.

The transformer body remains a constraint: gate/up/activation Jet arrays
scale with intermediate width times K, and prefix constants currently carry
K-entry zero-tangent tuples. A full896 attempt therefore remains unjustified
from head algebra alone. After pure synthetic equivalence, interval,
all-token, and refusal tests and independent review, a fresh bounded 64- or
128-column pilot could measure peak RSS, head versus body time, and enclosure
width. Retain the current32 run as the baseline. Do not extrapolate a promised
speedup or launch the eight-phase full Jacobian from this design note.

## Subsequent authorized synthetic implementation

A separate `streamed_head.py` now implements the two-pass head and the short
`StreamedHeadQwenJet` orchestration described above. Original evaluator,
Jet, linear, cache, and refinement sources were not changed. Source SHA-256:
`f6096528f5748392386e97e9a4722f5967aeb0ae24897aac1465d205fca4f24a`.
The new tests have SHA-256
`f36f42d3e1980985767385278a381a120b54121ea8b952cafcd793a9fecdf1d5`.

The 55 new cases plus existing Jet/linear/map tests passed: **141 passed in
2.65 s**, CPU1, pytest cache and bytecode disabled. Tests cover the analytic
two-token tanh derivative, comparison with the original head and tiny
transformer, box/direction sample containment, signed cancellation, the last
token's contribution, audited chunk coverage, changed weight-cache bytes,
nonfinite/shape/type refusals, and inherited cache guards. An optional exact
all-competitor argmax margin is recorded from already enclosed scalar logits;
only a strict positive margin establishes that local label, and ties/overlap
fail it. This metadata is explicitly separate from every orbit proof gate.
One pre-existing Torch deprecation warning occurred. No checkpoint or target
model was loaded and no timing or memory benefit has yet been measured.
Independent review is still required before a new target pilot.

Independent review identified a whole-embedding boolean finiteness temporary.
The released source now validates finiteness in vocabulary chunks, retaining
the all-token check with bounded O(bd) temporary storage. All 55 head cases
passed again in the subsequent 89-case pilot/head suite (6.92 s). No target
run accompanied this correction.
