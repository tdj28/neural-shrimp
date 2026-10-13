# Exact-real Qwen period-8 certificate development

## Current checkpoint — 13 October 2026

The assigned primitive P8 proposal has been independently recovered with relative
return `1.60e-14` and numerical divisor separation. There is **no orbit certificate**.

- `arb-pilot-001` (384 bits) failed during fixed-prefix enclosure at layer16.
- `arb-pilot-002` (1024 bits) completed the exact-real fixed prefix and one full
  point step. Maximum point-output radius was `1.215e-142`; the difference from
  the next binary64 proposal was at most `9.623e-15`. The radius`2^-80` box then
  wrapped to radius`1.6996` by layer2 and failed the RMS positivity gate entering
  layer3. This is a setup failure from dependency loss, not dynamical instability.
- All22 pilot002 artifact hashes were checked; receipts preserve both attempts.
- `signed-pilot-001` completed eight signed point-derivative columns in
  261.81 seconds at 1024 bits, with maximum derivative radius `2.526e-141`.
  All 27 artifact hashes and all eight comparison rows were audited. This
  covers eight directions at one point; full-box derivatives are still absent.
- `center-seed-001` completed all eight independent binary64 phase Jacobians
  and the cyclic inverse proposal in 677.27 seconds. All eight numerical
  closure gates passed and all 56 artifact hashes were checked. The numerical
  inverse defect is approximately `3e-15`; this is a proposal, not an inverse
  or derivative certificate.
- `arb-point-003` is constructing a fresh 2048-bit prefix and one point image
  under the existing CPU1/18-GiB/one-hour envelope. Its final receipt is still
  pending; the known-wide state-box probe is explicitly skipped.
- `center_refine.py` prepares one high-precision multiple-shooting correction
  followed by a fresh eight-phase residual check. It has not run; admission
  requires completed, hash-bound prefix and seed attempts and a pushed plan.

`ball_jet.py` provides batched signed first derivatives through the same reviewed
transformer architecture. `cache_io.py` preserves all prefix radii and checks the
construction/provenance chain. `proof_gates.py` implements conditional Krawczyk,
primitive-period and monodromy-power algebra. Passing those algebraic gates only
proves the desired statement once their full-domain enclosure contract has been
established; hashes and a declaration alone do not establish it.
`shooting_gates.py` adds full product-box root and conditional primitive-period
checks, with signed streamed blocks or a cheaper conservative factored bound.
It deliberately leaves dynamical attraction to a separate monodromy bound.
The current combined synthetic suite passes 224 tests; no test is an orbit proof.

The original pilot design below is retained as history. Current setup outcomes
are in `certificates/llm-p8-pilot-001.json` and `llm-p8-pilot-002.json`.


**Behavioral TLDR:** the assigned objective is to prove a locally attracting
primitive eight-state cycle in a specific continuous-feedback language-model
map. A certificate has not been obtained. This directory currently implements
rigorous forward arithmetic and a bounded diagnostic pilot.

The target is Qwen2.5-0.5B-Instruct at exact binary64 alpha 0.071674 and
log10(T) -0.115, the fixed 37-token open-chat prefix, one-token window and
philosophy steering after zero-based decoder block12 at every position.
The full 151,936-token expectation uses tied input/output embeddings. No sampled
text or conversation behavior is asserted by an orbit certificate.

## Exact map and arithmetic

Weights, vector, RMS epsilon, alpha, log10(T) and RoPE base are exact stored
binary rationals. Temperature is real exp(log(10)*log10(T)); RoPE frequencies
are real base^(-2j/head_dim), followed by real sin/cos. The alpha-dependent
prefix is recomputed in Arb balls. Rounded Torch activations, inverse frequencies,
temperatures or prefix caches are never treated as exact proof constants.

`ball_linear.py` converts bounded blocks of finite stored binary weights to
Arb exactly, then uses native Arb matrix multiplication; it never calls floating
BLAS for a proof operation. An optional bounded cache checks exact block bytes.
`ball_qwen.py` implements causal grouped attention, RMSNorm, SwiGLU, rotary,
steering, all logits and the complete embedding expectation with a directional
ball derivative. Every primal and derivative operation encloses rounding.
Inputs must remain immutable throughout a map/cache lifetime. The pilot owns
read-only NumPy weight views, uses no concurrent mutation, and guards common
parameter/attribute rebindings and precision changes. This is an explicit
caller contract, not protection against arbitrary mutation through hidden aliases.

Tests use small synthetic transformers and an independent tensor implementation,
exact affine positive controls and sampled interval interiors. They detect bugs;
they do not constitute a proof of the implementation or of the target orbit.
The 67 tests pass. See the [map review](map-code-review.md) and
[mathematical design review](proof-design-review.md) for their distinct scopes.

## First target pilot

The producer's table-1 `orbits.json` stores summary metrics and token strings,
but no full state. The pilot therefore uses our own hash-bound P8 fold state as
an initial guess, moves to the assigned parameters and iterates G=F^8 for at
most40 cycles. Two consecutive relative returns below1e-12, final return below
1e-10 and divisor1/2/4 separations above1e-6 are required before proceeding.
The decoded argmax token IDs must match the assigned eight-token loop up to
cyclic rotation (the tokenizer is hash-bound as a label check). Every numerical
residual and all nine phase/return states are retained.
This is a proposal for exact-real refinement, not a proof box.

At384-bit precision, the pilot computes and stores the complete rigorous prefix,
one exact-real point-map step and a one-step box-JVP diagnostic with state
radius2^-80 and tangent box[-1,1]^896. That box is far narrower than the existing
floating proposal residual; it is deliberately **not an inclusion test**.
The run records per-layer radii, derivative bounds, timings and any first failed
positivity/finite gate. A broad enclosure is an inconclusive setup result, not
instability. Both successful and failed attempts remain immutable.

A one-step norm diagnostic supplies neither the eight-step Krawczyk inclusion
nor primitive separation at a proved root nor the monodromy-power attraction
gate. All these are still required. In particular, one interval-box JVP is
mathematically valid but can lose cancellation; precision alone cannot remove
that dependency loss. A centered derivative remainder, adapted coordinates or
multiple shooting may be needed after the measured pilot.

## Resources and replay

CPU only, nice10, one Torch/BLAS/Arb thread,18GiB address-space and3600s hard wall.
No GPU, paid compute or API calls. Source and run card must be privately pushed
before execution. `launch.py` snapshots the exact pushed source, records remote
ancestry and owned PID, preserves stdout/stderr and artifact hashes, and only
terminates its own process group. Raw output is exclusive under
`/data2/neural-shrimp-data/codex/llm-p8-certificate/`.

Synthetic CPU1 affine measurements at128bits gave0.472s for4096×896 weights
and two columns (about7.8M stored coefficients/second);384-bit full-prefix cost
and wrapping are unmeasured. This motivates a bounded pilot, not a promised ETA.
Run the pushed pilot with:

```bash
/data2/neural-shrimp/.venv/bin/python -B codex/certify-llm/launch.py \
  --attempt arb-pilot-001 --precision 384 --seconds 3600
```

Tests:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/data2/neural-shrimp/.venv/bin/python -m pytest -q \
  tests/test_codex_llm_ball_linear.py tests/test_codex_llm_ball_qwen.py
```

| Component | Current evidence | Assumptions / artifact |
|---|---|---|
| Ball affine and transformer kernel |Synthetic numerical tests and code review|Arb/FLINT and analytic implementation trusted; not formally verified|
| Assigned orbit |Own numerical P8 proposal recovered; not an orbit proof|One fixed checkpoint/prompt/feedback map|
| Existence, uniqueness, primitive period, attraction proof|Not established|Future strict full-box inclusion, norm and divisor gates|
| Broader inference benefit or whole shrimp |Not tested by this task|Separate studies required|
