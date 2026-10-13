# Exact-real Qwen period-8 certificate development

## Current checkpoint — 13 October 2026, 06:15 UTC

The assigned numerical orbit is recovered and its shooting centres have been
refined to a rigorously enclosed residual of approximately **2.16e-64**.
**Existence, primitive period and attraction are not yet certified.**
This is an inference-state orbit of one exact-real soft-token map, not a
training-parameter cycle or evidence of a task-performance benefit.

The target is Qwen2.5-0.5B-Instruct at exact binary64 alpha 0.071674 and
log10(T) -0.115, the fixed 37-token open-chat prefix, one-token window and
philosophy steering after zero-based decoder block12 at every position.
The full 151,936-token expectation uses tied input/output embeddings.

| Attempt | Completed outcome | Scope |
| --- | --- | --- |
| arb-pilot-001 | 384-bit prefix failed at layer16 | Retained arithmetic setup failure |
| arb-pilot-002 | 1024-bit prefix/point passed; radius2^-80 box failed RMS positivity at layer3 | Wrapping failure, not instability |
| signed-pilot-001 | Eight signed point columns,261.81s | One point; no full-box coverage |
| center-seed-001 | Eight numerical896-square Jacobians and cyclic inverse proposal,677.27s | Binary64 proposals only |
| arb-point-003 | 2048-bit prefix/point; max point radius6.7203e-451 | All32 artifacts independently audited |
| center-refine-001 | One correction:9.6227e-15 to2.4361e-27 | Sixteen phases,77 artifacts audited |
| center-resume-001 | Two corrections:2.4361e-27 to6.5829e-40 to7.1572e-52 | Sixteen fresh phases,98 artifacts audited; strict exact residual decreases |
| center-secant-001 | One candidate:7.1572e-52 to2.1553e-64; all three prior secants accepted | Eight fresh phases,107 artifacts audited; no matched acceleration claim |
| box-derivative-001 | 32 signed columns over original phase-zero prototype box2^-800;988.57s forward,7.93GiB peak | Original prototype only |
| streamed-derivative-001 | 64 columns on same box;1787.97s forward,5.14GiB peak; all32 baseline column comparisons pass | Original prototype only; all67 artifacts audited |
| proposal-budget-001 | Numerical factored-bound budget; limiting common induced-infinity Jacobian discrepancy about2.59e-9 | Planning diagnostic, not rigorous operator bound |

All failed and completed attempts remain immutable under
`/data2/neural-shrimp-data/codex/llm-p8-certificate/`. Compact receipts are in
`certificates/llm-p8-*.json`; named result-review files retain independent
checks, exact receipt hashes and every configuration's outcome.

### Current owned computation

No Codex model job is active at this checkpoint. `center-secant-001` completed
from source `ac376355a37f6dc1368128ed8c9519a7bd2afb2a` in1422.10 seconds,
exit0, no timeout, peak5.68GiB. All107 artifacts and69 exact source blobs were
audited; all eight original phase enclosures and the strict exact residual
norm decrease passed. The `accelerated` formula achieved a reduction ratio of
about3.01e-13, comparable to earlier fixed-R passes; there was no matched
counterfactual, so this is not evidence of an acceleration advantage.

Development now targets independent parallel phase evaluation (up to four
CPU1 workers), restoration of the exact post-proposal secant history and removal
of a redundant repeated scalar dot product. These are reviewed engineering
changes, not target results. The first parallel candidate needs a committed
concrete plan and run card before execution. No GPU, paid pod or paid
consultation is in use.

### Remaining route to the certificate

1. Finish the reviewed continuation adapter and run a bounded one-candidate
   parallel pilot. Preserve all factors after proposal fallbacks, consume the
   pending consecutive secant exactly once, and measure actual wall time and
   aggregate memory before extending the approach.
2. Define exact candidate centres and phase radii after adequate refinement.
   The original prototype boxes do not contain the refined centres and cannot
   be reused as proof boxes. Small residual alone does not establish inclusion.
3. Evaluate all8x896 signed derivative columns over those same nominal boxes.
   The measured64-column stream suggests roughly56 CPU-hours for all112 shards
   if all other phases cost similarly; this is a single-phase extrapolation,
   not a campaign reservation or completion guarantee. Commit a concrete
   resource plan and run card, checking shared CPU/RAM before launch.
4. Authenticate every complete shard receipt, source bundle, exact map/backend,
   original live input containment and point-image receipt. Assemble every
   direction once. Point images must belong to exactly these centres.
5. Run the shooting root inclusion and contraction, all proper-divisor
   separations, and interval monodromy-power attraction gates. Independently
   review all witnesses before calling the result a certificate.

`phase_domain.py` checks original exact endpoint containment across precision
changes. `derivative_coverage.py` checks complete coordinate coverage and
original producer input/primal consistency before decoding; it does not
perform external artifact authentication. The new `derivative_shard.py`
producer and `shooting_certificate.py` conditional orchestrator passed
independent review. The latest focused combined synthetic
suite passed245 tests; those tests are not target-orbit evidence.

`shooting_gates.py` binds centre images, whole-box Jacobians and dyadic A/B
proposals to the full product-box contraction. `proof_gates.py` provides the
separate attraction-by-matrix-power check. Source hashes/declarations bind
evidence; they cannot establish an evaluator contract by themselves. The
analytic/evaluator review, exact source, Arb/FLINT backend and all raw evidence
remain part of the computer-assisted proof's trusted basis.

The first pilot's design and replay instructions below are retained as history;
they do not authorize repeating old attempts or changing current run budgets.

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
