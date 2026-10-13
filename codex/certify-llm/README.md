# Exact-real Qwen period-8 certificate development

## Current checkpoint — 13 October 2026, 08:14 UTC

The assigned numerical orbit is recovered and its shooting centres have been
refined to a rigorously enclosed residual of approximately **6.66712478562819091E-151**.
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
| center-parallel-001 | One candidate:2.1553e-64 to3.5238e-77; history3 to4 | Eight fresh phases,201 artifacts audited;325.21s total,20.775GiB sampled aggregate RSS |
| center-parallel-002 | One candidate:3.52379101053601862E-77 to4.16828019605685895E-89; history_cap_restart | Eight fresh phases,204 artifacts audited;286.70s total |
| center-parallel-003 | One candidate:4.16828019605685895E-89 to7.40229711559188591E-101; accepted_midpoint_rank_one | Eight fresh phases,211 artifacts audited;319.52s total |
| center-parallel-004 | One candidate:7.40229711559188591E-101 to4.87833414421185684E-114; accepted_midpoint_rank_one | Eight fresh phases,219 artifacts audited;384.88s total |
| center-parallel-005 | One candidate:4.87833414421185684E-114 to3.57846981214675381E-126; accepted_midpoint_rank_one | Eight fresh phases,225 artifacts audited;386.70s total |
| center-parallel-006 | One candidate:3.57846981214675381E-126 to1.74332023675537826E-138; accepted_midpoint_rank_one | Eight fresh phases,230 artifacts audited;429.03s total |
| center-parallel-007 | One candidate:1.74332023675537826E-138 to6.66712478562819091E-151; history_cap_restart | Eight fresh phases,233 artifacts audited;376.92s total |
| box-derivative-001 | 32 signed columns over original phase-zero prototype box2^-800;988.57s forward,7.93GiB peak | Original prototype only |
| streamed-derivative-001 | 64 columns on same box;1787.97s forward,5.14GiB peak; all32 baseline column comparisons pass | Original prototype only; all67 artifacts audited |
| gate-budget-001 | Synthetic 8x896/1024-bit full gates and report:988.804s,6.731GiB peak | CPU1;129 artifacts/108 Git blobs audited; no target evaluation or certificate |
| proposal-budget-001 | Numerical factored-bound budget; limiting common induced-infinity Jacobian discrepancy about2.59e-9 | Planning diagnostic, not rigorous operator bound |

The synthetic gate resource pilot spent 283.310s on construction, 689.484s
on gates and 13.847s serializing an 845,393,489-byte report. This measures
one reviewed synthetic workload; it is neither a target proof nor an upper
bound on final proof runtime. See `gate-budget-001-review.md`.

All failed and completed attempts remain immutable under
`/data2/neural-shrimp-data/codex/llm-p8-certificate/`. Compact receipts are in
`certificates/llm-p8-*.json`; named result-review files retain independent
checks, exact receipt hashes and every configuration's outcome.

### Current owned computation

No Codex model job is active at this checkpoint. `center-parallel-007` completed
from source `a34c347d4190e256a70a63772624213899d37db1`, exit0 without timeout,
in376.92 seconds. All233 artifacts,122 exact source blobs,
eight original phase enclosures and termination rows passed both independently written
audit scripts. The new residual upper bound is strictly below the old lower bound,
with ratio about3.82438329175646357E-13. Peak sampled aggregate RSS was
20.156GiB; this is not a hard cgroup cap.
No matched runtime or convergence comparison establishes acceleration.

The complete audited capsule has0 factors and an unconsumed
tuple10-to11 pair. The reviewed next single-candidate plan is
`center-parallel-008-plan.json`; check its exclusive attempt and final receipt before
doing anything that could duplicate it. The immediate capsule bytes are
authenticated again; older archives rely on the disclosed prior independent
audit rather than being silently claimed as rehashed. No GPU, paid pod or
paid consultation is in use.

### Remaining route to the certificate

1. Continue complete audited parallel candidates until centre refinement is
   adequate for final derivative boxes. Preserve every update, restart, fallback
   and regression; never reuse a partial attempt. The first parallel pass
   measured325.21s total; future costs remain measured outcomes.
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
