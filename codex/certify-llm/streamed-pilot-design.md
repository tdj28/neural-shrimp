# Bounded 64-column streamed-head pilot

2026-10-13. Preparation and synthetic tests only. No actual model run or
completed-baseline admission was performed by this implementation task.
`streamed_pilot.py` is separate from the original 32-column driver and every
common map/cache/refinement source. Nothing in the running or completed
source snapshots is changed.

The proposed measurement uses the original binary64 prototype's phase zero,
1024-bit Arb, all 151936 tokens, the same actual box constructed from nominal
radius 2^-800, and signed coordinate directions e0 through e63. The box is
not proposed as an isolating periodic-orbit box. Results omit 832 derivative
columns and seven phases; all orbit-certificate flags remain false.

The new head's token chunk is **1024**, not the earlier draft's 64. A larger
vocabulary batch avoids repeatedly copying/checking the 896-by-64 derivative
right-hand side for 2374 tiny chunks. There are 149 vocabulary batches. Native
weight conversion still uses 64-by-896 chunks with the same 262144-element
cache cap. This is a bounded engineering choice, not a speedup claim.

## Plan and admission

The plan schema is `codex-llm-streamed-pilot-v1`. Required fields are:

- `baseline_plan`: the exact captured original 32-column plan, including its
  original prefix, proposal, and signed-point pins.
- `baseline_attempt`: the owned `box-derivative-001` directory.
- `baseline_receipt_sha256`, `baseline_step_sha256`,
  `baseline_actual_inputs_sha256`, `baseline_actual_balls_sha256`: externally
  supplied lowercase SHA-256 strings. No placeholder values are created here.
- `baseline_source_sha256`: explicit hashes for the original boxed driver,
  BallQwen, Jet, affine kernel, cache reader, phase-domain helper, and
  independent Qwen map (`BASE_SOURCES` in the new driver).
- `precision=1024`, `direction_indices=[0,...,63]`, `token_chunk=1024`;
  `envelope={cpu_threads:1,nice:10,memory_gib:18,timeout_seconds:3600}`.
  Optional `purpose` and `basis` are descriptive only.

Admission authenticates the entire baseline receipt and source manifest,
requires successful completion without timeout, and matches the recorded
common-source hashes to current bytes. It checks backend versions, model
assets, exact captured plan, all 32 full output columns, all basis directions,
source/input-file bindings, and complete status. It independently reopens
and authenticates the original prefix/proposal/signed-point lineage and
recomputes the baseline's entire eight-column point comparison.

The actual receiver domain is rebuilt using the existing checked constructor.
Its full records, endpoint witnesses, and canonical hash must equal the
baseline actual domain exactly. The saved domain artifact, constructor,
proposal, plan, and receiver file hashes are cross-bound. A nominal radius
match alone is insufficient. Partial or failed baselines cannot be imported.
The root agent will select real audited pins and commit/push a concrete plan;
this implementation contains no invented research receipt.

## Retention and measurements

Before model construction, retain source/runtime bindings, the full domain,
all actual input balls, and all 64 exact direction vectors. Use the inherited
streamed subclass and the existing receiver-bound 1024-bit prefix cache.
Immediately retain all 896 primals and all 64 full finite columns before any
comparison with the baseline. Check exact dyadic endpoint overlap for every
one of the 32 shared columns and every primal entry; report all 64 column
radius/finite summaries. Overlap is a compatibility check, not a proof of
identical numerical intervals or of an attracting orbit.

Save the head's full-vocabulary argmax diagnostic with the actual domain hash.
A strict margin, if present, concerns only this prototype input box; it says
nothing about an unknown periodic orbit. Save total forward time, decoder
start-to-last-block time, and the remaining final-normalization/readout time.
The latter explicitly includes final RMSNorm and return overhead; it is not
presented as a perfectly isolated head timer. Chunk progress events and peak
RSS remain available. Failed comparisons retain the complete output.

The existing launcher must enforce CPU1, nice10, an 18 GiB address-space cap,
a fresh owned process group, and at most 3600 seconds. Parent-reported baseline
forward time is 988.570 seconds; doubling gives 1977.14 seconds before overhead
as a planning estimate only. Streamed accumulation and larger native batches
may change both time and width. No completion or memory-fit guarantee follows.

## Synthetic verification

Source SHA-256: `3400b36477168ffbbc2ba9d922641e1627c18bd264d63f700ecff9f811de82fc`.
Tests SHA-256: `c6b99a7f35c732f6d1e9452c45984ab33e917eef3820acd909cb7db34002915a`.

The new pilot's 34 cases plus 55 head cases passed: **89 passed in 6.92 s**,
CPU1, bytecode and pytest cache disabled. They cover completed lineage,
source/plan/backend/domain/input/comparison tampering, incomplete/timeout
baselines, all 32 overlap rows and all 64 finite rows, unchanged parents,
full-output retention before comparisons, forward failure, committed-plan
scope, and isolated `-I --help`. Tiny fixtures substitute a 64-coordinate
synthetic state to exercise all 64 directions; they do not evaluate Qwen or
establish target runtime. Independent review and a pushed run card remain
required before launch.
