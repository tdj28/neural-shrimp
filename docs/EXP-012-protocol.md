# EXP-012: local GPU qualification for batched short previews

15 September 2026. Prospective engineering qualification using already observed
development cases. No EXP-012 preview integration is permitted until this
protocol, the complete machine plan, both executors, qualification analysis,
tests and dependency bindings are independently reviewed, committed and pushed
to the private origin. Synthetic validation and compilation may precede that
freeze; they do not establish numerical parity or useful acceleration.

This packet asks whether one optional local CUDA implementation preserves the
existing 75-unit preview calculation and saves measured execution time. It
does not create fresh scientific observations, validate geometry utility,
repair the original EXP-008 strict gate, resolve EXP-009's long trajectories,
or authorize paid compute. Existing reference sources, tolerances and raw
attempts remain unchanged.

## Complete known-case input contract

Use every one of EXP-010's 3,456 original preview identities: 24 physical
origins, two phases, four radii and 18 distinct candidate IDs per context.
Retain both historical roles as known development evidence. There are exactly
192 complete contexts; duplicate physical directions retain separate IDs.
No outcome-dependent filtering, favorable-case replacement or missing-input
substitution is allowed.

Sort cases lexicographically by `(role, anchor, phase, radius, candidate)`.
The timing subsets are nested prefixes of 18, 144, 576 and 3,456 cases, each
containing complete 18-entry contexts. The smaller prefixes are workload-size
diagnostics, not representative scientific cohorts. Numerical qualification
requires the complete 3,456-case census at both native step sizes.

Bind the exact EXP-010 receipt, manifest, terminal records and all 6,912
same-step native traces and score records used as references. Bind the
original preview input records, raw input arrays, unperturbed reference
certificates and their transitive source dependencies. Verify receipt
membership, byte counts, hashes and exact identities before use. The plan
must record the realized complete IDs, all subset IDs, their ordered hashes,
the source commit and every source/input binding. A size-only match is
insufficient.

Each arm receives identical binary64 destination alpha/gamma, control delta,
initial state, original period, reference activation samples and phase epoch.
Reconstruction must match the original values, including the initial state
at the declared phase of the certificate. The reference remains the original
2,700 endpoint-excluded activation samples; the epoch remains
`phase_index * original_period / 2700`. Record and validate realized arrays,
not only requested IDs.

## Numerical paths and unchanged waveform meaning

Run fixed-step RK4 independently from the exact initial state for
`0 <= t <= 75`, at `dt=0.0025` and `dt=0.00125`. These mean 30,000 and 60,000
steps, output strides 40 and 80, and the exact original 751 binary64 sample
times at cadence 0.1, including both endpoints. Do not use prefixes of longer
traces or interpolate a different recording grid.

The CPU baseline calls the unchanged `native/hopfield_drift.cpp` ABI and its
included original model/RK4 implementation through `ctypes`. It uses a
persistent `ThreadPoolExecutor(max_workers=4)` and one BLAS/OpenMP thread.
Each CPU task integrates and scores its own case. This avoids charging the
CPU baseline one process launch per trajectory. The underlying native routine
has only call-local mutable state; its `ctypes` calls release the GIL.

The separate CUDA implementation maps independent trajectories to GPU work
items. It preserves the original RHS expressions, RK4 stage order, endpoint
derivative reuse, sampling, finite-state checks, final state and progress
semantics. Inputs, constants, intermediate state/arithmetic, outputs and
storage are binary64; use double-precision `tanh`. No FP32/TF32, mixed
precision, fast math or reduced-precision intermediates are permitted.
Disable fused multiply-add contraction for the initial qualified kernel and
record all compiler flags, target architecture, division/square-root and
subnormal settings. A later kernel, compiler, precision or architecture change
requires its own qualification rather than inheriting this result.

Both arms score their returned states using the same unchanged CPU scoring
routine, with four CPU scoring threads available per worker. The GPU arm
integrates its batch before dispatching CPU scores. Both calculate
`tanh(states)` on CPU, then use the existing 128-bin phase scan and bounded
refinement to fit one constant offset over the whole preview. All component
and window errors reuse that phase. Retain synchronous and aligned aggregate,
component and window scores, phase/search metadata, cadence diagnostics and
the error at the original saved phase. There is no frequency fit, time warp,
new orbit, windowwise rephasing or changed quality endpoint.

This packet does not qualify adaptive DOP853, parameter sensitivity,
eigenvectors, ridge/SVD/QR fitting, atlas construction, event detection,
primitive-period or Floquet calculations, or any 300-unit trajectory.

## Separate correctness gates

All gates are fixed engineering compatibility requirements, not certified
integration-error bounds or task-quality margins. Report each gate separately
and require all of them before allowing use of this GPU path.

1. **CPU reference reproduction.** The current threaded native CPU arrays
   must equal EXP-010's saved same-step state and time arrays exactly, for
   every required case and both step sizes. Reproduce the complete saved CPU
   score tree with the existing recursive verifier: numerical
   `rtol=1e-12, atol=1e-14`, exact keys, shapes, integers and categories.
   This prevents two new arms from agreeing on a shared input or scoring bug.
2. **Complete output identity.** Require exact planned IDs/order, float64
   state/time arrays, shape `(N,751,3)`, unchanged initial rows, complete
   per-case statuses, correct steps/retained-row progress and final-state
   equality with the last retained row. All required values must be finite.
   CPU progress may be constructed only after the native ABI's actual return
   status, progress and final/sample checks pass; missing output never
   implies success.
3. **Same-step CPU/GPU waveform parity.** At every stored state coordinate
   require
   `abs(cpu-gpu) <= 1e-10 + 1e-10 * max(abs(cpu),abs(gpu))`.
   The maximum absolute activation difference must be at most `1e-10`.
   Record state and activation maximum, RMS and per-component differences,
   including locations of maxima; aggregate agreement alone is insufficient.
4. **Same-step metric parity.** Each synchronous and aligned aggregate and
   per-component RMSE, each corresponding window RMSE and the original
   fixed-phase RMSE must differ by at most `1e-10`. Global phase differences
   are measured circularly modulo one cycle and must be at most `1e-8`
   cycles. Metric-tree keys, window identities/membership, sample times/counts,
   period, cadence categories, phase-search method/bin count and refinement
   success must agree exactly. Retain all phase-search fields;
   differences in refinement iteration/evaluation counts are reported
   diagnostics because tiny changed inputs can alter optimizer iterations.
   They do not substitute for any waveform, metric or phase gate.
5. **Decision parity.** For all 192 contexts at each step size, independently
   sort all 18 errors by `(RMSE, candidate_ID)`. Require exact complete order,
   minimum candidate, ordered top four and minimum candidate within each
   fixed control: axes `[0,4,8,12]`, diagonals `[2,6,10,14]` and local coverage
   `[0,8,16,17]`. Save all candidate scores, gaps and signed changes. A small
   error change does not excuse an altered decision.
6. **Repeat and layout parity.** At each step size compare all three full-size
   warm GPU runs pairwise, using complete output bits rather than sums or
   means. Compare one additional full-census run in reversed ID order and one
   run partitioned into consecutive chunks of 257 cases, with a final
   115-case chunk, against the canonical output after restoring ID order.
   Require exact output bits for states, final states and sample times, and
   exact statuses/progress. There are three pairwise repeat comparisons and
   two layout comparisons per step size. CPU repeated results must also
   reproduce the saved reference. These observations qualify the specified
   layouts/runtime; they do not prove determinism under every possible load.

Save arrays and metric packets from every required batch, including cold,
repeat and layout runs. Independently read each saved packet back and replay
its complete score tree under the original strict recursive scoring
tolerance. This saved-array check is distinct from cross-device parity. The
new engineering tolerances never replace or relax an older verifier.

Any missing required case, nonfinite output, invalid identity, CPU-reference
mismatch, parity/ranking difference or nondeterministic tested output fails
qualification. Keep every observed discrepancy. Do not choose the better
step size, omit a radius, accept a completed subset, switch math flags or
relax a threshold after results.

## Fixed timing schedule and cost boundary

Use two separate persistent spawned workers, one CPU and one GPU. The parent
does not initialize a CUDA context. Each child loads the read-only bound
contexts once and owns its four-thread CPU pool; only the GPU child loads and
initializes the CUDA runtime. Never fork a CUDA-initialized process.

The first invocation in each worker is a separately reported cold 18-case
batch at `dt=0.0025`. Its timing includes child startup, context loading,
first thread-pool/runtime initialization, host input preparation, integration,
scoring and the storage/verification steps below. Record native/CUDA build
times separately and include them in total attempt/startup accounting. Cold
calls are retained evidence but are excluded from warm medians.

For step sizes in order `[0.0025,0.00125]`, batch sizes in order
`[18,144,576,3456]`, and repetitions in order `[0,1,2]`, run matched arms
sequentially. Even repetitions use CPU then GPU; odd repetitions use GPU then
CPU. This produces **48 required warm batches**, 24 per backend, without
concurrent execution of the timed CPU and GPU arms. Both arms receive the
same subset and task before the next pair begins. Record the complete order;
never discard the first repetition or select the fastest run. Run the four
additional GPU layout batches separately from the warm timing comparison.
The complete dispatch census is 54 batches: two cold, 48 warm and four
layouts, totaling 64,188 case integrations with repetitions included. There
are 6,912 distinct same-step case pairs in the required numerical comparison.

The primary per-batch wall timer starts before construction of host input
arrays from the already loaded contexts and ends only after all of:

- integration, including device allocation, transfers, launch and explicit
  GPU synchronization where applicable;
- unchanged CPU scoring for every case;
- writing the complete batch compressed NPZ and JSON artifacts;
- computing their hashes and reading them back; and
- validating saved arrays/identities and replaying every saved score.

The timed JSON is `scores.json`; the timed array file is `traces.npz`.
The worker's `record.json`, which stores the completed timing, and the final
attempt receipt/inventory are written after this per-batch timer stops. Their
cost remains in total attempt wall time, and the distinction is retained in
reported timer metadata.

Use the same compression, storage root and verification requirements for
both arms. Array allocation, required host/device copies and GPU resource
release belong inside this boundary. Resource inventory immediately before
each batch is outside the per-batch timer for both arms, and its cost remains
in total attempt wall time. Record initialization/build, input preparation,
integration, transfer/kernel/synchronization where available, scoring,
serialization/hash and readback-verification time separately. CUDA event time
alone is a secondary kernel measurement, never the primary speedup.

Report all three paired ratios `CPU_wall/GPU_wall`, their median, absolute
times and ranges for every size/step cell. The intended deployment cell is
the complete 3,456-case batch at `dt=0.00125`. Declare this optional GPU path
worth using only if every required correctness/layout test passes, all three
GPU wall times in that cell are less than their paired CPU times, and the
median paired speedup is at least **1.25**. Otherwise retain the CPU default.
A favorable smaller batch, kernel-only result or other step size cannot
replace this rule. Report startup/compilation costs and the break-even number
of full batches using measured cold costs and median warm wall times; if
there is no positive measured saving, report no finite break-even.

The batch timing measures this preview integration/score/storage/verification
pipeline on this shared machine. It excludes acquisition of new reference
orbits, sensitivity, geometry and model training, and therefore does not
establish complete controller cost or net scientific utility. State cache,
device contention and input-loading scope explicitly. Do not call the old
process-per-trace EXP-010 wall time the optimized CPU comparator.

## Shared resources, ownership and failure evidence

Before launch and before each batch, record host CPU load, available RAM,
output capacity, GPU model/UUID/index, free/used VRAM, utilization and active
processes. A sandbox visibility failure means unknown capacity, not an idle
GPU. Use authorized host-level inventory for capacity and final cleanup.
Other agents' allocations remain theirs; never reset the device, kill their
processes or change the driver to make room.

Cap application-owned GPU allocations at **1 GiB** and at no more than half
the currently free VRAM. Account separately for runtime/context memory when
reporting total process VRAM. Preserve at least **2 GiB available host RAM**
beyond the estimated peak task working set, rechecking before each batch.
Refuse an unsafe launch rather than reducing precision or changing cases.
Keep four CPU worker threads and numerical-library threads of one; do not
silently increase concurrency after inspecting timings.

The whole attempt ceiling is **3,600 seconds**, including builds, cold calls,
all warm/layout batches and qualification analysis. Each worker startup has
a **180-second** ceiling; each dispatched batch has a **180-second** ceiling.
Use explicit owned process IDs, parent-death handling, bounded messages and
joins. A timeout terminates only that task's owned worker/process group.
Preserve available raw buffers, partial progress, errors and driver/compiler
messages. Never infer an absent case result from another case in the batch.

Run all fixed tests even when a numerical or performance gate has failed,
provided the runtime remains safe and usable. A resource, worker, device or
whole-attempt failure may prevent later tests: every unexecuted planned batch
and case must then receive an explicit reason/status. No automatic replacement
worker retry, hidden CPU fallback in the GPU arm, favorable rerun, in-place
resume or paid launch is authorized by this protocol. Failed/incomplete
attempts remain immutable; an amended attempt requires a new complete freeze.

Final failure preservation and receipt sealing remain possible after the
execution deadline. Report that sealing interval separately; it is not an
excuse for further integration after the deadline. Verify termination of both
owned workers using host process identity, including start-time identity when
available, and inspect host GPU ownership after completion. PID invisibility
inside a sandbox is not host-cleanup proof.

## Freeze, storage and audit

Require the exact clean committed source, complete plan and dependency hashes,
and a live private origin branch matching the source HEAD before execution.
Record the verified remote revision and UTC time. Bind both frozen CPU source
files and new CUDA code, compiler commands/output/library hashes, device and
driver/runtime/library versions, arithmetic flags, numerical-library versions,
thread environment and precise timer definitions. Recheck source/input
bindings after collection.

Write a new exclusive attempt below
`/data2/neural-shrimp-data/raw/EXP-012/`, outside the entire resolved Git
checkout, including symlink targets. Refuse existing paths. Keep bulky batch
arrays and detailed results external; retain source, tests, protocol, compact
plan/summary and artifact receipts in Git. Save expected/observed batch and
case counts, complete immutable outputs, append-only status/process journals,
all timing repetitions, gate details, every failure and a full SHA-256/byte
inventory. Verify the realized file census without silently skipping files.

Pre-freeze synthetic tests cover invalid dimensions/dtypes/nonfinite inputs,
grid/stride/capacity violations, odd batch lengths, exact identity and
reordering, invalid output/status/progress, missing/duplicate cases,
cross-device and ranking gate failures, timeout/worker-loss bookkeeping,
source/receipt/output-path violations and resource limits. Test scoring and
timing schemas without integrating known cases. Independently reconstruct the
plan census and all gates; review the native arithmetic and ownership paths.

After execution, audit source/receipt/input hashes, complete planned statuses,
saved-array metric replay, same-step CPU/GPU checks, all decisions and layout
comparisons, matched timing arithmetic and host cleanup. Any additional
post-run analysis must be labeled separately and must preserve this decision
rule and the original evidence.

Execution uses the project environment:

```bash
.venv/bin/python tools/run_gpu_preview_qualification.py prepare
.venv/bin/python tools/run_gpu_preview_qualification.py validate
# Commit, review, test and verify the private push before this command.
.venv/bin/python tools/run_gpu_preview_qualification.py run
.venv/bin/python tools/run_gpu_preview_qualification.py audit \
  --audit-output /data2/neural-shrimp-data/audits/EXP-012/post-run-replay-001
```

The audit command verifies the immutable raw file inventory and reproduces
the original gate/timing/cost results into a new exclusive audit directory.
It does not reintegrate or modify the research attempt. This branch builds on
the unmerged server follow-up at `dca1133a0caeea8fd0ee92cb08f9977b2a37317c`;
private branch pushes are the execution freeze, not a claim that the retained
EXP-008 main-branch gate has passed. GitHub PR creation remains pending account
authentication; the locally reviewed branch must not be merged around that
gate.
