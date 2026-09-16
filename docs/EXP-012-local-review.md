# EXP-012 local implementation review

16 September 2026 UTC (15 September locally). This is pre-qualification
engineering evidence. No EXP-012 known-case preview has been integrated by
this review. The CPU reference remains the default.

## Optional backend

`native/hopfield_cuda.cu` and `tools/hopfield_cuda.py` provide one persistent
CUDA thread per independent trace, with float64 parameters, states and RK4
stages. The model expressions, stage order, first/endpoint RHS reuse,
sampled endpoints, finite checks and attempted-step/retained-row counters
were checked against the frozen CPU implementation. An independent reviewer
found no blocking arithmetic, ctypes ABI or partial-prefix discrepancy.

The wrapper validates shapes, finite values and integer grids, caps steps at
1,000,000, and checks both a caller-specified byte ceiling and half the current
free VRAM. Requested device storage is exactly `24*N*T + 84*N` bytes.
It retains numerical-failure prefixes, final attempts and counters; CUDA API
errors retain the available host payload. The implementation frees only its
own allocations/events and never resets the device. Import/build does not
initialize CUDA. Loading and device queries belong in the dedicated spawned
GPU worker; the parent must not fork after CUDA initialization.

Compilation succeeded with nvcc 12.4.131 and explicit g++ 12.5.0 using
`sm_89`, `--fmad=false`, `--ftz=false`, `--prec-div=true`,
`--prec-sqrt=true`, and no fast math. Timeout/raised-interruption handling
terminates the owned compiler process group and preserves compiler output;
the qualification parent supplies its normal raising SIGTERM handler.
Different host/device `tanh` implementations can still round differently:
this review does not establish trajectory parity.

Backend source hashes at this review:

| File | SHA-256 |
| --- | --- |
| `native/hopfield_cuda.cu` | `a5970662ed47b22d4a497003c8953cefc477983468001ae63b6d477abb685213` |
| `tools/hopfield_cuda.py` | `b69b57170b5ab51128da54dfea887f2c4e4271d33d7406cb0acea9360306dfd4` |
| `tests/test_hopfield_cuda.py` | `7f015bcc5b9aa5bd5e54fbe31d53ca1d5b040ae05fa1455b45a6ddc1c5458e38` |

## Synthetic checks and retained GPU smoke

The CPU-only invocation
`.venv/bin/python -m pytest tests/test_hopfield_cuda.py -q` passed **35 tests**
with the one optional GPU test skipped. It covers input/grid/memory rejection,
full and partial native output contracts, CUDA-error payload retention,
architecture/headroom guards, compilation flags/receipts and compiler-timeout
cleanup. The independent reviewer separately reproduced 35 passes and one
skip with both GPU-test environment variables unset.

The exact optional GPU test subsequently passed in 0.28 seconds on the host,
bringing all **36 collected tests** to a pass across these invocations.
It used four predetermined dummy initial states at one and four RK4 steps,
including the zero equilibrium, with preset `rtol=2e-13, atol=2e-14` against
an independent scalar implementation. All eight tiny traces had zero maximum
absolute difference. These were not saved scientific initial states, and this
result establishes neither 75-unit compatibility nor useful acceleration.

Evidence is external and exclusive at
`/data2/neural-shrimp-data/audits/EXP-012-cuda-synthetic/attempt-001/`:

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| `receipt.json` | 3,380 | `a88fb708e20aa1ef760c0b4fb31aef795f23f50477a93d407db8bbcaf2af9ac9` |
| `steps-1.npz` | 2,960 | `8f7fc92e0e88f7aca72b6419a5cb050ad0339e88eb837cf6b882ecafd0e07fa0` |
| `steps-4.npz` | 3,560 | `cbc04d3a79b07c01305ed8fb88ab4908aeaf0d03c44289537696d57dca8b5612` |
| `post-smoke-host-check.json` | 10,326 | `7cbc25c84d336f1558e6134c05a43e0a6dc1490ba8f92a2b2f01e5b11d0ab9a8` |

The unchanged smoke receipt covers 05:13:49.937136–05:13:50.098747 UTC.
The separate host inventory at 05:16:49.577067 UTC found zero exact synthetic
command-signature matches and the same ten GPU-owner PIDs observed before
the smoke test. The original test command exited zero. Its receipt did not
capture its PID, so the amendment explicitly records command-signature
absence rather than claiming a retrospective PID/start-time identity check.
No other process was signaled and no device reset occurred.

The smoke device was RTX 2000 Ada, compute capability 8.9, 22 SMs, with a
reported single/double precision throughput ratio of 64. The actual loaded
CUDA runtime reported **12.3**, driver API **13.0**; compiler version alone
does not identify the loaded runtime. `ldd` resolved `libcudart.so.12` to
`/usr/local/cuda-12.3/targets/x86_64-linux/lib/libcudart.so.12`.
The qualification must bind this actual runtime. The tested library SHA-256
is `9a43f9de0129d6c996a6b74ba50ad117acc847d2771c245a9d5d12d8f438be2e`;
the second successful compilation receipt is
`/tmp/neural-shrimp-cuda-build-002/hopfield_cuda-build.json`, 1,582 bytes,
SHA-256 `40c009baf624a63224604288e6e5a51d522631039d41b5e3642dd653b001246e`.
Unchanged copies of that receipt and the tested 68,048-byte library are now
preserved in the separate exclusive directory
`/data2/neural-shrimp-data/audits/EXP-012-cuda-synthetic/synthetic-builds/build-002/`.
Its 646-byte `manifest.json` has SHA-256
`135e74d2a44bbca3071b58375ce191638d644373d83f48e1952c7fe3d3f221c5`;
this preservation did not alter the original smoke receipt or its inventory.

## Performance scope and runner review closure

An earlier read-only saved-array profile replayed 19 long and 16 short
sentinels without integration. Short-preview CPU scoring had median
0.00804 seconds and in-memory compression 0.000723 seconds; long-trajectory
scoring had median 0.01793 seconds. Phase interpolation dominated the scoring
profile. Existing task timers combine integration, scoring and serialization;
they cannot isolate RK4 throughput. The old process-per-trace scheduler also
adds overhead, so the GPU comparison needs the planned persistent threaded
CPU arm and the same scoring/storage/replay boundary.

The local runner review is complete with **no remaining blocking finding**
in the reviewed implementation. This closes implementation review, not the
prospective freeze or the empirical qualification. The final generated plan,
integrated test result and verified private push still precede known-case
execution.

The review found and closed missing saved-array score replay, incomplete
repeat/case checks, numerical-leaf validation gaps, unpreserved partial work,
and interruption/receipt-sealing gaps. The final runner:

- validates all 54 planned batch identities and every realized case, dtype,
  array shape, complete status, input, sample, final state and progress field;
- compares all 25 CPU batches with the saved same-step native reference,
  including exact arrays and the unchanged strict score-tree replay;
- compares all 29 GPU batches with their canonical CPU reference and retains
  waveform maximum/RMS/component/location diagnostics, complete candidate
  rankings, control winners, score gaps and signed score changes;
- checks all three pairwise full-size GPU repeats and both layout changes at
  each step size, giving ten explicit determinism comparisons; missing
  evidence creates failed/undefined entries rather than disappearing;
- uses the same timed NPZ/score-JSON writes, hashes, readback and strict
  saved-array score replay for both arms; worker timing records and final
  inventory are outside that batch timer and remain in attempt wall time;
- drains submitted CPU futures before another batch, preserves successful
  CPU cases and GPU chunks before later failure, and retains failed-batch
  IDs, partial arrays and available scoring errors/results;
- registers owned workers before launch, records process start identity,
  bounds startup/batches/the whole execution and analysis, and preserves a
  complete terminal census and receipt after headroom, startup or audit
  failure; only owned processes and allocations are released;
- checks the complete source/input bindings before and after execution,
  retains host resource inventory, and disables qualification after any
  infrastructure or postflight failure; and
- provides a separate audit command that checks immutable raw inventory and
  the committed plan, reconstructs the full qualification audit, gates,
  timings and startup/break-even arithmetic, compares them with the original
  results, and writes only to a new external audit directory.

The gate reviewer implemented the four audit/load/ranking/comparison functions
after the initial independent review. A separate test author exercised their
failure contracts; the runner author reviewed the integration. This is local
collaborative review, not an external paid review or an independent empirical
replication. A concurrent edit temporarily removed the runner's interstitial
helper block; lifecycle tests caught it, the block was restored, and the
combined tests passed before any known-case execution.

The final focused invocation, with both optional GPU-test environment
variables unset, was:

```sh
env -u NEURAL_SHRIMP_GPU_TEST_LIBRARY -u NEURAL_SHRIMP_GPU_TEST_RECEIPT_DIR \
  .venv/bin/pytest tests/test_gpu_preview_qualification.py \
  tests/test_gpu_preview_lifecycle.py tests/test_hopfield_cuda.py -q
```

It passed **123 tests with one optional GPU test skipped in 0.73 seconds**.
These include 20 mocked lifecycle tests covering failed headroom, all
54/64,188 planned batch/case statuses, audit failure, process-start failure
after PID assignment, postflight source/input changes, exclusive output paths
and symlink guards. Other tests reject missing/duplicate cases, malformed
scores and arrays, nonfinite or boolean numerical labels, unchanged-infinity
comparisons, artifact substitutions and incomplete rankings. No known-case
ODE, fitting or GPU execution occurred in this review invocation. The full
repository test result is recorded separately by the freeze owner.

The CPU remains the default unless the later complete frozen experiment
passes every required numerical/layout gate and the specified matched
end-to-end deployment speed rule. A successful tiny synthetic smoke does
not replace either requirement.
