# gdmap CUDA backend: engineering qualification

**Engineering qualification on artificial fixtures, 6-7 October 2026. Not a
scientific result.** Only artificial fixtures with seeds 99990-99999 were run.
No planned study system or seed (EXP-028 uses 28001, 28002, 28011, 28012) and
nothing under `experiments/` was touched. The NumPy engine
(`src/neural_shrimp/gdmaps.py`) remains the reference and the default. This
backend is an optional engineering candidate: a study that uses it needs its
own frozen qualification against its own endpoints (compare the failed
EXP-012 GPU parity gate).

**Human-friendly TLDR.** This note teaches nothing about how neural networks
behave in shrimp regimes. It qualifies a faster instrument for drawing the
learning-rate maps that will ask that question. On the GPU each map cell
repeats the CPU reference's arithmetic operation for operation; only tanh and
log differ, in the last bit, for about one input in four and one in 250
respectively. That is enough for chaotic cells to part ways after tens
to hundreds of steps, as any two correct float64 implementations would.
Fixed points, periodic orbits (up to phase), drift labels, divergence and the
Sohl-Dickstein convergent flags agree on the tested grids, with the exceptions
listed below. Throughput has not been formally measured yet, because the GPU
is reserved by other work; that benchmark is deferred.

## What was built

| File | Role |
|---|---|
| `native/gdmap_cuda.cu` | Kernels: one thread advances one learning-rate cell; recurrence screen; evaluation, device-math and screen entry points for qualification |
| `src/neural_shrimp/gdmaps_cuda.py` | ctypes wrapper. `run_grid_cuda(grid, theta0, steps, **options)` takes the options of `iterate_and_screen` (defaults read from its signature) and returns the same keys, order, dtypes and shapes as `run_grid`, including `window_norm_growth`, `last_evaluated_loss` and the drift label. `return_metadata=True` adds a metadata dictionary (device, NVIDIA driver, CUDA runtime and driver API versions, nvcc and host compiler versions, library hash, chunking, device memory, timings) |
| `tools/build_gdmap_cuda.py` | Builds the library outside Git and writes a JSON build receipt (command, versions, source and library SHA-256, ptxas register report) |
| `tools/qualify_gdmap_cuda.py` | The qualification recorded here |
| `tools/bench_gdmap_cuda.py` | Throughput benchmark, ready but not yet run for the record (see Throughput) |
| `tests/test_gdmaps_cuda.py` | 27 tests: CPU-side contracts always; GPU checks skip cleanly without nvcc, g++-12 or a CUDA device |

Design. The state of a cell is the full parameter vector, stored
structure-of-arrays in device memory together with its tangent, rates and
accumulators; the data `x`, `y` sit in shared memory. Per step, pass 1 computes
activations (and pre-activation directional derivatives during the FTLE
window) into per-cell scratch, then predictions, residuals and the loss; pass
2 reduces over samples per hidden unit and hands each parameter to an update
that writes `theta - rates * grad` and `v - rates * (H v)` in place. Kernels
are instantiated for each input dimension 1-16; width up to 64 and
`n (d + 1) <= 6144` (n up to 300 at d = 16) are supported. Cells run in chunks
sized to a device-byte budget (default 2 GiB, never more than half the free
memory); the final `tail_steps` updates run in sub-chunks that fill a tail
buffer, screened on the device (default) or copied back to the CPU
`screen_tails` (`screen="host"`). Tangents are drawn on the host exactly as the
CPU does; the Sohl-Dickstein measures apply the CPU function to the device's
early losses; the drift label and `window_norm_growth` use the CPU expressions
on the device's window norm and final states.

## Build

```
nvcc -O3 -std=c++17 --shared -Xcompiler=-fPIC --ftz=false --prec-div=true \
     --prec-sqrt=true -arch=sm_89 -Xptxas=-v --fmad=false -ccbin g++-12 \
     native/gdmap_cuda.cu -o gdmap_cuda.so
```

Toolchain: nvcc 12.4 (V12.4.131), g++-12 12.5.0, static CUDA runtime 12.4,
NVIDIA driver 580.178.04 (driver API 13.0), RTX 2000 Ada (sm_89, 22 SMs, FP64
at 1/64 of FP32). Registers per thread for the step kernel: 122 (d = 2), 154
(d = 8), 255 with 92 bytes of spill (d = 16). Library bytes differ between
rebuilds of the same source because nvcc embeds temporary names, but the device
code does not: SASS from two builds of one source hashed identically.

## Precision and determinism contract

| Role | Contract |
|---|---|
| Storage | float64 for parameters, tangents, activations, residuals, losses, window norms and tails; no reduced-precision copy anywhere |
| Multiply | IEEE float64, round to nearest, one rounding per operation, except inside the two FMA chains below |
| Accumulate | float64 in the CPU reference's order, per thread: FMA chains for both `tensordot` contractions, over the input dimension for pre-activations and over samples for the first-layer gradient, split into the K blocks of OpenBLAS's level-3 driver (GEMM_Q 256, UNROLL_M 4); numpy's two-lane SSE `einsum` loop over hidden units (reverse four-pair blocks, separate multiply and add); sequential sums over samples; numpy pairwise summation for the ridge term, tangent norm and window norm. No atomics, no cross-thread reductions |
| Output | float64 arrays and int64 or bool labels, keys, order and shapes identical to `run_grid` |
| Division, square root | IEEE correctly rounded (`__ddiv_rn`, `__dsqrt_rn`) |
| tanh, log | CUDA libdevice; measured accuracy below |
| Subnormals | Not flushed (`--ftz` affects only single precision) |

Every operation is written with explicit intrinsics (`__dmul_rn`,
`__dadd_rn`, `__dsub_rn`, `__fma_rn`), so neither the compiler nor `--fmad`
can contract or reorder it. The orders were established on this host against
exact rational emulation of numpy 2.5.3 with scipy-openblas 0.3.34
(DYNAMIC_ARCH Haswell kernels).

Determinism. Each thread computes sequentially and independently, so results
do not depend on block size, chunking, sub-chunking, launch segmentation or
scheduling. Observed: repeated runs bit-identical on all fixtures; a run with
different chunk, tail-chunk and launch sizes plus host screening bit-identical
to the default run; device iteration bit-identical to the evaluation entry
point plus the NumPy update. This is "no difference observed on this stack",
not a proof for other drivers, toolchains or devices.

### FMA decision

Build with `--fmad=false` and use explicit FMA exactly where the CPU
reference's OpenBLAS dgemm uses FMA chains. Disabling contraction does not mean
avoiding FMA: the reference itself fuses in its BLAS contractions and rounds
separately in its elementwise and `einsum` loops, and this policy matches both.
Evidence (64 random parameter and tangent vectors per fixture):

| Build | Identity net, even d: loss / grad / HVP elements bitwise equal to CPU | tanh nets: grad / HVP bitwise, max relative difference |
|---|---|---|
| mirror (`--fmad=false`, intrinsics) | 100% / 100% / 100% on S1-like, S2-like, W16 | 29-49% / 25-38%; at most 2.6e-16 / 3.4e-16 |
| contract (`--fmad=true`, plain operators) | 61-66% / 48-73% / 44-64% | 23-38% / 18-32%; at most 3.8e-16 / 3.4e-16 |

Free contraction therefore destroys exact agreement without improving
accuracy. At grid level both builds are dominated by the tanh and log
differences and agree with the CPU about equally well (tables below). Its
throughput benefit is unmeasured: the interleaved A/B in
`tools/bench_gdmap_cuda.py --ab-library` is deferred with the benchmark. The
contract build is diagnostic only.

## Qualification results

Receipt: `../neural-shrimp-data/audits/gdmap-cuda-qualification/attempt-002/qualification.json`
(SHA-256 `4d255eebd051a5f6c633e58cda3a03188a6a2cf56db9474d2c766e3e8dd4cf80`, copied
to [gdmap-cuda-qualification.json](gdmap-cuda-qualification.json)), from source
commit `d7862d187eced4a616964c541f20e7a7fc197375` with no tracked changes, two
CPU cores (30, 31), one BLAS thread, mirror library SHA-256
`f677ded68702972bd4400fcf2b4be8dd415b4293c073570ca885f4b44a379b2e`, contract
library `19bf1fd9327800bd0e46e433779a45ca633b27405225ef779fdf168b058b04ed`.
Peak GPU memory of the process was below 100 MiB. Attempt-001 (pre-rebase
source) ended with its session before writing any output; it is preserved with
an `INCOMPLETE.md` note and has no result.

Fixtures. S1-like: tanh, d = 8, width 8, no bias, `in_scale = sqrt(2/8)`,
`out_scale = 1/8`, `loss_scale = 1`, n = 72 = parameter count, data and
initial parameters standard normal (seed 99990). S2-like: tanh, d = 2, width 3,
biases, `loss_scale = 0.5`, ridge 1e-3, 24 teacher examples (seed 99991),
`TanhRegressor` initialization (seed 99992). W16: as S1-like with d = width =
16 and n = 272 (seed 99998). Odd-d control: d = 3, width 5, n = 30 (seed
99997). Rates are decades around the Gauss-Newton edge rates `2 / lambda` at
initialization.

### Transcendental functions (15,000 samples each, against a 60-digit reference)

| Implementation | tanh correctly rounded | tanh max error | log correctly rounded | log max error |
|---|---|---|---|---|
| numpy 2.5.3 (own SIMD tanh) | 80.0% | 1.14 ulp | 99.85% | 0.51 ulp |
| glibc (Python `math`) | 72.7% | 2.00 ulp | 99.85% | 0.51 ulp |
| CUDA libdevice | 90.0% | 1.26 ulp | 99.67% | 0.53 ulp |

CUDA's tanh equals numpy's for 76.1% of inputs, its log for 99.6%. These are
the only arithmetic differences from the CPU reference on even-d networks.

### Single steps

Loss, gradient and HVP from the evaluation entry point, which shares the step
kernel's code: identity networks with even d are bitwise identical to the CPU
(table above), including n = 272 where OpenBLAS splits the sample contraction
into two blocks; the S1-like, S2-like and W16 tanh networks agree to at most
2.6e-16 (gradient) and 3.4e-16 (HVP) relative to the largest element (the
odd-d control: 2.7e-16 and 7.2e-16). The odd-d control shows the one known
gap: with identity activation its loss is bitwise equal but 86.5% of gradient
and 87.1% of HVP elements are (the last first-layer row goes through
OpenBLAS's single-row remainder kernel, whose order depends on the batch width).

### Grids with the production windows

S1-like and S2-like: 12 x 12 cells, 6,000 updates with the EXP-028 windows
(tail 384, exponent 4,000 after 1,000 alignment, score horizons 100, 500,
1,000, default tolerances). W16: 8 x 8 cells, 600 updates (tail 128, exponent
300 after 100, horizon 100). Horizons are shorter than EXP-028's 20,000 updates
to keep GPU use minimal.

| Fixture (mirror build) | Status agreement | CPU labels | Divergent agreement | Divergence step equal | Chaotic: \|dFTLE\| median / max |
|---|---|---|---|---|---|
| S1-like | 143/144 | 39 divergent, 54 chaotic, 51 drift | 38/39 | 35/39 (early escapes 20/21) | 0.051 / 0.38 |
| S2-like | 143/144 | 40 divergent, 14 periodic, 7 chaotic, 58 unresolved, 25 drift | 40/40 | 40/40 | 0.0077 / 0.022 |
| W16 | 64/64 | 21 divergent, 26 chaotic, 17 drift | 21/21 | 18/21 (early escapes 13/14) | 0.0080 / 0.24 |

The two disagreements: an S1-like cell that escapes after a long chaotic
transient on the CPU had not escaped by update 6,000 on the GPU (labelled
chaotic), and an S2-like chaotic cell settled on a period-2 orbit on the GPU.
Both are finite-horizon labels of sensitive trajectories.

Regular cells. All 14 S2-like periodic cells (period 2) have equal status and
period. After aligning phase (iterating the CPU map from the CPU final state),
final states agree to at most 1.6e-15 for the 9 cells converged to a recurrence
residual of at most 1e-3, and to at most 1.1e-7 for all 14 (within the
screen tolerance); one cell had reached the orbit at the opposite phase. On the
test-suite S2-like grid (8,000 updates) the fixed point agrees to 5.6e-16,
deeply converged period-2 cells to at most 1.6e-15, and cells still on a slow
approach (residual 0.03-0.13) to 1.5e-11 to 8.5e-9. Unresolved S2-like cells:
58/58 agree; drift cells: 25/25 (S2-like, \|d window_norm_growth\| at most
8.9e-16), 51/51 (S1-like, median 3.2e-8, max 0.018), 17/17 (W16, max 0.0052).

Sohl-Dickstein measures. Convergent flags agree for every cell and horizon on
S1-like and S2-like, and for 63/64 W16 cells at horizon 100. Scores agree to
the rounding level where the early trajectory is not chaotic (S2-like
divergent and drift cells: at most 4e-15 relative at horizon 100; S2-like
unresolved cells: median 0, max 1.7e-5) and differ where it is: up to 11% on
S1-like chaotic cells, and up to 130% at horizon 100 (29-fold at horizon 1,000)
on S1-like divergent cells whose escape step moved.

Trajectory separation. Stepping all cells on the CPU and, with the GPU
gradient, by the same NumPy update shows where differences come from: CPU and
GPU states first differ at update 1, by one to a few ulps (3e-16 to 8e-15),
and the separation then grows exponentially at 0.59-1.53 per update in the
six cells with the largest score differences. In the W16 cell whose horizon-100
flag differs, separation stays near 1e-15 for 50 updates, then grows at 0.77
per update to 0.2 at update 100, exactly where the flag reads its last twenty
losses. Rounding-seeded chaos, not a code difference, explains the chaotic
and transient disagreements.

Contract build, same grids: status agreement 142/144 (S1-like), 144/144
(S2-like), 64/64 (W16); phase-aligned periodic agreement 2.0e-15 for converged
cells; otherwise similar statistics.

## Throughput

**Deferred.** The formal benchmark (`tools/bench_gdmap_cuda.py`: S1-like and
W16 at 8,192 and 65,536 cells, plain and FTLE steps, one-core NumPy comparison,
nvidia-smi load before and after every run, optional interleaved
mirror/contract A/B) was not run because the GPU is reserved by other work and
the CPU is shared with the 14-worker EXP-028 run. Run it when both are free.

Preliminary development timings, not receipted and from the pre-rebase kernel
(same arithmetic; only the recurrence screen and window norm changed), with
two Blender processes holding the GPU at 100% utilization: S1-like with
16,384 surviving cells cost 0.86-1.07 microseconds per cell-update for plain
updates and 1.46-1.58 with an HVP every update; W16 with 4,096 cells cost 9-11
and 18. The NumPy engine on one core cost 11-17 (S1-like) and 58 / 177 (W16,
plain / HVP) on comparable cells. tanh is about two thirds of a plain S1-like
update (identity activation: 0.39). Small grids leave the GPU latency-bound and
can be slower than one CPU core: 64 W16 cells for 3,000 updates took 68 s of
kernel time against 14 s on the CPU. Do not plan compute from these numbers.

## Limitations and deviations from CPU semantics

1. tanh and log differ from numpy's in the last bit. Chaotic trajectories
   therefore decorrelate at the local Lyapunov rate; periodic orbits can be
   reached at another phase; escapes after chaotic transients shift or move
   past the horizon; score values and finite-horizon labels of sensitive cells
   can change (status agreement 99.3-100% on the tested grids).
2. Odd input dimension: the reference's OpenBLAS reduces the last first-layer
   gradient row in a batch-width-dependent order that is not mirrored.
3. The CPU reference is not batch-invariant: when exactly one cell remains in
   its batch, NumPy sums the loss and bias gradients pairwise instead of
   sequentially. The GPU always uses the multi-cell order.
4. The mirrored orders are those of this host's numpy and OpenBLAS kernels. On
   another CPU or BLAS the reference itself changes at the rounding level.
5. Invalid screen settings or score horizons raise the CPU's errors before
   iteration instead of after it.
6. Supported: TwoLayerNet only; d <= 16, width <= 64, `n (d + 1) <= 6144`,
   steps <= 2^30, sm_89 only (other devices are refused pending their own
   qualification).
7. Horizons here are 6,000 updates for S1-like and S2-like, not EXP-028's
   20,000; the formal benchmark and the FMA throughput comparison are deferred.
8. The host thread spin-waits on CUDA events (one busy core per run). No
   multi-GPU or stream concurrency.

## Reproduce

```
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src python tools/build_gdmap_cuda.py OUT/build-mirror
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src python tools/build_gdmap_cuda.py --variant contract OUT/build-contract
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src taskset -c 30,31 python -u tools/qualify_gdmap_cuda.py \
    --library OUT/build-mirror/gdmap_cuda.so --contract OUT/build-contract/gdmap_cuda.so \
    --out OUT/qualification.json --side 12 --steps 6000 --w16-steps 600
NEURAL_SHRIMP_GDMAP_CUDA_LIBRARY=OUT/build-mirror/gdmap_cuda.so PYTHONPATH=src python -m pytest tests/test_gdmaps*.py
```
