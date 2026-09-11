# EXP-004 native engine and driver audit

Recorded 2026-09-11T18:46:01Z, before target outcomes. This is a numerical
implementation audit and a non-target performance calibration, not a neural
shrimp result. The calibration uses alpha = gamma = 0, outside both frozen
target rectangles. No calibration states, periodic classifications or task
outcomes were used to select the experiment.

## Code and environment

| Item | Value |
| --- | --- |
| Native source | `native/hopfield.cpp` |
| Source SHA-256 | `d77afbf22b10abb829eecb41ff6ace262713a8b00e7544f2bfe4d389d661eba4` |
| Test source | `tests/test_hopfield_engine.py` |
| Test SHA-256 | `a88041f9496c560450a0f19e9230b061e22e0394bb6b95cd2ad96b317d649f3a` |
| Calibration library SHA-256 | `1fb7d762c91a8e955750fcb9848221b54428aeac54a8371e15a107852c42e959` |
| Compiler | Apple clang 21.0.0, clang-2100.1.1.101 |
| Platform | macOS 26.5.1, arm64 |
| Python / NumPy / SciPy / pytest | 3.13.11 / 2.5.3 / 1.18.1 / 9.1.1 |

Compilation used `-O3 -std=c++17 -dynamiclib -fPIC`; a separate build added
`-Wall -Wextra -Wpedantic` and emitted no warnings. The calibration library
was temporary and is not a portable runtime artifact. Each target attempt
builds its own library and records its compiler, flags and binary hash.

## Numerical checks

All eight cases in `tests/test_hopfield_engine.py` passed in 0.83 seconds.
The checks cover:

1. The RHS and analytic Jacobian against an independent matrix expression
   and centered finite differences, including nonzero coupling coefficients.
2. Fourth-order step convergence and terminal-state agreement with DOP853
   at relative tolerance `1e-12` and absolute tolerance `1e-14`.
3. Identical flow across different burn/measurement divisions.
4. Exact zero-state moments and explicit tangent-alignment timing.
5. Invalid arguments returning the documented negative error code.
6. Peak times and complete event states against DOP853 event finding,
   plus exact agreement between a truncated chronological ring buffer and
   the corresponding tail of an untruncated buffer.
7. The variational exponent at the zero equilibrium against the exact
   constant-Jacobian matrix exponential.

The parameterized RHS/Jacobian check supplies two test cases. Time integration
fixtures use alpha = gamma = 0. The event test uses a 20-unit horizon and
`dt=0.01`; both event times and states meet an absolute tolerance of `2e-5`.
The convergence test retains its terminal-error requirement of `2e-8` at
`dt=0.0125`. An earlier test ended at `dt=0.025`, for which the measured error
was `1.211245116546949e-7`; an additional step refinement was added rather
than relaxing the terminal-error requirement. This was an implementation
test adjustment before target outcomes.

## Performance calibration receipt

The ctypes binding releases the GIL. The benchmark called the compiled
engine directly with the following fixed settings:

```json
{
  "alpha": 0.0,
  "gamma": 0.0,
  "initial_state": [0.13, -0.09, 0.2],
  "dt": 0.01,
  "burn_steps": 100000,
  "sample_steps": 100000,
  "event_capacity": 64,
  "single_cell_seconds": 0.03581554163247347,
  "parallel_cells": 16,
  "parallel_workers": 4,
  "parallel_total_seconds": 0.07789591699838638,
  "parallel_aggregate_seconds_per_cell": 0.004868494812399149
}
```

Timing used `time.perf_counter()`. The parallel measurement followed the
single call and therefore includes warm-cache effects. These short timings
establish local execution feasibility; they are not a throughput guarantee
for the target parameters, compressed output, certification or plotting.
No network call, paid compute or paid review was used. The target plan keeps
four workers and 39,523 grid/transect cells regardless of observed timing;
the first completed target rows will supply actual workload timing.

## Driver contract audit

The independent read of `tools/run_hopfield.py` confirmed the following.
An additional non-target execution exercised its actual build and ctypes
binding with alpha = gamma = 0, initial state `(0.13,-0.09,0.2)`, `dt=0.01`,
one unit of burn and two units of measurement. All checks below passed.

- The C ABI has ten arguments in the declared order, all arrays contain
  contiguous float64 data, and the return count indexes retained events.
- The driver labels all 20 statistic columns consistently with the native
  source. State and activation moments use the entire measurement interval.
  FTLE omits its first 100 time units solely for tangent alignment.
- A grid cell executes 100,000 burn steps and 40,000 measurement steps:
  2,000 and 800 physical time units respectively; its exponent uses 700 time
  units. Transect exponents use 900 time units; refinement exponents use 1,100.
- `events[:n]` excludes unused buffer rows. Event time starts after burn-in,
  increases chronologically and stays within the measurement interval.
- The row tuple maps to terminal states, event buffers, statistics, retained
  event counts, peak periods, recurrence residuals and physical periods in
  precisely that order.
- The classifier compares all three coordinates of peak-section states.
  It uses four repeated blocks, not four elapsed-time samples. A synthetic
  nine-state event sequence with three time units between maxima was
  classified as peak period 9 and physical period 27; changing one full-state
  event rejected that recurrence. This is a constructed array test, not an
  integrated neural result.
- Grid rows restart from the declared initial state and continue in
  increasing alpha within each row. Transect points each restart independently.
- The two grids have 6,561 and 32,761 cells; the transect has 201, totaling
  39,523. The selected-point refinements and behavior probes are additional
  predeclared diagnostics, not additional grid cells.

The initial driver audit identified three issues, which were corrected before
the execution freeze. Certification now requires agreement across **both**
refinement steps. The driver journals each completed cell before proceeding,
retains the next cell's coordinates and native failure arrays, and verifies
the completed row before deleting the journal. An injected second-cell
exception, with no integration, confirmed that the first cell's exact numeric
checkpoint and the failed position remain recoverable. Finally, a separate
timer writes a timeout receipt and exits the process at the declared ceiling,
including during selected-point diagnostics. The timer is cancelled in the
driver's `finally` block. These corrections preserve the original numerical
thresholds and sample plan.
