# Run FS: fixed-shape control of the batch-layout effect, and a precision-dose test (frozen before any data)

Frozen 18 Oct 2026, before any FS data; requested by the external review (18 Oct) and approved by the human
("Do all of them"). Exploratory parts are marked.

**Question.** Does the layout effect in float64 come only from the matrix shapes the kernels see (so that the feedback
dynamics decide its growth), with no coupling between batch rows? And does an injected displacement grow on the
schedule set by the Lyapunov exponent across doses?

**Cells.** docs/h2-horizon-cells.json: the 448 labelled cells of the horizon check (192 chaotic, 192 periodic, 64 other;
Qwen2.5-0.5B-Instruct, philosophy at layer 12), as in REPRO. Float64 throughout, 1500 lockstep steps from the fixed
start, one H100 pod (tools/repro_noise.py at the pushed commit, job set repro).

**Comparisons.**
- repeat: the same 448 rows twice, same order and shape (determinism of the computation itself).
- fixedperm: the same 448 cells in batch 448, the second run with rows permuted (seed 20261018): other neighbours and
  row indices, identical shapes.
- batch64: the REPRO comparison (448 rows against the same rows plus 448 more), as the positive control.
- dose: 448 x 7 rows in both runs; in the second run the start of each cell is displaced by d |x| along a seeded
  random unit direction, d in {1e-15, 1e-13, 1e-11, 1e-9, 1e-7, 1e-5, 1e-3}.

**Predictions and criteria.**
1. repeat and fixedperm: the largest relative distance over all steps is exactly 0 in every cell. Any nonzero
   distance means nondeterminism or coupling between rows, and our account must be revised.
2. batch64: at least 180 of the 192 chaotic cells pass 1e-2, and at least 170 of the 192 periodic cells stay below it
   (REPRO: 190 and 182).
3. dose (primary). For the chaotic cells, at every dose from 1e-15 to 1e-7, the median over cells of
   t / t_pred lies in [0.7, 1.3]. Here t is the first step at which the distance exceeds 1e-2, and
   t_pred = ln(1e-2 / d) / lambda, with lambda the cell's map exponent; cells whose t_pred exceeds 1500 steps are
   left out at that dose and counted.
   Secondary: the slope of the median t against ln(1/d) lies within 30 % of the median 1/lambda; periodic cells pass
   1e-2 at doses up to 1e-5 in at most 10 % of cells.
4. Exploratory: per-cell scatter, doses 1e-5 and 1e-3 (finite-size saturation expected), and the "other" cells.

All results are reported whatever they show. A failed criterion is reported as failed.
