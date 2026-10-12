# Task 2: independent numerical cross-check of soft-token LLM dynamics

Scope: the 12 October 2026 handoff, Task 2 only. This is an exploratory,
known-outcome replication (rung 2), not a fresh confirmatory test or an interval
certificate. The pretrained BF16 weights are exactly converted to float64;
this does not imply they were trained at that precision. The transformer
forward pass and dynamical solvers are independently implemented here, with
no repository-tool imports. PyTorch tensor primitives/autodiff and the same
pinned weights, prefix and saved seeds are shared with the producer.

The map has 896 continuous state coordinates, fixed 37-token prefix, one live
soft token, philosophy steering after zero-based block 12 including the prefix,
and a full 151936-token expected embedding. Rotary frequencies/tables, RMSNorm,
attention, softmax and all other arithmetic use float64 CPU. No token sampling,
growing context, training, GPU, paid compute, reserved seeds or sealed data.

## Staged execution plan (written before evaluating the pretrained model)

1. Profile: check the hash-bound current-map fold seed closure (<1e-10 relative),
   cached/uncached consistency (<1e-13 absolute), one-step and eight-step JVP
   central-difference ladders (relative h = 1e-3, 3e-4, 1e-4, 3e-5; best relative
   derivative error <1e-5 and inspect h-squared convergence), and batches 4/16.
   Four CPU threads, 16 GiB address-space limit, 600-second wall limit. Preserve
   failures. Admit heavier stages only after measured throughput and memory.
2. Independently Newton-refine stable P2 (t=-0.115, alpha=0.0624) and P8
   (t=-0.1205, alpha=0.0717), full-state relative closure <1e-11, all proper-divisor
   returns and phase separation, dense monodromy/eigenpair residuals <1e-8 and
   chain/composed JVP agreement. Keep failed solve histories and iteration counts.
   The P8 seed predates the rotary precision fix: its seed displacement and
   changed spectrum must be reported, not hidden as replication roundoff.
3. Refine the P2 fold via an independent augmented spectral Newton solve from
   a perturbed saved seed, classify the isolated real +1 multiplier and other
   moduli before computing coefficients. Use physical alpha, differentiating
   its prefix contribution. Require augmented residual <1e-11. Align critical
   eigenvector sign and phase to the reference. Compute a with parameter AD,
   b with nested state AD, and independent finite-difference ladders; retain
   discrepancies from the producer's finite-step averaged b, not tune to it.
4. Repeat all 18 controlled passages using the original fixed centre, modes,
   U, six epsilons and three entrances. Compare every count, crossing margins,
   and pre-crossing u trajectories. Original historical holdout outcomes are
   already known here. Discrete quadratic and continuous arctan predictions
   are approximations and are compared separately with the full-model counts.

Each stage needs committed code and a prior `tools/research_log.py run` card.
CPU concurrency stays <=12 total, initial worker 4, at most 8 after profiling,
with nice 10. Recheck capacity before a substantial launch. A run snapshots
committed task files into an exclusive attempt directory under
`/data2/neural-shrimp-data/codex/task2-independent-crosscheck/`; the launcher
records source hashes, PID, limits, status and output hashes, and makes finished
files read-only. Timeout terminates only its own process group. Raw arrays stay
outside Git; compact receipts and reports are delivered via private main.

Example (from the isolated task worktree, following its run card):

```
/data2/neural-shrimp/.venv/bin/python codex/crosscheck/launch.py \
  --attempt profile-001 --stage profile --threads 4 --seconds 600 --memory-gib 16
```

The scientific report must separate successful checks, finite precision or
reference-map changes, unresolved discrepancies, and unattempted work. This
Task 2 supplies no shrimp-boundary proof and no claim about ordinary sampled
LLM inference or training utility.

## Profile-informed amendment: wider batches and parallel passage replay

Profile001 (source bf35052990e3db9a951ba16bdb5a0ce72934ab7a) completed in
approximately73s, with fold closure3.5531e-14, one-step best FD error4.8917e-9
and eight-step best FD error1.6231e-8. Both ladders exhibit quadratic truncation
convergence before rounding. Batch16 predicts426s for denseP2 and1705s forP8;
therefore profile batches32/128/256 (600s limit, same4threads/16GiB cap),
checking first-direction agreement<1e-8 across batch sizes, before choosing
a dense batch. Separate passage replay (1800s limit,4threads/16GiB) can run
in parallel while capacity admits two workers: total8threads, not8each.
This changes throughput only; no scientific tolerance/entrance is changed.

Throughput001 passed: batch128 and256 first-direction differences5.51e-14 and7.35e-14; peakRSS at256 was6899056KiB. Choose128 for the dense stages: estimated203s/P2 and810s/P8 with smaller memory than256 for nearly the same throughput. Fold cap3600s; combined stableP2/P8 cap5400s, each4threads/16GiB/nice10; at most two workers concurrently. Augmented fold uses at most4 Newton corrections and2 backtracks per correction; near-seed ordinary Newton at most6 corrections, GMRES restart20 x2 cycles. No tolerance change.

Auxiliary saved-data check: reconstruct the stable-branch ab extrapolation from the18 pre-existing points with5e-5<=alpha_c-alpha<=4e-4, as stated in the producer researchlog. Retain every selected row/gate and indices; fit y=(1-mu)^2/[4(alpha_c-alpha)] against sqrt(alpha_c-alpha), degrees1/2. This checks the stated123.1 extrapolation recipe, not a second independently recomputed branch. No additional transformer evaluations.60s walllimit,1CPUthread; still below12 total.

Branch001 retained a failed stricter1e-11 screen: selectedrow3 has savedclosure3.755297017484666e-11; exact18selection andallothergatespassed. The producer archivedargs.tol is1e-10. Branch002 explicitly reconstructs the historical18point arithmetic using that original criterion, preserving the strictfailure. It does not relax independentP2/P8/fold precision gates, independently validate the saved eigenvalues, or establish an extrapolation error bound.

Manuscript table alignment: the stable P2 point near the fold is distinct from the table P2 at(log10T=-0.17263,alpha=0.061579). Add a current-map replay of that already pinned historical seed, with identical orbit gates, chunk128,4threads,16GiB and1800s. Start after fold worker completion so at most two model workers run. Report its rotary-map shift and all multiplier discrepancies separately; this lets Claude update that exact table row.
