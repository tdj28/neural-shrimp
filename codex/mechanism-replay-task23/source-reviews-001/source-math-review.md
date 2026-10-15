# Task23 source and finite-gain convention

The independent replay will test how perturbation growth is divided between the transformer and the soft readout at four specified points of one hostile-steering line. The division is an exact finite-twin accounting identity when all required norms are positive; reproducing it alone does not establish why the line is chaotic, a causal role for entropy, or a benefit for native generation.

This is source/design review only. No model was loaded, no target gain/trajectory outcome was decoded, and no control or profile was executed here. Source packet `/tmp/task23-source-002/manifest.json` has SHA256 `957f293f43a34aa62f99d2ef4c60a0531904b6b132427507e003ad5fa23b6d40`. Original source is `b203fb56d0daa41441a926097baa7c6743d42b8a`; `steer_gain.py` is `ccb427ec06f21620858808a8d51220c0056bfb218f742d54c625ba61d7d0e8a5`. Source-only metadata identifies the comparison file but its gain outcomes remain unopened.

## Map, dimensions and source association

Let column state x have dimension d=896. The fixed prefix has 37 tokens and one appended soft slot at a fixed position. There are 24 decoder blocks; steering adds alpha times the exact stored hostility-vs-kindness vector after zero-based block 12, on both the fixed prefix and appended slot. Alpha is fixed at 0.0655, and beta=0: no output-logit bias or extra steering block is included. The prefix is the nonempty original `PROMPTS['open']` system text and user `Hello!`, with generation prompt. Original `SoftLoop.prepare` initializes the soft state as the embedding of the last prefix token. This matches the accepted independent map's exact `PREFIX_IDS` and `initial_state()`. Original runtime prefix IDs are not separately saved in MECH1; the finite authority is the pinned source plus prior accepted fixed-prefix review, not invented historical tokenizer observation.

The accepted independent `qwen_map.py` SHA `493a3d1d57fec227f5ba2e3f8f9cfcfdd02d22495d1d44e6449ede901310ae6e` computes every norm, rotary phase, attention operation and readout in CPU float64. Its configuration requires tied embeddings, full attention, deterministic SwiGLU and the exact tensor inventory. The hostile asset helper SHA `06477ee36a78f15ee8f1e8ae196dc53ae230ff19368fbda638eb6a6e0ef758e0` supplies the exact hostile vector; its unrelated Task7 seed is not needed. The generic map loader selects philosophy and must not be used for this task. No accepted source file is edited; the new adapter exposes the existing final normalized hidden state through the same block and final-norm operations.

The pinned checkpoint revision is `7ae557604adf67be50417f59c2c2f167def9a775`; model SHA is `fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`, config SHA `18e18afcaccafade98daf13a54092927904649e1dd4eba8299ab717d5d94ff45`. The hostile vector archive is `d18395239dba63c7deacb8aa9431d768269617c9cc44a98f89cc1e22423b5958`, key `hostility-vs-kindness`, binary64 length896. Its scale is used unchanged. Runtime must verify these assets before loading; historical hashes alone are not a fresh runtime check. Float64 arithmetic operates on the stored checkpoint values converted exactly from their stored representation, not newly trained float64 weights.

The final normalized hidden state h(x) has dimension896. With input embedding E of shape151936x896 and output matrix W of the same shape, z=W h, p=softmax(z/T), F=E^T p. The general derivative is

`DF = E^T [diag(p)-p p^T] W Dh / T`.

For this tied model W=E, giving covariance operator `A=E^T[diag(p)-p p^T]E/T`, a positive-semidefinite896x896 matrix. The transformer factor Dh includes the final RMS normalization and excludes the output head. An untied head would not justify the PSD covariance formulation. No covariance eigensolve, power iteration or rank claim is required for the assigned replay.

## Exact finite-twin accounting and clocks

Original `steer_gain.py:120–128` uses two finite perturbations, relative offsets1e-6 and1e-9, renormalized after every measured step. For a base x, twin x+delta, normalized outputs h,h_twin and feedback F,F_twin, write dx=||delta||, dh=||h_twin-h|| and dF=||F_twin-F||. All norms are Euclidean. The source defines

`log g_net = log(dh/||h||) - log(dx/||x||)`,

`log g_read = log(dF/||F||) - log(dh/||h||)`.

Their sum is `log(dF/dx)+log(||x||/||F||)`. Along N consecutive base steps F_n=x_(n+1), therefore

`lambda_N = mean(log g_net)+mean(log g_read)+log(||x_N||/||x_0||)/N`.

The executable `boundary_term` at line161 has this positive endpoint sign. The header's `log(||x_0||/||x_N||)/N` describes sum-minus-lambda, not the correction to add to the sum. x_0 is the base state after1000 transient updates and before the first measured update; x_N is after measured update2000. Both endpoints and every actual dx/dh/dF and base/hidden norm must be retained. Use actual representable perturbation norms after renormalization, not the nominal epsilon as a substitute. The source splits only the first twin; own records may retain the second split as an additional diagnostic without extra model work. Both finite-twin exponents remain primary comparison fields.

This is a finite-secant identity, not an infinitesimal leading-eigenvector computation. A single renormalized perturbation approaches a leading tangent direction only under additional asymptotic/nondegeneracy assumptions. A separately labeled AD versus central-difference diagnostic checks the implementation/precision but must not relabel the full finite-twin replay as an AD exponent. A pointwise readout Jacobian or an eight-iteration Rayleigh estimate is not an upper bound on a finite secant gain; the task does not need those original auxiliary statistics.

Zero or nonfinite x/h/F/dx/dh/dF norms make one or more logarithms/renormalizations undefined. Preserve the exact row and condition; do not clamp a norm, replace the twin with a new random direction, fill a rate or silently skip the step. A failed renormalization stops that admitted attempt with preserved partial evidence. Exact coalescence is a finite numerical outcome, not a successful finite relative-gain identity. Tiny manufactured controls must cover this branch.

## Four coordinates, initialization and comparison scope

Original MECH1 uses a201-cell tgrid0.1..0.3 in one chunk1300; source cell order is temperature, beta, alpha. Requested order selects original indices25,29,19,35. The third stored coordinate is **0.11900000000000001**, one binary64 ULP above the requested literal0.119; replay uses the stored coordinate and reports that distinction. The other stored coordinates are0.125,0.129,0.135. The exposed requested chaotic/periodic labels are hypotheses/comparison labels, never own qualification gates.

The original run initializes one `default_rng(0)` per chunk, draws an entire201x896 standard-normal block for epsilon1e-6 and another for1e-9, then normalizes each row. The own implementation must generate these two complete blocks and select rows25/29/19/35; drawing only four rows would change the original initial-direction protocol. The original third random block for the excluded readout power iteration has no bearing on later randomness. No target array is needed to reproduce these draw positions. Base trajectories begin from the exact same last-prefix-token embedding and fixed parameter convention; the actual physical T values and exact floats must be saved.

CPU versus original GPU arithmetic and four-cell versus201-cell batching can perturb chaotic trajectories. Even aligned seeds and formulas do not entail pointwise equality after1000 transient steps. Report each own/producer scalar and its signed/absolute difference; do not invent an outcome-selected tolerance or call disagreement a failed arithmetic identity. Retain each twin separately, both finite horizon labels, the signed boundary and the first-twin gain split. Descriptive agreement across four selected cells does not establish global or causal steering effects.

The source's period diagnostic is only a last-state recurrence against lags1..64 with threshold `1e-8*max(1,||x_N||)` using a66-state tail. It is not all-phase primitive-orbit/stability verification. The original chaotic flag is period0 and both finite-twin rates>1e-3; preserve this literal finite criterion and numerical qualification separately.

## Readout and retention

Entropy is in nats, from the full151936-token probability vector. For literal comparison retain the producer's `-sum p*log(max(p,1e-300))`; zero-probability terms contribute zero. Gap is the difference between the two largest **unscaled** logits divided once by physical T. Save both token IDs/logits, gap and top probability, along with per-step norm/gain records. No decoded text is needed. The original summary spans2000 measured steps, while its saved step arrays retain only the first1000 in float32. Own complete2000-step float64 scalar records avoid confusing those horizons. Entropy/gap correlation or a gain accounting identity does not show that either causes a change in accuracy.

## Bounded representative profile proposal, not admission

Use the exact four-row parameters and common alpha cache, with a4-row base path and12-row base/twin measured path. Proposed profile: load/hash assets once; record load/prefix cost;32 base transient steps and16 measured steps with exact original RNG positions; serialize all profile scalar rows and native start/end/tail states. Timing should separate base step,12-row twin step, scalar reduction/serialization, and diagnostic overhead. This fixed-shape probe estimates throughput and storage, not the requested1000/2000 scientific result and not asymptotic labels. It must not be substituted into the final averages.

At a fixed profile state retain a declared relative central-difference ladder (1e-2,3e-3,1e-3,3e-4,1e-4,3e-5,1e-5) for one-step h/F and a two-step composed F direction against own AD, with every row/error/rate retained; compare split outputs with unchanged map logits/forward, repeat determinism and batch/scalar consistency. Exact diagnostic acceptance is to be frozen in the reviewed profile card before execution, with no favorable plateau selection. No full dense monodromy or multiplier search is needed for this finite-twin task.

A conservative proposed profile envelope is CPU2 worker plus one supervisor, nice10,12GiB AS,600s wall/1100s process CPU,64MiB output,28GiB startup and16GiB runtime available-memory floors. These are a proposal for Root/source/lifecycle review, not permission to launch. Historical own Task7 peak5,825,380KiB and timings support feasibility only; they do not measure this new batch/gain path. The sole profile's actual costs will support a separate fixed-cap1000/2000 card. No automatic retry, extension, GPU/pod use or optional crossover is authorized.
