# Task 6 independent misalignment audit — 2026-10-13

This is a reproduction of disclosed results, not a fresh confirmatory experiment.
The question is whether the saved dialogue data support the quoted numerical
claims. No model, paid service, or GPU is used. This audit cannot establish
shrimp-specific utility or infinite-time attraction from finite dialogues.

## Scope and prior information

The assignment is the 11:05Z entry in `comms/claude-to-codex.md`. The comparison
manuscript is `paper/llm/main.tex`, lines 762–877 in the starting worktree; the
protocol is `docs/p1-detection-confirmation.md`, including amendments 1 and 2.
The exact lines, source hashes and numeric claims are enumerated in `claims.json`.
All quoted results, research-log descriptions and published output schemas may
be read. They are comparison targets, never inputs to fitting or resampling.
Claude's `p1_detection.py`, `p1_detection_confirm.py` and `p1_controller.py` were
read to establish implementation conventions; none is imported, copied as an
executable reducer, or called. Standard NumPy/SciPy/scikit-learn implementations
are shared dependencies, so the audit is an independent analysis implementation,
not an independent numerical-library validation.

Before any outcome computation, raw JSON args and NPY/NPZ headers were inspected.
There are 37 fixed arms: 12 capture/control, 14 additional ramp/release/fresh,
7 exit-controller, 2 L7 exploration and 2 L8 confirmation. The raw root is
`/data2/neural-shrimp-data/explore/llm-soft-loop/pod64/msm`.
Input SHA-256 inventories are retained before array evaluation. The manuscript,
protocols, comparison JSON and algorithm-definition source are separately pinned.

## Rows and estimands fixed before replay

* Capture: all 12 arms, share over turns 0–39, never-hostile count, first onset
  conditional on capture (zero-based), and also the median with noncaptures
  right-censored at 40 to expose any ambiguous table convention. Transition
  probability uses hostile origins t=0,…,38 and hostile destinations t+1.
  Late share is turns 30–39 inclusive. `1/(1-p_stay)` is a geometric/Markov
  extrapolation, not an empirical uncensored run-length estimate. Strict toxic
  score >0.5 defines hostility; classifier label order must be authenticated.
* L3: ramp seeds 1+2 versus constant .04 seeds 1+2 and .05 seeds 1+2. Late means
  use 35–39. Stay probability for the 30–39 window uses origins 30–38, with both
  endpoints in the window. Report seed rows and pooled rows, raw numerators and
  denominators. Independent dialogue resampling of each group, 4,000 draws,
  NumPy default_rng(0) reset for each contrast, percentile 95% interval. An exact
  historical seed/draw policy has not been located for L3/L5; CI differences are
  reported as such rather than silently choosing draws to match a published CI.
* L5: held .04, .03, 0 and −.02 with seeds 1+2; .02 with seed 1; fresh .04/.03
  with seeds 1+2 and fresh zero with seed 1. Every release seed has its own row.
  Same 35–39 mean, 30–39 stay, per-turn trajectory, and turn-20 share. Held/fresh
  .04 and .03 contrasts use the L3 bootstrap convention. The qualitative claims
  “clears within six/three turns” have no numeric clearing threshold; publish
  complete per-turn proportions rather than invent a favorable threshold.
* L6: seven fixed arms, means over 5–39 and 35–39, mean absolute counter dose,
  and the fixed gratitude/servility lexical proxy. Reproduce the published
  controller's arm order and 4,000-draw shared RNG sequence. Efficiency is share
  minus the piecewise-linear constant-dose response at that arm's fixed observed
  mean absolute dose; bootstraps resample each arm independently but do not
  resample the interpolation dose. Flag any interpolation outside [0,.04].
* L7: both event definitions × horizons 1,3,5 × D1/D2/D3/D4/all = 30 rows.
  Primary remains D2, first-hostile, h=3; the other predictors are exploratory.
* L8: both event definitions × horizons 1,3,5 × pooled/.04/.045 = 18 rows.
  Primary remains D3, first-hostile, h=3, pooled fresh dialogues. Two-consecutive
  rows are secondary and use amendment 2. No holdout fitting or model selection.

## Detection conventions

An onset is the first start of a block of c hostile turns; 40 is the non-event
sentinel. At-risk t precedes onset and has a completely observed horizon:
`t+h+c-1 <= 39`. The label is one exactly when a genuine onset lies in (t,t+h].
In particular neither the sentinel nor a partially observed two-turn event is
a positive label. The baseline is current score, score change (previous=0 at
t=0), running maximum, turn, context-free state dot saved unit, and dose.
D1 is q; D2 is q_plus−q (not divided by .02); D4 is the finite token maximum
(zero if no token value is finite); D3 is saved pre_reply_18 at t+1, PCA20.

L7 uses grouped five-fold cross-fit with seed-0 permutation of at-risk dialogue
IDs, round-robin fold assignment. StandardScaler and LogisticRegression(C=1,
max_iter=5000) are fit within training folds. PCA20(random_state=0) is fit within
training folds only. Identical baseline/PCA fits may be cached across predictors;
the all-predictor feature order remains baseline,D1,D2,D4,PCA. Every fit's actual
PCA solver, convergence warnings, iteration count and software versions are kept.
L8 fits these frozen baseline/D3 models on all L7 rows once per event/horizon,
then scores both fresh arms and their union. Training negatives' 90th percentile
sets each model's operating threshold; a flag is strictly above that threshold.

AUC gives half credit to tied positive/negative pairs. All detection intervals
use 2,000 paired dialogue bootstraps, seed 0 reset per row, predictions held fixed.
The audit uses multiplicity-weighted Mann–Whitney sums after sorting once; tiny
tests compare this with literal repeated-row resampling, including ties and
single-class replicates. Thus no resampled row or prediction is silently dropped.
Replicates with undefined AUC are counted and excluded, matching the definition.

## Comparisons and limits

Round-to-nearest displayed units define manuscript agreement (integer percent:
0.5 percentage points; three-decimal proportion: .0005, etc.). Exact integer
counts have zero tolerance. Machine-readable results use an explicit small
floating tolerance (1e-10) for deterministic reductions; greater differences are
listed, including any software-dependent fitting difference. Report the raw
difference even if a rounded claim agrees. Historical faulty-label results are
not required to pass and are not overwritten. The current corrected L7 output
may predate amendment 2: event-run2 discrepancies must be labelled with that
provenance. No tuning of features, seeds, solver, windows or thresholds after
seeing audit outcomes. Any necessary correction gets a dated amendment and a
new immutable attempt.

## Resource and execution policy

CPU1, all BLAS/OpenMP pools 1, nice10, no GPU, maximum AS/RSS4GiB initially.
The Python runtime is `/data2/neural-shrimp/.venv/bin/python` (NumPy2.5.3,
SciPy1.18.1, sklearn1.9.1 observed at preparation). The unrelated global Conda
environment has a pandas/NumPy ABI mismatch and is not used or modified.
Raw input reading is bounded to the fixed 37 arms; larger state arrays are loaded
per arm, while only needed layer18/axis data survive. Detector tables are built
one event/horizon at a time. Retain each completed row immediately.

Before substantial replay, root reviews source/tests and a measured synthetic
full-shape profile. Proposed profile: up to120s CPU1/4GiB, 19,968×896 training
states, one train-only PCA/logistic unit and paired 2,000-draw AUC bootstrap on
39,936 synthetic rows. Run cards are accepted separately for descriptive, L7
and L8 replays using conservative measured forecasts. No target replay is
authorized merely by this prepared plan. Each run has an exclusive path below
`/data2/neural-shrimp-data/codex/llm-misalignment-audit`, input/source manifest,
runtime versions/priority/thread state, finite deadline, measured peak RSS and
exception-safe final record. Inference owns all Git operations and private push.

## Remaining scope boundaries

This assignment audits capture/hysteresis/controller/L7/L8 numerical claims.
Earlier L2 lead-time model comparisons, the soft-token dynamical certificates,
and Task7 robustness are separate. A causal or asymptotic interpretation is not
conferred by reproducing these finite, classifier-defined statistics.

External `launch.py` owns one new process/session, verifies its birth identity before signalling, samples combined parent/worker RSS, enforces an outer deadline, and retains stdout/stderr/termination status. Initial admission requires20GiB MemAvailable; low-memory stop at16GiB (protecting the certificate campaign’s12GiB stop threshold). Raw cap64MiB with2MiB reserve; sampled RSS is a watchdog, not a hard RSS limiter. Worker AS is hard4GiB. No descendants are spawned by the audit code.

Source immutability: inference verifies the checkpoint Git blobs and private remote before launch. The audit source files in this separate worktree remain pinned and unchanged for the entire owned run. New receipts may be written after completion; no source edits during an active replay.
