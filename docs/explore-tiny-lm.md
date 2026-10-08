# Small causal-transformer language-model pilot

**TLDR.** A causal transformer trained on actual text can now be studied with
our matrix-free gradient-map derivatives at low local cost. In this pilot,
moderate rates memorized a small fixed batch while the full parameter state
continued to drift; higher rates produced large fluctuations or hit the
numerical guard. No tested run qualified as a full-state fixed point or
period-2 candidate. This is an architectural feasibility result, not evidence
of Neimark–Sacker bifurcations, invariant circles, chaos, shrimp-specific
utility, or improved language modeling.

Status: **exploratory, evidence rung 0**. All ten main cases, the timing unit,
raw trajectories, numerical checks and source commits are bound by the
[generated receipt](explore-tiny-lm-receipt.json). The first six rates were
specified before their outcomes; the four higher rates were a disclosed
adaptation after all six completed without full-state recurrence.

## Model, text and state convention

The model is an independently implemented one-layer causal transformer with
width 32, one attention head, context 32, a 4x GELU feed-forward block,
pre-LayerNorm, learned positions and a final LayerNorm. It has **15,776
parameters**. The token embedding is reused as the output projection and
occurs exactly once in the flat parameter state. Attention uses explicit
masked matrix operations so the required derivatives do not depend on a
flash-attention backward implementation. There is no dropout.

The public corpus is the 1,115,394-byte
[char-rnn tiny-Shakespeare text](https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt).
The upstream [README](https://github.com/karpathy/char-rnn/blob/master/Readme.md)
states MIT; no separate dataset license file was found. Both files are retained
with hashes and attribution in the private data store. Corpus SHA-256:
`86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`.

Eight fixed sequences begin at character offsets `0, 4096, ..., 28672`.
For each sequence, inputs are characters `start:start+32` and targets are
`start+1:start+33`: every update uses **256 next-character targets**. The
vocabulary is sorted characters from the first 90% of the corpus plus an
unknown token. The final 10% is disjoint and has **no evaluated validation
losses** in this pilot. Training loss on this tiny batch cannot support a
generalization claim.

The objective is mean token cross-entropy plus `0.001/2 * ||theta||^2`.
Full-batch gradient descent is autonomous: its complete optimizer state is
`theta`, with no moments, schedule or bias-correction counter. All cases start
from seed 28141 and the same parameters/data. AdamW, stochastic batches,
weight-decay exclusions and modded-nanogpt schedules were not tested. Their
additional state/time dependence must be modeled explicitly in any extension.

LayerNorm and parameter symmetries were not quotiented out. Loss or function
convergence can therefore coexist with parameter drift; this report requires
full-state recurrence before considering an ordinary periodic-orbit claim.
No near-neutral eigenvalue cluster is interpreted as an NS crossing.

## Numerical and resource checks

Runs used the local RTX 2000 Ada, float32, deterministic Torch algorithms,
TF32 disabled and one CPU thread. The measured 100-update profile took
0.511 seconds; subsequent cases required roughly 2–5 milliseconds per update.
Peak live CUDA tensor allocation in the six-rate pilot was 71.5 MiB. This
allocation metric excludes CUDA context/reserved-memory overhead. After the
runs, `nvidia-smi` showed no compute process, 4 MiB used and 0% utilization.
No rented compute or paid review was used.

The checks were substantive but have limited scope:

- Four tests passed: causal masking/tied-state accounting; one- and two-step
  JVPs against dense reduced derivatives and finite differences; Lanczos
  against a known symmetric matrix; reserved-seed rejection and AR controls.
- On a 66-parameter reduced transformer in CPU float64, dense and matrix-free
  one-/two-step derivative errors were below `5e-16` relative.
- On the actual 15,776-parameter architecture, evaluated at exactly the same
  rounded initial state, CPU float64 versus GPU float32 relative differences
  were `9.60e-8` for the gradient, `1.35e-7` for the HVP and `2.08e-8` for one
  update. This is a one-state comparison, **not trajectory precision
  equivalence**. Full-model float32 finite-difference update-JVP errors were
  `7.50e-5` and `2.26e-4` at steps `0.003` and `0.001`; smaller steps expose
  rounding error. A bilinear Hessian-symmetry check was below `8e-10`.
- Matrix-free Hessian Lanczos returns recomputed Ritz residuals. Its initial
  largest estimate was `7.60232`, with relative residual `0.00287`; this is a
  rate scale, not an exact stability threshold. Some final minimum Ritz
  values are unresolved (the unit's minimum residual is 50%). Small residuals
  establish approximate returned eigenpairs, not completeness of the spectrum.

The initial point is nonstationary and has negative-curvature directions.
Consequently `2 / initial_sharpness` does not divide the following cases into
stable/unstable attractors. In the timing unit, the maximum Hessian estimate
grew from `7.60` to `50.94` in only 100 updates.

## All main outcomes

Rates use `eta = factor * 2 / 7.60232` (the unrounded estimate is retained).
The first six factors are `0.1, 0.4, 0.8, 1, 1.2, 1.6`; factors
`2.5, 4, 6, 8` are the disclosed adaptive extension. Each requested 1,000
updates from the identical initialization. Initial training CE was `4.181448`.

| Rate factor | eta | Completed updates | Final recorded train CE | Outcome |
|---|---:|---:|---:|---|
| 0.1 | 0.026308 | 1000 | 0.616610 | drifting full state |
| 0.4 | 0.105231 | 1000 | 0.016375 | drifting full state |
| 0.8 | 0.210462 | 1000 | 0.010128 | drifting full state |
| 1 | 0.263078 | 1000 | 0.009498 | drifting full state |
| 1.2 | 0.315693 | 1000 | 0.009386 | drifting full state |
| 1.6 | 0.420924 | 1000 | 0.009448 | drifting full state |
| 2.5 | 0.657694 | 1000 | 2.649972 | drifting full state |
| 4 | 1.052311 | 551 | 1378.702609 | numerical guard stop |
| 6 | 1.578466 | 10 | 6502.083843 | numerical guard stop |
| 8 | 2.104622 | 7 | 2155.853719 | numerical guard stop |

A guard stop means the next objective exceeded `1e4`, became nonfinite, or
its gradient became nonfinite. It is not a mathematical divergence theorem.
The table's final CE is the last retained pre-guard value, not the offending
value. The offending final state and all finite prior traces are retained.

The maximum relative state recurrence over each final tail was checked at
lags 1, 2 and 4. A fixed-point candidate would require lag 1 <= `1e-5`; a
period-2 candidate would require lag 2 <= `1e-5` and lag 1 > `1e-4`.
**None passed.** At the smallest rate, lag-1/lag-2 residuals were `0.01621`
and `0.00623`, and the full state moved 7.09% over the tail: an alternating
but drifting trajectory. At eta `0.657694`, the lag-2 residual was `0.69263`
and parameter norm grew 16.98% over the tail. The latter is strongly
fluctuating finite-horizon training, with no qualified chaos label.

## Loss AR(2) is a baseline, not an orbit detector

The final 512 losses (100 for the timing unit) were analyzed with AR(1) and
AR(2). The first 75% fits the trend and autoregression; the last 25% measures
one-step prediction using observed preceding losses. This is a **time-held
prediction check within the training trajectory**, not corpus validation or
multi-step autonomous forecasting.

The timing unit gives weak complex AR(2) roots while its loss is descending
and its full state is not periodic. In the moderate-rate completed runs,
AR(2) often improves one-step prediction by orders of magnitude by fitting
smooth drift; those fitted roots are real. At the lowest rate AR(2) is worse
than AR(1), and at eta `0.657694` both extrapolate the changed tail poorly.
Root angles, AR prediction errors, state recurrence, parameter displacement
and norm drift are all retained together in the receipt. None is substituted
for a full-state rotation number or multiplier.

## Evidence, limits and next decision

| Component | Evidence | Limit |
|---|---|---|
| Real-text transformer training | Rung 0, exploratory, one seed and fixed batch | No generalization or prevalence claim |
| Matrix-free derivatives | Reduced dense/FD controls and same-state full-model precision comparison | No float32/float64 trajectory agreement claim |
| Local Hessian estimates | Matrix-free Ritz pairs with explicit residuals | Nonstationary local geometry; no Floquet spectrum |
| AR diagnostics | Time-held one-step prediction against AR(1) | Drift and harmonics can create apparent structure |
| Periodic dynamics | No candidate met the full-state screen | No NS, circle, chaos or ancestry result |

The bridge is cheap enough locally that an H100 is not currently justified
for this model. The next useful dynamical experiment would continue a
well-specified constant-rate branch longer and require stationarity/recurrence
and derivative convergence before interpreting two-step spectra. A separate
optimizer/noise/schedule experiment can test relevance, but should retain
this deterministic control and explicitly include optimizer state.

## Reproduce the receipt

Source: `src/neural_shrimp/tiny_lm.py`; runner:
`tools/explore_tiny_lm.py`. Raw data and all three attempts live under
`/data2/neural-shrimp-data/`; this is local retention, not an independent
backup. The receipt stores source commits, manifest hashes and the exact
plan/command trail is in the research log.

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /data2/neural-shrimp/.venv/bin/python tools/analyze_tiny_lm.py \
  --receipt docs/explore-tiny-lm-receipt.json
```

This checks all stored file hashes and recomputes the receipt without new
trajectories or network access.
