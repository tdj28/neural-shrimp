# Brief: the feedback dynamics of language models, second synthesis

14 October 2026. A private manuscript of about 60 pages, aimed at cs.LG with cross-list nlin.CD. Consult 6
(13 October) narrowed two claims ("shrimp-like organization"; hostile behaviour as a classifier-defined proxy);
both narrowings were applied. The paper has since changed substantially. This brief asks for a director-level
judgement of the new parts before posting.

## System

Soft-token map x_(n+1) = E^T softmax(z(x_n)/T): a frozen model's expected next-token embedding fed back at a fixed
position after a fixed chat prefix, steering alpha v added at one layer. Smooth; T -> 0 gives one-token greedy
decoding. Float64 throughout (float32 paths of Hugging Face code patched). Evidence rungs: 1 = float64 numerics;
2 = independent implementation or hardware; 3 = interval proof (one period-8 orbit).

## What changed since consult 6

1. **Theory section.** For tied embeddings DF(x) = (1/T) Sigma_p J_h(x), Sigma_p = Cov_{t~p}(e_t) (verified against
   forward-mode derivatives to 1.3e-13). With J_h = I the map is exactly a modern Hopfield update (Ramsauer et al.),
   the gradient of a convex function, which lowers an energy each step and cannot cycle; any cycle or chaos needs DF
   to fail to be symmetric positive semidefinite somewhere. T -> 0: the piecewise-constant greedy map, every orbit
   eventually periodic. T -> infinity: a contraction with a fixed point near the mean embedding (a high-temperature
   sweep finds fixed points at all 25 temperatures, entropy to log V, exponent slope -2.14 per decade toward
   -ln 10). In the four chaotic models the chaotic temperatures lie in bands within log10 T -0.20..0.28.
   "Stability belongs to the loop": along a period-8 orbit every step expands (largest singular value 1.2-21), yet
   the cycle's Jacobian has norm 0.19 against 1.1e7 for the product of the step norms. A proposed explanation of
   the orbits' compression by near-rank-one Sigma_p was refuted by the data and is not used.
2. **Chaos behind a repeating text.** Along a hostile cascade line, a long test (2000 steps, twins at 1e-6 and 1e-9)
   finds 182 chaotic cells; in 65 the leading-token text repeats exactly (periods 7 and 28 mostly) while the state
   is chaotic. At the strongest cell the states form seven disjoint bands, and the return map of one phase's
   leading-token probability is a smooth unimodal map whose exponent matches the full map's within 1.7 %. Across
   the 62 reproducing cells, with an amended analysis declared before computing it (cross-validated polynomial
   degree; median over the single-valued phases, by conjugacy invariance), 51/62 match within 10 % (the
   pre-specified fixed-degree analysis gave 49/62; one cell was underfitted).
3. **Horizon check of finite-horizon labels.** Random cells of two float64 maps retested at seven times the horizon
   with two twin offsets: chaotic labels hold in 86/96 and 92/96, periodic labels in 96/96 each; on the cascade line,
   only 135/235 chaotic labels hold. Both are stated.
4. **Steering controls.** Unsteered (alpha = 0) sweeps are chaotic in Qwen2.5-0.5B, Qwen2.5-7B, OLMo-2 7B and
   Llama-3.1-8B. Norm-matched random and sparse-autoencoder (SAE) feature directions produce chaos as readily as
   concept directions (Llama-3.1-8B at the Goodfire SAE's layer: concept 28 %, frozen-rule hostility feature 29 %,
   random live feature 27 %, Gaussian 39 % chaotic cells).
5. **Scale (tonight).** Float64 with bfloat16 weight storage (bitwise identical to float64 on the 0.5B model), layers
   spread over GPUs:
   - Unsteered sweeps (128 temperatures, log10 T -1..0.3): Qwen2.5-72B and Llama-3.3-70B not chaotic anywhere (largest
     exponents -2.76 and -0.27). Qwen2.5-72B stays in a two-step "!Hello" echo with entropy <= 0.15 nats up to T = 2;
     at higher T (to T = 16) the entropy jumps from 0.5 to 11.4 nats within log10 T 0.357..0.386 and the map
     contracts to a fixed point; no chaos. Llama-3.3-70B spreads to 5-10 nats at high T but still contracts.
     Qwen3.5-35B-A3B (mixture of experts) not chaotic unsteered either.
   - Steering controls at the Goodfire SAE layers, all directions at the concept norm, hostility plane: Llama-3.1-8B
     (layer 19) concept 28 %, frozen-rule SAE hostility feature 29 %, random live feature 27 %, Gaussian 39 % chaotic;
     Llama-3.3-70B (layer 50) concept 4.3 %, SAE hostility feature 3.5 %, Gaussian 1.1 %, random live feature 0.2 %.
     Qwen2.5-72B hostility (layer 40) 0.8 %. At 70B the chaotic cells of the hostility directions lie on a ridge that
     steering moves from log10 T ~0.3 to ~0.1.
   - Gain split (new tool): each step's growth = network factor (relative sensitivity of the final hidden state to the
     fed-back token) x readout factor (relative amplification by the softmax readout); the log-averages add up to the
     exponent exactly. Llama 3.1-8B vs 3.3-70B at matched entropy: 1-3 nats, network -0.05 vs -0.59, readout +0.19 vs
     -2.33 (medians); 6-10 nats, network +0.28 vs -0.19, readout -0.20 vs -0.24. A possible unsteered chaotic cell in
     Llama-3.3-70B at log10 T 0.304 (just above the sweep range) is being confirmed.
   - Running: Llama-3.1-70B (same recipe as the 8B: scale versus post-training); a scaling series (Qwen2.5 1.5B, 3B,
     14B, 32B; OLMo-2 1B, 13B, 32B) with predictions frozen before measurement (docs/scaling-predictions.md: decline
     above 7-8B within each family; non-increasing order above 7B; both gain factors at or below the 7B's); a 64 x 64
     zoom of the 70B ridge.
6. **Zero-temperature bridge.** K-token greedy decoding: K = 1 reproduces the soft map's T -> 0 attractors; for
   K >= 2 the loops leave the one-token cycles; unsteered looping falls to the full-context level at K = 16-32.
7. **Discussion.** New paragraphs on implications for interpretability and alignment (predictability horizon of
   20-50 steps where chaotic; step-level explanations do not compose; feature dictionaries do not give attractor
   boundaries; acting on the feedback loop beats pushing against the behaviour; results at one operating point do
   not transfer) and on which findings might reach frontier models (GPT-6 Astra, Claude Opus 5.5, GLM-5).
8. **Independent audit (Codex, in progress).** Corrections already applied: the low-temperature exclusion is a
   pointwise limit (chaos at small positive T is excluded only on invariant sets with a uniform logit gap); the second
   period-8 multiplier is negative and the cycle's norm depends on the starting phase (0.06-0.64 over the 8 phases);
   "single maximum" became "single turning point" where curvature was not checked; the amended return-map analysis is
   disclosed as chosen after inspecting an outlier.
9. **Title.** Current: "Ghosts, cusps and chaos: bifurcation theory predicts the dynamics of a language model fed its
   own output". The human proposes "Chaos theory predicts the dynamics of multiple large language models fed their
   own output"; the agent suggested "Chaos theory predicts the dynamics of language models fed their own output".

## Unchanged principal results

Fold normal forms predict bottleneck passage times without fitting (prospective test at two new folds: 23 of 24
forecasts within one iterate; the all-row criterion failed by one row). Numerically nondegenerate cusps with
regular unfoldings: frozen local predictions of orbit counts at 12/12 offsets and stability at 11/12; one cusp's
coefficients reproduced by an independent implementation. Interval proof of a primitive period-8 orbit. Huginn:
a frozen spectral test of where chain-of-thought accuracy breaks, confirmed on held-out data and replayed
independently. Sampled self-dialogue: hostility capture with hysteresis; detect-and-replace controls it at a quarter
of the dose of constant prevention; counter-steering backfires past the dose-response peak at 1.5B.

## Questions (please answer each concisely; exact citations only, we check every one at the primary record)

Q1. Claims against evidence in the new parts: the theory section (especially the Hopfield/energy argument and
"chaos needs intermediate temperatures"), the logistic-type reduction behind a repeating text, and the
interpretability/alignment paragraph. What is overstated, and what would a referee attack first?

Q2. Scale. The human wants a definite answer in the main paper on how this chaos depends on model size (including
whether trillion-parameter models should be expected to show it). Given the results in item 5 and the frozen test,
what claim can the paper make, what is the most likely dynamical explanation, and which additional measurement (within
about [amount redacted] of rented GPU time) would make the scale statement decisive? Are the frozen predictions the right test?

Q3. Title and framing for cs.LG with nlin.CD: which title matches the evidence best?

Q4. Literature (to October 2026): published work on (a) Jacobian or Hopfield-type analysis of softmax feedback
loops in language models; (b) chaotic dynamics hidden behind periodic symbolic output in neural networks;
(c) chaos or contraction of language-model feedback dynamics as a function of scale.
