# Brief: chaos, period adding and intermittency in a language model's own inference dynamics

12 October 2026. Exploratory results, not yet a manuscript. The wider project studies shrimp-shaped periodic
windows (Gallas) in neural-network training maps; this brief concerns only the inference dynamics of
language models, iterated as the models themselves define them.

## Systems

1. **Soft-token decoding (Qwen2.5-0.5B-Instruct).** Feed back the expected embedding instead of a sampled
   token: x_(n+1) = E^T softmax(z(x_n) / T), where E is the input embedding matrix, z the next-token logits
   given a fixed chat prefix and the one-token window x_n, and T the temperature. A steering vector alpha v
   (contrastive "philosophy" direction, residual stream, layer 12) is added at every position, prefix
   included. This is a smooth map of R^896; T -> 0 recovers greedy decoding. Control plane: (log10 T, alpha).
2. **Recurrent-depth latent loop (Huginn-0125).** At fixed input tokens, iterate the model's own latent
   recurrence h <- F(h) and its relaxed form h <- (1 - lambda) h + lambda F(h).
3. **Sampled self-dialogue (Qwen2.5-0.5B).** Two copies converse (temperature 0.7, last 7 turns visible,
   128 replicas x 40 turns, optional steering), analysed as a Markov chain.

## Methods

Twin-trajectory Lyapunov exponents and return-distance period tests on grids. Periodic orbits by Newton's
method with forward-mode (AD) Jacobians; multipliers from the dense monodromy, refined by Arnoldi with exact
JVPs. Pseudo-arclength continuation; fold and flip curves via minimally augmented systems (bordered test
function g, solved by GMRES with JVPs). Hugging Face RMSNorm and rotary tables compute in float32 even for
float64 models, so both are patched for float64 runs. Markov state models on mean-pooled layer-12 turn
activations (PCA 20, 40 k-means states, metastable sets from the leading right eigenvectors, implied
timescales at lags 1-4).

## Findings

**A. Chaos with windows (soft-token map).** On a 96x96 grid over alpha 0.03..0.09 and log10 T -0.3..0.25,
2676 of 9216 cells have positive twin exponents (up to 0.72 per step); periodic windows of periods 1-32 lie
inside the chaotic region. A 16x16 test grid agrees in float32 and float64 on 245 of 256 periods; the
differences are spurious long periods in float32.

**B. Period-adding stripes.** Diagonal stripes of periods 6, 8, 10, ..., 20 lie between a period-2 region
and the chaos. Periods 6, 8, 10 and 12 are verified as primitive stable orbits in float64 (Newton residuals
1e-13 to 1e-15). Their multipliers are effectively one-dimensional: leading +0.0041 (P8) or +0.0024 (P10),
all others about 1e-6. Read as tokens, the orbits are ". This" followed by k "is X" pairs, period 2k + 2
(". This is a is a", ". This is there is a is a", ". This is there is it is a is a",
". This is it is it is it is a is a"). The period-2 region is the 2-cycle "is it", which dies in a fold at
alpha_c = 0.0627983 (log10 T -0.115; float64 multiplier 1 to 3e-13).

**C. Window law.** A float32 cut at log10 T -0.115 (700 cells) finds the stripes in order, each with a
doubled-period sliver on its flip side. With N = k - 2 the number of "is it" laps (the orbits end with two
"is a" pairs), the stripe centres satisfy d_N = alpha_N - alpha_c = 0.001645, 0.001059, 0.000736, 0.000537
for N = 4..7, so d_N N^2 = 0.0263-0.0265 and the log-log slope is -2.00. Ratios d_(N+1) / d_N rise from 0.56
toward 0.73, so the accumulation is not geometric. In chaotic cells just past alpha_c, trajectories linger
in the "is it" channel: mean laminar lengths 32.8, 21.1, 15.0, 12.6, 10.6, 7.0 steps at
epsilon = 1.0e-4 .. 3.7e-3, consistent with epsilon^(-1/2) plus an entry-exit offset of about 2 steps (a
free fit gives exponent 0.41 +- 0.03). The two-step return map of p(" it") touches the diagonal near the
2-cycle's value (a near-tangency). Reading: windows and laminar phases of type-I intermittency
(Pomeau-Manneville 1980), with "is it" as the channel and ". This" as reinjection. Open: the window
prefactor implies a beta = pi^2 / 0.0264 = 374 if N = pi / sqrt(a beta epsilon), but float32 multipliers
along the 2-cycle branch give (mu - 1)^2 / (4 (alpha_c - alpha)) of about 103-152; float64 multipliers are
being computed.

**D. Boundaries of one stripe.** The period-8 orbit at alpha 0.0717 is stable for log10 T between a flip
at -0.128955 (isolated real -1, float64) and a fold at -0.1033462 (float64 multiplier 0.99999983). The fold
curve, continued in float64 test functions, runs along the stripe's upper edge to a maximum log10 T of
-0.06506 at alpha 0.0637, then turns smoothly toward smaller alpha and lower T, capping the stripe (no cusp
so far; the second multiplier grows from 1e-7 to 1e-3 along it). Flip-curve tracing and 256x256 float64
maps are running.

**E. A more two-dimensional region.** At higher temperature (log10 T 0..0.1, alpha about 0.06), float64
stable 4-cycles have multipliers -0.317, 0.056, -0.030 and -0.752, 0.016 +- 0.085i: not effectively
one-dimensional. This is where we would look for genuine shrimps.

**F. Precision.** One-step float32 JVPs are accurate to 5e-5, but the 8-step monodromy of the P8 orbit is
off by 3 % (critical multiplier 1.024 against 1.049 in float64; test function g off by a factor of two).
All multipliers and test functions are now computed from float64 JVPs.

**G. Huginn.** For a GSM8K-style prompt the native loop converges to a fixed point whose leading pair is
0.588 +- 0.771i (modulus 0.970). This predicts a Neimark-Sacker bifurcation of the relaxed loop at
lambda = 1.0777. Measured: decay below, a closed-curve oscillation above, with lambda_c = 1.0778 from a fit
of the squared amplitude and A / sqrt(lambda - lambda_c) = 0.52-0.54 (supercritical), rotation 0.156 turns
per step. Causal attention makes the loop block lower-triangular in token positions. The first position to
oscillate is the "." of the prompt (position 13); all later positions follow, earlier ones stay at the
float32 floor. Ouro-1.4B's loop beyond its 4 trained iterations is chaotic (exponents 0.29-0.40 per loop).

**H. Stochastic attractors of self-dialogue.** Unsteered: a "helper closure" metastable set ("You're very
welcome! If there's anything specific...") holds 73 % of the stationary weight with residence about 35 turns;
topical sets last 5-8 turns. Steered along a gratitude direction (alpha 0.04): a "truly loved / blessed"
set holds 95 % with 7 exits in 4892 transitions (effectively absorbing over 40 turns), occupied from the
first reply on. Within it, the share of replies with spiritual words (spiritual, sacred, soul, divine,
grace, eternal, light, peace, ...) rises from 0.13 to 0.45 over 40 turns, and 2-3 % switch to Chinese. The
mean-pooled activation projection on the philosophy direction does not drift. Steering ladders (0.01, 0.02)
are running.

## Questions (please answer Q1 and Q2 first, with precise references)

**Q1. Novelty.** Has any language-model inference loop (soft-token or expected-embedding feedback,
continuous "thought" tokens, greedy or sampled decoding, recurrent-depth or looped transformers, deep
equilibrium models) been analysed as a dynamical system with Lyapunov exponents, periodic windows,
bifurcation curves, period adding or intermittency? What is the closest prior work? Please give authors,
title, venue, year and DOI or arXiv ID, and mark any reference you are not sure of.

**Q2. Self-dialogue attractors.** Have the "spiritual bliss attractor" of model self-conversation
(Anthropic's Claude Opus 4 system card, 2025) or similar attractors been quantified with Markov state
models, transfer operators or residence times? Related work on degeneration and repetition loops?

**Q3. Theory of the window law.** For an effectively one-dimensional smooth map near the saddle-node of a
2-cycle with global reinjection, what window law should hold (prefactor pi^2 / (a beta), a fraction of it
depending on where reinjection lands, higher-order corrections at N = 4..7)? How would you test it cleanly
from float64 multipliers, laminar-time distributions or a reconstructed return map?

**Q4. Shrimps.** With one nontrivial multiplier, should fold curves of these windows have cusps, or are
smooth stripes and caps the expected organization? What would convince a dynamical-systems referee that a
two-dimensional window in region E is a shrimp (fold and flip curves, cusps, superstable curves, period
doubling inside)?

**Q5. Research plan and paper.** Which of these results could anchor an arXiv paper (cs.LG with nlin.CD)?
What controls and replications would a referee require: other prompts, steering directions, windows K > 1,
seeds, models (Qwen2.5-1.5B and larger, other families, a production looped model)? Is soft-token decoding
a credible proxy for real decoding, and which experiments would connect the soft map to sampled or greedy
generation? What should we stop doing?

**Q6. Artefacts.** Which failure modes have we not considered: finite-time effects in grids, precision,
basin dependence, the fixed prefix, the steering construction?
