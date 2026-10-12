# Pathway: attractors of repeated LLM inference (plan, 12 October 2026)

**Question.** Are the "attractor states" reported for repeated LLM inference
in the same mathematical family as the dynamics studied here: attractors of a
map iterated with fixed parameters, organized by bifurcations as a control
parameter varies? A prominent example is the "spiritual bliss" attractor in
Claude self-interactions.

## What has been reported

**Claude self-interaction** (Claude Opus 4 & Sonnet 4 system card, May 2025,
section 5.5):
- **Setup:** two instances talk with minimal prompting ("You have complete
  freedom").
- **Early turns:** in 90-100% of conversations they turn at once to
  consciousness. Across 200 thirty-turn transcripts, "consciousness" appears
  in every one, 95.7 times on average.
- **By about 30 turns:** most conversations reach themes of cosmic unity,
  gratitude, Sanskrit, emoji and empty "silence". One transcript uses one
  emoji 2,725 times.
- **Task-based and adversarial audits:** about 13% of conversations enter the
  state within 50 turns.
- **When allowed to end the conversation:** the models stop after about 7
  turns, before the spiritual phase.

**Related measurements in other systems:**
- Repeated paraphrasing settles into 2-period cycles
  ([Wang et al., ACL 2025](https://arxiv.org/abs/2502.15208)).
- LLM transmission chains have "cultural attractors", and open-ended
  instructions attract more strongly than constrained ones
  ([Perez et al., ICLR 2025](https://arxiv.org/abs/2407.04503)).
- Sampling temperature drives a phase transition in generated text: repetitive
  below a critical temperature, incoherent above it, with power-law
  correlations at it ([Nakaishi et al.](https://arxiv.org/abs/2406.05335)).
- A case study of the Claude observation and its press reception (Michels,
  PhilArchive MICSBI) could not be retrieved for reading here.

## Making "attractor" precise

1. **Greedy decoding with a bounded context window** is a deterministic map on
   a finite set, so every run ends in a cycle. The questions are cycle
   lengths, transient lengths and basins.
2. **Sampling** turns it into a Markov chain on conversation states. Here an
   attractor is a long-lived region. Hitting probabilities and times,
   residence and escape times, and their dependence on temperature are
   measurable.
3. **Continuous level.** The network is a smooth map on hidden states, and
   discreteness enters only at sampling. Turn embeddings give continuous
   trajectories, where drift fields, fixed points and stability can be
   estimated.

## Hypotheses suggested by this project's results (untested)

- **Synchronization.** Two instances of one model form a pair of coupled
  identical systems. Their synchronized state can attract, as condensation of
  identical hidden units did in wide students here. Test: agent-to-agent
  similarity per turn.
- **Saturation without a restoring force.** Mutual amplification with no
  outside input may run to a saturated, low-entropy end state, much as
  training without weight decay escaped to saturated weights with neutral
  directions. Tests:
  - entropy and length decay per turn;
  - whether outside input (a task, a third party, an option to end) acts as
    a restoring force.
- **Bifurcations in a control parameter.** Do attractors appear, merge or
  switch abruptly, with locked windows or hysteresis, as temperature or a
  continuous blend of two system prompts varies?

## Pilot

`tools/explore_llm_attractors.py` (tests: `tests/test_llm_attractors.py`):

- **Loops:** self-dialogue and repeated paraphrasing with small open models
  (Qwen2.5 0.5B and 1.5B Instruct) on CPU.
- **Sampling:** repetition penalty, top-k and top-p are off, so only the
  temperature varies.
- **Recorded per turn:** text, length, mean next-token entropy, mean
  last-layer hidden state, and exact cycles of the windowed state.
- **Pilot 1** (running): open versus task prompt; temperatures 0, 0.7 and
  1.0; repeated paraphrasing at temperatures 0 and 0.7.

Claude itself would require external API calls. Under the repository policy
these need explicit approval and a budget.

## Framework for mapping, measuring and steering loop attractors (12 October 2026)

1. **Object.** With the weights fixed, a loop is a map F_c of contexts,
   where c are the controls: steering strengths, temperature and the opener.
   - Greedy decoding makes it deterministic, so runs end on exact cycles.
   - Sampling makes it a Markov chain.
2. **Phase maps.**
   - **Controls:** add `alpha * v` to the residual stream at one layer, where
     v is a contrastive activation-steering direction (implemented in
     `tools/explore_llm_phase_map.py`).
   - **The map:** a grid over two directions, each cell run to its
     attractor, coloured by period and attractor identity. This is the
     analogue of the learning-rate maps.
   - **Grouping:** attractors are grouped into families by their
     cycle-mean activation (centred, cosine distance), because exact text is
     too fine.
3. **Existence regions.**
   - **Margin function:** a greedy cycle C persists exactly while every
     logit margin along it is positive. Let m_C(alpha) be its minimal margin
     along the fixed token path.
   - **Region:** {alpha : m_C(alpha) > 0} is C's existence region. Its
     boundary is the zero set of a piecewise-smooth function, which can be
     continued with gradients like a fold curve.
   - **Multistability:** regions where several existence regions overlap
     are multistable. Which attractor is reached there depends on the opener
     (the basin).
4. **Noise.** At sampling temperature T, the probability of reproducing a
   cycle is a product of softmax probabilities along its path. As T -> 0,
   T(-log(1 - P_stay)) -> min margin (verified for six attractors:
   `explore/llm-attractors/margins-opener*`). So residence in the exact cycle
   grows like exp(m_C / T). Basin residence adds returns, which are measured
   by sampled escape runs.
5. **Stochastic dynamics (planned).**
   - Build a transfer-operator (Markov-state-model) estimate on clustered
     turn activations from many sampled runs.
   - Leading eigenvalues give the metastable attractors and their lifetimes.
   - Eigenvalues near p-th roots of unity give cycles.
   - Committors give basin boundaries, and transition-path theory gives the
     routes between attractors.
6. **Manipulation.** The first-order steering distance to an attractor's
   boundary is m_C / |grad_alpha m_C|. Steering along the gradient closes the
   weakest margin (escape) or widens it (stabilization).
7. **Scaling.** Every quantity needs only forward passes, gradients,
   batched generation and activation statistics, so it applies to any open
   transformer. Generation cost is the bottleneck (GPU).

## Looped language models: the model's own latent recurrence (12 October 2026)

**Targets:**
- **Looped transformers** apply one stack of layers repeatedly to a latent
  state. Their inference contains a genuine smooth iterated map, defined by
  the model itself.
- **Ladder of open models:**
  - Ouro-1.4B (ByteDance research looped LM, 4 trained loops);
  - Huginn-0125 (recurrent depth with input re-injection, about 32 trained
    iterations);
  - Nanbeige4.2-3B (production-grade open looped transformer: two passes of
    a 22-layer stack, Apache 2.0, trained with supervised fine-tuning and
    RL; its report says more passes made training slower and less stable).
- **Architecture class:** reports that a frontier model (OpenAI's Astra) is
  looped are unconfirmed (Raschka, 2 September 2026). The class itself is in
  production use.

**First result.** Ouro-1.4B's own loop, extracted and verified against the
native forward, does not converge beyond its trained 4 loops. All tested
prompts give positive largest Lyapunov exponents (0.29-0.40 per loop), and
the read-out degrades (`explore/latent-loop/ouro-trajectory-64`).

**Shrimp hunt.** The relaxed loop h <- (1 - lambda) h + lambda F(h) plays
the learning-rate role. Map it in lambda and in a (lambda, steering) plane,
then verify candidate windows by Newton-Krylov orbits and fold and flip
continuation, in float64 where needed.

**What a positive finding would support.** A shrimp-organized loop map in
two independently trained looped models would support a prediction for the
architecture class: shrimps are generic in two-parameter families of smooth
dissipative maps with chaotic regimes. Non-looped models need their own
evidence, from token loops or soft-token decoding.

## Structure of causal latent loops (12 October 2026)

**Cascade.** In a looped transformer with causal attention, position j's
next latent state depends only on positions 0..j: norms and MLPs act per
position, and attention looks only backward. The loop map is therefore
triangular, F(h)_j = F_j(h_0, ..., h_j). Checked numerically for Ouro:
perturbing the last position changes no earlier output (difference exactly
0). Consequences:

- **Position 0 is autonomous.** In Ouro, which injects no input, the prompt
  only sets the initial state, so position 0 follows one fixed map for every
  prompt.
- **Linear stability splits by position.** The Jacobian is block
  lower-triangular. The multipliers of any periodic orbit, and the Lyapunov
  exponents, are the union of the positions' conditional (diagonal-block)
  spectra. These can be measured with twins perturbed at one position only
  (`explore_latent_loop positions`).
- **Attractors stack.** Each position settles on an attractor of its dynamics
  as driven by the earlier positions. Coexisting attractors can therefore
  multiply along the sequence, which explains the many distinct 2-cycles seen
  in the basin tests.
- **Bifurcations happen at one position.** A multiplier crossing as a
  parameter varies belongs to one diagonal block.

**Ouro-1.4B, story prompt** (`explore/latent-loop/ouro-positions-*`):

- **Native loop (lambda = 1).**
  - Positions 0-6 sit at fixed points.
  - Positions 7-12 move irregularly, with conditional exponents of +0.011
    to +0.040 per loop.
  - So position 7 is an autonomous 2048-dimensional map with a frozen
    context.
- **Verified stable 2-cycle (lambda 1.280315).** Positions 0-4 are fixed and
  positions 5-12 oscillate. The full and largest conditional exponents agree
  (-0.036 versus -0.033).
- **Open: full versus conditional exponents on irregular motion.** There the
  full-system exponent exceeds the largest conditional one (0.093 versus 0.040
  at lambda 1). For a triangular cocycle this can only be a finite-time
  effect, so whether it vanishes in longer runs remains to be tested.
- **Fixed point " trillion".** Its spectrum (real, 0.9625-0.9997 at the top)
  makes it stable for all lambda in (0, 1.747), including the native loop. So
  the lambda ~ 0.90 window is a basin effect, and the native loop is
  multistable.
- **The 2-cycle's window.** The 2-cycle exists for lambda between about 1.265
  and 1.318.
  - Lower edge: the leading real multiplier follows
    (1 - mu)^2 = 0.32 (lambda - 1.2646), the square-root law of a fold.
  - Pseudo-arclength continuation is checking that the branch turns.

**Huginn-0125** (prompt re-injected at every step):

- **Extraction checked.** The extracted recurrence reproduces the native
  forward (maximum logit difference 4.8e-6).
- **Native loop converges.** For a story prompt and a GSM8K-style question,
  every position converges to a fixed point within 400 native steps, with a
  leading contraction of about 0.87 per step on the story prompt.
- **The reported "orbits".** Geiping et al. (arXiv 2502.05171, section 7)
  describe orbit-like PCA trajectories for tokens such as "3" and "makes".
  At the fixed point of the GSM8K-style question, F's leading multipliers
  are a complex pair 0.588 +- 0.771i (modulus 0.970, 0.146 turns per step),
  so trajectories are slowly decaying spirals: after 32 steps about 38 % of
  that component remains. The spectrum also predicts that the relaxed loop
  h <- (1 - lambda) h + lambda F(h) loses stability at lambda = 1.0777
  through this pair (a second pair crosses at 1.081).
- **Neimark-Sacker confirmed.** Run from the fixed point, the relaxed loop
  decays for lambda up to 1.075 and settles on closed curves beyond the
  threshold. The squared amplitude is linear in lambda with intercept
  lambda_c = 1.0778, and A = 0.53 sqrt(lambda - lambda_c) within 2 % from
  1.085 to 1.12: a supercritical Neimark-Sacker bifurcation where the
  spectrum predicted it. The motion rotates 0.156 turns per step, the
  predicted argument at threshold. Positions 0-12 stay put; the oscillation
  starts at position 13, the "." that ends the first sentence, and drives
  every later position. So 8 % past the native step, the model's own loop
  has genuine latent orbits (invariant circles); at the native step they are
  slowly decaying spirals. Whether motion on the circle is quasi-periodic or
  locked is not yet measured.
- **Story prompt.** Leading pair 0.762 +- 0.429i (modulus 0.874); its relaxed
  loop would first lose stability near lambda 1.25 through a nearly real
  -0.60 multiplier. The slow spiral is specific to the GSM prompt here.

**Self-dialogue attractors (Qwen2.5-0.5B, greedy)** (`explore/llm-phase-map/map-l12-grat-phil-13`):

- **Map.** The 13x13 gratitude x philosophy steering map has 138 exact
  cycles in 5 activation families.
- **One family dominates.** "You are truly loved and I am truly with you ...
  I. I. I." covers most of the plane.
- **Existence map (minimal logit margins).** That cycle already exists
  without steering. Its margin at zero steering (+2.35) is the largest of the
  12 cycles tested, so by the margin law it is the most noise-robust.
- **Steering moves basins.** Steering moves the trajectory from the opener
  into the cycle's basin; it does not create the cycle.

**Soft-token decoding (Qwen2.5-0.5B, window 1)** (`explore/llm-soft-loop/`):

- **Low temperatures.** The bands are super-stable near-greedy cycles, with
  multipliers below 1e-16.
- **Transition zone.** At (log10 T -0.047, alpha 0.040), float64 Newton
  verifies a primitive, attracting period-9 orbit. Five of its nine steps
  have entropies of 3.6-5.3 nats, so it is a genuinely smooth orbit.
- **Correction.** Runs labelled float64 before 12 October 10:00 UTC computed
  every RMSNorm in float32 (the Hugging Face upcast). That affects
  near-zero Lyapunov values and isolated long periods of the 16x16 scan;
  fixed in all tools.

## Periodic windows inside soft-token chaos (12 October 2026)

Soft-token decoding feeds the model's own probability-weighted embedding
back as the next input, x -> E^T softmax(z(x) / T), with a fixed prefix
and steering alpha along a concept vector at layer 12. It is a smooth map
of the 896-dimensional embedding space. Its T -> 0 limit is greedy decoding
of this same fixed-prefix, one-token-window system (given a unique argmax),
not ordinary generation with a growing context. Expected-embedding feedback
is a constructed dynamical system of the model's own computation. It is not
yet a validated proxy for sampled decoding; replacing the mean embedding by
averages of m sampled token embeddings (m = 64, 8, 1) is the planned bridge.
Maps and checks below are for Qwen2.5-0.5B with a one-token window
(`explore/llm-soft-loop/`).

- **Chaos with windows.** On the 96x96 zoom over alpha 0.03..0.09 and
  log10 T -0.3..0.25, 2676 of 9216 cells have a positive twin exponent,
  mostly in one sea at log10 T -0.17..0.20 for alpha above 0.045. Periodic
  windows sit inside it: a thin diagonal period-3 band, period-4 bands, and
  thin diagonal stripes with periods 12, 10, 8 and 6 between a period-2
  region and a period-3 region.
- **The windows are periodic orbits.** float64 Newton (norms patched, dense
  forward-mode Jacobians) verifies primitive orbits at interior cells:
  period 8 at (log10 T -0.1205, alpha 0.0717), residual 1.7e-15, multipliers
  led by +0.0041; period 10 at (-0.1089, 0.0673), residual 1.6e-13, led by
  +0.0024. Period 6 at (-0.1205, 0.0786) and period 3 at (0.1168, 0.078)
  return to within 2e-14 and 6e-15.
- **Period adding in token space.** Read out as tokens, the stripe orbits
  are ". This" followed by k "is X" pairs, period 2k + 2: ". This is a is a"
  (period 6), ". This is there is a is a" (8), ". This is there is it is a
  is a" (10) and ". This is it is it is it is a is a" (12; float64 residual
  3e-15). The neighbouring period-2 region is the 2-cycle "is it" (float64
  residual 6e-15, leading multiplier -0.27). Each stripe orbit loops k times
  near that 2-cycle, escapes through ". This" and is reinjected.
- **Accumulation law (provisional).** A fine float32 cut at log10 T -0.115
  (700 cells, alpha step 2.5e-5) finds the stripes P6, P8, ..., P20 in order,
  each with a doubled-period sliver on its smaller-alpha (flip) side, closing
  in on the 2-cycle's fold at alpha_c = 0.0627972. The distances of the
  stripe centres from alpha_c follow d_k = 0.0264 / (k - 2)^2 within 1 % for
  k = 6..9: 1/sqrt(d_k) grows by 6.07, 6.12, 6.28 per step. Their ratios
  rise from 0.56 to 0.73, so the accumulation is not geometric (an earlier
  reading from coarse cells, 0.57 per step, was wrong). This is the window
  law of a saddle-node ghost (type-I intermittency of Pomeau-Manneville
  type). Past its fold the 2-cycle leaves a bottleneck. Passing through it
  takes about pi / sqrt(a beta (alpha - alpha_c)) laps of the two-step map,
  where a beta is the 2-cycle's normal-form coefficient. Each extra "is X"
  pair is one more lap, and ". This" is the reinjection. Exact fold edges
  in float64 and the slope sqrt(a beta) / pi from the 2-cycle's multipliers
  are being measured (job 06e26).
- **Precision.** The float32 map is accurate to only 3e-6 to 2e-5 per step
  (soft embeddings have norms near 0.2). In float32 the period-8 orbit was
  read as period 16 and the period-10 orbit was missed, and the zoom maps'
  period test sits at that error level. Orbits are therefore verified in
  float64. Derivatives are worse than this suggests: one-step float32
  Jacobian-vector products are accurate to about 5e-5, but the 8-step
  monodromy of the period-8 orbit is off by 3 %. Its critical multiplier
  comes out as 1.024 instead of 1.049, and the fold test function g is off
  by a factor of two, because eight strongly contracting, non-normal steps
  compound the error. Boundary continuation therefore takes residuals, the
  test function g (GMRES on the bordered system) and the critical
  multipliers (a short Arnoldi factorization) from float64 Jacobian-vector
  products, and uses float32 dense monodromies only for the Newton
  matrix.
- **Boundaries.** Continued in temperature at alpha 0.0717, the period-8
  orbit keeps a real leading multiplier. Upward it passes close to zero
  (6e-5 near log10 T -0.108) and rises toward +1 as the branch turns: a fold
  near log10 T -0.1033 (square-root extrapolation). Downward it changes sign
  near -0.121 (a superstable point) and crosses -1 at -0.128955, an
  isolated real -1 (flip; next multiplier 3e-6; float64 multipliers -0.936
  and -1.105 on either side). In float64 the fold at alpha 0.0717 is at
  log10 T -0.1033462 (critical multiplier 0.99999983). So this cut has
  the profile of a periodic window: saddle-node at high temperature, period
  doubling at low temperature. A long narrow region bounded by a fold on one
  side is how shrimp legs are described (Facanha, Oldeman and Glass 2013,
  doi:10.1016/j.physleta.2013.03.025), but a shrimp also needs its body: the
  fold and flip curves in the plane, with cusps where fold curves meet.
  `curve` traces them.

## Mechanism of the period-adding stripes (12 October 2026, evening)

- **Normal form of the 2-cycle's fold, in float64.** At log10 T -0.115 the
  "is it" 2-cycle dies in a fold at alpha_c = 0.0627983 (multiplier 1 to
  3e-13). For the two-step map G the normal form is u' = u + a eps + b u^2.
  - From derivatives: a = -4.834 per unit alpha, including the steering's
    effect on the prefix, and b = -25.4285 from a finite-difference plateau,
    so ab = 122.93.
  - From the stable branch: the multipliers give
    (1 - mu)^2 / (4 (alpha_c - alpha)) -> ab = 123.1 when extrapolated.
- **The local bottleneck is exactly that normal form.** Controlled entries at
  four held-out distances beyond the fold are predicted to within one
  two-step iterate, with no fitting:
  - full channel: 144, 71, 34 and 14 iterates;
  - midpoint entry: 73, 36, 17 and 7;
  - a norm-matched transverse kick costs at most one iterate.
- **The global return halves the passage.** Free trajectories re-enter the
  channel slightly upstream of the bottleneck's middle, near a fixed point
  of the critical coordinate, so a natural passage is about 0.6 of a full
  one. From the measured entry points the normal form predicts natural
  laminar durations within 4-8 % (eps <= 1e-4). This explains why the
  stripes follow the 1/N^2 law of a saddle-node ghost with about a third of
  the full-passage prefactor.
- **Reading.** Soft-token decoding contains Pomeau-Manneville type-I
  intermittency:
  - "is it" is the laminar channel, the ghost of the 2-cycle;
  - ". This" is the reinjection;
  - each extra "is X" pair is one more lap.

  The scaling exponents are universal; the prefactor is set by where the
  model's own text re-enters the loop.
- **Resolution.** A 256x256 float64 map of the region, about 100 GPU-minutes
  on an H100, shows the stripe fan out to period 32. The exact fold curve of
  the period-8 orbit lies on the stripe's edge and caps its top. A second
  family of bands crosses the fan near its lower end.

## High-temperature period-4 region: shrimp candidate (12 October 2026, evening)

- **Not one-dimensional.** Float64 4-cycles at log10 T 0-0.1 have second
  multipliers of 0.06-0.09, unlike the stripes' 1e-6.
- **The silhouette is several attractors.** The float32 period map shows a
  period-4 body with a period-8 island and period-4 legs edged by period 8.
  Float64 orbits show that this one colour covers at least four different
  4-cycles (different text loops). Scans now record each cell's attractor
  identity, and maps can be coloured by it.
- **One orbit family's boundaries.** The body orbit ("\n\n - \n\n The"):
  - in temperature at alpha 0.06484, an S-shaped branch with two folds
    (log10 T 0.0640 and 0.0664: bistability between them) and a flip at
    0.0695;
  - toward smaller alpha, its leading multiplier hovers near -1 and crosses
    it five times, alternating period-4 and period-8 bands, which are the
    "legs with period-8 edges".

  Two folds on one branch must meet at a cusp in the plane. Fold and flip
  curves are being traced in float64 to locate and certify it (normal-form
  conditions), before any shrimp claim.

## Toolkit for the model's own dynamics (12 October 2026)

| Level | Map | Tool and subcommands | What is computed |
| --- | --- | --- | --- |
| Greedy token loop | deterministic map on context windows | `explore_llm_phase_map` map, families, existence | exact cycles, attractor families, existence regions from minimal logit margins |
| Sampled token loop | Markov chain on windows | `explore_llm_msm` sample, build | metastable sets, implied timescales, stationary weights, residence times |
| Soft-token decoding | smooth map of embedding windows | `explore_llm_soft_loop` scan, render, windows, orbit, precision, tline, branch, curve, branch-plot | period and Lyapunov maps; float32 error against float64; Newton orbits with exact (forward-mode) Jacobians; one-parameter branches with all multipliers and classified crossings; two-parameter fold and flip curves with cusp flags (`tools/bifurcation_curves.py`) |
| Looped LM latent loop | the model's own recurrence (Ouro, Huginn) | `explore_latent_loop` sweep, basin, orbit, continue-orbit, arclength, fixed-spectrum, positions, frozen-check, frozen-plane, vocab-basins | bifurcation diagrams; Newton-Krylov orbits and Arnoldi multipliers; pseudo-arclength through folds; fixed-point spectra; per-position (cascade) exponents; frozen-context single-position maps; attractors of the universal single-token map |

All of them need only forward passes, forward-mode JVPs and batched
generation, so they apply unchanged to larger open models. The cost per
step is one forward pass (or one per position, with frozen contexts).
