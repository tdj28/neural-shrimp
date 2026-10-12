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
