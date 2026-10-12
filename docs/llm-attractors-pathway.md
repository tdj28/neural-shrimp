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
