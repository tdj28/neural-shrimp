# Consult 12 brief: new ideas for practical consequences of chaos and bifurcations in language-model feedback

16 October 2026. This is a request for ideas, not a review. Please think broadly and creatively, as a research
director who knows both nonlinear dynamics and how large language models are trained, deployed and used. Propose
practical, ideally deep, consequences of the findings below that we have not thought of, and experiments that could
establish them.

## What we found (released models, measured carefully)

- **The system.** Feed a language model the probability-weighted average of its next-token embeddings (the "concept
  token" of Soft Thinking), with a fixed chat prompt. This gives a deterministic map x -> E^T softmax(z(x)/T) of the
  embedding space, which becomes greedy decoding as T goes to 0.
- **Regimes.** In fifteen released models (Qwen2.5, OLMo-2, Llama-3; 0.5B to 72B parameters), temperature and
  steering vectors move this loop between repeating text (periodic attractors) and chaos (positive Lyapunov
  exponents, 0.1-0.9 nats per step). Chaos was found in all three families, up to 70B.
- **Low-dimensional organization**, as in classical maps:
  - period-doubling cascades with Feigenbaum ratios;
  - saddle-node bottlenecks with type-I intermittency (laminar phases near a vanished periodic orbit, a 1/N^2 law
    for periodic windows);
  - coexisting attractors: from different starts, the same model and settings write different texts forever;
  - shrimp-shaped periodic windows;
  - "hidden" chaos: the text repeats exactly while the internal state moves chaotically.
- **Mechanism.** The loop's derivative is DF = T^{-1} Sigma_p J, the covariance of token embeddings under the
  model's own next-token distribution times the network Jacobian. Chaos appears where tokens compete and 1/T
  amplifies. Steering creates chaos mainly by making tokens compete, and a logit bias alone suffices. Normal forms
  computed from the model's derivatives predicted, in advance, how long trajectories linger near folds.
- **Consequences measured so far.**
  - *Reproducibility.* With only the batch size changed, chaotic loops separate at the rate the Lyapunov exponent
    predicts, while periodic loops stay together. Batch-invariant kernels remove one rounding source, but a chaotic
    loop amplifies any other.
  - *Answers.* In bfloat16 soft-thinking inference with a growing context, the batch layout changed 55-56 % of
    arithmetic answers, against 0-10 % for greedy decoding. Resetting the fed-back state removes this, so the
    feedback is the amplifier.
  - *Recurrent-depth models.* A bifurcation of the model's own iteration predicted an accuracy cliff, confirmed by a
    test fixed in advance.
  - *Interaction.* Two chaotic loops writing one shared text often lock each other into periodic orbits.
- **Nulls.** Whether a prompt's loop is chaotic does not predict ordinary answer accuracy beyond the model's
  first-token uncertainty. No "edge of chaos" accuracy advantage.

## Resources

A local 16 GiB GPU (a 7B model in bfloat16 at about 190 tokens/s in batches), about [amount redacted] of H100 time for float64
measurements, and a few days. Models up to 7B locally, up to 70B on rented GPUs.

## What we would like from you

Propose 10-15 concrete ideas for practical or deep consequences that matter to people who build, align or interpret
language models: inference engineers, interpretability and alignment researchers, agent builders. Range widely. For
example (not limits):
- agents and tool loops;
- self-refinement and debate;
- test-time compute and long chains of thought;
- sampling and diversity;
- uncertainty and hallucination;
- watermarking or fingerprinting;
- security: adversarial control of a loop by tiny perturbations;
- training on model outputs, distillation and RL;
- steering safety;
- interpretability limits;
- energy, efficiency or caching;
- evaluation reliability.

For each idea give:
1. the claim, in one sentence;
2. why the dynamics found here would predict it;
3. the smallest experiment that could test it with our resources (models, data, measure, control);
4. what result would be a headline, and what would kill the idea.

Then rank your ideas by expected value (probability of a striking, true result times its importance) per day of
work. Already planned, so please go beyond these:
- reproducibility of Soft Thinking against its top-n filtering;
- a gain-limiting stabilizer;
- majority voting over perturbed runs;
- batch flips in ordinary decoding;
- repetition against the chaos-onset temperature.
