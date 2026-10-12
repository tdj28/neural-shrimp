# Adjudication: deep-research prior-work report (12 October 2026)

The human ran a separate deep-research query (GPT-6 Astra) on prior work for our LLM-dynamics findings and
supplied the report plus one paywalled PDF. The report is archived outside Git at
`../neural-shrimp-data/consults/2026-10-12-deep-research-prior-work/deep-research-report.md`
(sha256 67ea04428ab9adf954f5ab30718cb98741de1216d84cc19175a46f135f556fce). Every reference used below was
checked at its primary record (arXiv API or Crossref) on 12 October 2026.

## Verified references

| Reference | Relevance |
| --- | --- |
| Zeng et al., "PonderLM: Pretraining Language Models to Ponder in Continuous Space", arXiv:2505.20674 (2025) | the expected-embedding feedback operation itself (trained to exploit it); cite for the construction |
| Hao et al., "Training Large Language Models to Reason in a Continuous Latent Space", arXiv:2412.06769 (2024) | continuous latent feedback (Coconut) |
| Li et al., "Cognitive Activation and Chaotic Dynamics in Large Language Models: A Quasi-Lyapunov Analysis of Reasoning Mechanisms", arXiv:2503.13530 (2025) | Lyapunov-type chaos diagnostics in LLMs (layerwise) |
| Jaca, Benedek, Török, "Chaos in reason: How chain-of-thought LLMs can look for an answer", arXiv:2607.27805 (2026) | autoregressive generation as a dynamical system, chaos signatures |
| Arnold et al., "Phase Transitions in the Output Distribution of Large Language Models", arXiv:2405.17088 (2024) | temperature transitions of output ensembles |
| Nakaishi, Nishikawa, Hukushima, "Phase transition in large language models and the criticality of natural languages", arXiv:2406.05335 (2024) | temperature criticality of generated text |
| Sun, Haghighat, "Phase Transitions in Large Language Models and the O(N) Model", arXiv:2501.16241 (2025) | temperature phase transition |
| Ruan et al., "Generative Criticality in Large Language Model Temperature Scaling", arXiv:2606.06238 (2026) | susceptibility peak across Qwen3 sizes |
| Johnson, Huo, Li, "Temperature-driven inversion and nonlinear dynamics in ChatGPT-like AIs", arXiv:2608.00939 (2026) | closest competitor: frozen, periodic and intermittent output regimes, fitted effective nonlinear map, steering of a hidden coordinate |
| Wang et al., "Unveiling Attractor Cycles in Large Language Models: A Dynamical Systems View of Successive Paraphrasing", arXiv:2502.15208 (2025) | periodic attractors (2-cycles) of a paraphrasing loop |
| Tomihari, Karakida, "Recurrent Self-Attention Dynamics: An Energy-Agnostic Perspective from Jacobians", arXiv:2505.19458 (2025) | Jacobian dynamics of recurrent self-attention |
| Zhu et al., "Scaling Latent Reasoning via Looped Language Models", arXiv:2510.25741 (2025) | Ouro |
| Yang et al., "Stabilizing Recurrent Dynamics for Test-Time Scalable Latent Reasoning in Looped Language Models", arXiv:2605.26733 (2026) | STARS: Jacobian spectral radius and stable fixed points of looped LMs |
| Luo, Yu, "Prediction Dynamics in Depth-Recurrent Language Models", arXiv:2609.21383 (2026) | depth trajectories of Huginn and Ouro |
| Ko, Geiping, "Attractor States Emerge in Multi-Turn LLM Conversations", arXiv:2606.30571 (2026) | attractor states of model-to-model conversations |
| Herbster et al., "Activation Steering for Aligned Open-ended Generation without Sacrificing Coherence", arXiv:2604.08169 (2026) | activation steering |
| Zhang et al., "Personalized Text Generation with Contrastive Activation Steering", ACL 2025 (Long), pp. 7128-7141, doi:10.18653/v1/2025.acl-long.353 | activation steering |
| Stenzinger, Luz Oliveira, Tragtenberg, "Shrimp hubs in the Hindmarsh-Rose model", Chaos 35, 023148 (2025), doi:10.1063/5.0239268 | shrimp hubs in a neuron model; method: high-resolution phase and Lyapunov diagrams, maximum-stability (superstable) curves fitted inside shrimp bodies |

Two report details were corrected against the records: the Nakaishi et al. title (above), and the Claude 4
system card, which the report itself marks unverified (we cite it only after re-reading the source).

## Consequences for claims (accepted)

1. **Not claimable:** "first chaos in LLM inference" (Li et al. 2025; Jaca et al. 2026); "temperature drives
   regime changes" (Arnold, Nakaishi, Sun-Haghighat, Ruan, Johnson); "first expected-embedding feedback"
   (PonderLM; Coconut for continuous latents); "LLMs have periodic attractors" (Wang et al. 2025); "first
   Jacobian or fixed-point stability of looped LMs" (STARS); "LLM conversations have attractor states"
   (Ko and Geiping 2026); "first observation of intermittency in LLM output" (Johnson et al. 2026 use the
   word for a phenomenological regime class).
2. **Claims that the report and our own checks find unanticipated:** primitive periodic orbits of a frozen
   LM's exact inference map verified by Newton with monodromy multipliers; period adding tied to text
   structure (one "is X" pair per period step) accumulating on a 2-cycle's fold; type-I (Pomeau-Manneville)
   intermittency with the normal form computed from derivatives and passage times predicted without fitting;
   pseudo-arclength continuation of fold and flip curves in (temperature, steering); a Neimark-Sacker
   bifurcation of a released recurrent-depth LM predicted from the spectrum and confirmed with the
   square-root amplitude law; Markov-state-model metastability of self-dialogue with steering.
3. **Lead result:** the quantitative type-I intermittency (fold -> derivative normal form -> passage times
   -> 1/N^2 window accumulation, with the global reinjection measured), with period adding as its symbolic
   companion and continuation as the method. Shrimp language only after the topology is verified.
4. **Contrast to draw explicitly in any paper:** Johnson et al. fit an effective scalar map to sampled,
   growing-context generation; ours is the exact differentiable map of a fixed-prefix, one-token-window
   system, whose invariant sets are computed, not fitted. Our bridge experiment (m sampled embeddings) tests
   how much survives sampling.
5. **Method borrowed from the shrimp literature:** render periodic regions by their contraction (largest
   exponent, grey scale) so that maximum-stability curves appear, as in Stenzinger et al.; added as
   `render --style lyapunov` (commit c1cae4a0).

## Scoop risk noted by the report

Johnson, Huo and collaborators (effective maps, steering, intermittency); Jaca, Benedek and Török (chaos
diagnostics); Yang, Li and collaborators (STARS); Geiping's group (Huginn; conversation attractors); Luo and
Yu (depth-recurrent dynamics). This favours writing up the intermittency mechanism promptly once the
remaining checks (stripe edges from continuation, bridge to sampling) are in.
