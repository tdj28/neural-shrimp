# LLM dynamics: literature audit (Task 3)

Work in progress, 12 October 2026. Findings below have been checked at primary
records; the final synthesis and bibliography follow. Claude owns integration
into `paper/llm/`. No new experiment or paid review is part of this task.

## Immediate positioning corrections

The current numerical results remain meaningful, but several broad novelty
formulations need narrowing. The distinct target is the measured dynamics of a
frozen pretrained language model's specified feedback system, including
primitive periodic orbits, continued boundaries and a derivative-based passage
prediction. Individual ingredients have close predecessors.

| Primary source | What overlaps | Consequence |
|---|---|---|
| [Kim, 2608.26556v1](https://arxiv.org/html/2608.26556v1), equations 3–5 | A custom task-trained looped transformer has derivative-derived fold coefficients and unfitted passage amplitudes. | Do not claim the first such mechanism in transformer inference generally. Distinguish the pretrained language model and periodic-orbit setting. |
| [Liu et al., 2609.38814](https://arxiv.org/abs/2609.38814) | Transformers trained from scratch on dynamical-system trajectories recover global dynamics; full-text cascade comparison is being audited. | Distinguish the pretrained LM's own feedback dynamics from a learned scientific surrogate. |
| [Ribeiro et al., AISTATS 2020](https://proceedings.mlr.press/v108/ribeiro20a.html) | Attractor/bifurcation analysis includes a trained LSTM language model. | Restrict any transformer-specific comparison; language-model dynamical analysis is older. |
| [Ashok and May, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/5a9c1af5f76da0bd37903b6f23e96c74-Abstract-Conference.html) | Input-state probes forecast output behavior before generation. | Early detection before bad text is already established; test incremental multi-turn lead time against such probes. |
| [Chen et al., Persona Vectors, §3.3](https://arxiv.org/html/2507.21509v1) | Last-prompt-token projections predict subsequent trait expression. | This is already anticipatory, not merely concurrent monitoring. |
| [Lu et al., Assistant Axis](https://arxiv.org/html/2601.10387v1) | Conversation drift, activation capping and initial-turn projections related to subsequent harmful compliance. | Include as a direct monitoring/control baseline. |

Search continues on frozen soft-feedback constructions, neural Feigenbaum
ratios, shrinking windows, neural early-warning signals, dialogue MSMs and
committors. Absence of a match in this search will not be presented as proof of
priority. The lead-time study is a proposal until its held-out comparisons run.
