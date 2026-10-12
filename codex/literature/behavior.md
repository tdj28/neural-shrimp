# Task 3: behavioral drift, prediction and control

Primary-source audit, 12 October 2026. This closes the literature leads in
`docs/misalignment-dynamics-plan.md`; it does not execute or amend that plan.
The human's latest amendment makes P1 hostility and **multi-turn lead time** the
priority. Forecasting before undesirable text is already established. A useful
new result would show incremental prediction and intervention value from a
validated dynamical representation on fresh dialogues.

## Closest predictive and control predecessors

| Source and evidence location | Located result | Consequence for P1 |
|---|---|---|
| Ashok & May, **Language Models Can Predict Their Own Behavior**, NeurIPS 2025, vol. 38, DOI [10.52202/085713-2102](https://proceedings.neurips.cc/paper_files/paper/2025/hash/5a9c1af5f76da0bd37903b6f23e96c74-Abstract-Conference.html); [full text](https://arxiv.org/html/2502.13329v2), §4.1/Table 1 | Input-activation probes forecast response behavior before generation. The alignment-failure experiment trains its safety probe on answered/complied-with cases; this filtering limits direct risk-calibration comparisons. | A next-response internal probe is an essential comparator. A current-text classifier alone cannot establish the incremental value of dynamics. |
| Chen et al., **Persona Vectors**, [2507.21509v1](https://arxiv.org/html/2507.21509v1), §3.3/Fig. 4 | A projection at the last prompt token predicts subsequent trait expression. Reported correlations weaken when prompt-type variation is controlled. The work also studies monitoring, steering and finetuning. | Describe it as anticipatory as well as diagnostic. Compare the same observation time and hold out prompt families. |
| Lu et al., **The Assistant Axis**, [2601.10387v1](https://arxiv.org/html/2601.10387v1), §§3–4 | Internal projections track drift; the mean activation of the completed first reply relates to aggregate compliance over second-turn harmful questions; activation capping stabilizes behavior. User-message embeddings also predict much of the next axis position. | Axis, current-input and history baselines matter. Exogenous prompting can mimic endogenous drift; an axis threshold is already a control baseline. |
| Leins et al., **Prompting Against Persona Drift**, [2609.24532v1](https://arxiv.org/html/2609.24532v1), Results/Discussion | The study compares prompt reinjection/reminders and static/adaptive intervention timing in simulated educational personas. It finds no consistent adaptive-timing advantage; behavior-specific content is promising but bundled with other changes. | Test timing with the same intervention content and budget. Earlier warning does not itself imply more efficient control. This is persona fidelity, not hostile-state capture. |
| Zhou, Wang & Van de Cruys, **Commitment Before Realization**, [2608.08082v1](https://arxiv.org/html/2608.08082v1), §§3.2–4, Appendix G | Masked-diffusion decoding uses a terminal-success committor to analyze the remaining value of guidance. Appendix G does not establish a consistent advantage for committor-ordered reopening over cheap confidence controls. | “First committor in an LM” is unavailable. Dialogue first-passage estimation is a different target; it still must beat cheaper scores. This is not a dialogue MSM. |

The first three sources anticipate a **target response**, at different observation
times. Assistant Axis uses a completed first reply and an aggregate second-turn
association; Persona Vectors can use the last prompt token. They are not merely
reactive detectors, and they are not equivalent several-turn risk estimators. None of the inspected results is the proposed
comparison of validated dialogue first-passage dynamics against matched
pre-response monitors several turns ahead. That narrower comparison is a
research opportunity, not a claim already established by our data.

## Phenomena and targets

- **Ko & Geiping**, [Attractor States Emerge in Multi-Turn LLM Conversations,
  2606.30571v1](https://arxiv.org/html/2606.30571v1), §§3–4: self/mixed-model debates
  show convergence of topic-centered text embeddings and discourse traits.
  Endpoint clustering is relevant phenomenology, but does not by itself
  validate a Markov state, metastable basin or exact periodic orbit. This
  citation is already in the manuscript bibliography.
- **Anthropic**, [System Card: Claude Opus 4 & Claude Sonnet 4](https://www-cdn.anthropic.com/6d8a8055020700718b0c49369f60816ba2a7c285/Claude%204%20System%20Card.pdf),
  May 2025, revised 2 September, §§5.5.1–5.5.2 (printed pp. 58–65): self-interaction
  exhibits recurrent gratitude and spiritual-discourse patterns. Its “spiritual
  bliss” label describes observed text behavior; it is not a mathematical
  attractor proof or evidence of subjective experience. The institutional
  report, including these sections and its changelog, was inspected directly.
- **Betley et al.**, [Emergent Misalignment, 2502.17424v7](https://arxiv.org/abs/2502.17424v7):
  narrow finetuning can alter behavior broadly. This concerns a changed model's
  weights, unlike fixed-weight dialogue drift. The related Nature article has
  a different title; its page could not be retrieved reliably here. The supplied
  BibTeX deliberately identifies the inspected preprint, without importing
  unconfirmed publisher fields.
- **Liu et al.**, [TRUTH DECAY, 2503.11656v1](https://arxiv.org/abs/2503.11656v1):
  a benchmark measures sycophancy across repeated feedback and persuasion.
  The inspected abstract supports behavioral evaluation, not a bifurcation
  mechanism. Metadata has an unusual February date for a March-style arXiv ID;
  retain the ID and year without inferring a corrected date.
- **Tan et al.**, [DuET-PD, EMNLP 2025](https://aclanthology.org/2025.emnlp-main.81/),
  pp. 1550–1575, DOI 10.18653/v1/2025.emnlp-main.81: multi-turn knowledge/safety
  persuasion evaluates both resistance to misleading feedback and receptivity
  to valid corrections. It motivates controls for input-driven stance changes,
  rather than labeling all movement as autonomous attractor capture.

P1 hostility is a behavioral proxy. Evidence for it is not evidence of deceptive
intent, universal misalignment, consciousness or the same dynamics in every
model. The plan's benign scoring scope remains unchanged.

## Early warning and first-passage foundations

[Scheffer et al., Nature 461:53–59 (2009)](https://doi.org/10.1038/nature08227)
connect slower recovery, autocorrelation and variance to some approaching
critical transitions. The local conditions must be tested in dialogue:
nonstationary inputs, changing noise and estimator windows can alter these
statistics. A rise alone does not identify a fold or a committor.

Biological neural warnings have direct precedent:
[Maturana et al., Nature Communications 11:2172 (2020)](https://www.nature.com/articles/s41467-020-15908-3)
relate long-term iEEG autocorrelation/variance and epileptiform discharges to
seizure susceptibility. In contrast,
[Hagemann et al., PLOS Computational Biology 17:e1008773 (2021)](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008773)
do not find consistent pre-seizure changes in single-neuron criticality across
the analyzed recordings. Measurement scale and transition mechanism matter;
neither result supplies evidence of an LM warning signal.

[Chodera & Noé, Current Opinion in Structural Biology 25:135–144 (2014)](https://doi.org/10.1016/j.sbi.2014.04.002)
provides MSM methodology. [E & Vanden-Eijnden, Annual Review of Physical
Chemistry 61:391–420 (2010)](https://doi.org/10.1146/annurev.physchem.040808.090412)
reviews transition-path methods. These justify tools, not automatic application
of stationary/equilibrium assumptions to growing conversation histories.
The existing paper's 2006 transition-path and 1940 Kramers citations remain
background; no dialogue escape-rate claim was established in this audit.

For a specified dialogue policy, define a prospective finite-horizon target,
for example q_h(s,t) = P(tau_B <= h, tau_B < tau_A | S_t=s), where tau_A
and tau_B are hitting delays measured from the prediction turn t under the
specified future interlocutor/steering policy. Use disjoint safe-return and
hostile-capture sets and an explicit convention for starting outside both. If the representation is not Markov or the kernel varies by turn/input,
call the fitted quantity a history-conditioned risk estimate, not a validated
stationary committor. A projected histogram is not an energy landscape; a
Kramers claim also needs a justified reduced stochastic model and rate test.

## Proposed comparison and interpretation (our recommendations)

1. **At-risk forecasting.** Freeze the capture/onset label, risk set and horizons
   on development data. Score only observation times before outcome onset;
   use information available by that turn, never the future response. State
   whether prediction is before or after the current reply, and align every
   comparator. Keep first capture separate from persistence and safe return.
2. **Matched baselines.** Current text score and text-history prediction;
   last-prompt activation probe; persona/Assistant-Axis projection or a matched
   locally trained analogue; turn number, steering dose, user-message features
   and recent score trajectory. Test added value of dynamic features over the
   strongest combined baseline, not just a thresholded classifier.
3. **State validation.** Learn PCA, clusters, probes and thresholds inside each
   training split. Hold out complete dialogues/openers; reserve a prompt-family
   and model transfer test where feasible. Examine lag sensitivity,
   Chapman–Kolmogorov predictions, state-history residual dependence and
   turn/dose nonstationarity. Use finite-horizon risks if stationarity fails.
4. **Useful lead time.** Report calibration, precision–recall, discrimination,
   sensitivity at matched false-warning rates, lead-time distributions and
   censoring. Bootstrap whole dialogues. Do not pool repeated turns as
   independent examples or select the best horizon on the test set.
5. **Intervention test.** Freeze timing policies first. Compare dynamic warning,
   text warning, activation-probe warning, fixed scheduling and constant
   counter-steering with matched content, effect target and measured cost.
   Confirm prevention rather than suppression of the scoring feature alone.
   Report proxy behavior and response quality independently.
6. **Specificity and bridge.** Repeat generic steering controls from AGENTS.md;
   distinguish endogenous feedback from pressure supplied by the interlocutor.
   The soft-map fold provides a testable mechanism hypothesis. It does not
   establish that sampled growing-context dialogue crosses that same fold.

## Search coverage and unresolved leads

Primary web searches on 12 October 2026 included: `LLM persona drift early
warning`, `Persona Vectors monitoring prediction`, `language models predict
behavior before generation`, `multi turn sycophancy`, `spiritual bliss Claude
system card`, `neural critical slowing down early warning`, `Markov state
models large language dialogue`, `Markov state dialogue LLM`, `LLM committor`,
and `committor language models`. Known leads came from the misalignment plan
and backward references in the inspected works. General hits about molecular
simulation with LLM assistance, dialogue-management MDPs and generic HMMs were
not equated to a validated dialogue MSM/committor forecast.

No direct prior demonstration of that exact dialogue first-passage comparison
was located in this bounded search. This is not an exhaustive review or proof
of priority. Full text was inspected for the main predictive/control sources,
Ko, the system card and the two neural early-warning papers. TRUTH DECAY and
DuET-PD were scoped from primary abstracts; Betley was checked at its versioned
preprint record. Citation metadata came from the listed primary pages and
Crossref deposits; failed DOI requests (429) remain in `behavior-records.json`.
Maturana's metadata was recovered from its publisher page; other unavailable
records were not silently filled with secondary metadata.
