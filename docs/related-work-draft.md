# LLM dynamics: related-work draft and novelty audit

Task 3, completed 12 October 2026. Claude owns integration into `paper/llm/`.
The accompanying `paper/llm-references.bib` is an audited supplement, not a
replacement for the manuscript's existing bibliography. Merge by citation key;
some rechecked existing references occur in both. Source-level evidence,
metadata qualifications and search logs are in `codex/literature/`.

## Positioning decision

The strongest contribution is the **quantitative connection between primitive
periodic states, their parameter-plane boundaries and derivative-predicted
passage in a specified feedback map of a released pretrained language model**.
Several ingredients already exist separately, including close transformer
examples. State the actual computed objects and prediction errors. Avoid a
broad first-inference-dynamics or first-transformer-bifurcation claim.

The bounded search found no primary paper matching the complete combination
above. It does not prove priority. Equally, it does not diminish the value of
an independently reproduced result in a language-pretrained model. The map's
fixed prefix, fixed position and soft feedback must remain explicit: ordinary
sampled growing-context generation is a separate system whose connection
requires a measured bridge.

## Closest comparisons and required corrections

| Primary source | What it already establishes or reports | Distinction to preserve |
|---|---|---|
| [Kim 2026](https://arxiv.org/html/2608.26556v1), `kim2026phase` | Derivative fold coefficients and unfitted passage/spectral-gap predictions in a small task-trained looped transformer; a complex-pair crossing in another phase. | Our released pretrained LM and fold of a primitive periodic state. Its workload tail is not our window-parameter accumulation law. |
| [Soft Thinking](https://arxiv.org/html/2505.15778), `zhang2025softthinking` | Training-free probability-weighted embedding feedback in pretrained LMs. | Our autonomous fixed-prefix/position map and dynamical analysis. Frozen-model feedback itself is established. |
| [Ribeiro et al. 2020](https://proceedings.mlr.press/v108/ribeiro20a.html), `ribeiro2020attractors` | Language-model RNN attractor/bifurcation diagrams. | Full-state primitive-orbit computations and continued curves in our pretrained transformer. |
| [Liu et al. 2026](https://arxiv.org/html/2609.38814v1), `liu2026learningchaos` | AD cycle/flip calculations and a finite cascade through period 128, ratio 4.6687, in a transformer scientific surrogate. | Its learned scalar dynamical-system map versus our full live-embedding state in a language-pretrained model. |
| [Dinc et al. 2026](https://doi.org/10.1103/mjcl-lb4x), `dinc2026ghost` | Neural saddle-node ghost delays in task-trained RNNs. | Continuous neural state and learning task versus discrete pretrained-LM periodic-orbit passage. |
| [Ashok & May 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/5a9c1af5f76da0bd37903b6f23e96c74-Abstract-Conference.html), `ashok2025behavior` | Forecasting eventual response behavior before generation from input activations. | P1 must test added multi-turn predictive value, not claim the first preemptive monitor. |
| [Persona Vectors](https://arxiv.org/html/2507.21509v1), `chen2025persona`; [Assistant Axis](https://arxiv.org/html/2601.10387v1), `lu2026assistant` | Anticipatory monitoring and steering/capping; Assistant Axis uses completed first-reply activations to assess second-turn compliance. | Compare matched internal monitors and input/history controls. These methods are not merely concurrent detectors. |
| [Zhou et al. 2026](https://arxiv.org/html/2608.08082v1), `zhou2026commitment` | A committor analyzes terminal constraint success and guidance in masked-diffusion decoding. | Dialogue safe-return/capture is different, but committor use in language models is established. |

The neural-systems scope is also well populated: measured Feigenbaum ratios
([Battaglia et al.](https://doi.org/10.1103/PhysRevLett.99.238106),
[Papasavvas et al.](https://doi.org/10.1103/PhysRevE.92.032723)); neural normal
forms/continuation ([Marichal & Piñeiro](https://doi.org/10.1016/j.neucom.2015.04.001));
and neuronal shrimp phase diagrams/periodicity staircases
([Gomez et al.](https://doi.org/10.1093/bioinformatics/btu332),
[Stenzinger et al.](https://doi.org/10.1063/5.0239268),
[Caixeta et al.](https://doi.org/10.1063/5.0250342)). Clark's
[2610.12426](https://arxiv.org/abs/2610.12426v1) concerns a large-N random-RNN
Lyapunov spectrum, a different object from a finite periodic-orbit monodromy.

## Suggested manuscript prose

The following paragraphs are editable synthesis. Their factual source
comparisons are checked; any sentence about *our* completed results must remain
matched to the live results ledger and the paper's evidence gates.

**Inference and the map.** Continuous feedback has been studied in latent
reasoning and expected-embedding methods. Soft Thinking applies the latter
without retraining a pretrained model, while later analysis probes the
relationship between soft and greedy feedback (`zhang2025softthinking`,
`wu2025singlethreaded`). Bifurcation diagrams have also been studied in LSTM
language models (`ribeiro2020attractors`). Kim derives fold-based passage
predictions in small task-trained looped transformers (`kim2026phase`). Our
study defines a different, autonomous map by fixing the prefix and live-token
position in a released pretrained transformer. We connect its primitive
periodic states and local spectra to continued boundaries and test passage
predictions computed from the derivatives of a periodic orbit's fold.

**Neural nonlinear dynamics.** Period doubling, saddle-node bottlenecks and
organized periodic windows have a long neural-systems history, including
quantitative ghost passage in task-trained RNNs (`dinc2026ghost`). Neural rate
and mass models have quantified finite-order Feigenbaum scaling
(`battaglia2007temporal`, `papasavvas2015gain`), and two-neuron recurrent maps
have been analyzed with normal forms and continuation (`marichal2015multiple`).
Shrimp phase diagrams and periodicity staircases also occur in neuronal models
(`gomez2014universal`, `caixeta2025staircase`). More recently, a transformer
trained on dynamical-system trajectories recovered a numerical flip ladder
(`liu2026learningchaos`). These precedents motivate our methods; the application
here concerns feedback dynamics of a language-pretrained model, with the
computed orbit and its full live-embedding-state derivative as the object of
analysis.

**Behavioral drift and prediction.** Conversational attractor-like patterns
and persona drift have been observed in LLM interactions (`ko2026attractor`,
`anthropic2025claude4`, `lu2026assistant`). Activation-based methods already
anticipate response behavior (`ashok2025behavior`, `chen2025persona`). Our
proposed lead-time experiment therefore asks whether a dynamical state estimate
adds calibrated several-turn warning over matched text, activation and history
baselines, and whether the added warning improves intervention timing at a
matched cost. Critical slowing down and transition-path methods provide
candidate estimators (`scheffer2009early`, `e2010transition`); their validity
for a growing conversation must be tested. This is an experimental question,
not a consequence of the deterministic soft-map result.

Retain the manuscript's Johnson, Wang and Jaca citations. Their precise scope
is detailed in `codex/literature/inference.md`: fitted projected generation
maps, similarity-based paraphrase recurrence, and genuine Jacobian/trajectory
chaos diagnostics respectively. Do not describe all three as superficial
analogies, or paraphrase recurrence as an exactly closed full-state orbit.
Also retain the standard numerical-continuation and intermittency references;
new application does not mean a new mathematical method.

## P1: what would make the lead-time result useful and distinct

The current plan compares a dynamical warning against the current-turn text
classifier. Strengthen that comparison before a headline confirmatory test:

1. **Forecast the same target with the same information.** Freeze hostile
   onset/capture, safe return, eligible pre-onset times and several-turn
   horizons on development data. Specify the future interlocutor and steering
   policy on which the risk is conditioned. State whether a forecast sees the
   current reply. The baseline and dynamical monitor must see the same temporal prefix.
2. **Include strong baselines.** A pre-response activation probe, persona-axis
   score, text-history predictor, turn/dose/current-user-message features and
   recent score changes. Compare dynamics against their strongest combination,
   not only a reactive classifier. The first three predictive papers in the
   table make this necessary for an incremental-utility claim.
3. **Validate the state model.** Fit projections, clusters and thresholds only
   inside training splits; hold out whole dialogues and prompt families.
   Check lag dependence, Chapman–Kolmogorov predictions and residual dependence
   on history, turn and dose. If the coarse state is non-Markov or
   nonstationary, estimate explicit finite-horizon, history-conditioned risk.
4. **Measure usable warning.** Calibration, precision–recall, sensitivity at a
   fixed false-warning rate, lead-time distributions and censoring complement
   horizon-specific ROC. Bootstrap dialogues, not pooled turns.
5. **Separate timing from intervention content.** Compare warning-triggered,
   regularly scheduled and constant counter-steering with matched content and
   cost. Leins et al. found no consistent advantage of adaptive timing in
   their persona setting (`leins2026drift`). A better prediction score alone
   does not demonstrate better control.
6. **Test the proposed mechanism separately.** A committor is a hitting
   probability under specified dynamics, not any latent score. A histogram
   is not an effective potential; a Kramers rate requires a justified stochastic
   reduction. CSD proxies are mechanism-dependent, as biological neural work
   illustrates (`maturana2020slowing`, `hagemann2021criticality`). Preserve the
   specificity controls and independently scored behavior/quality outcomes.

Representation Engineering supplies earlier representation reading and control
(`zou2023representation`); Many-shot Jailbreaking supplies a clear example of
input-context pressure (`anil2024manyshot`). Neither inspected source establishes
the proposed several-turn first-passage forecast. The source audit records
Ashok's filtered safety-probe design and the differing observation times of the
monitoring methods; reproduce the intended baselines on the same P1 risk set.

These are design recommendations for Claude's plan, not an amended or run
confirmatory protocol. No model experiments or protected holdouts were touched.

## Live-manuscript accuracy flags

- Replace the introduction's broad exclusion of previous invariant-set and
  bifurcation analysis with the positive, scoped contribution above. Kim is
  the key correction; Soft Thinking corrects the frozen-feedback comparison.
- Replace **“complete period-doubling cascade”** with the actual resolved
  finite ladder. The inspected manuscript table still contains placeholders;
  the saved-results audit found three completed flips and one early spacing
  ratio, approximately 2.844. That neither establishes nor refutes an eventual
  Feigenbaum limit. See the dated, hash-bound saved-output check in
  `codex/literature/cascades.md`; it is not an independent numerical replay.
- State finite-order ratios with uncertainty and the actual parameter
  coordinate. Finite ratios can change under nonlinear reparameterization.
- Distinguish Kim's one-sided passage clock from our full-channel passage
  convention and the two-step map. Their shared fold mechanism does not
  identify the window-law prefactor or global reinjection automatically.
- Use “numerically validated” or “independently cross-checked” for our
  floating-point orbit/derivative results. A mathematical certificate requires
  the separate interval/ball-arithmetic proof gate.
- Keep predicted P1 benefit and planned replications visibly prospective.
  Emergent misalignment after finetuning (`betley2025emergent`) is a useful
  comparator, not evidence about fixed-weight conversation dynamics.

## Coverage, citation quality and integration

This was a targeted, reproducible search organized by the three assigned
questions, not an exhaustive database review. Search date/cutoff: 12 October
2026. Primary records and methods were prioritized; no external authors were
contacted, no paid review or compute was used. Queries and exclusions are in
`codex/literature/inference.md`, `cascades.md` and `behavior.md`. Abstract-only
or inaccessible full-text comparisons are explicitly marked there. No located
predecessor for the exact combined result is an absence-of-evidence finding.

Metadata was checked against arXiv, publisher pages or publisher Crossref
deposits. The supplement avoids invented venues/pages. Soft Thinking's arXiv
and NeurIPS author lists differ, so it cites the identified arXiv version.
Betley's Nature successor could not be retrieved reliably, so its entry cites
the inspected versioned preprint. Stenzinger's publisher author list takes
precedence over a conflicting secondary index. Failed DOI retrievals are
retained in `behavior-records.json`; successful publisher checks are described
in the notes. Use each primary version consistently when integrating.
