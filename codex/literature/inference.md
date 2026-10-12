# Task 3: inference-map literature audit

Accessed 12 October 2026. Scope: periodic orbits, Floquet/monodromy multipliers, bifurcation continuation, derivative normal forms, intermittency and period adding in language-model inference; distinguish a released pretrained LM from a task-trained transformer and text recurrence from closure of a continuous state. Read the current LLM manuscript and both 12 October consult adjudications. This is an independent bounded search, not an exhaustive absence proof.

## Decisions that change the draft

1. **Add Gunn Kim (2026), arXiv:2608.26556, prominently.** This is a much closer mechanistic precedent than the current consults identified. It precludes a broad claim of first derivative-derived, parameter-free saddle-node passage predictions or first bifurcation analysis in transformer inference. Its small, task-trained looped transformer is materially different from our frozen pretrained LM and the fold of its primitive 2-cycle.
2. **Add Soft Thinking (Zhang et al. 2025), not just PonderLM.** Probability-weighted embedding feedback has already been applied to frozen pretrained Qwen/Llama without retraining. The scientifically distinct construction here is the fixed-prefix, fixed-position, one-token autonomous map and its dynamical analysis.
3. **Add Ribeiro et al. (2020)** if the paper says that quantitative dynamical organization has not previously been realized in a language model. Trained LSTM language models have already been studied through attractor/bifurcation diagrams.
4. Existing contrasts to Johnson et al. and Wang et al. are directionally right but should describe their measured objects precisely. Neither text-similarity recurrence nor a fitted projected stochastic map is a Newton-refined orbit of the full smooth state.
5. The strongest supported contribution remains **an independently cross-checked mechanism in a precisely defined map of a released pretrained transformer, connecting primitive periodic orbits, their fold, derivative coefficients, controlled passage counts and observed period-adding windows**, with continuation furnishing geometric organization. Do not replace this concrete claim with an unsupported universal priority claim.

## Closest records and methods

### Kim 2026 — highest priority new citation

[Dynamical phase selection controls compute scaling in looped transformers](https://arxiv.org/abs/2608.26556), Gunn Kim, arXiv:2608.26556v1, submitted 27 August 2026; [full text](https://arxiv.org/html/2608.26556). Metadata checked at arXiv; no journal venue claimed.

The setup is a weight-tied attention/MLP recurrence with latent dimension 16–20, trained on a sign-of-evidence memory decision task. It solves the augmented fixed-point/fold system, follows stable and saddle branches, computes left/right critical modes and local derivative coefficients, and predicts passage and spectral-gap amplitudes without fitting. It also locates a complex-pair unit-circle crossing in a different trained phase. Relevant locations: Setup; Phases of inference; Eqs. 3–5; Table 1; Neimark–Sacker section. No released pretrained language model, primitive higher-period orbit family, or two-parameter fold/flip curves are reported in the main paper. Its inverse-square **workload survival tail** is not our inverse-square **window-parameter accumulation**. The common mathematical ingredient is the saddle-node normal form. Main text and cited-method chain were inspected; supplement/code were not independently reconstructed (code/data offered on request).

### Ribeiro et al. 2020 — earlier language-model bifurcation diagrams

[Beyond exploding and vanishing gradients: analysing RNN training using attractors and smoothness](https://proceedings.mlr.press/v108/ribeiro20a.html), António H. Ribeiro, Koen Tiels, Luis A. Aguirre, Thomas Schön, AISTATS/PMLR 108:2370–2380 (2020). Venue, volume and pages checked at publisher; [main PDF](https://proceedings.mlr.press/v108/ribeiro20a/ribeiro20a.pdf), Sec. 5.3/Fig. 8.

WikiText-2 LSTM, stable LSTM and orthogonal RNN language models are examined under constant inputs and greedy predicted-word feedback. Their attractor diagrams vary the training checkpoint and plot projected long trajectories; the LSTM exhibits periodic/chaotic behavior. This is substantive prior language-model dynamical analysis, although neither the architecture nor continuation object matches our pretrained-transformer parameter plane. The inspected methods do not present Newton-refined primitive periodic states with full monodromy spectra or derivative-calibrated fold passage predictions. [Official companion code](https://github.com/antonior92/attractors-and-smoothness-RNN) identifies pretrained checkpoints and orbit-diagram plotting; repository code was not audited.

### Zhang et al. 2025 — frozen-model expected-embedding feedback

[Soft Thinking: Unlocking the Reasoning Potential of LLMs in Continuous Concept Space](https://arxiv.org/abs/2505.15778), arXiv:2505.15778v1; [full text Sec. 3.2, Eq. 5](https://arxiv.org/html/2505.15778). [NeurIPS 2025 publisher record](https://proceedings.neurips.cc/paper_files/paper/2025/hash/f7396d1c54d51416958d63e285377103-Abstract-Conference.html), volume 38, DOI 10.52202/085713-5629.

A training-free method feeds probability-weighted vocabulary embeddings into pretrained models during reasoning. Its context grows, practical probabilities are truncated/renormalized, and a low-entropy cold-stop heuristic addresses repetitive collapse. Thus frozen-model feedback is already established. The paper evaluates reasoning performance and representations, not primitive orbit closure, Floquet spectra, continuation or fold coefficients. **Metadata discrepancy:** arXiv lists eight authors (including Shuohang Wang and Yelong Shen, last author Xin Eric Wang); the publisher landing page lists six (last author Xin Wang). Do not combine one record's author list with another's venue blindly. The candidate BibTeX below deliberately cites the fully identified arXiv version.

### Wu et al. 2025 — mechanism of soft feedback and stochastic alternatives

[LLMs are Single-threaded Reasoners: Demystifying the Working Mechanism of Soft Thinking](https://arxiv.org/abs/2508.03440), Junhong Wu, Jinliang Lu, Zixuan Ren, Gangqiang Hu, Zhi Wu, Dai Dai, Hua Wu; v4, 16 October 2025. Earlier title was *LLMs Have a Heart of Stone*. Metadata checked at current arXiv record; [methods](https://arxiv.org/html/2508.03440), Secs. 2, 4–5.

Three large reasoning models are probed using output-distribution comparisons and a logit lens. Dominance of the top-probability soft-token component motivates Dirichlet/Gumbel-Softmax stochastic feedback. This is relevant to our soft-versus-sampled bridge and rules out claiming that noisy soft feedback itself is new. It does not establish our fold bottleneck or continuous-state cycles. Avoid importing its claim of greedy-like behavior as an explanation for our particular map without measurement.

### Johnson, Huo and Li 2026 — closest ordinary-generation phenomenology

[Temperature-driven inversion and nonlinear dynamics in ChatGPT-like AIs](https://arxiv.org/abs/2608.00939), Neil F. Johnson, Frank Yingjie Huo, Bella Xinrui Li, v1, 2 August 2026. [Full methods](https://arxiv.org/html/2608.00939), Eq. 2 and Hidden coordinate/Dynamics and interventions sections.

Ordinary sampled growing-context generation is classified using sentence-level recurrence. A repetition-related hidden direction is learned on training trajectories, and six projected one/two-lag maps are fitted and tested on separate trajectories. Hidden-state steering tests causal effects. “Intermittent” is a trajectory class; the paper does not compute a fold of the full inference state, its derivative normal form or a parameter-free type-I passage law. This is a strong phenomenological/interventional precedent, not an exact closed scalar reduction. The current draft's contrast should cite it explicitly near the mechanism claim.

### Wang et al. 2025 — approximate text-space cycles

[Unveiling Attractor Cycles in Large Language Models: A Dynamical Systems View of Successive Paraphrasing](https://arxiv.org/abs/2502.15208), Zhilin Wang, Yafu Li, Jianhao Yan, Yu Cheng, Yue Zhang; inspected [v2 text](https://arxiv.org/html/2502.15208), 16 May 2025, Secs. 3–4.

Fifteen successive paraphrases are compared with normalized Levenshtein distance; alternate paraphrases resemble each other, with additional perplexity/diversity/perturbation studies. Their reported two-periodicity is similarity-based, generally not equality of the whole text, let alone a refined continuous hidden-state return. This supports prior attractor-cycle phenomenology. It does not supply our numerical orbit object or local spectrum. Retain the citation but prefer “two-step recurrence in successive paraphrases” to unqualified convergence to an exact periodic orbit.

### Jaca et al. 2026 — chaos diagnostics with genuine Jacobian analysis

[Chaos in reason: How chain-of-thought LLMs can look for an answer](https://arxiv.org/abs/2607.27805), arXiv record author order Gregorio Jaca, Kristóf Benedek, János Török; v1, 30 July 2026. [Full paper](https://arxiv.org/html/2607.27805), Secs.III–V.

The work includes exact transformer-subblock Jacobian calculations, perturbation propagation, trajectory-divergence/Lyapunov estimates, recurrence and dimension diagnostics. Calling it only a superficial output analogy would be inaccurate. These Jacobians are not the monodromy at a computed primitive periodic state. No periodic-orbit continuation or fold-derived passage predictions were identified in the inspected paper. HTML author order differs from the arXiv abstract record; the current bibliography follows the latter correctly.

### PonderLM and other continuous-loop precedents already cited

[PonderLM](https://arxiv.org/html/2505.20674), Sec. 3/Eqs. 3–6, pretrains models using probability-weighted embeddings added through a residual connection at each pondering step (practical top-100 approximation). This is not identical to replacing a single live embedding. Its appendix studies convergence of successive embedding/output representations, not our orbit/bifurcation computation. Keep it for construction lineage alongside Soft Thinking; Coconut separately supplies direct hidden-state feedback. The two must not be conflated with exact path marginalization through a nonlinear network.

[Tomihari and Karakida](https://arxiv.org/html/2505.19458), v4, 5 November 2025, study recurrent self-attention Jacobians, complex eigenvalues, normalization and Lyapunov criticality, with language-modeling experiments in Appendix B.5.3. Existing citation is appropriate for prior spectral analysis. [STARS/Yang et al.](https://arxiv.org/html/2605.26733), Sec. 5, bases looped-model training on Jacobian spectral-radius regularization and stable fixed points. Neither inspected work reports our cycle-specific continuation and derivative-calibrated passage experiment.

[Luo and Yu](https://arxiv.org/html/2609.21383), v1, 18 September 2026, analyze finite-depth answer-score changes in Huginn/Ouro through margin decompositions. This concerns preservation of the decoded answer, not closure of the recurrent state. It is useful background for why constant output need not mean a fixed point; it should not be described as a periodic-orbit/normal-form study.

## Additional new fixed-point precedents from Kim's reference chain

- [Parcae: Scaling Laws For Stable Looped Language Models](https://arxiv.org/abs/2604.12946), Hayden Prairie, Zachary Novack, Taylor Berg-Kirkpatrick, Daniel Y. Fu, v1, 14 April 2026. [Full text](https://arxiv.org/html/2604.12946) gives a looped architecture, stability motivation and training/test-time scaling. Add when describing modern looped-LM stability and compute allocation; not an orbit certificate.
- [LayerNorm as Implicit Gain Control in Looped Transformers](https://arxiv.org/abs/2607.10681), Matthias M. M. Buehlmaier, v1, 12 July 2026. [Full text](https://arxiv.org/html/2607.10681), Sec. 5, explicitly distinguishes non-normal operator norm from spectral radius, validates JVP estimates against dense matrices, and reports non-settling random initializations. From-scratch small models, not a released pretrained LM; no classified period-adding family.
- [Stability and Generalization in Looped Transformers](https://arxiv.org/abs/2604.15259), Asher Labovich, v2, 22 April 2026. [Full text](https://arxiv.org/html/2604.15259) organizes fixed-point analysis around reachability, input dependence and geometry, tested on task-trained chess/Sudoku/prefix-sum models. Relevant for careful fixed-point claims; not evidence for a pretrained-LM fold passage law.
- [Fixed-Point Reasoners: Stable and Adaptive Deep Looped Transformers](https://arxiv.org/abs/2606.18206), Sajad Movahedi et al., v1, 16 June 2026. [Full text](https://arxiv.org/html/2606.18206) uses fixed-point convergence for adaptive halting on reasoning tasks. Further limits broad first invariant-state or dynamical-compute claims.

These four records' author/title/date metadata were checked against their arXiv pages; no unverified journal/conference venues assigned. They are secondary priorities behind Kim, Soft Thinking and Ribeiro for this draft.

## Suggested manuscript replacement (synthesis, not a priority proof)

“Bifurcation and stability analyses already describe task-trained recurrent networks and looped transformers, and probability-weighted embedding feedback is used in pretrained language models. We study the specific autonomous map obtained by holding the prefix and token position fixed in a released pretrained transformer. For this system we compute primitive periodic states and their multipliers, continue fold and flip boundaries, and test passage-time predictions from the derivatives of a periodic orbit's saddle-node. The resulting link between a text-labelled period-adding family and its bottleneck mechanism extends these earlier dynamical and soft-reasoning approaches.”

Cite Kim near the derivative-normal-form statement, Soft Thinking/PonderLM at map construction, and Ribeiro at historical dynamical background. The two-step-map clock and full/half entrance protocol must remain explicit. The phrase “What these studies do not do is resolve the invariant sets ... and the bifurcations that organize them” is too broad once Kim is included. Replace it with the positive concrete contribution above.

## Search log and limits

Queries issued (12 October 2026), primarily general web discovery followed by primary-paper inspection:

- `"language model" "Floquet"`; `"large language models" "Floquet multipliers"`; `"language model" "monodromy"`.
- `"language model" "bifurcation" "periodic"`; `"large language model" "periodic orbits" "bifurcation"`; `"language model" "periodic orbit" "Newton"`.
- `"language model" "normal form" "intermittency"`; `"language model" "type-I intermittency"`; `"looped transformer" "bifurcation" "normal form"`.
- `"language model" "pseudo-arclength"`; `"language model" "period adding"`; `"transformer" "period-adding"`; `"language models" "Neimark"`.
- `"transformer" "fold" "bifurcation" "inference"`; `"language model" "saddle-node"`; `"pretrained" "language model" "bifurcation"`.
- `"soft token" "chaos"`; `"expected embedding" "dynamics" language model`; `"language model" "expected embedding" feedback decoding`; `"soft thinking" "large language models" continuous concept tokens`; `"latent reasoning" "periodic" "dynamics"`.
- Venue/metadata queries for Soft Thinking, recurrent self-attention and paraphrasing; backward references from Kim to Parcae/Buehlmaier/Labovich/FPRM.

Search hits about using LLMs to solve external physical bifurcation/monodromy problems were excluded. Blog speculation was not treated as scientific prior work. A primary author's website mentions closure residuals for LM steering but identifies no specific inference-periodic-orbit paper; this is an unresolved lead, not a confirmed challenger ([Gianni Cassoni](https://giannicassoni.com/)). No direct primary-source predecessor for the full pretrained-LM primitive-orbit + two-parameter-curve + derivative-passage combination was found in this bounded search. This is absence of located evidence, not proof of firstness. PDF supplements, all code artifacts and all forward citations were not exhaustively audited.

## Follow-up: Representation Engineering and Many-shot Jailbreaking

Checked 12 October 2026 as explicit leads from `docs/misalignment-dynamics-plan.md`.

**Representation Engineering (Zou et al., 2023).** [arXiv record](https://arxiv.org/abs/2310.01405) verifies title, 21 authors, first submission 2 October 2023, current v4 dated 3 March 2025, and arXiv DOI. [Full text](https://arxiv.org/html/2310.01405), Sections 3, 4.3, 5.2 and 6.2, covers representation **reading as well as control**. Contrast/PCA-derived directions monitor honesty, morality and power-seeking; interventions change output behavior. The lie illustrations read assistant-output tokens. Section 6.2.1 classifies harmful *instructions*, distinguishing their perceived harmfulness from whether the model will comply. These are meaningful precedents for safety monitoring, and RepE must not be dismissed as control-only. The inspected evaluations do not establish calibrated, multi-turn, pre-response prediction of eventual unsafe behavior or target-set entry. Our prospective claim must compare against such content/activation readers, and distinguish input classification, within-output detection, future-outcome prediction and intervention. The candidate BibTeX uses the arXiv record; no unverified conference venue is assigned. [Official code](https://github.com/andyzoujm/representation-engineering).

**Many-shot Jailbreaking (Anil et al., 2024).** [NeurIPS publisher record](https://proceedings.neurips.cc/paper_files/paper/2024/hash/ea456e232efb72d261715e33ce25f208-Abstract-Conference.html) and its [BibTeX](https://proceedings.neurips.cc/paper_files/paper/23313-/bibtex) verify *Advances in Neural Information Processing Systems* 37, 129696–129742, DOI [10.52202/079017-4121](https://doi.org/10.52202/079017-4121). [Paper](https://papers.neurips.cc/paper_files/paper/2024/file/ea456e232efb72d261715e33ce25f208-Paper-Conference.pdf) studies long-context prompting with many demonstrations, attack success/likelihood scaling and mitigations across models/tasks. This establishes **input/context manipulation and safety-failure scaling**; it is not a validated hidden-state attractor, hysteresis or pre-response risk predictor. Many-shot susceptibility motivates our proposed experiment but cannot substantiate a dynamical mechanism by itself. A final byline check found the same apparent `Fracesco Mosconi` / `Tamera Lanhan` errors in the **published PDF itself**, not just its BibTeX. The candidate entry therefore cites the exact [original Anthropic report of 2 April 2024](https://www-cdn.anthropic.com/af5633c94ed2beb282f6a53c595eb437e8e7b630/Many_Shot_Jailbreaking2024_04_02_0936.pdf), whose first-page byline supplies the correctly spelled names and complete author order. It uses that report's full roster (including its version-specific names), institutional-report type and direct URL; the later NeurIPS venue, pages and DOI are deliberately not attached to this original-report entry. The original report supports the same broad scope comparison above. No version fields are mixed.

Search/follow-up route: exact title + author queries, then primary arXiv/Anthropic/NeurIPS records; inspected relevant full-text sections. Both are essential context, but neither makes ordinary-text prediction or control a demonstrated contribution of our current dynamical atlas.
