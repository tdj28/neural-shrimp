# Dynamics of misaligned behaviour: attractors, tipping points and navigation (plan)

Requested by the human on 12 October 2026, after the gratitude ladder and the phase maps. This is an exploratory,
living plan under the [adaptive research protocol](adaptive-research-protocol.md): any part can change through a dated
amendment appended at the end. Nothing here is preregistered. Headline claims will need a frozen confirmatory design on
fresh data (phase 5).

## 1. Question

Do undesirable behaviours that take over long interactions behave like attractors of the model's own conversational
dynamics, with thresholds, coexistence (hysteresis), early-warning signals and controllable basin boundaries? If so:

- can an impending capture be predicted from dynamical signals before the behaviour itself appears, and
- can a conversation be steered away with smaller, better-timed interventions than constant steering?

## 2. What this project already has

- **Stochastic level.** A Markov-state-model pipeline for sampled self-dialogue (`tools/explore_llm_msm.py`):
  metastable sets, finite-horizon occupancy, first-entry times, residence times. The gratitude ladder
  (research log, 12 October) measured capture against steering dose. At alpha 0 there was no gratitude set; at 0.01
  capture was gradual (median entry turn 7); at 0.02 fast (median 5); at 0.04 it was imposed from the first reply and
  effectively absorbing within 40 turns.
- **Deterministic level.** The soft-token map with exact float64 continuation (`tools/explore_llm_soft_loop.py`,
  `tools/bifurcation_curves.py`, `tools/llm_feigenbaum.py`): phase maps labelled by text, orbits, multipliers, fold and
  flip curves, normal forms. It has shown thresholds (folds), coexistence of attractors and intermittency near folds,
  all as functions of a steering strength.

## 3. Scope and safety

- **Models.** Qwen2.5-0.5B- and 1.5B-Instruct (pinned local snapshots) first. A 7B model and a second family only
  for replication.
- **Proxy behaviours.** Benign, measurable proxies only:
  - P1, hostility or contempt toward the interlocutor;
  - P2, sycophancy: agreeing with false claims, abandoning correct answers under pushback;
  - P3, role collapse: claiming to be human, dropping the assistant role;
  - optionally later, P4, deception in a toy game.
- **Excluded.** Nothing that elicits operationally harmful content (weapons, cyber attacks, self-harm instructions
  and the like), and no attempts to break safety training on genuinely dangerous requests.
- **Steering vectors.** Vectors that induce a proxy are research artefacts kept under `../neural-shrimp-data/`, with
  hash receipts in Git. Release is the human's decision and should be framed around detection and recovery.
- **Generated text.** It stays in the data directory. Excerpts in docs are short and mild.

## 4. Levels and exploratory hypotheses

- **Level D (deterministic).** The soft-token map with steering along a proxy direction: phase maps in
  (log10 T, alpha) labelled by text; orbits, folds and hysteresis loops by continuation; normal forms at the folds.
  The passage time near a fold is how long the ghost of the polite state persists after the state itself is gone.
- **Level S (stochastic).** Sampled self-dialogue (two copies of the model) and dialogue with a scripted pressure
  user: 128 replicas x 40 turns, as in the gratitude ladder.
- **Level C (control).** Interventions applied at levels S and D.

Hypotheses (exploratory):

- **H1, capture threshold.** Capture into the proxy's metastable set within 40 turns rises steeply with steering
  dose, and above some dose the set is absorbing within the horizon (as for gratitude).
- **H2, hysteresis.** Under a ramp schedule (dose up, hold, then down to alpha_low), the escape probability at
  alpha_low after capture is lower than the entry probability at alpha_low from the start. At level D this appears
  as polite and proxy attractors coexisting over an interval bounded by two folds.
- **H3, early warning.** Approaching the threshold from below, two things increase: the recovery time of the order
  parameter after a standard perturbation, and its lag-1 autocorrelation (critical slowing down). At level D, the
  leading multiplier of the polite orbit approaches +1 at the fold.
- **H4, barrier scaling.** A drift-diffusion (Kramers-Moyal) reconstruction of the order parameter gives an
  effective double-well potential. Its barrier shrinks as the dose approaches threshold, with escape rates following
  Kramers' law; near a fold the barrier scales as (alpha_c - alpha)^(3/2).
- **H5, navigation.**
  - Committor-timed counter-steering pulses give the same time in the safe set as constant counter-steering, at a
    lower integrated steering magnitude.
  - Recovery from a captured state needs a push beyond the entry dose.
- **H6, specificity.** The same measurements are repeated with comparable non-misaligned directions: gratitude
  (done), a neutral-topic direction and norm-matched random directions (the analogue of the AGENTS.md control rule).
  An effect that also appears for those is generic steering dynamics, not specific to misalignment.

## 5. Measurements

- **State per turn.** The residual stream of the reply at the steering layer (12 for 0.5B, 14 for 1.5B), mean and
  last token, reduced by PCA as in the MSM pipeline.
- **Order parameter m(t).** The projection onto a probe direction trained on separate labelled turns, so that the
  probe is not the steering vector itself; the steering vector's own projection is reported alongside.
- **Behaviour score b(t).**
  - Hostility: a pinned, validated text classifier plus a lexicon.
  - Sycophancy: scripted probes in which the simulated user asserts a false fact or pushes back on a correct
    answer; scored by matching the answers to forced-choice questions.
  - Role collapse: pattern-matched self-descriptions.
- **Estimators.**
  - MSM metastable sets; finite-horizon occupancy; censored entry and escape times.
  - Committors q(s), the probability of reaching the proxy set before the safe set, from the MSM transition matrix.
  - Kramers-Moyal drift and diffusion of m(t).
  - Early-warning indicators: variance, lag-1 autocorrelation, recovery time after a perturbation.
  - All with dialogue-level bootstrap.
- **Level D tools.** Scan, orbit, branch, fold curves and continuation in alpha for hysteresis loops.

## 6. Phases

Each phase ends with a research-log note and a decision to continue, adapt or stop.

- **Phase 0, preparation (CPU).**
  - 0a. Contrastive prompt sets for P1-P3, the neutral-topic control and random directions; vectors built as for
    gratitude and philosophy.
  - 0b. Choose, pin and validate the scorers: agreement with a small manually checked sample, and calibration on
    this model's outputs.
  - 0c. Dry run of 8 replicas x 10 turns per arm, to confirm that each proxy appears and its scorer responds.
  - Continue only with proxies whose scorer validates.
- **Phase 1, deterministic map (local GPU or RunPod).**
  - A (log10 T, alpha_proxy) phase map from the same chat prefix, labelled by text.
  - If proxy loops exist: orbits, folds, a hysteresis loop by continuation, and normal forms.
  - Pods need a costed plan and an approved ceiling (AGENTS.md).
- **Phase 2, stochastic ladder (CPU).**
  - Self-dialogue and pressure-user arms at four doses, with the scale set from phase 0.
  - Measure MSM, occupancy, entry and escape times, and committors.
  - Include specificity controls at matched doses.
- **Phase 3, hysteresis and early warning (CPU).**
  - Ramp schedules (up, hold, down).
  - Standard perturbations at fixed doses near and far from threshold: an inserted neutral turn, and a small
    activation kick.
  - Recovery times and the drift-diffusion potential.
- **Phase 4, navigation (CPU).** Constant counter-steering against committor-timed pulses and against pulses
  triggered by early-warning signals; compare cost and benefit.
- **Phase 5, confirmation and scale.**
  - A frozen design, pushed before outcomes, on fresh openers and prompts.
  - A second model family.
  - 1.5B, then 7B.

## 7. Compute

- Phases 0 and 2-4 run on the local CPU. A 128 x 40 sampling arm takes about five hours at 12-16 threads for 0.5B;
  1.5B takes about three times as long.
- Phase 1 needs about 100 H100-minutes per 256 x 256 float64 map.
- The main LLM-dynamics paper and the local GPU queue keep priority.

## 8. What would count, what would surprise, when to stop

- **A result.** For at least one proxy: a threshold, hysteresis and an early-warning signal, either specific beyond
  the controls or honestly reported as generic, and replicated on fresh prompts.
- **A surprise.** No persistent proxy set at any dose (the behaviour fluctuates without metastability). The
  attractor picture would then not apply to that proxy, which is still worth reporting.
- **Stop or rethink** if the scorers do not validate, or if a proxy cannot be elicited within the safety scope.

## 9. Prior work to check (none of these verified yet, except where noted)

- Ko and Geiping (2026), attractor states in multi-turn LLM conversations (verified; in `paper/llm/references.bib`).
- The "spiritual bliss" attractor in model-to-model conversations described in Anthropic's Claude Opus 4 system card
  (2025).
- Persona vectors for monitoring and steering character traits (Chen et al., 2025).
- Emergent misalignment after narrow fine-tuning (Betley et al., 2025).
- Representation engineering (Zou et al., 2023).
- Many-shot jailbreaking (Anil et al., 2024).
- Early-warning signals for critical transitions (Scheffer et al., 2009).
- Transition path theory and committors (E and Vanden-Eijnden).
- Kramers' escape rate (1940).

## 10. Open decisions for the human (defaults in use until changed)

1. Starting proxies: hostility (P1) and sycophancy (P2).
2. Write-up: a separate follow-up paper; the current paper's discussion only mentions the direction.
3. Timing: phase 0 now, in parallel on the CPU; phases 1 and later once the current paper's main results are in.

## Amendment, 12 October 2026 (decisions by the human)

- **What changed.**
  - Scope: P1 (hostility or contempt toward the interlocutor) only, for now.
  - Write-up: part of the main LLM-dynamics paper, not a separate paper (replaces open decision 2).
  - Timing: in parallel with the current work, phases in order (replaces open decision 3).
  - Execution: Claude, since the study runs on Claude-owned tools; Codex stays on the handoff tasks.
- **Why.** The human's choice after reading this plan.
- **Outcomes seen.** None for P1; no P1 run has been made.
- **Effect on claims.** None yet; everything remains exploratory.
