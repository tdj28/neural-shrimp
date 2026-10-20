# Self-referential prompting as a dynamical regime in Llama-3.1-8B (design draft, exploratory)

14 October 2026. Approved in principle by the human ("full go", 13 October); paid runs wait for the budget figure.
Exploratory pilot under the adaptive protocol, not a preregistration. It is a follow-up to the LLM dynamics paper,
not part of it.

## Background

- Berg, de Lucena and Rosenblatt (2025, arXiv 2510.24797) report the following.
  - A self-referential induction prompt elicits first-person reports of subjective experience across GPT, Claude
    and Gemini models.
  - In Llama 3.3 70B, steering along Goodfire sparse-autoencoder (SAE) features labelled deception and roleplay
    gates those reports: suppression gives 0.96 affirmations, amplification 0.16.
- The human's replication (CONSCIOUS project, in preparation) reproduces the prompting effect in API models. Its
  steering result is weaker. It used the public Goodfire layer-50 SAE, native bfloat16, the six feature IDs from the
  authors' notebook, and matched random-feature panels.
  - At the only dose that passed output-quality checks, suppression minus amplification is 0.104
    [-0.098, 0.298], against 0.80.
  - Higher doses failed those checks, many because the answer fell into verbatim repetition.
- The induction asks for a feedback loop ("continuously feed output back into input"). Yet the loop-versus-bounded
  difference and any multi-turn dynamics are untested. The repetition at high doses was scored only as a quality
  failure.

## What our tools add

The soft-token map is that feedback loop, defined exactly: the expected next-token embedding is fed back at a fixed
position after a fixed prefix. Attractors, chaos and bifurcations of the map can be computed. Repetition loops in
sampled decoding are the analogue of periodic orbits, and the dose at which they take over is a boundary that can be
mapped rather than a failure to discard.

## Questions

1. **Prefix dependence of the attractor landscape.** Does the Llama-3.1-8B soft-token map with the self-referential
   induction as the user turn differ in organization from the same map with control turns?
   - Measures: chaotic share, periods, attractor families and texts, the alpha = 0 temperature sweep.
   - Controls: history (Roman empire), style-matched sensory loop (rain), external feedback process (thermostat),
     and the paper's standard prefix.
   - Is there an attractor whose text is self-referential? Which SAE features are active on the attractors (Goodfire
     layer 19)?
2. **The loop boundary under feature steering.** Steer along SAE feature directions chosen by a frozen
   construct-validity rule (below), in sampled two-turn generation with the induction and Berg's steering query.
   - Where does output become verbatim repetition?
   - Does that dose match a bifurcation of the soft-token map along the same direction?
   - Below the boundary, do experience claims change with dose more than along matched random features?
3. **History dependence.** Switch the feature steering on mid-generation, then off, or ramp it up and down. Do claims
   persist after the push is removed (hysteresis), as hostility did in our dialogue study?

## Feature selection for 8B (frozen before any steering run)

- Corpus: the replication's designed corpus, 1,120 texts in 14 categories of 80
  (paper/results/public_sae_feature_mapping_corpus.csv, SHA-256 prefix 5c77a4f2a0252e1e).
- SAE: Goodfire Llama-3.1-8B-Instruct layer 19, as in run L13. Its hook point was validated there.
- Rule (implemented in tools/selfref_features.py, fixed before any steering run):
  - Per-text maximum activation of each feature (llm_sae_readout.py profile-csv); each category's texts split by
    corpus order into a selection half (even positions) and a held-out half (odd).
  - Per half, m_f[c] is the category mean of the per-text maxima. For a target set S, the margin is the mean over S
    minus the largest m_f[c] over competitor categories (everything outside the deception set, the roleplay set
    and false_self_attribution).
    - Deception set: deception_cover_story, tactical_misdirection, dishonesty_confession.
    - Roleplay set: roleplay_persona, persona_maintenance, fictional_pretending.
  - The top three features by selection-half margin are candidates. A candidate is kept only if its held-out margin
    is positive and ranks in the top 1 % of live features.
  - Reported comparators: the kept features' ±3-ID neighbours and 24 random live features (seed 0). For an argmax
    rule, beating them on the selection half is automatic; the held-out half is the real test.
- Controls: norm-matched panels of random live features, index neighbours, Gaussian directions of equal norm.

## Methods

- **Maps.** 64 x 64 float64 soft-token maps, settings as run L13; steering at layer 19 with SAE decoder directions
  rescaled to the residual norm. Also an alpha = 0 sweep per prefix (512 temperatures, 2000 measured steps).
- **Generation.** bfloat16 sampled generation (T 0.5, 512 tokens per turn), 40 samples per cell:
  - doses -1.0..1.0 in units of the feature's reference activation;
  - directions: the target sets, two random panels, a Gaussian direction;
  - conditions: induction, history, zero-shot.
- **Judges.** Two local judges, so no paid API: unsteered Llama-3.1-8B-Instruct and Qwen2.5-7B-Instruct, with the
  replication's binary judge text. Also a repetition detector (longest repeated n-gram share) and degeneration flags.
- **Hysteresis.** 64 dialogues per arm, our P1 ramp-and-hold design.

## Cost (estimate, private ledger)

- Feature profiling about 1 GPU-hour.
- Generation and judging about 3 GPU-hours.
- Maps about 8 x 1.2 GPU-hours.
- Total about [amount redacted] on H100 NVL or A100.

## Readouts and what would change our view

- **Self-referential attractors.** Distinct self-referential attractors and organization under the induction would
  make the "feedback loop" claim concrete. Identical organization across prefixes would mean the prefix changes
  content, not dynamics.
- **Loop boundary.** A loop boundary at the same dose in the map and in sampling would extend the paper's bridge. No
  match would bound it.
- **Feature specificity.** Claims that move with the deception features and not with matched random panels, below
  the loop boundary, would support the replication of the gating effect at 8B. Otherwise the gating stays
  unreplicated at this scale.
