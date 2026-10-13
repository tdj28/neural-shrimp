# Huginn causal test: does the fixed-point spectrum predict how test-time relaxation changes accuracy?

Status: **draft v2, not frozen** (development amendment, 13 October 2026). Every parameter marked TBD is fixed from
development data only (GSM8K train and ARC-Challenge train; no test-split outcome is computed before the freeze).
The completed file is then reviewed by Codex and pushed before any test run (adaptive research protocol, confirm
mode). Owner: Claude. Suggested by Codex after the human asked about ambition; the v1 draft and Codex's preflight
(codex/huginn-design-preflight.md, commit 323d5083) are superseded as recorded below.

## Question

Huginn-0125 iterates a recurrent block at every position. Relaxing the recurrence, s <- (1 - lam) s + lam Core(s, e),
keeps its fixed points and maps each multiplier mu of a fixed point to 1 - lam + lam mu. The spectrum of the native
fixed point therefore predicts, before any task is scored, how relaxation changes the convergence of the latent state:
the asymptotic rate rho(lam) = max |1 - lam + lam mu|, the best relaxation lam*, and the instability threshold lam_c > 1
where a negative real multiplier reaches -1 (flip) or a complex pair the unit circle (Neimark-Sacker). Does the
predicted change of the dynamics appear in task accuracy at a matched number of recurrence steps, and where predicted?

## What development showed (before this design)

- GSM8K train prompts 0-3 (instruction template, about 68 positions; float32 fixed points, residual 1-2e-6): the
  leading spectrum surrounds the origin (real 0.914, complex pairs of modulus 0.85-0.89 at many angles, negative real
  -0.72 to -0.82). Predicted lam* = 1.03-1.06 with rho under 1 % below native, so relaxation cannot speed convergence
  much; lam_c = 1.094-1.126 (median 1.113), set by the negative modes; lam 0.55 slows the rate from 0.914 to 0.953.
- Measured float32 convergence on prompt 0: lam 0.55 0.9505 (predicted 0.9525), 0.75 0.9298 (0.9352); near lam = 1
  the accessible window decays 0.02-0.04 faster than the spectral radius (small amplitude in the slowest modes).
- Wang et al. 2026 (TAPS, arXiv 2609.36653) report for pretrained Huginn that a constant step 1.2 collapses accuracy
  (GSM8K 8.34 % against 19.41 %; ARC-C 35.15 against 37.80) and 0.8 underperforms, without explanation. This is the
  pattern the spectra predict; it was not used to choose any parameter below except as context.
- ARC-Challenge development (train questions, job 11c): TBD (spectra of 5 train questions with their correct choice;
  convergence; accuracy against budget and lam).

## System and data

- Model: tomg-group-umd/huginn-0125, snapshot bb6621b65e90b6a4b9b29ef88dc83866d450470c (weights and remote code
  raven_modeling_minimal.py), tokenizer from the same snapshot; transformers 4.55 with the cache patch in
  tools/huginn_relax_eval.py; bfloat16 on one GPU.
- Primary task: ARC-Challenge test, all 1172 questions (allenai/ai2_arc revision
  210d026faf9955653af8916fad021475a3f00453; test.jsonl SHA-256
  062fe98a0d64b0bb86b43ee49faf8de6b4c02417f0b4a6b7f9607208c6f1a9d6; tools/export_benchmark.py manifest). Each choice
  is scored by the log-likelihood of " " + text after "Question: {q}\nAnswer:" (lm-eval-harness format); primary
  metric acc_norm (log-likelihood divided by the character length of the continuation), secondary acc. Scoring
  needs one prefill per choice, so the scored system is exactly the prefill recurrence whose fixed point is analysed.
- Every arm reads out all budgets from one prefill (the model's own head applied to the latent state after r steps;
  checked bit-identical to the model's output at the last step). The initial latent state is drawn with
  torch.manual_seed(100003 * seed + index), identical across arms (paired). Seed 0.
- Secondary task (if run): GSM8K test problems 0-299 (openai/gsm8k revision 740312ad...; test.jsonl SHA-256
  3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14), greedy generation, 256 new tokens, exact match
  of the final number, arms TBD, on a costed pod run.

## Predictions (fixed from development spectra; TBD values filled before the freeze)

Let lam_c,lo and lam_c,hi be the smallest and largest spectral lam_c over the development prompts of the primary task
(TBD), and lam_c,med their median.

- H1 (primary, the cliff). At r = 32 recurrence steps (Huginn's trained mean):
  - H1a: lam = TBD (below lam_c,lo by at least 0.03) is equivalent to native within +-2 percentage points (paired
    two one-sided tests, 90 % question-bootstrap interval inside the margin).
  - H1b: lam = TBD (above lam_c,hi by at least 0.07) is less accurate than native (exact one-sided McNemar, 0.05).
  - H1c: over the lam grid {TBD}, the breakpoint of a flat-then-linear fit of accuracy against lam lies in
    [lam_c,lo - 0.05, lam_c,hi + 0.05].
- H2 (secondary, under-relaxation). At r = 4 and r = 8, lam = 0.55 is less accurate than native (exact one-sided
  McNemar, 0.05 each).
- H3 (secondary, no gain available). lam* = TBD is equivalent to native within +-1.5 points at r = 4, 8 and 32.
- H4 (mechanism, secondary). Over the lam grid, the median native residual ||F(s) - s|| / ||s|| at the last recurrence
  step rises sharply between lam_c,lo and lam_c,hi + 0.05 (TBD threshold), and accuracy changes track it.
- Exploratory (not confirmatory): Anderson acceleration aa:3:1.0 and aa:5:1.0 at all budgets.

A null or reversed result is reported as such. Nothing in this file is changed after test outcomes are seen; any
later analysis is labelled adapted. The analysis code is tools/huginn_arc_analysis.py at the frozen commit.

## Changes from v1 (all before any test data)

- The primary task changed from GSM8K generation to ARC-Challenge log-likelihood scoring: the scored system is then
  the analysed prefill recurrence (generation adds per-token recurrences with a step-indexed cache), and the test fits
  the local GPU.
- The v1 hypothesis (a spectrally chosen lam* improves accuracy) is replaced: development spectra predict that no
  relaxation can improve convergence by more than about 1 %, so the gain claim is narrowed to an equivalence test
  (H3), and the confirmatory weight moves to the predicted instability (H1) and the predicted slowdown (H2).
- The random-relaxation arm is dropped: the lam grid is the dose-response. The untargeted-damping control of the
  earlier mode-damping idea does not apply, as no mode-targeted intervention is tested.
- Convergence is measured by the native residual in float32 (HuginnLoop) with a residual-level fit window
  (below 1e-2, above 30 x the floor), per Codex's preflight; development attempt 1 (bfloat16, on its floor) was
  discarded and logged.

## Development record

- GSM8K train spectra and convergence: research log 13 October (dev-spectrum-0..4, dev-conv-0..4).
- ARC-Challenge train: TBD.
