# Huginn causal test: does the fixed-point spectrum predict how test-time relaxation changes the recurrence and the task?

Status: **draft v3, not frozen** (13 October 2026). Parts A and B are complete. The file goes to Codex for review
and is pushed (frozen) before any test-split computation (adaptive research protocol, confirm mode). Owner: Claude. Suggested by Codex after the human asked about ambition. The
history of v1 and v2 is at the end.

## Question

Huginn-0125 iterates a recurrent block at every position. Relaxing the recurrence,
s <- (1 - lam) s + lam Core(s, e), keeps its fixed points and maps each multiplier mu of a fixed point to
1 - lam + lam mu. The spectrum of a prompt's fixed point therefore predicts, before anything is run, how that
prompt's latent state responds to relaxation:
- the asymptotic rate rho(lam) = max |1 - lam + lam mu|;
- the stability interval of lam, whose upper end lam_up is where a negative real multiplier reaches -1 (flip) or a
  complex pair reaches the unit circle (Neimark-Sacker); lam_up < 1 means the native loop is itself unstable there.

Does this prediction hold question by question on fresh data (Part A)? And does the instability matter for a task in
which step-to-step changes can compound (Part B)?

## What development showed (no test-split data used)

- GSM8K train prompts 0-4 (zero-shot instruction template): fixed points with leading multipliers around the origin.
  lam_c = 1.074-1.126; best relaxation lam* = 1.03-1.06, within 1 % of the native rate. Measured float32 rates follow
  the spectrum: 0.951 measured against 0.953 predicted at lam 0.55, and 0.955 against 0.956 at lam 1.05 on the
  prompt with lam_c 1.074.
- ARC-Challenge train 0-29 (question + correct choice; jobs 11c/11d; all 30 pass the fixed-point gate):
  - lam_up quantiles: 0.88 (min), 1.015 (20 %), 1.088 (median), 1.149 (max);
  - 6 of 30 are natively unstable (rho(1) = 1.009-1.273);
  - the share unstable is 20 % at lam 1.0, 23 % at 1.05, 43 % at 1.08, 70 % at 1.10 and 100 % at 1.15.
- ARC readouts (run H2, train 0-99, bfloat16, steps 24-32):
  - past lam_c the answer margins oscillate step to step, with answer changes for 12 / 16 / 33 / 63 % of questions
    at lam 1.0 / 1.10 / 1.15 / 1.25;
  - mean accuracy holds until 1.15 and falls at 1.25.
- Item level (tools/huginn_item_level.py; dry run of the Part A analysis on train 4-29,
  huginn-relax/dryrun-items): within each lam arm, the Spearman correlation across questions between rho_q(lam)
  and the log native residual at step 32 is 0.754 at lam 1.0 and 0.756 at 1.10. The mean is 0.755
  [0.58, 0.87]. Answer changes occur in 4 of 24 unstable (question, lam) pairs and 1 of 28 stable ones.
- Wang et al. 2026 (TAPS, arXiv 2609.36653) report for pretrained Huginn that a constant step of 1.2 collapses
  accuracy (GSM8K 8.34 % against 19.41 %) and that 0.8 underperforms. This is context only.
- GSM8K 8-shot chain-of-thought generation against lam (run H1b, train 0-119 complete in all arms, r = 32,
  tools/huginn_gsm8k_dev.py). Flexible accuracy: native 0.375; 0.80 0.408; 1.05 0.425; 1.10 0.425; 1.15 0.292
  (17 lost / 7 gained, one-sided McNemar p = 0.032); 1.20 0.158 (29 / 3, p = 1.3e-6). The break lies between 1.10
  and 1.15, at or just above the spectral thresholds of the zero-shot GSM8K prompts (1.074-1.126) and of the ARC
  questions (median lam_up 1.09). Generation breaks earlier than log-likelihood scoring (ARC held to 1.15). The
  dry run of the Part B analysis on these rows (huginn-relax/dryrun-generation) gives B1 p = 1.3e-6, B3 lower 90 %
  bound 0.00 and B2 p = 0.032.

## Part A (primary): item-level prediction on fresh ARC-Challenge questions

**System.**
- Model: tomg-group-umd/huginn-0125, snapshot bb6621b65e90b6a4b9b29ef88dc83866d450470c (weights, remote code
  and tokenizer). Run with transformers 4.55 and the cache patch in tools/huginn_relax_eval.py.
- Data: allenai/ai2_arc ARC-Challenge revision 210d026faf9955653af8916fad021475a3f00453. test.jsonl has SHA-256
  062fe98a0d64b0bb86b43ee49faf8de6b4c02417f0b4a6b7f9607208c6f1a9d6 (tools/export_benchmark.py manifest).
- Questions: test indices 0-63 of that file, in file order. They are fixed now and no question is read before
  the freeze.

**Predictor, computed first.**
- For each question, the float32 fixed point and leading spectrum of the prefill
  "Question: {q}\nAnswer: {correct choice}". Tool: explore_latent_loop.py fixed-spectrum --model huginn --dtype
  float32 --arc-split test --arc-indices 0..63 --lam 1.0, with defaults transient 3000, k 8, arnoldi-iter 400 and
  fallback relaxed transients 0.75 and 0.5.
- A question whose relative fixed-point residual exceeds 1e-4 is excluded and reported.
- ARPACK may not converge for every eigenvalue set (LM, SR, LR). The tool keeps the converged eigenvalues and
  flags the set; all kept eigenvalues are used, as in development (18 of 31 development spectra converged in all
  three sets). The flags are reported.
- From each spectrum: rho_q(lam) and lam_up(q). The correct choice is used only to select which prefill is analysed;
  no outcome below depends on correctness.

**Outcome, computed second.**
- huginn_relax_eval.py arc --split test --start 0 --count 64 --steps 24 25 26 27 28 29 30 31 32
  --lams 1.0 1.05 1.08 1.10 --seed 0 (bfloat16; paired initial latent states, torch.manual_seed(100003 seed + index)).
- Per question and arm, two outcomes:
  - the native residual ||F(s) - s|| / ||s|| at recurrence step 32;
  - whether the chosen answer (argmax of the log-likelihood over choices) changes across steps 24-32.

**Primary prediction (A1).** S is the mean, over the four arms, of the Spearman correlation across questions
between rho_q(lam) and the log native residual at step 32. Success requires both:
- the lower end of a question-bootstrap 95 % interval (2000 resamples, the same questions in every arm) is above
  0.3;
- at most 25 % of the questions are excluded by the fixed-point gate. More than 25 % is reported as a failed setup.

**Secondary (descriptive, no multiplicity correction).**
- A2: the per-arm correlations.
- A3: the answer-change rate in spectrally unstable (rho_q(lam) >= 1) against stable (question, arm) pairs, pooled
  over arms, with a one-sided Fisher exact test.
- A4: the share of questions predicted unstable in each arm.

**Analysis code.** tools/huginn_arc_analysis.py items at the frozen commit:

    items --spectra <spectra folder> --rows <read-out folder> --start 0 --count 64 --lams 1.0 1.05 1.08 1.10

## Part B (secondary): GSM8K generation accuracy against lam

**System.**
- Data: GSM8K test problems 0-199 in file order (openai/gsm8k revision 740312ad; test.jsonl SHA-256
  3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14).
- Prompt: 8-shot chain of thought in the lm-eval gsm8k_cot format (exemplars verbatim, gsm8k-cot.yaml SHA-256
  7c9dd256...).
- Decoding: greedy, generation until 'Q:' or 256 new tokens, r = 32 recurrence steps per token, bfloat16, paired
  initial latent states (seed 0).
- Command: huginn_relax_eval.py accuracy --split test --start 0 --count 200 --steps 32
  --lams 1.0 1.05 1.15 1.20 --interleave --prompt-style cot8 --max-new 256 --seed 0. The problems may be split
  across pods as contiguous ranges with identical settings.

**Predictions** (from the development lam response; the arms straddle the spectral thresholds):
- B1: flexible accuracy at lam 1.20 is lower than native (exact one-sided McNemar, alpha 0.05).
- B3: lam 1.05 loses no more than 4 points against native: the 5th percentile of the paired problem bootstrap of
  the difference (10,000 resamples, seed 0) is above -0.04.
- Part B succeeds if B1 and B3 both pass. This is an intersection-union test, so each is at alpha 0.05.
- Secondary, descriptive: B2 (lam 1.15 lower than native, one-sided McNemar); the same contrasts for strict
  accuracy; median new tokens; the share of replies in the strict 'The answer is N' format.

**Analysis code.** tools/huginn_arc_analysis.py generation at the frozen commit (coverage of every problem in every
arm is checked; duplicate rows are refused):

    generation --rows <folders> --start 0 --count 200

**Spectral context.** Before the test, development spectra of the 8-shot prompts (gpu job 13, GSM8K train 0-2,
huginn-relax/cot8-spectrum-*) are reported with this design. They do not change the arms or the success criteria.

## Rules

Nothing in this file changes after test outcomes are seen; any later analysis is labelled adapted. A null or reversed
result is reported as such. The spectra are computed and saved before the read-out run starts. Both runs and the
analysis use the commit at which this file is frozen.

## Changes from v2 (before any test data)

- The primary moves from population accuracy to an item-level dynamical prediction. ARC development showed:
  - mean accuracy tolerates the instability up to lam 1.15, because answer changes go both ways (run H2), so the
    v2 cliff hypothesis H1 had no development support;
  - each question's own spectrum predicts its own convergence (Spearman about 0.75 within arms).
- The read-out arms move into the range where questions split between stable and unstable (lam 1.0-1.10; 20-70 %
  unstable in development).
- The GSM8K generation test becomes Part B, with arms and criteria fixed from H1b.

## History

- v1: GSM8K generation with a spectrally chosen lam*. Superseded when the development spectra showed that no
  relaxation improves the rate by more than about 1 %.
- v2: ARC-Challenge population tests (H1 cliff at the spectral lam_c, H2 under-relaxation, H3 equivalence at lam*,
  H4 residual mechanism). Superseded as above.
- Codex's preflight of v1 (codex/huginn-design-preflight.md, commit 323d5083) set the float32 convergence
  measurement and the residual-level fit window used in development.
