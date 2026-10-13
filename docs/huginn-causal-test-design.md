# Huginn causal test: does the fixed-point spectrum predict how test-time relaxation changes the recurrence and the task?

Status: **draft v3.1, not frozen** (13 October 2026). Parts A and B are complete and revised after Codex's
pre-outcome review (codex/huginn-design-review/, 9ea0384e). The file is pushed (frozen) before any test-split
computation (adaptive research protocol, confirm mode). Owner: Claude. Suggested by Codex after the human asked about ambition. The
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

## Part A (primary): a gold-conditioned spectral feature against the finite-budget residual, question by question

**Scope (narrowed after Codex's pre-outcome review, codex/huginn-design-review/, 9ea0384e).** Part A tests a
held-out association. Across fresh ARC questions, does an estimated spectral feature of each question's fixed point
rank that question's finite-budget residual under relaxation? It is teacher-forced and gold-conditioned: the
analysed sequence is the question with its correct answer, so this is a mechanistic probe, not an answer-blind
warning. It is not a mediation test. Persistent differences between questions can make all within-arm correlations
positive. Together with Part B the results would be consistent with the mechanism, but would not identify it.

**System.**
- Model: tomg-group-umd/huginn-0125, revision bb6621b65e90b6a4b9b29ef88dc83866d450470c, passed explicitly to every
  loader (weights, remote code, tokenizer). Run with transformers 4.55 and the cache patch in
  tools/huginn_relax_eval.py.
- Data: allenai/ai2_arc ARC-Challenge revision 210d026faf9955653af8916fad021475a3f00453. test.jsonl has SHA-256
  062fe98a0d64b0bb86b43ee49faf8de6b4c02417f0b4a6b7f9607208c6f1a9d6.
- Questions: test indices 0-63 in file order, fixed now.

**Predictor, computed first and saved before the read-out run.**
- explore_latent_loop.py fixed-spectrum --model huginn --dtype float32 --arc-split test --arc-indices 0..63
  --arc-ids batch --lam 1.0, with defaults transient 3000, k 8, arnoldi-iter 400, fallback transients 0.75 and 0.5.
- The token IDs are exactly the gold row of the read-out batch (huginn_relax_eval.gold_sequence_ids: context, then
  the continuation tokenized separately). The spectrum records their SHA-256, and the analysis requires it to equal
  the read-out row's.
- Disposition per question:
  - 'ok' needs a finite relative fixed-point residual <= 1e-4, a finite nonempty union of the computed LM, SR and LR
    eigenvalues, and matching token IDs;
  - every other case is excluded with its reason, and no excluded question is replaced.
- Feature: rho_q(lam) = max |1 - lam + lam mu| over the computed eigenvalues. This is an estimated Ritz feature,
  not a complete spectrum. The computed sets can miss the dominant mode of the shifted operator; Codex's
  counterexample is recorded as a test. 'Unstable' therefore means an unstable mode was detected; the converse is
  'no unstable mode detected', not proven stability. ARPACK convergence flags and eigenvalue counts are reported per
  question.

**Outcome, computed second.**
- huginn_relax_eval.py arc --split test --start 0 --count 64 --steps 24 25 26 27 28 29 30 31 32
  --lams 1.0 1.05 1.08 1.10 --seed 0. The recurrence runs in bfloat16 on all choices; the initial latent states are
  paired across arms, torch.manual_seed(100003 seed + index).
- Endpoint: for the gold sequence only, R(s_31) = ||Core(s_31) - s_31|| / ||s_31||. This is the native residual of
  the state after 31 updates, i.e. the input of update 32. The difference is taken in float32 from the bfloat16
  states, so the recurrence's bf16 floor remains. It is a finite-budget endpoint, not a rate. Zero residuals are
  kept, and ties get average ranks; no log or clamp is applied, since Spearman correlation uses ranks.
- Secondary outcome: whether the chosen answer (argmax log-likelihood over choices) changes across steps 24-32.

**Admission, enforced before any statistic.**
- Exactly one row per requested (question, arm) cell; duplicates, extras and missing cells are refused.
- Each row must match the frozen split and seed, have read-outs at steps 24-32, a residual trace of 32 entries, and
  a finite non-negative gold residual.
- Exactly one spectrum disposition per requested question, and no spectrum outside the panel.
- The cohort is the questions with disposition 'ok', the same in every arm. It must hold at least 48 of the 64;
  fewer is a failed setup.

**Primary prediction (A1).** S is the mean, over all four arms, of the Spearman correlation across the cohort
between rho_q(lam) and R(s_31) at that arm.
- S is undefined, and the primary fails, if any arm's correlation is undefined (for example, a constant input).
- The interval is a question bootstrap: 2000 resamples of the cohort, the same questions in every arm.
- A draw is undefined under the same rule. More than 1 % undefined draws fails the primary; otherwise the interval
  uses the defined draws, and their count is reported.
- Success requires a 95 % lower bound above 0.3.
- The interval is approximate and conditional on this seed, this numerical pipeline and the qualifying questions.

**Secondary (descriptive only, no tests).**
- The per-arm correlations.
- Answer-change counts and rates where an unstable mode was detected and where none was. Repeated cells of the
  same question are not independent, so no p-value is reported.
- The share of questions with a detected unstable mode in each arm.

**Analysis code.** tools/huginn_arc_analysis.py items at the frozen commit:

    items --spectra <folder> --rows <folder> --start 0 --count 64 --lams 1.0 1.05 1.08 1.10 --split test --seed 0

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

**Predictions** (from the development lam response; the arms straddle the spectral thresholds). Write
Delta(lam) = accuracy(lam) - accuracy(native), flexible match.
- B1 (decline): Delta(1.20) < 0, by an exact one-sided McNemar test at 0.05. B1 tests any decline; the magnitude
  is reported, with lost and gained counts and an interval. A 'collapse' is not claimed from B1 alone.
- B3 (non-inferiority within 4 points): the 5th percentile of the paired problem bootstrap of Delta(1.05)
  (10,000 resamples, seed 0) is above -0.04. This is an approximate one-sided 95 % bound. Four points is a
  tolerated loss, chosen as half the smallest development loss called a break (8 points at 1.15). It is not a
  no-loss claim.
- Part B succeeds if B1 and B3 both pass. The global null is {Delta(1.20) >= 0} or {Delta(1.05) <= -0.04}, so
  this is an intersection-union test with each component at 0.05. B3's bootstrap makes the joint level nominal,
  not exact.
- Secondary, descriptive: Delta(1.15) with its counts (B2); the same contrasts for strict accuracy; median new
  tokens; the share of replies at the 256-token cap; the share in the strict 'The answer is N' format. A capped
  reply is an observed outcome, not a missing row.
- Admission: every row must match the frozen split (test), recurrence budget (32), prompt style (cot8), seed and
  lam. Coverage of every problem in every arm and no duplicate rows. An interrupted run is completed to all 200
  problems; it is not analysed as a prefix.

**Analysis code.** tools/huginn_arc_analysis.py generation at the frozen commit (coverage of every problem in every
arm is checked; duplicate rows are refused):

    generation --rows <folders> --start 0 --count 200 --split test --steps 32 --prompt-style cot8 --seed 0

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

## Changes in v3.1 (Codex's pre-outcome review, before any test computation)

- Part A is now a same-prefill probe:
  - the spectrum uses exactly the gold row's token IDs, checked by hash;
  - the endpoint is the gold sequence's own residual, not the all-choice aggregate;
  - the clock is stated as R(s_31), the input of update 32;
  - the residual difference is taken in float32.
- The spectral feature is an estimated Ritz feature, with detected-instability wording.
- Admission rules: exact panel, one disposition per question, a common cohort of at least 48, no dropped arms,
  undefined-draw rules.
- A3 is descriptive without a p-value.
- Part B reports accurate test labels (decline; non-inferiority within 4 points) with the margin's rationale.
  Rows are admitted against the frozen split, budget, prompt, seed and lam, and the cap rate is reported.
- Explicit model revision in every loader. On the 100 development questions the gold-row IDs from both paths were
  identical, so the development association was not affected by tokenization.
