# REPRO-T: does an irrelevant execution choice change the answer of continuous-feedback inference?

Frozen design, 15 October 2026, pushed before any held-out run. Recommended by consult 9 (direction 1; see
docs/consults/2026-10-15-missing-anything-huge/adjudication.md). Exploratory in the project's sense, but frozen
because it may carry a practical claim. Code: tools/repro_task.py at the commit that adds this file. Problems:
docs/repro-task-problems.json (fixed before any run).

## Units and conditions

- **Units.** 96 held-out two-digit additions (seed 1015002; 48 with a units carry), Qwen2.5-0.5B-Instruct, unsteered.
  - Eight development problems (seed 1015001) check the pipeline only and do not enter any endpoint.
  - Three fixed distractor problems (seed 1015003) fill the batched layout.
- **Pipeline (soft).** After the chat-templated question ("What is A + B? Answer with just the number."), 32 steps each append the expected embedding E^T softmax(z/T) to the context: Soft Thinking's growing context, without top-k.
  - Then the fixed cue "\n\nAnswer: " is appended, and 8 tokens are decoded greedily.
  - The canonical answer is the first integer in them, or INVALID if there is none.
- **Baseline (greedy).** Ordinary greedy inference with the same budget: 32 greedy tokens, the same cue, the same extraction.
- **Temperatures.** log10 T 0.0 (nominally attracting) and 0.2 (nominally expanding: inside the 0.5B one-token map's unsteered chaotic band on the paper's prompt), fixed from prior maps. log10 T 0.4 is run as a secondary condition.
- **Layouts.** The problem alone (batch 1), against the problem in row 0 of a batch of four with the three distractors.
  - Every prompt has the same token length, so nothing is padded and positions are identical; only the matrix shapes differ.
  - The single layout is rerun as a determinism control.
- **Precision.** bf16 (model in bfloat16; logits, softmax and soft token in float32) is primary. float64 (norms and rotary patched) is secondary.

## Endpoints

- **Primary.** Delta = 100 x mean over held-out problems of [ (d_soft(0.0) + d_soft(0.2)) / 2 - d_greedy ], in bf16.
  - d = 1 if the canonical answers of the single and batched layouts differ. INVALID counts as an answer value, and a pair of INVALIDs is reported separately.
  - 95 % interval by a paired bootstrap over problems (10,000 resamples, seed 0).
  - Two-sided sign-flip permutation test on the per-problem differences (10,000 permutations, seed 0).
  - Positive if the interval excludes 0.
- **Secondary.**
  - The same contrast per temperature, in float64, and at log10 T 0.4.
  - Exact-answer accuracy per condition and layout; INVALID rates.
  - The per-step relative distance between the two layouts' soft tokens, and its value at step 32.
  - The first step at which greedy tokens differ.
  - Determinism: share of identical reruns.
- **Mechanism subset.** For the first six held-out problems, in float64 and the single layout, at both temperatures, the first soft token is displaced by delta |x| with delta in {1e-12, 1e-8, 1e-4}. Report the distance growth and answer changes against the unperturbed run.

## Interpretation (fixed now, after the consult)

- **Positive, with growth of the layout distance over the 32 steps:** operational sensitivity for this implementation and budget.
- **Positive without growth:** a discrete answer-boundary effect. Retain the reproducibility result, but not a chaos attribution.
- **Null:** short-horizon robustness at this budget. It does not refute long-run chaos.
- **No numerical difference between layouts:** an operational null. The injected perturbations then estimate the threshold.
- Accuracy differences are reported, but this test makes no accuracy-harm claim.

## Extension, frozen before its run (15 October 2026, after the 0.5B result): Qwen2.5-7B-Instruct

The same design, problems (docs/repro-task-problems.json), distractors, temperatures, budget, cue, extraction and endpoints, with Qwen2.5-7B-Instruct (pinned a09a3545) instead of the 0.5B model. The primary endpoint is again the bfloat16 Delta, with the same interval and test. Run REPRO-T7, job set reprotask, at the commit that adds this section. The interpretation rules are unchanged. The 0.5B result is not used to choose anything here.
