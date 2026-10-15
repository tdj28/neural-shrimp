# BPC: does a prompt's loop regime predict the accuracy of ordinary answers beyond difficulty? (frozen)

Frozen 15 October 2026 and pushed before any BPC data. This is the confirmatory follow-up of the exploratory run BP (research log, 15 Oct 05:53 UTC). The human chose option B at about 06:30 UTC ("Go with B"). At about 08:25 UTC the human asked "What about our statistical test on solving problems in a chaotic vs periodic area?" and then "We should also test deeply periodic regions and deeply chaotic regions and regions that are on the edge". Consult 9 (adjudication item 4) shaped the design: a single task family, covariates fixed before sampling, and held-out predictive value.

## Data

- **Model.** Qwen2.5-7B-Instruct (pinned a09a3545), unsteered.
- **Prompts.** docs/bpc-tasks.json, 4000 fresh products of two two-digit numbers ("Compute a * b. Reply with only the number."), seed 20261116, with none of run BP's products. One task family only, which removes the task-mix confound of BP.
- **Label.** The soft-token loop after each prompt's own chat-templated prefix (window 1, float64, alpha 0) at log10 T = 0.40. Transient 500, record 1000, twins at 1e-6 and 1e-9, classified by tools/hidden_chaos.classify.
  - Computed with tools/behavior_prompts.py label --batched (16 prompts per batch) on 12 shards.
  - "chaotic" means visible or hidden chaos and "periodic" means periodic. Other labels are excluded from the primary analysis.
- **Covariates,** all computed before any sampled outcome (tools/behavior_prompts.py covariates and the greedy answer of generate):
  - greedy answer correct;
  - entropy and largest probability of the reply's first-token distribution at T;
  - log-probability of the correct first token at T;
  - number of digits of the product;
  - a carry from the units digits;
  - a + b.
- **Outcome.** 10 sampled answers per prompt at T (top_p 1, top_k 0, float32, at most 12 new tokens), scored by the first integer. Per prompt: the fraction correct.

## Analysis (tools/bpc_analyze.py at the commit that adds this file; tests/test_bpc_analyze.py)

- **Primary.** The coefficient of the chaotic indicator in y ~ 1 + chaotic + X (least squares, covariates standardised).
  - 95 % bootstrap interval over prompts (10,000); two-sided permutation test of the labels (10,000).
  - **Positive:** coefficient < 0 with p < 0.05, meaning chaotic-loop prompts are answered less accurately beyond the covariates.
- **Secondary:**
  - (a) the raw chaotic-minus-periodic difference;
  - (b) the same among greedy-correct prompts;
  - (c) held-out predictive value: logistic regressions on even-numbered prompts, scored on odd ones; the log-loss improvement per sample from adding the label, with a bootstrap interval;
  - (d) the coefficient of the continuous exponent in place of the label;
  - (e) the regime gradient and the edge of chaos. The bins of the exponent are fixed now: deep periodic (lambda <= -0.3), edge (|lambda| <= 0.1) and deep chaotic (lambda >= 0.3). Report the covariate-adjusted accuracy per bin, the edge contrast (edge minus the two deep bins) and the quadratic coefficient of adjusted accuracy on lambda, with bootstrap intervals. An edge optimum predicts a positive contrast and a negative quadratic coefficient.
- **Power.** With about 1800 prompts per label (BP had 45 % chaotic at this temperature), the interval half-width of an accuracy difference is about 3 points.
- **Interpretation.**
  - A positive primary result supports the loop regime as a diagnostic beyond these covariates. It is not evidence of causal harm.
  - A null result rejects the diagnostic at this precision, not the dynamics.
  - Any deviation from this plan will be a dated amendment.
