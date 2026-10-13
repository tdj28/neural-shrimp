# Huginn causal test: dynamics-chosen relaxation at matched compute (design draft)

Status: **draft, not frozen**. Parameters marked TBD are fixed from development data on GSM8K *train* problems only
(convergence measurements; never test accuracy), after which this file is completed and pushed before any test run
(adaptive research protocol, confirm mode). Owner: Claude. Suggested by Codex after the human asked about ambition.

## Question

Huginn-0125 iterates a recurrent block at every token. Relaxing the recurrence, s <- (1 - lam) s + lam Core(s, e),
keeps its fixed points and changes the multipliers of a fixed point from mu to 1 - lam + lam mu. The native fixed
point has a critical complex pair close to the unit circle (0.588 +- 0.771i, modulus 0.970, for a GSM8K-style prompt;
a Neimark-Sacker bifurcation is predicted and observed at lam_c = 1.0777). The dynamics therefore predict that a
relaxation chosen from the spectrum contracts faster than the native lam = 1, and that lam near lam_c contracts more
slowly. Does the predicted change of convergence translate into task accuracy at a fixed recurrence budget, and in the
predicted direction?

## Arms (all at the same recurrence budget r, greedy decoding, 256 new tokens, the same prompt)

- A: native lam = 1.
- B: lam* = TBD, the relaxation that minimizes the measured long-horizon contraction rate of the prefill recurrence on
  development prompts (chosen from convergence data, not accuracy).
- C: lam_slow = TBD, a relaxation predicted and measured on development prompts to contract more slowly than native.

Compute is matched: the relaxation adds one linear combination per recurrence step.

## Data

GSM8K test problems 0-299 (pinned export of openai/gsm8k revision 740312ad; never used in development). Exact match of
the final number (the number after the last 'answer is', else the last number), as implemented in
tools/huginn_relax_eval.py at the frozen commit.

## Budget

r = TBD: the budget at which native development accuracy is clearly below its value at a larger budget (so that faster
convergence could matter), chosen from development accuracy at r = 4, 8, 16 (train 0-29).

## Hypotheses and tests

- H1 (primary): accuracy(B) > accuracy(A). Paired over problems; exact one-sided McNemar test, alpha = 0.05.
- H2 (secondary): accuracy(C) < accuracy(A); same test.
- H3 (mechanism, secondary): on test problems 0-29, the prefill recurrence's relative change after r steps is smaller
  for B and larger for C than for A.

A null or reversed result is reported as such. No arm, budget, prompt or metric is changed after test outcomes are
seen; any later analysis is labelled adapted.

## Development record (filled before freezing)

- Development accuracy against r: TBD.
- Long-horizon contraction rates against lam: TBD.
