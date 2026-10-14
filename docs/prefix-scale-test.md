# Prompt-blocked size test (run PS): frozen before any run

14 October 2026, about 09:05 UTC. Adopted from consult 7 (recommendation 1, in a cheaper form; adjudication in
docs/consults/2026-10-14-llm-paper-synthesis-v2/adjudication.md). The human asked for a definite main-paper answer on
size. The scale results so far use one prefix and fixed temperature grids; this test varies the prefix and searches the
entropy transition, where Qwen2.5-72B's narrow chaotic window was found (run B7).

## Design

- Models: Qwen2.5-7B-Instruct and Qwen2.5-14B-Instruct (pinned revisions a09a3545, cf98f3b3), unsteered, float64 with
  bfloat16 weight storage (exact).
- Prefixes: docs/prefix-scale-prefixes.json, ten system/opener pairs fixed in this commit. Index 0 (the paper's prefix)
  and 1 are development prefixes, reported separately; indices 2-9 are the held-out units.
- Stage 1 (coarse): 48 temperatures over log10 T -0.4..0.9; tools/hidden_chaos.py line, 1000 transient and 2000
  recorded steps, twins at offsets 1e-5 and 1e-7 (both well above the noise floor measured at 72B, about 2e-8).
- Stage 2 (transition): 24 temperatures over the interval of the largest increase of next-token entropy between
  adjacent stage-1 cells, widened by one cell each side (tools/prefix_transition.py), same settings.
- Label: chaotic = exponent above 1e-3 at both offsets and no period up to 64. Exponents at both offsets are reported.

## Endpoints

- C1(model, prefix): chaotic temperatures in stage 1 (of 48). C2: chaotic temperatures in stage 2 (of 24).
- M(model, prefix): the largest min(exponent at 1e-5, exponent at 1e-7) over both stages.

## Predictions

- PS1 (primary): over the 8 held-out prefixes, C1(14B) < C1(7B) in at least 6, and C1(14B) > C1(7B) + 2 in none.
- PS2: M(14B) < M(7B) in at least 6 of the 8 held-out prefixes.
- PS3 (descriptive, no direction predicted): the number of held-out prefixes with any chaotic stage-2 cell, per model.

## Scoring

Labels as returned; no reruns to change them. A failed run is rerun once with the same settings and reported; a
prefix with a missing result for either model is reported missing, never imputed. Every prefix's numbers are
reported in the paper, whether the predictions pass or fail.
