# Prompt-blocked test at 72B (run PS72): frozen before any 72B prefix run

14 October 2026, about 11:00 UTC. Written after the frozen 7B/14B prompt-blocked test (docs/prefix-scale-test.md)
failed both predictions: across held-out prefixes Qwen2.5-14B was chaotic at more stage-1 temperatures than
Qwen2.5-7B, all of them above log10 T 0.36, at and above its entropy transition. On the paper's prefix alone,
Qwen2.5-72B was chaotic only in a narrow window at its transition (runs B1, B4, B7). This test asks whether the 72B
model, too, has broad chaos at other prefixes. No 72B prefix run exists at this commit; the window confirmation B7b
(paper's prefix) is running and does not inform this design.

## Design

- Model: Qwen2.5-72B-Instruct, pinned revision 495f3936, unsteered, float64 with bfloat16 weight storage (exact; the
  checkpoint is stored in bfloat16), layers over two GPUs.
- Prefixes: the first four held-out units of docs/prefix-scale-prefixes.json, indices 2, 3, 4 and 5 (a fixed rule,
  chosen for cost; the 14B values of these prefixes are partly known: 4 and 5 are, 2 and 3 are not yet).
- Stage 1 only: 48 temperatures over log10 T -0.4..0.9, as stage 1 of the 7B/14B test; tools/hidden_chaos.py line,
  1000 transient and 2000 recorded steps; twins at relative offsets 1e-5 and 1e-6 (1e-7, used at 7B/14B, is only five
  times the 72B noise floor of about 2e-8, so the second offset is raised before any data).
- Label: chaotic = exponent above 1e-3 at both offsets and no period up to 64.

## Endpoints and prediction

- C1(72B, prefix): chaotic temperatures of the 48. M: the largest min-offset exponent.
- PS72 (primary): C1(72B) < C1(14B) at at least 3 of the 4 prefixes (14B from the frozen 7B/14B test; a prefix with a
  missing 14B or 72B result is reported missing and counts against the prediction).
- Descriptive: per prefix, C1, M, the chaotic temperatures and the temperature of the largest entropy increase.

## Scoring

Labels as returned, no reruns to change them; a failed run is rerun once with the same settings. Reported in the paper
whether the prediction passes or fails.
