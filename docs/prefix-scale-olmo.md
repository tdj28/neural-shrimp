# Prompt-blocked test in OLMo-2 (run PSO): frozen before any OLMo prefix run

14 October 2026, about 12:25 UTC. On the paper's prompt OLMo-2-32B (release 0325) was the most chaotic OLMo-2 model
(16.2 % of the temperatures in log10 T [-1, 0.3], against 6.2 % for OLMo-2-7B, release 1124), the result that broke the
frozen scaling predictions P1-P3 for this family. The Qwen2.5 prompt-blocked test (docs/prefix-scale-test.md) showed
that one prompt can mislead about size. This test asks whether the OLMo ordering holds across prompts. No OLMo prefix
run exists at this commit.

## Design

- Models: OLMo-2-1124-7B-Instruct (revision 470b1fba) and OLMo-2-0325-32B-Instruct (b9602434), unsteered, float64 with
  bfloat16 weight storage (exact; both checkpoints are stored in bfloat16), one GPU each.
- Prefixes: the held-out units 2, 3, 4 and 5 of docs/prefix-scale-prefixes.json (the rule of the 72B test).
- Stage 1 only: 48 temperatures over log10 T -0.4..0.9, tools/hidden_chaos.py line, 1000 transient and 2000 recorded
  steps, absolute twin offsets 1e-5 and 1e-7 (on one GPU the two agreed within 0.02 at every chaotic temperature of the
  Qwen2.5 prompt-blocked test).
- Label: chaotic = exponent above 1e-3 at both offsets and no period up to 64.

## Endpoints and prediction

- C1(model, prefix): chaotic temperatures of the 48; M: the largest min-offset exponent; the chaotic temperatures.
- PSO (primary): C1(32B) > C1(7B) at at least 3 of the 4 prefixes (the paper-prompt ordering holds across prompts).
  A prefix with a missing result for either model counts against the prediction and is reported missing.
- Descriptive: per prefix, C1, M, the chaotic temperatures and the temperature of the largest entropy increase.

## Caveat

The two models are different releases (1124 and 0325), so size and release are confounded; the test asks whether the
ordering is prompt-specific, not why it holds.

## Scoring

Labels as returned, no reruns to change them; a failed run is rerun once with the same settings. Reported in the paper
whether the prediction passes or fails.
