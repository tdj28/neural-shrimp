# Scaling series: predictions frozen before the new models are measured (run SC)

14 October 2026, about 08:30 UTC. Written after the 70B-class results (B1, B2, S70M, G1, G2 partial) and before any
result of the models below exists. The human asked for a definite answer in the main paper on how the loop's chaos
depends on model size; this file fixes the test.

## Models measured in run SC (new)

Qwen2.5-1.5B, -3B, -14B, -32B (Instruct); OLMo-2 1B (0425), 13B (1124), 32B (0325) (Instruct). Llama-3.2-1B and -3B
are gated and not yet accessible. Each: an unsteered float64 sweep (160 temperatures over log10 T -1..0.6, 1000 +
2000 steps, twin 1e-6, max period 64; chaotic = exponent > 1e-3 and no period), and the gain split
(tools/gain_split.py; 24 temperatures over -0.3..0.9). Existing: Qwen2.5-0.5B, -7B, -72B; OLMo-2 7B; Llama-3.1-8B,
-3.3-70B, -3.1-70B (running); Qwen3.5-35B-A3B.

## Outcome measure

Unsteered chaotic share of the temperatures in log10 T [-1, 0.3] (the range of the existing sweeps; the wide range is
reported too).

## Predictions

P1 (decline above 7-8B). Within each family, every model above 8B parameters has a lower chaotic share than the
family's 7-8B member: Qwen2.5-14B and -32B below Qwen2.5-7B (65/512 = 12.7 %); OLMo-2-13B and -32B below OLMo-2-7B
(32/512 = 6.3 %).
P2 (ordering above 7B). Within each family, the chaotic share does not increase with size above 7-8B
(Qwen 7B >= 14B >= 32B >= 72B; OLMo 7B >= 13B >= 32B), ties at zero allowed.
P3 (mechanism). At matched next-token entropy (bins 1-3, 3-6, 6-10 nats), the medians of both gain factors of the
14B and 32B models are at or below those of their family's 7B model.
No prediction is made for the models below 7B (the 0.5B is chaotic in a narrow band only, so small size alone does
not order the share).

## Scoring

Each prediction is scored on the sweeps as returned (no reruns to change a label), and reported in the paper whether it
passes or fails, with every model's numbers.
