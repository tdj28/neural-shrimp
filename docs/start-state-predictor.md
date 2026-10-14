# Loop-free cheap predictor of the loop's chaos (run CP): frozen before the cross-model results

14 October 2026, about 10:20 UTC. The human asked whether the measured decline of chaos with size gives a solid
reason to expect larger models to be stable, and asked for the scaling series, the gain split and a test of cheap
predictors. The scaling series showed that size does not order the chaos (OLMo-2-32B is the most chaotic OLMo; frozen
P1 and P2 fail for OLMo-2). An exploratory, adapted analysis found that the breadth of the intermediate-entropy regime
measured on the loop's own orbits orders the 15 models' chaos (Spearman +0.69) better than parameter count (-0.46),
but that measure is partly circular. This test asks whether a property read off one forward pass, without running the
loop, does the same.

## Predictor (tools/start_state.py)

At the loop's start state (the prefix, then the last prefix token's embedding as the first soft token): the entropy
H(T) of the start logits on log10 T in [-1.5, 1.5] (601 points); breadth B = decades of log10 T with 1 < H < 10 nats;
also the network's relative sensitivity g_net (median over 8 random directions, relative twin offset 1e-5) and the
one-step gain max_T [log g_net + log g_read(T)] (log10 T in [-1, 1]). Computed for the ten prefixes of
docs/prefix-scale-prefixes.json; index 0 is the paper's prefix, the one used by every unsteered sweep.

## Models

The 14 dense models of the scaling table (tools/scale_summary.py): Qwen2.5 0.5B, 1.5B, 3B, 7B, 14B, 32B, 72B;
Llama-3.1 8B and 70B; Llama-3.3 70B; OLMo-2 1B, 7B, 13B, 32B; pinned revisions as in their sweeps; float64 compute
with bfloat16 weight storage. The mixture of experts (Qwen3.5-35B-A3B) is excluded before any result: its float64
path (tools/qwen35_float64.py) has not been tested with this probe. Development: Qwen2.5-0.5B was run on the local
CPU (native float64 weights) while the tool was written, before this file (prefix 0: B = 0.25 decades, log g_net
-1.47, one-step gain max -1.34); its pod value is the one scored.

## Outcome

Chaos of each model = decades of log10 T labelled chaotic in its unsteered sweep (exponent > 1e-3 and no period,
spacing-weighted; tools/scale_summary.py sweeps): Qwen2.5 0.018, 0.231, 0.030, 0.165, 0.050, 0.010, 0.000; Llama-3.1
0.244, 0.031; Llama-3.3 0.000; OLMo-2 0.050, 0.081, 0.000, 0.252 (already known; computed before this file).

## Predictions

- CP1 (primary): Spearman correlation across the 14 models between B at prefix 0 and the chaos decades is at least 0.5.
- CP2: the same with B the median over the ten prefixes, at least 0.5.
- CP3 (descriptive, no direction predicted): Spearman of the chaos decades with log g_net at prefix 0, with the
  one-step gain maximum at prefix 0, and with log parameter count (-0.46 on 15 models; recomputed on the 14).

## Scoring

Values as returned; a failed model is rerun once with the same settings and otherwise reported missing (never
imputed); the correlation is then computed on the models present and the count is stated. Reported in the paper
whether the predictions pass or fail.
