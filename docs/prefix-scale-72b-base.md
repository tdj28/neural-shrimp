# Base against Instruct at 72B (run PB72): frozen before any base-model run

14 October 2026, about 17:05 UTC. Qwen2.5-72B-Instruct is nearly free of unsteered chaos on five prompts: the frozen
test PS72 (docs/prefix-scale-72b.md) found it chaotic at 0, 1, 0 and 0 of 48 temperatures on held-out prefixes 2-5,
against 14, 9, 11 and 15 for Qwen2.5-14B-Instruct. Llama-3.1-70B and Llama-3.3-70B, which differ only in post-training,
differ in unsteered chaos. This test asks whether the 72B's near-absence of chaos comes from post-training: the same
measurement on the pretrained base model of the same size and architecture. No base-model run exists at this commit.

## Design

- Model: Qwen/Qwen2.5-72B (base), pinned revision efba10c8e54e91e0d9570ab5f7b51a958474d4cb; unsteered; float64 with
  bfloat16 weight storage over two GPUs, with the float64 hook fix of 14 October (commit 6a4775ae). Its chat template
  gives the same prefix token ids as the Instruct model's for all prefixes used (checked before the run).
- Prefixes: 2, 3, 4, 5 of docs/prefix-scale-prefixes.json (the PS72 prompts) and 0 (the paper's prompt, descriptive).
- Stage 1 as PS72: tools/hidden_chaos.py line, 48 temperatures over log10 T -0.4..0.9, absolute twin offsets 1e-5 and
  1e-6, but 500 transient and 1000 recorded steps (half of PS72's, for cost).
- Label: chaotic = exponent above 1e-3 at both offsets and no period up to 64 (hidden_chaos.classify).

## Endpoints and prediction

- C1(base, prefix): chaotic temperatures of the 48; M: the largest min-offset exponent.
- PB72 (primary): C1(base) >= C1(Instruct) + 3 on at least 3 of the 4 prefixes 2-5, with C1(Instruct) from PS72
  (docs/prefix-scale-72b-result.json). A missing base result counts against the prediction.
- Descriptive: per prefix, C1, M, the chaotic temperatures, the temperature of the largest entropy increase and the
  intermediate-entropy breadth (decades of the grid with mean entropy between 1 and 10 nats), for base and Instruct;
  prefix 0 against the Instruct model's sweeps on the paper's prompt (B1, B4).

## Reading fixed now

- PB72 holds: the near-absence of unsteered chaos in Qwen2.5-72B-Instruct is attributable to post-training, not to
  size, for this checkpoint pair.
- PB72 fails with the base also nearly free of chaos: at 72B the pretrained model already suppresses it on these
  prompts; post-training is not needed for the suppression.
- Caveats: one checkpoint pair; PS72 used 1000 + 2000 steps and the two-GPU computation before the fix (its labels
  need both offsets positive, and the 1e-5 offset is above that computation's noise).

## Scoring

tools/prefix_pair_score.py pb72, written before the data. Labels as returned; a failed run is rerun once with the same
settings. Reported in the paper whether the prediction passes or fails.
