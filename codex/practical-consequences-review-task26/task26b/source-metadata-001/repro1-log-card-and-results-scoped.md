## 2026-10-15 06:51 UTC: run (exploratory): REPRO, do the incidental numerical differences of ordinary inference change what the loop does?

- **Question.** Production inference is not bitwise reproducible: batch composition, the kernel chosen for a matrix shape, and the precision all change the rounding. Is that harmless in periodic regimes and decisive in chaotic ones? The twin method sets offsets by hand. This test uses only the differences that real inference makes.
- **Prompt for the test.** The batched labeller's control, done by the subagent on local CPU in float64 (scratchpad control.json): duplicating the cells in a larger batch changed chaotic exponents by up to 1.6e-5 and decorrelated the chaotic trajectories by step 1500, while non-chaotic cells agreed to rounding.
- **Tool.** tools/repro_noise.py (new; CPU smoke test passed). It runs the same cells twice in lockstep from the same start, differing in one detail:
  - batch64 / batch32: the cells in a batch of n rows, against the same cells followed by the n cells reversed (batch 2n), in float64 or float32;
  - precision: float64 against float32;
  - bf16: the model cast to bfloat16 (state in float32), batch n against batch 2n.
  - Per cell and step: the relative state distance, and whether the two most probable next tokens agree.
- **Cells.** All 448 cells of docs/h2-horizon-cells.json (Qwen2.5-0.5B, philosophy at block 12): 192 chaotic, 192 periodic and 64 other cells of two float64 maps, as labelled by the long H2 test. 1500 steps from the fixed start. One H100 pod, job set repro.
- **Readings, fixed now:**
  - (R1) For each comparison and label: the share of cells whose two runs end with disagreeing most probable tokens (tail of 500 steps), and the median first step at which the distance exceeds 1e-2.
  - (R2) For chaotic cells: compare that step with ln(1e-2 / d_1) / lambda, where d_1 is the first nonzero distance and lambda the H2 exponent.
  - Expected: periodic cells stay at the rounding level and agree throughout; chaotic cells diverge. A periodic cell that ends disagreeing would be reported (for example, a cell near a boundary).

## 2026-10-15 07:09 UTC: REPRO1 result: in float64, a batch-size change alone makes 190 of 192 chaotic cells diverge (median step 102, as the Lyapunov exponent predicts) while 182 of 192 periodic cells stay within 1e-2; in bfloat16 every cell diverges within a few steps

Pod frtqm5urstn2d3 (SECURE H100 NVL; runner rc 0, hash verified, terminated; 560 s; commit dbd125a7). Output repro/h2cells. The 448 cells of docs/h2-horizon-cells.json (192 chaotic, 192 periodic, 64 other; Qwen2.5-0.5B, philosophy at block 12); 1500 steps from the fixed start. Inline analysis in this session.

| comparison | label | cells reaching distance 1e-2 | median step | median final distance | ending with disagreeing tokens |
|---|---|---|---|---|---|
| batch, float64 | chaotic | 190 / 192 | 102 | 0.91 | 90.1 % |
| | periodic | 10 / 192 | (92) | 2.4e-14 | 3.6 % |
| | other | 14 / 64 | (70.5) | 3.6e-13 | 15.6 % |
| batch, float32 | chaotic | 192 / 192 | 20.5 | 0.98 | 92.2 % |
| | periodic | 72 / 192 | 16.5 | 2.3e-5 | 22.9 % |
| float32 vs float64 | chaotic | 192 / 192 | 20 | 0.94 | 92.7 % |
| | periodic | 75 / 192 | 16 | 2.5e-5 | 19.3 % |
| batch, bfloat16 | chaotic | 192 / 192 | 2 | 1.07 | 99.5 % |
| | periodic | 192 / 192 | 2 | 1.01 | 95.3 % |

"Ending with disagreeing tokens" means the most probable tokens disagree at some step of the last 500.
- **float64.**
  - The first nonzero distance between the layouts is about 6e-15, the rounding level.
  - At the chaotic cells, the step at which the distance passes 1e-2 agrees with ln(1e-2 / d_1) / lambda, with lambda the map exponent: median ratio of observed to predicted 0.91, log-log correlation 0.73 over 190 cells. The two chaotic cells that never reach 1e-2 have lambda = 0.013 and 0.0013, where the predicted steps exceed the run.
  - The 182 periodic cells that stay below 1e-2 have a median maximum distance of 2.3e-12.
- **float32.** Every chaotic cell diverges within about 20 steps. 23 % of periodic cells also end with disagreeing tokens.
  - Their float64 labels need not hold for the float32 map.
  - A disagreement can also be a phase shift on the same cycle, or a switch between coexisting attractors. This is not resolved here (tokens were not saved).
- **bfloat16.** The two layouts differ by about 1e-2 after two steps in every cell, and almost all cells end with different tokens, periodic ones included. At this precision the loop is not reproducible across batch layouts anywhere in this steered region, whatever its float64 regime.
- **Reading.** In exact-enough arithmetic, the dynamical regime decides whether rounding-level execution differences grow into different outputs, and the Lyapunov exponent predicts when. In deployment precision (bfloat16), batch-layout noise is large enough to change the trajectory regardless.

