## 2026-10-15 07:14 UTC: run (exploratory): XO, the orbit-preserving four-arm crossover (consult 9, direction 2), and FB extensions toward the asymptote (human: "Yes I'd like them to approach the asymptote")

- **XO.** tools/crossover.py (new; CPU smoke test: arm 11 reproduces the twin exponent exactly, -0.132 at a short-transient chaotic cell). Along the steered reference orbit it carries four tangent dynamics:
  - arm 11 = R(p_s) J_s; arm 10 = R(p_s) J_u; arm 01 = R(p_u) J_s; arm 00 = R(p_u) J_u;
  - J_u and p_u come from the unsteered model at the same states, with its own unsteered prefix cache.
  - Cells: docs/xo-cells.json, a seeded draw (seed 20261015) fixed before any XO data. 16 chaotic cells of the block-12 alpha map (two per temperature band of eight; both twin exponents > 0.05), and 8 attracting periodic cells of the same rows. The first two chaotic cells are development cells.
  - Transient 1000, record 1000 in windows of 250. One H100 pod.
  - **Readings fixed now:**
    - Validation: arm 11 must match the twin exponent at every cell within 0.02.
    - Per chaotic cell, report arms 10 - 00 (readout effect), 01 - 00 (network effect) and 11 - 10 - 01 + 00 (interaction), with means and 95 % intervals by a bootstrap over held-out cells.
    - Readout sufficiency if arm 10 is within 0.1 nats per step of arm 11 on average and its interval excludes arm 00.
- **FB extensions.** On three H100 pods; the root acceptance is set above the measured floors (--g-accept 1e-4).
  - **FBS768m:** the superstable point s768 at log10 T -0.14798. Bracket [0.0661249646696678, 0.0661249836696678] around the estimate s384 + (s384 - s192)/4.65. Band positions are scaled from -0.156, where s sits at 0.519 of its band.
  - **FBC148:** flips at -0.14798 from period 96: start s96, next s192, ratio guess 4.62, 3 flips.
  - **FB15D:** the 1.5B route (alpha 0.1, philosophy at block 14, free log10 T) from period 64: start -0.0018311, next -0.0018207, ratio guess 4.3, 3 flips.

## 2026-10-15 07:21 UTC: XO result: along the same steered orbits, steering's change of the readout creates the expansion; its change of the network response damps it

Pod slk055l1wkl6hq (COMMUNITY H100 NVL; runner rc 0, hash verified, terminated; 270 s; commit 92f1a20e). Output xo/cells: 24 cells of docs/xo-cells.json, transient 1000, record 1000. Statistics from tools/plot_mechanism.py (paper/llm/figures/fig-mechanism.json; bootstrap over held-out cells, 10,000).
- **Validation (frozen: arm 11 within 0.02 of the twin exponent at every cell).** Met at 23 of 24 cells, all within 0.012. Missed at one periodic cell contracting at about -12 per step (tangent -12.12, twin -11.21): there the twin at relative offset 1e-9 shrinks to the float64 rounding floor each step, so the twin, not the tangent, is limited.
- **Labels.** One development cell labelled chaotic by the alpha map (log10 T -0.400, alpha -0.0624) is strongly contracting in this run (exponent -4.0). The batch differs from the map's, so this is a transient or a coexisting attractor. It is a development cell and enters no estimate.
- **Contrasts.** 14 held-out chaotic cells, nats per step:

| contrast | mean | 95 % interval |
|---|---|---|
| readout effect, arm 10 - arm 00 | +0.81 | [0.29, 1.35] |
| network effect, arm 01 - arm 00 | -0.49 | [-0.63, -0.35] |
| interaction, 11 - 10 - 01 + 00 | +0.26 | [0.10, 0.45] |
| arm 11 - arm 10 | -0.23 | [-0.41, -0.03] |

- **Split by temperature (descriptive):**
  - log10 T < 0 (6 cells): arm means 00 -1.03, 10 +0.72, 01 -1.62, 11 +0.34. Readout effect +1.75, network effect -0.58. The steered readout alone turns a contracting derivative into an expanding one, and does so more strongly than the full steered map.
  - log10 T >= 0 (8 cells): arm means 00 +0.41, 10 +0.51, 01 -0.01, 11 +0.40. At these states the unsteered derivative already expands, so steering acts by moving the orbit into competitive states, not by changing the derivative there.
- **Frozen sufficiency criterion** (arm 10 within 0.1 of arm 11 on average, interval excluding arm 00). The second part holds. The first fails by overshoot: arm 10 exceeds arm 11 by 0.23, because the steered network response partly damps the growth.
- **Reading.** At low temperature, steering destabilises the loop through the readout: the token competition it creates at fixed states. Its effect on the network response works against the instability. Together with MECH3 (the readout factor carries the difference; network flat) and MECH4 (a logit bias alone gives chaos), the answer to "why does steering induce chaos" is: it creates competition between next tokens, which turns the softmax readout from a contraction into an expansion.

