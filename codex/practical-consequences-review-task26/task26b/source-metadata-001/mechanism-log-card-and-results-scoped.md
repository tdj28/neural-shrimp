## 2026-10-15 06:42 UTC: run (exploratory): MECH, why steering makes the loop chaotic: the gain split along steering, steering block and a readout-only logit bias (human: "Let's turn that speculation into solid fact", about 06:30 UTC)

- **Question.** The paper's gain split (tools/gain_split.py, runs G1 and G2) factors each step's twin growth exactly into a network factor g_net and a readout factor g_read, but only on unsteered temperature lines. What does steering change: the readout (token competition), the network, or both? And why does early-block steering carry chaos to low temperature (LS)?
- **Tool.** tools/steer_gain.py (new; tests/test_steer_gain.py, 4 pass). Same split and twin rules as gain_split.py, extended:
  - any grid of (alpha, beta, log10 T), twins at 1e-6 and 1e-9 (chaotic: both above 1e-3, no period up to 64);
  - tangent-free readout statistics per step: entropy; top-2 logit gap divided by T; the largest eigenvalue r1 of the readout Jacobian over the whole vocabulary (power iteration, warm-started); its exact trace and the stable rank trace / r1;
  - an optional logit bias beta * b, with b = W v centred and scaled to unit standard deviation over the vocabulary. This is a readout-only intervention: the network is left untouched.
  - Transient 1000, record 2000 (map pods: 1500), float64.
  - MECH1: alpha lines at blocks 12 and 4 (philosophy; alpha -0.12 to 0.12 in 241, log10 T -0.6, -0.3, 0.0, 0.15, 0.25); the hostile line (alpha 0.0655, log10 T 0.10-0.30 in 201); the 30 KY cells (docs/ky-cells-*.json).
  - MECH2: the same alpha lines at blocks 8, 16 and 20.
  - MECH3a/b: an alpha map at block 12, alpha -0.12 to 0.12 in 101, log10 T -1.0 to 0.3 in 53 rows, split by rows.
  - MECH4a/b: the logit-bias map at alpha = 0, beta -8 to 8 in 129, the same 53 rows.
- **Readings, fixed now** (exploratory, descriptive unless stated):
  - (R1) Which factor carries steering. At each temperature of the alpha lines, compare the chaotic cells with the cells at |alpha| <= 0.01. Report the mean differences of log g_net and log g_read, with per-group counts. If the readout difference is the larger one, steering acts mainly by creating token competition; otherwise mainly through the network.
  - (R2) Token competition. At the same temperature, compare chaotic and periodic cells: mean gap, entropy and r1_rel.
  - (R3) Block. At matched temperature, compare the two factors of the chaotic cells of blocks 4, 8, 12, 16 and 20. No direction is predicted.
  - (R4) Readout-only intervention: chaotic share of the beta map and the count of chaotic cells.
    - If the share is at least 0.5 % and at least 20 cells, then creating token competition at the readout suffices for chaos, given the unmodified network.
    - If it is below that, the network change made by residual steering is necessary.
    - Also compare its chaotic cells' factors with those of the alpha map.
  - (R5) Dimension. At the KY cells, Spearman correlation of D_KY with the mean stable rank of the readout.
  - Would surprise us: chaotic cells with g_read below its alpha = 0 value at the same temperature.

## 2026-10-15 07:19 UTC: MECH4 result: a pure logit bias, with the network untouched, makes 3.5 % of the plane chaotic (R4 met); its chaos lies at higher temperature than that of residual steering

Pods v6zndolr40ecev, 026rfinp1b3h8d (runner rc 0, hashes verified, terminated; commit b203fb56). Output mech/bmap-{a,b}: alpha = 0, logit bias beta * b with b = W v (philosophy at block 12, centred and scaled to unit standard deviation over the vocabulary), beta -8 to 8 in 129 steps, log10 T -1 to 0.3 in 53 rows, 6837 cells.
- **R4 (frozen).** 241 chaotic cells, 3.5 % of the plane. The threshold (at least 0.5 % and 20 cells) is met. Creating token competition at the readout suffices for chaos, given the unmodified network.
- **Comparison.** Residual steering at block 12 on the same temperatures (alpha map, same protocol): 331 of 5353 cells, 6.2 %.
  - The logit bias's chaos lies mostly at log10 T 0.1-0.3, with only a few cells near and below log10 T 0.
  - Residual steering's chaos reaches log10 T -0.2 (fig-mechanism panel d).
  - This continues the block trend of LS and L70D. The later the push enters, the higher the temperature its chaos needs; a logit bias is the latest possible entry point.
- **Statistics** (tools/plot_mechanism.py, committed with this entry; output paper/llm/figures/fig-mechanism.json), alpha map, row-matched over 18 temperature rows:
  - chaotic minus periodic, with bootstrap 95 % intervals over rows and row-stratified permutation tests (10,000):
    - log readout factor +1.36 [0.92, 1.92], p = 1e-4;
    - log network factor +0.23 (p = 1e-4), a smaller shift;
    - entropy +1.62 [1.26, 1.91] nats, p = 1e-4;
    - top-2 gap / T -1.37 [-1.98, -0.87], p = 1e-4;
  - chaotic minus |alpha| <= 0.01 cells: readout +2.58 [1.79, 3.58], network -0.07 [-0.21, 0.11].

