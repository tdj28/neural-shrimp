## 2026-10-15 04:21 UTC: run (exploratory): KY, Kaplan-Yorke dimensions of the chaotic attractors (human: "Full go")

- **Question:** How far is a chaotic attractor of the soft-token map from one-dimensional? The Kaplan-Yorke dimension D = j + (lambda_1 + ... + lambda_j) / |lambda_{j+1}| (j the largest index with a nonnegative partial sum; D = 1 + lambda_1 / |lambda_2| for one positive exponent) measures it. The paper's theory paragraph separates the exact local reduction from a global scalar return map; D says where the global one holds.
- **Already measured, no new run:** the six chaotic TC cells (run TC, hidden/tc-spectrum, four exponents by QR, 1000 + 2000 steps) give D = 1.063, 1.196, 1.239, 1.322, 1.325 and 1.645. The values rise with lambda_1 (0.063 to 0.415), while lambda_2 is -0.64 to -1.10.
- **Cells, chosen by a rule fixed before any spectrum of them:** in each of three groups, the chaotic cells at the 5th, 15th, ..., 95th percentile of the exponent (nearest rank, ties by index), 10 per group.
  - high: the high-temperature map map-highT-256-families (log10 T -0.02 to 0.14). Chaotic means the map label: exponent above 1e-3 and no period.
  - head: the H2 long-test chaotic cells of the headline maps (hidden/h2-cells).
  - host: the hostile temperature line hidden/tline-a0.0655, where the paper fits a scalar return relation (visible or hidden chaos).
  - Files: docs/ky-cells-philosophy.json (high, head) and docs/ky-cells-hostility.json (host).
- **Setup:** two pods (H100-class, new job set spectrum). tools/lyapunov_spectrum.py, float64, k 4, transient 1000, record 2000, checkpoints 1000 and 2000. KY1 runs the philosophy cells, KY2 the hostility cells (vectors-l12-p1b).
- **Reading:** D per cell, with its two checkpoints, reported by group. A cell that is not chaotic at this horizon (lambda_1 <= 0) is reported as such. No threshold is set: the result is the distribution of D, which goes into the paper's theory paragraph.
- **Output:** `../neural-shrimp-data/explore/llm-soft-loop/pod64/hidden/spectrum-ky-philosophy`, `spectrum-ky-hostility`


## 2026-10-15 05:08 UTC: KY2 result: the hostile band's chaotic attractors have Kaplan-Yorke dimensions 1.008-1.156

Pod cjlui0q6npnllj (SECURE H100 NVL; runner rc 0, hash verified, terminated; 2018 s; commit f8219106). Output hidden/spectrum-ky-hostility: four exponents by QR, 1000 + 2000 steps, float64, checkpoints at 1000 and 2000 steps.
- **The 10 percentile cells of the hostile temperature line (alpha 0.0655):**
  - 8 are chaotic at this horizon, with D = 1.008, 1.013, 1.013, 1.045, 1.050, 1.052, 1.100 and 1.156 (lambda_1 0.008-0.141; lambda_2 -0.85 to -1.31). At 1000 steps D differs by at most 0.06 (1.111 against 1.052 at log10 T 0.18984), and by at most 0.02 in the other cells.
  - 2 (log10 T 0.17968 and 0.18005) are not chaotic at this horizon: lambda_1 -0.007 and -0.009, with a repeated pair -0.35, -0.35 next. They are reported as such.
- **Reading:** where the paper fits a scalar return relation (the chaos behind a repeating text), the chaotic attractors are within 0.16 of one-dimensional. At the six TC cells of the headline maps, D is 1.06-1.65.


## 2026-10-15 05:25 UTC: KY1 result: the dimension of the chaos grows with temperature, from about 1 to about 2; never more than one positive exponent

Pod 3en7yg96v92ysm (COMMUNITY H100 NVL; runner rc 0, hash verified, terminated; 3751 s; commit f8219106). The 60-minute tool limit stopped it after 18 of 20 cells: the last two head cells are missing, reported as such. Output hidden/spectrum-ky-philosophy.
- **High-temperature group** (map-highT-256, log10 T -0.02 to 0.13):
  - 9 of 10 cells are chaotic, with D = 1.373, 1.413, 1.605, 1.683, 1.812, 1.858, 1.973, 2.009 and 2.015 (lambda_1 0.21-0.52, lambda_2 -0.32 to -0.60).
  - At three cells lambda_1 + lambda_2 >= 0 (D >= 2). 1000-step values differ by at most 0.06.
  - One cell (log10 T 0.0101) is not chaotic at the horizon (lambda_1 -0.038).
- **Headline group** (H2 chaotic cells): 8 of 8 measured cells are chaotic, D = 1.038-1.368. With the six TC cells, the headline maps give 1.04-1.65.
- **Reading:**
  - Every measured attractor has exactly one positive exponent (no hyperchaos).
  - D grows with temperature: hostile band 1.01-1.16, headline maps 1.04-1.65, high-temperature region 1.37-2.02. The global scalar reduction holds for the low-temperature chaos and fails in the high-temperature region, consistent with the high-temperature bubble's second multipliers (0.28-0.38).
  - Paper: the dimension paragraph of the theory section.

