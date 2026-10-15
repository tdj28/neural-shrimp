## 2026-10-15 07:08 UTC: DSS4 result: the period-6 hook has no superstable curve at all; its leading multiplier never falls below 0.028 in modulus

Pods 6lini88kmg7dxs, adx6wj2vsij4gd (runner rc 0, hashes verified, terminated; 3628 and 3199 s; commit ec5c0e30). Output dss/p6-hook2-{a,b}: --multipliers-only, period 6, philosophy at block 12, alpha 0.0782-0.0812, log10 T -0.076 to -0.046, 31 x 31.
- **Coverage.** The orbit continued to 675 of 961 cells (372 with the seed's token loop) and is attracting (|mu_1| < 1) in 303.
- **Multipliers.** The leading modulus over the continued cells has quantiles 0.028 / 0.14 / 1.28 / 4.19 / 5.82. Its minimum, 0.028, sits in the valley of strongest contraction, which runs diagonally through the attracting region.
  - 55 cells have a complex leading pair, mostly along the edge where the orbit loses stability.
  - Elsewhere the second modulus is small (median ratio |mu_2|/|mu_1| = 0.021).
- **Reading.** No cell of the continued orbit comes near mu = 0, so the hook carries no superstable curve and no shrimp centre of any kind for this orbit. Its attracting region is bounded by multiplier crossings, partly through complex pairs. Together with DSS3 (seeds with complex or comparable multipliers) and the coexistence result at the strip crossing, the period-6 hook is not a shrimp.
- Inline analysis in this session.

