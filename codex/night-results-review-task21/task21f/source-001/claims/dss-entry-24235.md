## 2026-10-15 06:06 UTC: run (exploratory): DSS4, the period-6 hook as a two-dimensional map: valleys of the leading multiplier modulus

- **Why:** DSS3 showed that the hook's period-6 orbits have two multipliers of similar size, at times a complex pair, so superstable curves are not defined. In two-dimensional dissipative maps the skeleton of a shrimp is the set of curves of strongest contraction (valleys of the leading multiplier modulus). A shrimp centre is where two such valleys cross.
- **Tool:** tools/doubly_superstable.py grid --multipliers-only (new). It accepts a complex or unclean leading multiplier at the seed and records the three leading multipliers (real, imag) per cell of the continued orbit.
- **Grid:** period 6, philosophy, layer 12. alpha 0.0782-0.0812 in 31 columns, log10 T -0.076 to -0.046 in 31 rows, split by rows on two pods:
  - DSS4a: rows 0-15, log10 T -0.076 to -0.061, seed (0.080667, -0.065059);
  - DSS4b: rows 15-30, log10 T -0.061 to -0.046, seed (0.080196, -0.054118).
  - Both seeds are period-6 cells of zoom Z3 with exponent -0.35 and -0.45.
- **Reading:** descriptive. The map of the leading modulus, the region of complex pairs, and whether two valleys of the modulus cross inside the body. Any crossing is a candidate centre for a finer, locating follow-up, not a claim.

