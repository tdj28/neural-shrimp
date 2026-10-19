# Consult 18 adjudication: why the superstable curves of the language-model shrimps avoid crossing (19 October 2026)

Review: `review.md` (gpt-6-astra, research-note-scientific, completed 19 October 2026 about 05:35 UTC; raw response and
manifest in `../neural-shrimp-data/consults/2026-10-19-avoided-crossing/execute-001/`). Verdict: not ready; the
numerical observations merit preservation, the proposed mechanism is not established. We agree on the main points.

| Finding | Decision |
|---|---|
| B01: the avoided-crossing mechanism (delta = eps s_a s_b - kappa, hyperbolic unfolding) is not established; det DF^p = prod det DF(x_i) holds in any dimension, so dimension alone does not force avoidance; the Henon map with b != 0 has no zero multiplier | Accepted. The papers give the consult's conditional geometric model, labeled as a model, not a causal explanation; the one-dimensional comparator is stated with distinct critical points |
| B02: the two-mode reduction fails at the key points (three comparable moduli: 0.0063, 0.0063, 0.0051 at the DSS6 minimum; 3.4-3.6e-3 at the period-5 head); delta = 0 marks a zero eigenvalue of the selected block, not of the leading multiplier | Accepted. Text distinguishes the spectral radius (leading modulus), the tracked block and its determinant; no validated two-mode explanation is claimed |
| B03: grid proxies are not resolved zero curves (529 of 961 cells solved; the 7/38 split is an endpoint rule) | Accepted. "Nonzero at all solved cells"; the classification is reported as an endpoint rule; contours are labeled as interpolated |
| B04: at R the multipliers -0.9952 and -1.0005 miss -1 | Reconciled: R is the intersection of the two traced curves (point sequences), whose nearest points lie 4.1 and 0.7 grid steps away (HR9 steps); each curve point is gated (critical multiplier within 1e-7 of -1). The orbit distance is the largest point distance after the best of the six cyclic alignments. The papers say "near the point where the two flip curves cross" and give both multipliers |
| B04: do not infer two coexisting attractors or a completed kite | Accepted. Coexistence and the kite's cusp are stated only as far as computed (the KITE continuations are running) |
| S01: census accounting (1946 + 945 + 811 = 3702 of 4224) and extrema labels | The follow's member classes are period 6: 1946, periods 12/24/48: 945, other periods: 35, chaos behind the text: 811, no period and exponent < 0.005: 487 (total 4224). The "most contracting cell" of the HR9 scan (0.073097, -0.15204; leading modulus 0.0161) and the DSS6 grid minimum (0.073053, -0.15120; 0.0063) are different searches; the papers name the search |
| S02-S04: specify the system, numerical criteria, certified orbits; window comparisons are not matched controls | Already in the article (Secs. II, IV, VI; the certified orbits are named there); the steering construction is now explicit (How we steer). Window comparisons are described as analogies |
| R01: Gallas sources | Verified: Gallas, PRL 70, 2714 (1993) for the shrimps of the Henon map; Gallas, Physica A 202, 196 (1994), Fig. 7 (journal p. 217) for the cell, the kite C-A-R-B and the tail T as a "pseudo-intersection" |
| F01: figures | The period-6 figures (HR10 close-up with curves, DSS6 invariants) show coverage and label contours as interpolated |

Text adopted (adapted from the consult's replacement, under 180 words): see the article's "A second shrimp" paragraph
and the report's period-6 section.
