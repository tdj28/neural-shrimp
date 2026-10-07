# Criticality of the period-2 Neimark-Sacker curve (exploratory, 7 October 2026)

Question: is the edge where the edge-of-stability period-2 orbit loses stability
soft (supercritical Neimark-Sacker: quasi-periodic training is born continuously)
or hard (subcritical: the orbit is surrounded by a repelling closed curve, and
training jumps away), and does the hard part coincide with the divergence
boundary seen in fixed-start maps? All results are exploratory.

## Method

- `src/neural_shrimp/criticality.py`: Kuznetsov's normal-form coefficient for
  n-dimensional maps, c1 = 1/2 <p, C(q,q,qbar) + 2B(q,(I-A)^{-1}B(q,qbar)) +
  B(qbar,(mu^2 I - A)^{-1}B(q,q))>, first Lyapunov coefficient d = Re(conj(mu) c1),
  |q| = 1, <p, q> = 1. d < 0: supercritical; d > 0: subcritical. B and C come from
  Richardson-extrapolated central differences of the exact two-step
  Jacobian-vector product of gradient descent; A is the exact dense two-step
  Jacobian.
- Validation (`tests/test_criticality.py`): equals the explicit planar formula
  (times the |q| = 1 normalization factor 2) to 1e-10 on random planar maps with
  quadratic and cubic terms; finite-difference forms agree with exact forms to
  1e-7; the sign predicts soft vs hard loss of stability in simulation.
- Gradient-map control (`tools/explore_criticality.py control`): on the
  engineered planar gradient map with a prescribed 2-cycle
  (`neural_shrimp.minimal.EngineeredTwoCycle`) the circle's normal-form
  amplitude matches sqrt(log(1 + delta) / -d) to 0.9% at delta = 1e-5 and 0.05%
  at delta = 1e-6 (kappa = -2). The kappa term adds only non-resonant cubic
  terms (z zbar^2) to the two-step map, so it does not change d, contrary to the
  docstring's claim that it tunes criticality; at delta = 1e-3 its large
  non-resonant terms still change the amplitude through higher-order effects.
- Curve points: for each point of the merged S2-28011 trace the period-2 orbit
  is re-found below the boundary, continued to it (bisection 1e-10) and d is
  evaluated at the last attracting point; a second finite-difference step size
  agrees to all printed digits; eigen-gaps 0.88-1.01; no strong resonance.

## Results

- **S2-28011, 69 curve points:** d < 0 (supercritical) from log10 output -0.4121
  to -0.0026 (most negative, -5.08, near -0.062 next to the birth-rotation
  maximum) and at +0.0605; d > 0 (subcritical) for output <= -0.4215 and from
  +0.0132 to +0.0447. Three sign changes, generalized Neimark-Sacker (Chenciner)
  points: near -0.417, +0.005 (bottom of the second dip) and +0.05.
- **Agreement with fixed-start maps:** the quasi-periodic band beyond the curve
  (output -0.37 to -0.046) lies on the supercritical arc; the far left, where
  maps turn chaotic immediately, is subcritical for output <= -0.42.
- **The "hard" arc of the maps is a basin effect, not subcriticality:** from
  -0.034 to +0.06 the fixed-start map is divergent on both sides of the curve,
  including just below it where the period-2 orbit is stable, and d is negative
  on part of this arc (-1.81 at -0.034, -1.33 at -0.018). Starting near the
  2-cycle, a small attracting circle with square-root amplitude scaling exists
  beyond the curve there. The hypothesis that the trainability boundary
  coincides with the Neimark-Sacker curve on that arc is not supported.
- **Dynamical checks:** see the table below (corrected run, distances relative to
  the Newton-refined 2-cycle at each offset).
- **S1R-28002 (main's trace, 16 points):** d < 0 at 15 points (-0.064 to -1.57);
  d > 0 (+0.0375) at output 1.1414, a point with birth rotation 0.043 unlike the
  0.12-0.13 of the main arc. Both types occur; soft dominates.

### Dynamical checks (corrected run)

Offsets are in log10 hidden rate beyond (or before) the curve; w is the
normal-form amplitude |<p, x - x_c>| of the attracting circle about the
Newton-refined (unstable) 2-cycle, predicted as sqrt(log|mu| / -d); before the
curve, the escape threshold is the smallest perturbation along the critical
direction that does not return, with the predicted repelling-circle amplitude
in parentheses when d > 0.

| log10 output | d | beyond: status, measured w / predicted w (offset 1e-4, 4e-4, 1.6e-3) | before: escape threshold (predicted unstable circle) |
|---|---|---|---|
| -0.4405 | +0.081 | w 0.4646 / n/a (max dist 4.272); w 0.4622 / n/a (max dist 4.195); w 0.4648 / n/a (max dist 4.669) | 0.0335 (0.0217); 0.0636 (0.0433); 0.1148 (0.0865) |
| -0.2982 | -0.243 | w 0.0110 / 0.0111 (max dist 0.021); w 0.0221 / 0.0221 (max dist 0.042); w 0.0442 / 0.0443 (max dist 0.091) | >0.3; >0.3; >0.3 |
| -0.1844 | -0.468 | w 0.0083 / 0.0083 (max dist 0.015); w 0.0167 / 0.0167 (max dist 0.030); w 0.0333 / 0.0334 (max dist 0.062) | >0.3; >0.3; >0.3 |
| -0.0800 | -1.076 | w 0.0078 / 0.0078 (max dist 0.014); w 0.0155 / 0.0156 (max dist 0.028); w 0.0309 / 0.0313 (max dist 0.060) | >0.3; >0.3; >0.3 |
| -0.0342 | -1.813 | w 0.0102 / 0.0102 (max dist 0.018); w 0.0203 / 0.0203 (max dist 0.037); w 0.0405 / 0.0406 (max dist 0.089) | >0.3; >0.3; >0.3 |
| -0.0184 | -1.331 | w 0.0106 / 0.0106 (max dist 0.018); w 0.0212 / 0.0212 (max dist 0.040); w 0.0422 / 0.0424 (max dist 0.095) | >0.3; >0.3; >0.3 |
| -0.0026 | -0.109 | w 0.0224 / 0.0224 (max dist 0.039); w 0.0447 / 0.0448 (max dist 0.082); w 0.0845 / 0.0895 (max dist 0.202) | >0.3; >0.3; 0.2073 |
| +0.0132 | +0.111 | w 0.1337 / n/a (max dist 0.279); w 0.1351 / n/a (max dist 0.282); w 0.1395 / n/a (max dist 0.294) | 0.0335 (0.0205); 0.0636 (0.0410); 0.1267 (0.0820) |

- Supercritical points: amplitudes match the prediction within 1% (5.6% at the
  largest offset next to the Chenciner point near +0.005); no small escape
  threshold before the curve.
- Subcritical points: beyond the curve trajectories jump to distant states
  independent of the offset; before it the escape threshold scales like the
  square root of the offset and tracks the predicted repelling circle up to a
  constant normalization factor of about 1.5.
- Hysteresis (sweep up then down, carrying the state; first run, amplitudes
  relative to the boundary orbit point, valid for near/far): at the subcritical
  points the state jumps to a distant attractor just past the curve and stays
  there below it (amplitudes 0.54-1.49 at -0.44, 0.17-0.26 at +0.013); at the
  supercritical points the circle grows and shrinks back reversibly.

## Readiness for confirmation

The local statement is ready to freeze as a prediction on fresh losses: the
computed sign of d at each Neimark-Sacker point predicts (i) d < 0: started near
the 2-cycle just beyond the edge, training settles on a circle with normal-form
amplitude sqrt(log|mu| / -d) within a frozen tolerance; (ii) d > 0: it jumps
away, with escape thresholds before the edge scaling like sqrt(offset) and
hysteresis. It is not ready as a statement about trainability maps from a
shared initialization, which mix criticality with basin accessibility.

## Caveats

- d is a local, third-order quantity; it predicts the birth of a small circle,
  not how far the circle survives. On the far-left supercritical stretch
  (-0.41 to -0.38) maps show chaos almost at once beyond the curve.
- Chenciner points are located only to the spacing of the trace (0.01 in log10
  output on the left, 0.016 on the right).
- Fixed-start maps mix local criticality with basin accessibility from one shared
  initialization; criticality statements here concern the local bifurcation.

## Files

- Code: `src/neural_shrimp/criticality.py`, `tests/test_criticality.py`,
  `tools/explore_criticality.py`, `tools/analyze_criticality.py`.
- Data (outside Git): `/data2/neural-shrimp-data/explore/criticality/`
  (`s2-28011-curve/curve.json`, `s2-28011-steep/curve.json`,
  `s1r-28002-curve/curve.json`, `boundary.json`, `criticality.png`,
  `s2-28011-dynamics/`, `control/`).
