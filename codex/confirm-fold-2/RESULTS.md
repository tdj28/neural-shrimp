# PT2 results: fresh fold passages with the continuous passage integral as primary (15 October 2026)

**Freeze.** Design `design.json`, sha256 5a8ac959..., pushed in commit 6f1a53b6. The push was verified at 23:48:53 UTC,
before the first target evaluation (C at 23:48:57, D at 23:49:03). Hashes of every receipt, result and trace are in
`results-receipt.json`. The audit (`audit.py`, frozen with the design) reconstructed every first strict exit from the
saved traces and reproduced every frozen prediction.

**Targets.** Both fresh; setup history:

| target | setup | fold | a | b | U |
|---|---|---|---|---|---|
| C, 2-cycle fold, log10 T = -0.200 | setup-C-01 qualified | alpha_c = 0.071369030053 | 10.95048 | 25.33001 | 0.0039479 |
| D, period-6 fold, log10 T = -0.087439 | setup-D-01 failed (corrector, retained); setup-D-02 qualified after a pre-outcome amendment | alpha_c = 0.075632912551 | 72.69497 | 72.23677 | 0.0013843 |

## Verdicts

- **Primary**, |N - T_arc| <= 1 on every row: **fails, 18 of 24.**
  - Target C passes all 12 rows (errors +0.26 to +0.86). The measured count equals the ceiling of T_arc in every C row.
  - All six failures are target D's full-channel and kicked passages at q <= 0.05: errors -2.29, -2.29, -4.29, -3.29,
    -5.74, -4.74. The deficit grows as eps decreases.
  - D's midpoint passages pass (errors -0.20 to +0.63).
- **Secondary**, |N - N_quad| <= 1 on every row: **fails, 16 of 24.** The failures are C-3-midpoint (-2), the
  reproduction at a fresh fold of Task 5's failing configuration, plus seven D rows.
- **Secondary sign hypothesis**, N - N_quad <= 0 on midpoint and kicked rows: **holds on all 16** (residuals -6 to 0).

## Rows

| row | q | eps | measured | T_arc (primary) | error | discrete (secondary) | error |
|---|---|---|---|---|---|---|---|
| C-0-full | 0.08 | 2.307e-05 | 23 | 22.401 | +0.599 | 23 | +0 |
| C-0-midpoint | 0.08 | 2.307e-05 | 12 | 11.201 | +0.799 | 12 | +0 |
| C-0-full+transverse | 0.08 | 2.307e-05 | 23 | 22.401 | +0.599 | 23 | +0 |
| C-1-full | 0.05 | 9.013e-06 | 45 | 44.286 | +0.714 | 45 | +0 |
| C-1-midpoint | 0.05 | 9.013e-06 | 23 | 22.143 | +0.857 | 23 | +0 |
| C-1-full+transverse | 0.05 | 9.013e-06 | 45 | 44.286 | +0.714 | 45 | +0 |
| C-2-full | 0.03 | 3.245e-06 | 86 | 85.289 | +0.711 | 86 | +0 |
| C-2-midpoint | 0.03 | 3.245e-06 | 43 | 42.645 | +0.355 | 44 | -1 |
| C-2-full+transverse | 0.03 | 3.245e-06 | 86 | 85.289 | +0.711 | 86 | +0 |
| C-3-full | 0.018 | 1.168e-06 | 155 | 154.745 | +0.255 | 155 | +0 |
| C-3-midpoint | 0.018 | 1.168e-06 | 78 | 77.372 | +0.628 | 80 | -2 |
| C-3-full+transverse | 0.018 | 1.168e-06 | 155 | 154.745 | +0.255 | 155 | +0 |
| D-0-full | 0.08 | 1.219e-06 | 22 | 22.401 | -0.401 | 23 | -1 |
| D-0-midpoint | 0.08 | 1.219e-06 | 11 | 11.201 | -0.201 | 12 | -1 |
| D-0-full+transverse | 0.08 | 1.219e-06 | 22 | 22.401 | -0.401 | 23 | -1 |
| D-1-full | 0.05 | 4.761e-07 | 42 | 44.286 | -2.286 | 45 | -3 |
| D-1-midpoint | 0.05 | 4.761e-07 | 22 | 22.143 | -0.143 | 23 | -1 |
| D-1-full+transverse | 0.05 | 4.761e-07 | 42 | 44.286 | -2.286 | 45 | -3 |
| D-2-full | 0.03 | 1.714e-07 | 81 | 85.289 | -4.289 | 86 | -5 |
| D-2-midpoint | 0.03 | 1.714e-07 | 43 | 42.645 | +0.355 | 44 | -1 |
| D-2-full+transverse | 0.03 | 1.714e-07 | 82 | 85.289 | -3.289 | 86 | -4 |
| D-3-full | 0.018 | 6.170e-08 | 149 | 154.745 | -5.745 | 155 | -6 |
| D-3-midpoint | 0.018 | 6.170e-08 | 78 | 77.372 | +0.628 | 80 | -2 |
| D-3-full+transverse | 0.018 | 6.170e-08 | 150 | 154.745 | -4.745 | 155 | -5 |

## Reading (the verdicts above stand as frozen; what follows is exploratory)

- **The 2-cycle fold (C) follows the continuous integral closely.** N = ceil(T_arc) in all 12 rows, consistent with
  counting the first strict exit after a continuous passage.
- **The discrete recurrence fails again where Task 5 failed:** at the midpoint entrance and the smallest distance.
  There iterating the quadratic map adds steps to the integral (Task 5's arctangent note).
- **At the period-6 fold (D) the second half of the channel follows the normal form, the first half does not.** From
  the fold point (u = 0) to the exit, every passage is within one iterate. From the entrance u = -U, the measured
  passages are shorter, by 0.4 to 5.7 iterates, more so at smaller eps.
- **What the D deficit could be.** Candidates include terms beyond the quadratic normal form that act on the u < 0
  side (for example a cubic term) and the curvature of the centre manifold at the entrance, which lies U = 0.0014
  along the linear mode r from a fold with b = 72. Neither is tested here. Settling it would need an exploratory
  analysis (the cubic coefficient of D's reduced map; passages from entrances placed on the centre manifold) and,
  for a claim, a fresh test.
- **Scope.** The test does not support a uniform within-one-iterate claim for either predictor across folds. It
  supports the continuous integral at the 2-cycle fold, and at the midpoint entrance of both folds.
