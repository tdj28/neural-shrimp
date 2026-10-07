# Minimal models for the quasi-periodic route (exploratory, 7 October 2026)

Which small gradient-descent loss already has an edge-of-stability 2-cycle
that loses stability by a Neimark-Sacker bifurcation? Every result here is
exploratory. Raw outputs are in `/data2/neural-shrimp-data/explore/minimal-model/`;
code in `src/neural_shrimp/minimal.py`, `tools/explore_minimal_model.py`,
`tools/explore_minimal_hofmann.py` and tests in `tests/test_minimal.py`.

## Answer

Two small models qualify.

1. **Hofmann's 2x2 depth-two linear network** (eight weights, arXiv:2607.04993,
   Proposition 16). The aligned 2-cycle loses stability at a(s1 + s2) = 2,
   reproduced to machine precision for s2/s1 = 0.7, 0.8 and 0.9. The crossing is
   a transversal complex pair (d|mu|/da = 1.85, 1.90, 1.95) with birth angles
   theta/2pi = 0.1380, 0.1082 and 0.0737. Beyond the edge, training settles on an
   attracting invariant circle of the two-step map (non-periodic, nonzero loss,
   two near-zero Lyapunov exponents for the rotation symmetry and the circle,
   the rest negative), and its rotation number tends to the birth angle.
   - **Without weight decay** (Hofmann's model) the onset is soft (no
     bistability), but the amplitude grows like delta^0.24. Second-mode minima
     form a neutral continuum (uv = s2), so the bifurcation is degenerate.
   - **With weight decay 0.01** only the exact rotation-symmetry multiplier
     stays at 1, and the bifurcation is generic and supercritical: amplitude
     grows like delta^0.49, the rotation at delta = 1e-4 is within 3e-5 of the
     birth angle, and the onset is soft.
   - **Locking beyond the edge.** At s2/s1 = 0.7 there is a 1/8 plateau
     (period 16) and locked cells at 2/17 (period 34) and 5/42 (period 84, the
     mediant of 2/17 and 3/25). At 0.8 a fine scan resolves a narrow 1/10
     tongue (period 20). All periodic cells have rotation m/q with
     q = period/2.
2. **The scalar factorization 0.5 (uv - d)^2 + 0.5 lam (u^2 + v^2)** (two weights)
   with lam = 0.2 and unequal layer rates (ratio about 2.8 or 0.35). This case
   was found as the c = 0 member of the coupled four-weight family below, where
   the two modes decouple. Its 2-cycle loses stability by Neimark-Sacker at
   theta/2pi = 0.4288, and the circle just beyond looks supercritical
   (amplitude ratio 2.08 for a fourfold offset, against 2 for sqrt(delta)).
   With weaker weight decay (lam = 0.05, or the earlier ridge-0.1 model) every
   case ends in a flip.

## Controls

- **Engineered planar gradient map** (positive control). The 2-cycle has
  symmetric step derivatives S0 = diag(1, -1) and S1 = r R_phi S0. Neimark-Sacker
  is typed at r = 1 with theta/2pi = 0.13, and the bifurcation is supercritical
  for every kappa tested.
- **Earlier factorization model, ridge 0.1.** Complex 2-cycle multipliers occur
  on 5 of 8 branches, but their modulus never exceeds 0.65; every branch ends
  in a flip.
- **c = 0** reduces the four-weight model to two scalar factorizations, as
  expected. That is where the scalar Neimark-Sacker case above was found.

## Coupled four-weight model

The model is L = 0.5 z^T C z - d^T z + 0.5 lam |theta|^2 with z = (a1 b1, a2 b2),
C = [[1, c], [c, 1]] and d = C z* + lam, z* = (1, 0.5).

- **Grid.** c in {0, 0.3, 0.6, 0.9} x lam in {0.05, 0.2} x 8 rate ratios. 58
  cases flip; 6 end in Neimark-Sacker, all at lam = 0.2 and extreme ratios:
  (c, ratio, theta/2pi) = (0, 0.354, 0.4288), (0, 2.828, 0.4288), (0, 4, 0.3401),
  (0.3, 4, 0.3834), (0.6, 4, 0.3759), (0.9, 4, 0.4075).
- **Held-out couplings.** Predictions for c = 0.45 and 0.75 were committed
  before running them (`docs/explore-minimal-model-heldout-predictions.json`).
  32 of 32 types were correct; the interpolated angles were 0.3797 vs 0.3740
  observed and 0.3917 vs 0.3892.
- **Coupling is not needed.** Weight-decay strength and rate asymmetry are.

## Caveats

- One-dimensional criticality probes (two offsets), not computed first Lyapunov
  coefficients.
- Rotation numbers come from principal-plane projections of rotation-invariant
  observables.
- In the linear network, divergent starts interleave with circle states at
  larger rates.
- Hofmann's analysis of the edge is the prior result. What is new here is the
  invariant circle, the continuity of its rotation number with the birth angle,
  its amplitude scaling, and the locking beyond the edge.

## Figures

- `/data2/neural-shrimp-data/explore/minimal-model/hofmann/staircase-lam0.01.png`
  (coarse staircases for s2/s1 = 0.7, 0.8 and 0.9)
- `/data2/neural-shrimp-data/explore/minimal-model/hofmann/staircase-lam0.01-fine08.png`
- `/data2/neural-shrimp-data/explore/minimal-model/grid.png` (four-weight boundary types)
