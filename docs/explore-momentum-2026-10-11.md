# E1 pilot: does the edge-of-stability route survive heavy-ball momentum?

Status: exploratory pilot, one network (S2-28011), one rate ray. Nothing here
is certified. 11 October 2026.

## Question

Language models are trained with optimizers that carry state (momentum, Adam).
For plain gradient descent on the toy networks, the edge-of-stability 2-cycle
loses stability through a Neimark-Sacker bifurcation: training becomes
quasi-periodic on an invariant circle, and periodic windows are locking
regions. Does the same route exist with heavy-ball momentum?

Map (PyTorch SGD-momentum convention, per-layer rates P):
`m' = beta m + grad L(theta)`, `theta' = theta - P m'`.

## What is known analytically

At a minimum, each eigenvalue `h` of the (rate-weighted) Hessian gives
`mu^2 - (1 + beta - eta h) mu + beta = 0`. Complex pairs have modulus
`sqrt(beta) < 1`, so the minimum itself cannot undergo a Neimark-Sacker
bifurcation. It first loses stability by a flip at `eta h = 2 (1 + beta)`.

## Method

Rate ray: `log10 output = log10 hidden - 1.828`, through the region of the
period-18 window.
1. The minimum is the same for every rate and every `beta` (gradient
   1.4e-15). Its flip on the ray follows exactly from the largest eigenvalue
   of `P H`.
2. The 2-cycle is seeded just past the flip from the flip eigenvector of the
   26-dimensional step Jacobian, then continued by pseudo-arclength in
   state and ray parameter together.
3. The first unit-circle crossing of its two-step multipliers is classified.
4. Onset test: just before and after the bisected crossing, start next to
   the 2-cycle (or kick it hard to test bistability), iterate, and measure
   the attractor's distance from the cycle and its rotation around it.

Tool: `tools/explore_momentum_maps.py` (tests: `tests/test_momentum_maps.py`).
Outputs: `../neural-shrimp-data/explore/momentum/`.

## Results

| beta | flip (s) | first crossing of the 2-cycle | stable 2-cycle range in s | rotation at crossing (turns per two steps) | onset |
|---|---|---|---|---|---|
| 0 (control) | 0.6619 | complex pair, s near 1.653 | about 0.99 | 0.178 | soft: amplitude grows with distance (amplitude squared about linear), rotation matches the crossing angle |
| 0.5 | 0.8380 | complex pair, s = 1.0679622 (bisected) | 0.230 | 0.047 | hard: a large slow attractor coexists with the stable 2-cycle before the crossing (hysteresis); past it the trajectory jumps there (mean distance 0.047 against cycle separation 0.083, rotation 0.015) |
| 0.9 | 0.9407 | complex pair, s near 0.9412 | about 5e-4 | 0.0066 | unresolved: the crossing sits next to the flip (codimension-two interaction) |

## Reading

- **The route survives in this toy.** With momentum, the edge-of-stability
  2-cycle still loses stability by a Neimark-Sacker bifurcation.
- **Momentum moves and slows it.** Larger momentum shortens the range where
  the 2-cycle is stable and slows the rotation, so the loss modulation is
  slower.
- **Momentum can make the onset abrupt.** At `beta = 0.5` the onset is
  subcritical, with hysteresis and a jump to a slow large-amplitude
  oscillation, where plain gradient descent on the same ray has a soft
  onset. The first Lyapunov coefficient confirms it (12 Oct): d = -0.511 for
  plain gradient descent and d = +2.012 at `beta = 0.5`, both stable under
  doubled finite-difference steps (`criticality-beta*`). It is a genuine
  change in local criticality, not a basin effect. This bears on the spike-versus-soft-onset hypothesis (H1) and
  optimizer memory (H4), but one ray in one network is not evidence about
  language-model training.

## Failures kept

- **From-initialization scans diverged.** Iterating from the shared initial
  parameters with zero momentum diverged at these rates (`scan-beta*`).
- **Natural continuation collapsed.** It slid back onto the fixed point
  (`branch-*`), and the first `branch` run used a descent point already past
  the flip.

## Next

1. More rays, and the window region itself: does momentum preserve locking
   windows and their organization?
2. Criticality (first Lyapunov coefficient) of the momentum 2-cycle's
   crossing along the curve.
3. Adam with epsilon regularization (state `theta, m, v`; autonomous at late
   times).
