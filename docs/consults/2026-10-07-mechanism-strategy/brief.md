# Research strategy consultation: windows of gradient descent beyond the edge of stability

Date: 7 October 2026. Private, unpublished work; exploratory unless marked
confirmed. Target: arXiv cs.LG primary, cross-listed to nlin.CD.

## Objective and decision

We study full-batch gradient descent as a two-parameter family of maps
theta <- theta - P grad L(theta), with P = diag(eta_h on first-layer weights,
eta_o on second-layer weights), over the plane (log10 eta_h, log10 eta_o).
Beyond the edge of stability we find verified periodic training windows inside
chaos. The decision to advance: which decisive analyses establish the
mechanism(s) organizing these windows, before we freeze a confirmatory test on
fresh networks and scale to a real-data benchmark.

## Systems

- **S2**: tanh network 2-3-1 with biases, 13 parameters, 24-point teacher
  regression, ridge 1e-3, float64.
- **S1R**: mean-field tanh 8-8-1 without biases, 72 parameters, 72 Gaussian data
  points (setup of Sohl-Dickstein 2024, "The boundary of neural network
  trainability is fractal"), plus ridge 1e-3, float64.
- Labels from 20,000 updates: divergent, fixed point, periodic (full-state
  recurrence over the whole tail, period up to 64), chaotic (largest finite-time
  Lyapunov exponent > 0.005), unresolved (non-periodic, near-zero exponent).
- Periodic orbits are Newton-certified with the exact Jacobian I - P H; stability
  boundaries are found by continuing an orbit along 64 rays and bisecting where
  a Floquet multiplier crosses the unit circle (typed fold +1, flip -1,
  Neimark-Sacker complex).

## Evidence so far

| Finding | Status |
|---|---|
| 61 Newton-certified attracting periodic windows (periods 4-58, including 5, 11, 13, 17, 29) embedded in chaos (at least half of a surrounding ring chaotic), two networks x two seeds | Confirmed: frozen design, independent audit |
| The published binary trainability score labels 87-100% of non-fixed-point cells "convergent" | Confirmed (descriptive) |
| The edge-of-stability period-2 orbit loses stability by Neimark-Sacker in both networks; the birth angle arg(mu)/2pi predicts the rotation number of the quasi-periodic training that follows (S2: 0.218 vs 0.2175 observed; S1R: 0.132 vs 0.12-0.135) | Exploratory |
| Traced Neimark-Sacker curve (S2): U-shaped; birth rotation 1/6 at its bottom rising to 0.24 and 0.22 along the two branches | Exploratory |
| Rotation-number map (256 x 256, S2): just beyond the curve, median abs(rho - arg(mu)/2pi) = 0.0018, growing smoothly with distance; of 6,874 even-period cells, 74% have rotation m/q with q = period/2 and 17% are period-doubled inside an m/q tongue | Exploratory |
| Devil's staircases along slices with Farey-ordered plateaus: S2 2/9, 3/13, 1/4; S1R 1/9, 2/17, 1/8, 3/23, 2/15 | Exploratory |
| Of the 56 confirmed windows with at least 5 points, 35 are well-ordered (Birkhoff) m/q orbits of the two-step map (random-angle null expectation 3.6), including doubled orbits (2/5 = 2 x 1/5; 4/11 = 2 x 2/11) | Exploratory |
| Boundary census over 61 windows x 64 rays: fold 1,316, Neimark-Sacker 1,395, flip 1,012. S2 windows mostly Neimark-Sacker and flip bounded; S1R windows mostly fold bounded. Only 2 of 61 meet a classical shrimp criterion (fold and flip edges, flips opening onto period 2p); one period-18 (2/9) window has shrimp-like anatomy (fold body, flip edge into period 36, legs) | Exploratory |
| Maps from one shared initial point are speckled inside windows: S1R windows coexist with other attractors (87% of speckled cells return to the window orbit from the orbit point, 10% from the shared start even with 4x longer runs); S2 windows have long chaotic transients | Exploratory |
| Odd-period windows: all 5 confirmed ones (S2: 5, 11, 13, 17, 29) are not well-ordered circle orbits; 181 odd-period cells in the S2 map | Unexplained |
| Minimal model L = (uv - 1)^2 + 0.05 (u^2 + v^2) with separate rates for u and v: the period-2 orbit loses stability by flip on all 24 rays; period-doubling cascade, odd windows 3, 5, 7; no quasi-periodic cells anywhere | Exploratory |

Working thesis: beyond the edge of stability, per-layer gradient descent
follows the quasi-periodic route to chaos (period 2 -> Neimark-Sacker of the
two-step map -> invariant circle -> Arnold tongues with periods 2q in Farey
order -> torus breakdown, with period doubling and secondary Neimark-Sacker
inside tongues); classical shrimps are a minority phenomenon. Literature
checks found no report of invariant circles, tongues or staircases for
optimizer-on-loss maps; Hofmann (2026, arXiv:2607.04993) derives complex
unstable multipliers of a 2-cycle in a two-mode depth-2 linear network without
rotation numbers.

## Failed or retired approaches

- Symbolic words from the sign of the smallest eigenvalue of J at orbit points
  (curvature beyond 1/eta) were all-R: uninformative for unimodal (MSS) tests.
- Net winding of a closed cycle is automatically an integer; replaced by the
  well-ordering test with a random-angle null.
- Ray-continued edges and fixed-start map edges disagree where attractors
  coexist; orbit-following maps are needed for boundary figures.

## The three questions

**Q1. Odd-period windows.** These are not circle orbits. What is the decisive
test that they come from an effectively one-dimensional (unimodal) reduction
organized like a periodicity hub, versus other routes (saddle-node windows
inside torus-breakdown chaos, tongue-overlap regions, homoclinic mechanisms)?
Reference point: T. D. Jones, "Topological origins of a bi-parameter
periodicity hub for the Rössler attractor", arXiv:1201.4343 (2012): the hub
shape follows from conjugacy between the Rössler system and unimodal maps in
the two-branch region of parameter space; a nested spiral comes from a
topological feature of the flow; symbolic sequences of orbits are tracked
through the spiral transition up to period seven. Concretely, in a 13-
dimensional gradient map, what would you compute: kneading words of a
one-dimensional return map, a search for doubly superstable points (two orbit
points at critical locations) and spiral organization of windows around a hub
in the rate plane, or something else? What outcome would refute the hub
interpretation?

**Q2. Mechanism consistency.** Is the census above (S2 windows bounded by
Neimark-Sacker and flip curves, S1R windows by folds; period doubling inside
tongues; coexistence and long transients) consistent with known torus-breakdown
phenomenology (circle maps and Arnold tongues; Curry-Yorke and Aronson-Chory-
Hall-McGehee routes; shrimps inside tongues in driven oscillators), or does
anything contradict it? Which single diagnostic would best discriminate the
tongue picture from alternatives?

**Q3. Minimal model.** What is the smallest gradient-descent map (at most four
parameters) in which the edge-of-stability 2-cycle undergoes a Neimark-Sacker
bifurcation? Is there a structural reason the two-parameter factorization
model cannot (its single-step Jacobians I - P H are similar to symmetric
matrices, but the two-step product need not be)? Is the two-mode depth-2
linear network the right candidate?

## Resources and priorities

- Local 32-core CPU and an RTX 2000 Ada GPU (float64 about 0.19 TFLOPS) with a
  qualified float64 CUDA engine; a proposed cloud H100 budget up to $200; about
  two weeks before drafting is finalized.
- Priorities: correctness first, then novelty and generality for a cs.LG
  audience with nlin.CD rigor: (1) mechanism established on the toy networks,
  (2) the same route in a real-data benchmark (full-batch GD on a CIFAR-10
  subset as in Cohen et al. 2021, with weight decay), (3) a training-relevant
  consequence (for example, the loss spectrum revealing the regime, or rotation
  number predicting reproducibility beyond Lyapunov exponent and sharpness),
  (4) a frozen confirmatory test on reserved fresh seeds.
