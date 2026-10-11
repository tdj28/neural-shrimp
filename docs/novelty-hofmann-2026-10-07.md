# Novelty boundary: Hofmann (2026), read at the primary source

Read by the executing agent on 7 October 2026 at the human's request (the
human will read it later); arXiv:2607.04993v1, T. Hofmann, "The Map Behind the
Flow: Finite-Step Gradient Descent as a Dynamical System" (ETH Zurich, 6 July
2026). Quotations are from the arXiv PDF.

## What the paper establishes

- Fixed-step gradient descent studied as a discrete dynamical system in exactly
  solvable models (balanced scalar reductions of deep linear chains, wider
  linear networks, data coupling, activations, stochastic targets), all with a
  single learning rate a.
- The edge of stability is "the first bifurcation of the training map": a flip
  of the scalar cubic map, then sharpness hovering on the period-2 orbit, a
  period-doubling cascade, Feigenbaum universality, saddle-node windows inside
  chaos, a Ricker-type map in the large-depth limit, and negative-Schwarzian
  (Singer) arguments certifying complete one-dimensional bifurcation diagrams.
- Section 6.1: a "ladder of spectral edges" across singular modes of a wide
  linear network.
- Section 6.2, Proposition 16 ("Rotational edge"): in the two-mode, depth-two
  linear network, the aligned two-cycle of the top mode is stable against frame
  misalignment if and only if a(s1 + s2) < 2; "Beyond this rotational edge, the
  unstable multipliers are complex. Misalignment therefore grows with
  precession rather than along a fixed direction. Numerically, the full
  two-mode system then saturates in a persistently misaligned oscillatory
  state with nonzero loss." The edge lies inside the top mode's period-2 window
  iff s2/s1 > (sqrt(5) - 1)/2.
- The Neimark-Sacker bifurcation is named in the paper's background appendix
  as a bifurcation type that "appears briefly".

## What it does not establish (as far as this reading shows)

- No invariant circles, rotation numbers, phase locking, Arnold tongues,
  staircases or resonance ancestry; the post-edge state is described only as a
  "persistently misaligned oscillatory state".
- No per-layer learning rates or two-parameter learning-rate planes; no
  nonlinear networks with full-state certification of periodic orbits; no
  window census; no loss-based readout of rotation or multipliers.

## Positioning for the manuscript

1. Credit Hofmann for: edge of stability as a flip bifurcation of the training
   map, the one-dimensional period-doubling route with windows, and the first
   derivation (to our knowledge) of complex unstable multipliers of an
   edge-of-stability 2-cycle (rotational edge, two-mode linear network).
2. Our contribution starts where Proposition 16 stops: in nonlinear networks
   with per-layer rates, the 2-cycle's Neimark-Sacker crossing produces
   quasi-periodic training on invariant circles whose rotation number is set
   by the crossing multiplier, and periodic windows arise as resonances of this
   bifurcation (subject to the ancestry test); the two-parameter Neimark-Sacker
   curve organizes the post-edge plane.
3. Remove "the argument is ours" phrasing for the symmetric-similarity point
   unless independent derivation and chronology are stated precisely.
4. Bridge experiment (running in the minimal-model stream): reproduce the
   rotational edge numerically and test whether Hofmann's post-edge oscillatory
   state is quasi-periodic (an invariant circle with rotation continuous from
   arg(mu)/2pi) with locking windows. A positive result gives the paper a
   minimal, published model that shows the same route, credited to its source.
