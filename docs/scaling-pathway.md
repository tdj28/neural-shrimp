# Pathway: do these dynamics exist in larger networks? (plan, 12 October 2026)

The mechanism does not depend on dimension. A Neimark-Sacker onset is
organized by a two-dimensional centre manifold, however many parameters the
network has. What changes with scale is how much else happens near the unit
circle: several sharp directions, near-neutral weight-decay directions, and
weakly unstable directions. The plan relies on observables that do not need
full periodic-orbit continuation in large dimension.

## Scalable observables

1. **Loss-only signatures**, available in any training run:
   - a slow modulation at the rotation frequency;
   - plateaus of that frequency as the learning rate varies (locking windows);
   - type-I intermittency near window edges, with laminar times about
     pi/sqrt(|a b| d_eta) (verified against a certified fold, within 4%);
   - soft or abrupt period doubling at flip edges.
2. **A curvature-only rotation prediction.** Project the two step Jacobians
   onto the top two rate-weighted Hessian eigenvectors at each point of the
   2-cycle. This gives the rotation angle to about 0.001 turns
   (`docs/shrimp-dynamics.md`). It needs only Lanczos with Hessian-vector
   products, which is feasible at millions of parameters, and can be checked
   against Arnoldi two-step multipliers.
3. **Matrix-free Floquet checks.** Arnoldi on the two-step map with
   Hessian-vector products, using shift-invert near the expected pair.
4. **Exact reductions where they exist.** For an MLP's first layer, only the
   input span is active (CIFAR: 16,746 of 98,666 coordinates), which allows
   dense Newton.

## Ladder

| Rung | System | Question | Resources |
|---|---|---|---|
| 1 | S2 task, widths 3/6/12/24 (13-97 parameters) | Does the edge-of-stability 2-cycle still lose stability by Neimark-Sacker, with what criticality, and do locking plateaus persist? | CPU, hours |
| 2 | UCI digits MLP (real data; onset already seen with a single rate) | Locking plateaus and one intermittency edge on real data | CPU |
| 3 | CIFAR subset A in exact reduced coordinates, weight decay above the weak negative curvature (about 0.014) | An attracting 2-cycle, its Neimark-Sacker crossing, rotation versus prediction, four-mode check | CPU, dense 16,746-dim algebra |
| 4 | Wider MLPs and CNNs on CIFAR subsets, full batch | Loss-only signatures and the Hessian-vector-product rotation prediction across learning-rate scans | RunPod GPU (budget in the private ledger) |
| 5 | Small transformers, long full-batch runs near the edge | Loss spectra, modulation, locking | GPU |
| Noise | A certified cycle with minibatch noise | Escape times, phase diffusion, survival of locking | CPU |

## Known obstacles

- **Several outlier sharp directions** can make the centre manifold
  higher-dimensional, so that the attractors are 2-tori or 3-tori instead of
  circles.
- **Near-neutral and weakly unstable directions** blur the onset. On CIFAR at
  weight decay 0.01, the 2-cycle is a slow saddle with real multipliers
  1.00002-1.00007. Weight decay is the lever.
- **Stochastic minibatches and learning-rate schedules** break autonomy.
  Floquet language applies only to periodic objects of an autonomous map
  (with noise treated as a perturbation).

## Status

- **Rung 1, corrected (12 Oct):** at ridge 1e-3 the first instability in
  the 6-, 12- and 24-unit networks is a real +1 multiplier, not
  Neimark-Sacker. Stronger weight decay switches units off (the first
  instability is then a unit's recruitment) or makes all units identical
  copies (any Neimark-Sacker is then that of a one-unit network).
  - **Rotation maps (widths 6 and 12):** both show a precession region above
    the 2-cycle band, with about 0.2 turns at its rim.
    - At width 6, the rim is a genuine supercritical Neimark-Sacker
      crossing (d = -0.46, 0.167 turns).
    - At width 12, the symmetric 2-cycle first breaks symmetry.
  - **But the wider students condense.** The S2 teacher has 3 units. With
    small initialization and weight decay, extra units die or become
    identical copies, so the network behaves as an effective 3-unit network.
  - **Consequence:** S2 cannot test width with distinct units.
  - **Wider teachers and larger initialization do not help.** With a 6- or
    12-unit teacher, at initialization scale 0.2 or 1.0, the regularized
    minimum and the cycles still have identical copies of units: 5 of 6, or
    9-10 of 12. This matches the known preference of weight-decayed
    two-layer networks for few distinct units.
  - **A tension:** weight decay is what makes the route exist
    (confinement), yet it collapses wide students in small teacher tasks.
  - **What width needs:** a task that requires many features, i.e. real
    data (UCI digits, CIFAR), or the language models themselves, whose own
    dynamics are now mapped directly (`docs/llm-attractors-pathway.md`).
- **Rung 3 (CIFAR) is paused at a plateau.** Six escape rounds at weight
    decay 0.03 found successive saddles with weak negative curvature while
    the loss fell only from 0.12578 to 0.12553.
- **Rung 3:** running at weight decay 0.03.
- **Earlier evidence:**
  - 72-parameter S1R: most traced 2-cycle boundaries are Neimark-Sacker, and
    a period-58 cycle is certified.
  - UCI digits MLP: Neimark-Sacker onset with the rotation predicted to four
    digits.
