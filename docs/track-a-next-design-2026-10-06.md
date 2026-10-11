# Track A next stages: boundary organization (A2) and mechanism and consequence (A3)

Design draft, 6 October 2026, written **before EXP-028 outcomes were inspected**
(the run was still in its zoom stage). Not frozen. Each stage below becomes a
separately frozen protocol whose selection rules act mechanically on EXP-028
outputs, so no choice here depends on what EXP-028 finds.

## Why these stages decide the paper

A period-resolved trainability map that only shows periodic windows exist will
read as a collection of pictures. The paper needs (i) verified shrimp boundary
organization, (ii) a mechanism that predicts where and in what order windows
occur, and (iii) one consequence that matters for training. A2 supplies (i);
A3 supplies (ii) and (iii).

## A2 (EXP-030): boundary organization of confirmed windows

**Input.** EXP-028 components confirmed by its independent audit. If none are
confirmed, A2 does not run; the fallback is a finer zoom study and the minimal
analytic model below.

**Selection.** Up to two confirmed components per system, ordered by audit
confirmed-ring fraction, then size, then component index.

**Local map.** For each selected component, a 256 x 256 period-resolved map
centered on its representative cell, spanning in each log-rate axis three times
the component's extent in the source map (at least four source cells), with the
EXP-028 iteration and screening rules. The CUDA backend may be used only if it
has passed a qualification at the full 20,000-update horizon; otherwise CPU.

**Ray continuation** (`src/neural_shrimp/continuation.py`, implemented and
tested). From the certified orbit, follow the period-`p` orbit along 32 equally
spaced rays in log-rate coordinates normalized by the local half-widths, with a
secant predictor and the exact-Jacobian Newton corrector, until it stops being
attracting. Bisection locates the boundary; the dominant multiplier at the last
attracting point types it: fold (+1), flip (-1), Neimark-Sacker (complex pair)
or unresolved. Positive controls: the period-3 window of the quadratic-map limit
of the Hénon map (fold exactly at `a = 7/4`, flip where the 3-cycle multiplier is
-1) and the analytic gradient-descent product model's flip, both reproduced to
1e-6 or better.

**Shrimp-consistent organization (to be frozen).** All of: fold and flip each
type at least 4 of 32 rays; on at least half of the flip rays the first cell
beyond the boundary in the local map has period `2p`; the ray boundaries agree
with the edge of the period-`p` cells of the local map within two cells on at
least 75% of rays. Only this establishes "shrimp-consistent boundary
organization". The word "shrimp" without qualification additionally requires
A2b: pseudo-arclength continuation of the fold and flip curves themselves,
showing the cusp (codimension-two) points that form the legs.

## A3: mechanism, following Gallas (1994) and Jones (2012)

Jones (2012, arXiv:1201.4343) explains the periodicity hub of the Rössler
system by conjugacy of its return map with unimodal maps along parameter
slices running from the Andronov-Hopf curve to a homoclinic point. Along those
slices the order of periodic windows matches the logistic map's
Metropolis-Stein-Stein (MSS) order up to at least period seven. Shrimp centers
line up along a topological transition line where the return map gains a
second critical point: at their centers, orbits are doubly superstable and
contain both critical points (Gallas 1994). Shrimps there are the two-critical-
point signature.

**Hypothesis to test, not assume.** Full-batch gradient descent restricted to
its active (curvature-limited) direction acts like a one-dimensional map whose
derivative is `1 - eta * curvature`. Its critical points are where the local
curvature equals `1/eta`, half the edge-of-stability curvature `2/eta`. Where
this reduced map has one critical point, windows should follow MSS order along
one-dimensional rate slices. Where it has two (cubic-like losses already give
two), two-parameter families should show shrimps whose centers are doubly
superstable. Weakly contracting flat directions (multipliers near 1) make the
reduction approximate, which is why it must be measured rather than assumed.

**Measurements (A3a, frozen after EXP-028 and A2 designs are reviewed).**
1. *Symbolic signatures.* For each certified orbit, transport the active
   eigenvector of the monodromy around the orbit and record the sign of the
   per-step stretching factor, giving a word whose sign product equals the
   active multiplier's sign. Compare the words of windows with the MSS words
   for unimodal maps of the same period.
2. *MSS order along slices.* Along mechanically chosen one-dimensional slices
   of the coarse and local maps, compare the order of first appearance of
   window periods with the unimodal U-sequence. Agreement supports unimodal
   conjugacy on that slice; systematic violations locate a second critical
   point.
3. *Superstable loci and doubly superstable centers.* Inside each confirmed
   window, locate where the active multiplier crosses zero along the A2 rays
   (superstable points), continue those loci as curves, and test whether
   window centers coincide with intersections of two superstable loci.
4. *Critical curvature.* Check that superstable orbits pass through states
   whose active-direction curvature is `1/eta` within numerical tolerance.

A minimal analytic model, a two-parameter gradient-descent map on a
cubic-gradient (quartic) loss with separate rates, is the positive control: its
reduced map is exactly bimodal and its shrimp structure can be computed
directly.

## A3: consequence for training (A3b)

**Amendment, 6 October 2026 (before EXP-028 outcomes):** the
[novelty check](novelty-check-2026-10-06.md) found orbit averaging largely
anticipated (Chen et al. 2024; central flows 2025), so it is dropped as a
headline candidate. Reproducible islands remain only with quantification
against Ly and Gong (2025): uncertainty exponents and function-space
disagreement inside versus outside windows. The fractal-boundary mechanism
needs crisis-curve analysis and a window-masking test of the boundary's
fractal dimension. The strongest novel axis is C1 plus the mechanism tests
(MSS order, symbolic words, doubly superstable centers, Feigenbaum ratios).

Two candidates, chosen before A3b's freeze:

* **Reproducible islands.** Inside confirmed windows, final predictions are
  insensitive to tiny perturbations of initial parameters and learning rates;
  in nearby chaotic cells they are not. The training-relevant claim is the
  interleaving: arbitrarily small learning-rate changes switch between
  reproducible and irreproducible training at large rates. Includes a
  minibatch-noise arm to test whether windows survive stochastic gradients.
* **Fractal-boundary mechanism.** At the published binary trainability
  boundary, bounded non-convergent dynamics (windows, chaos, drift) are
  over-represented relative to the interior, and windows recur across zoom
  levels. This would explain the boundary's fractal structure by known
  window organization rather than by roughness alone.

Each needs a real-data network (for example an MLP on a low-dimensional
projection of a public dataset) and the learning-rate x momentum plane before
any general claim. A novelty check against the closest work (Hofmann 2026,
Liang and Montufar 2025, Liu 2024, Ly and Gong 2025) is running and will be
recorded before A3b is frozen.
