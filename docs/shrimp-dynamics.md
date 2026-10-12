# What the shrimp are: a dynamical explanation (living document)

Status: exploratory float64 numerics on one 13-parameter network (S2-28011),
anchored by interval-arithmetic proofs of selected points. Started 11 October
2026; update as results arrive. Details and run cards are in
`docs/window-organization.md`, `docs/explore-momentum-2026-10-11.md` and
`docs/research-log.md`.

## The phenomenon

Train a small tanh network by full-batch gradient descent with separate
learning rates for the hidden and output layers. Map the long-run behaviour
over the plane of the two rates. Beyond the edge of stability, most of that
plane is chaotic training. Embedded in it are small islands where training
settles into an exact cycle: the weights visit p distinct states and repeat
forever. The islands have the "shrimp" shape familiar from two-parameter
dissipative maps (Gallas 1993).

- **The proved shrimp.** The best-studied island is a period-18 window. An
  attracting period-18 cycle is proved there, and so are the key points of
  its boundary skeleton: fold, 1:1 point, cusps, a Neimark-Sacker curve of
  the cycle, and flip junctions.

## The mechanism, step by step

1. **Edge of stability.** Below a critical rate, training converges. Past it,
   the minimum loses stability by a flip: the weights bounce across the
   valley every step (a period-2 cycle). Each step overshoots along the
   sharpest direction.
2. **The bounce starts to rotate.**
   - At larger rates the period-2 cycle itself loses stability by a
     Neimark-Sacker bifurcation: a complex pair of its two-step multipliers
     crosses the unit circle.
   - Training then circles the (now unstable) 2-cycle. Every second state lies
     on a closed curve, and each two steps advance a fixed fraction of a turn
     around it: the rotation number, about 0.22 near this window.
   - The motion is quasi-periodic, a bounce plus a slow precession. In the
     loss, the precession shows as a slow modulation at the rotation frequency.
3. **Locking makes periodic windows.** Where the rotation number is a fraction
   p/q, the motion closes up into a cycle of period 2q, which is phase
   locking. The period-18 window is 2/9 locking: two full turns in nine
   bounces.
4. **Breakup makes the shrimp shape.**
   - Farther from the edge, the closed curve breaks up and chaos fills the
     gaps, as in the Aronson-Chory-Hall-McGehee scenario.
   - The locked cycles survive inside the chaos as windows. Their boundaries
     deform into the shrimp anatomy: fold curves with cusps, period-doubling
     curves (period 36 appears inside), and, in this higher-dimensional
     system, a Neimark-Sacker curve of the cycle itself.

So a shrimp is a phase-locking resonance of the precessing
edge-of-stability bounce, seen after the circle has broken up. That fits the
view that shrimp domains form from resonance tongues (Varga, Klapcsik and
Hegedus 2019).

## The period-18 shrimp in detail

The window's organization (`docs/window-organization.md`) has two layers.

- **Not reached from the main cycle's roots.** The main cycle's own 2/9
  locking near the edge is a closed "banana" between two 2/9 roots. Above it,
  the rotation falls below 2/9 and later climbs back. The window's lower
  boundary is a closed fold loop with two cusps, where the rotation returns
  to exactly 2/9.
- **Attached to a hidden tongue.** The window's upper boundary, including the
  certified fold, the 1:1 point and the cusps, belongs to a thin, unstable
  2/9 tongue. That tongue is rooted at a resonance of a second, coexisting
  2-cycle that is a saddle. One family of period-18 orbits joins the two
  layers across the window.
- **Why earlier tests missed it.** That tongue is invisible to maps of
  attractors, which is why the prespecified ancestry test could not find the
  window's origin. The test looked for the wrong parent.

## What the network does inside the shrimp

At the window centre (`explore/shrimp-anatomy/cycle-centre-001`):

- **It all happens in the hidden layer.** The hidden-layer rate is about 90
  times the output rate, so the bounce and the rotation are 99% hidden-layer
  weights and biases.
- **Two units see-saw; one is idle.**
  - One hidden unit is idle: it never saturates, and its output weight is
    near zero.
  - The other two have opposite output weights. The bounce pushes their
    biases in opposite directions, driving them in and out of saturation.
- **The loss peaks once per turn.** Two loss peaks per 18 steps coincide
  with peaks in hidden-unit saturation. Just outside the window, the loss
  looks almost the same over a few periods. The difference is that inside,
  the weights return exactly to the same values every 18 steps.
- **Sharpness straddles the edge.** Along the cycle the preconditioned
  sharpness `lambda_max(P H)` ranges from 1.4 to 3.7, around the
  edge-of-stability value 2. Training is unstable on some steps and stable on
  others, and balances on average.
- **Misaligned sharp directions predict the rotation.**
  - The sharpest directions on the two sides of the bounce differ by about
    42 degrees near the Neimark-Sacker curve, and by 50 degrees at the window
    centre. They differ because the hidden units sit at different places on
    their nonlinearity on the two sides.
  - That angle divided by pi predicts the rotation to within 5-10%. It is a
    useful proxy, not yet a mechanism: a two-direction model gets the angle
    but not the instability.

## How low-dimensional is it?

- **Cycles are planar and in the hidden layer.** Every census window examined
  so far (periods 5, 11, 13, 16, 17 and 18) is an attracting cycle lying
  almost entirely in one plane of parameter space (98.7-99.8% of its motion)
  and in the hidden layer (99.7-99.98%).
- **But the chaos is not two-dimensional.** The chaotic attractors next to
  the period-18 window have Kaplan-Yorke dimension 3.2-3.4, even though 79-96%
  of their motion is in one plane. Many directions contract only weakly per
  step, so they add fractal dimension without adding much amplitude. The
  analogy with two-dimensional maps, where shrimps are universal, is
  therefore qualitative. Farther out, training is hyperchaotic (dimension
  4.5-5.6).
- **The odd-period windows (5, 11, 13, 17)** have real leading multipliers
  near the fold and flip values. That is the classic shrimp signature, and it
  cannot come from locking on the period-2 circle, which only gives even
  periods. Their origin is open.

## What it implies for neural-network training (tentative)

- **Beyond the edge, two frequencies organize training:** the bounce and its
  precession.
  - Periodic windows are resonances between them, so their periods follow
    the fractions near the rotation number.
  - The widest windows sit where the rotation stays close to a simple
    fraction over a large region of rates.
- **The fractal-looking trainability boundary is structured,** not random. It
  is a mosaic of resonance windows and chaos.
- **Edges should leave signatures in the loss, by type:**
  - A fold edge gives intermittency. This is confirmed at the period-18
    window's lower fold (`explore/shrimp-anatomy/slips-002`). Just outside
    it, training repeats with period 18 for thousands of steps and then slips
    by one step of the pattern. The time between slips grows like the inverse
    square root of the distance to the edge (fitted exponent -0.54, and -0.52
    close to the edge; theory -0.5). Examples: about 670 steps at 1e-3
    decades, and 15,700 steps at 3e-6 decades. The mean loss does not change,
    so only the time structure reveals the nearby window.
  - A soft period-doubling edge doubles the period. This is confirmed at the
    window's left flip edge (`explore/shrimp-anatomy/edge-ray51`). Outside
    it, training follows a 36-step cycle whose two halves differ by an
    amount growing like the square root of the distance (fitted exponent
    0.49).
  - A hard flip edge, the window's right edge (`edge-ray13`), switches
    abruptly from the 18-step cycle to irregular training, with no 36-step
    stage.
  - A Neimark-Sacker edge should add a slower modulation (not yet tested).
  - None of these edges changes the mean loss appreciably. The information is
    in the time structure.
- **Optimizer memory.** Heavy-ball momentum keeps the route in our pilot, but
  can make its onset abrupt, with hysteresis.
- **Not yet shown:** any of this in large models or under minibatch noise, or
  any training benefit from knowing it.

## Open questions and the plan

1. **The precession, mechanistically.** Derive when the two-step product
   `(I - P H_b)(I - P H_a)` has complex eigenvalues of modulus one, from the
   curvature change across the bounce. The misaligned sharp directions point
   the way.
2. **A fully annotated shrimp map.** For every boundary curve, say what
   changes in training when you cross it, and test the predicted loss
   signatures. The fold edge is done (intermittency confirmed); the flip and
   Neimark-Sacker edges are next.
3. **Generality.**
   - Do other windows (1/5, period 10; others) show the same organization,
     including hidden tongues?
   - What organizes the odd-period windows? Locking on the period-2 circle
     only gives even periods.
4. **Weight decay.** The network without weight decay showed no windows.
   Is regularization what creates the bounded precessing motion?
5. **Noise.** Do shrimps survive minibatch noise as metastable cycles (escape
   times, phase diffusion)?
6. **Certification** of the new organization claims (the hidden-tongue
   collapse point and the connecting family), once the numerics settle.
