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
   - Locking does not need the circle to break: it already happens on the
     intact circle (the banana, and the window's lower fold).
   - Farther from the edge, the circle breaks up and chaos fills the gaps, as
     in the Aronson-Chory-Hall-McGehee scenario. The locked cycles continue
     through that breakup and survive inside the chaos as windows.
   - Their boundaries deform into the shrimp anatomy: fold curves with cusps,
     period-doubling curves (period 36 appears inside), and, in this
     higher-dimensional system, a Neimark-Sacker curve of the cycle itself.

So a shrimp is a phase-locking resonance of the precessing
edge-of-stability bounce. It is born on the intact circle and persists after
the circle breaks up. That fits the
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
- **How the hidden tongue becomes the visible window.**
  - This differs from ordinary tongues, whose locked orbits sit on an
    attracting circle. Near R3 the locked orbits sit on a saddle circle, so
    they inherit the 2-cycle's unstable direction, with multiplier 1.031 per
    two steps (1.031^9 = 1.316 per cycle).
  - Tracked continuously along the tongue boundary, this inherited multiplier
    stays real and positive. It crosses +1 at two 1:1 resonance points
    (double +1). One is Codex's certified 1:1 point. The other is at
    (eta_h, eta_o) = (84.0779, 0.80237).
  - For that second point, a computer-assisted proof now shows a locally
    unique primitive period-18 orbit whose multiplier +1 has exact algebraic
    multiplicity two, with one Jordan block and all 11 other multipliers
    strictly inside the unit circle (`explore/hidden-parent-certification/
    chain-proof-003`). Its nonlinear normal-form coefficients and its
    connection to the window's branch are not yet proved.
  - The tongue's root R3 is proved too. At (eta_h, eta_o) =
    (52.961196369999170639, 0.910699388167699003), each to about 2e-19, a
    primitive 2-cycle has two-step multipliers exactly exp(+-i 4 pi/9), a 2/9
    resonance on the unit circle (`resonance-proof-001`). The same 2-cycle
    has a proved real multiplier 1.0309819704993792 +- 9e-19, so it is a
    saddle (`resonance-proof-004`). The tongue's root is therefore rigorously
    a 2/9 resonance of a saddle cycle, unlike an ordinary tongue rooted on an
    attracting circle. The branch connecting R3 to the window (381
    continuation nodes) is still numerical.
  - Numerically, the same fold continuation passes through both certified
    cusp locations. The certificates themselves do not yet prove that the
    cusps belong to this connected branch; keep them as separate local
    results until a validated connection exists.
  - Between them it is below 1 (about 0.8), and the cycle is attracting: that
    stretch is the window.
  - At a 1:1 resonance of a map, a Neimark-Sacker curve emanates. That is the
    certified complex-unit branch, across which the cycle trades stability
    with the inherited direction.
  - So the certified skeleton has a dynamical meaning: it is the gate through
    which an unstable resonance becomes an attracting training cycle.

## What the network does inside the shrimp

At the window centre (`explore/shrimp-anatomy/cycle-centre-001`):

- **It all happens in the hidden layer.** The hidden-layer rate is about 90
  times the output rate, so the bounce and the rotation are 99% hidden-layer
  weights and biases.
- **Two units move together; one is nearly idle.**
  - One hidden unit is nearly idle: it never saturates, its output weight is
    near zero, and it carries 6% of the cycle's motion.
  - Compare the other two after accounting for tanh's sign symmetry (flipping
    a unit's incoming weights, bias and outgoing weight changes nothing).
    They then have similar output weights and similar second input weights,
    and 95% of their joint motion is in phase.
  - They are partly redundant units that bounce and precess together, driven
    in and out of saturation. An earlier "see-saw" description was a
    sign-convention artifact, caught by an external reviewer.
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
  - **A four-mode model predicts the rotation from curvature alone.** Project
    the two step Jacobians onto the top two rate-weighted curvature
    directions at each point of the bounce. Over 21 cycles the product's
    angle matches the exact rotation to within 0.0012 turns (maximum error;
    0.0007 with three directions each). The modulus, which decides stability,
    converges more slowly (about 1.05 instead of 1.00). So the precession
    comes from misaligned top-two curvature planes on the two sides of the
    bounce, and can be computed before any monodromy
    (`explore/shrimp-anatomy/rotation-geometry-002`).

## How low-dimensional is it?

- **Cycles are planar and in the hidden layer.** Every census window examined
  so far (periods 5, 11, 13, 16, 17 and 18) is an attracting cycle lying
  almost entirely in one plane of parameter space (98.7-99.8% of its motion)
  and in the hidden layer (99.7-99.98%).
- **Local organization is two-dimensional.** Near the Neimark-Sacker
  bifurcation the invariant circle, locking and tongues live on a
  two-dimensional centre manifold. The higher dimension below concerns the
  global chaotic attractors, not this local picture.
- **But the chaos is not two-dimensional.** The chaotic attractors next to
  the period-18 window have Kaplan-Yorke dimension 3.2-3.4, even though 79-96%
  of their motion is in one plane. Many directions contract only weakly per
  step, so they add fractal dimension without adding much amplitude. The
  analogy with two-dimensional maps, where shrimps are universal, is
  therefore qualitative for the global attractors. Farther out, one point
  has two positive Lyapunov exponents (0.216, 0.057; hyperchaotic,
  Kaplan-Yorke 5.6); another has one (Kaplan-Yorke 4.5). These are
  finite-horizon numerical spectra (40,000 QR steps, float64): estimates,
  not rigorous dimensions.
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
  - **The certified fold predicts the timescale, with nothing fitted.** The
    certified normal-form coefficients of the window's fold, a = -0.0156160
    and c/2 = -4.0000580, give a bottleneck passage time of
    18*pi/sqrt(a*(c/2)*d_eta) = 226/sqrt(d_eta) gradient steps, for a
    hidden-rate detuning d_eta beyond the fold. Measured median laminar times
    are 831, 1501, 2493, 4399 and 7440 steps, against predictions of 715, 1306,
    2263, 4131 and 7155, for d_eta = 0.1 down to 0.001. The ratio falls to
    1.04 as d_eta shrinks (`explore/shrimp-anatomy/fold-prefactor-certified`).
    Held-out checks at five new detunings, each from three random starts, give
    ratios of 1.16, 1.13, 1.08, 1.05 and 1.03 (at d_eta = 5e-4). They are
    identical across starts to three digits
    (`fold-prefactor-heldout-1..3`). The same calculation at the U-bottom
    fold, using its numerically computed coefficients, matches the longest
    laminar phases to 2-3% near the fold.
    This connects a computer-assisted bifurcation certificate to an
    observable property of the training-loss time series. (The prediction
    was suggested by an external reviewer from the paper's coefficients.)
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
  can make its onset abrupt, with hysteresis. On the tested ray the first
  Lyapunov coefficient changes sign: d = -0.51 for plain gradient descent
  (supercritical) against d = +2.01 with momentum 0.5 (subcritical). The hard
  onset is a genuine change in local criticality, not a basin effect
  (`explore/momentum/criticality-beta*`). Across a momentum sweep the sign
  change sits at a degenerate (Chenciner) point near beta = 0.07. Below it
  the onset is soft; above it, it is hard, with |d| growing steeply (up to
  +57 at beta = 0.7). One ray of one network so far.
- **Weight decay decides whether the route appears (12 Oct).**
  - **Same final sharpness at every weight decay.** At the window rates the
    minimum is far beyond the edge: rate-weighted sharpness 29-38 at the
    true minima for ridge 0, 1e-5 and 1e-3, against 2
    (`minimum-check-002`). From the same start, training settles back at a
    sharpness of about 2 whatever the weight decay.
  - **Without weight decay the minimum is nearly flat.** It is isolated, but
    its smallest Hessian eigenvalue is 9e-7, against 4e-4 with ridge 1e-3.
    So even at the minimum, one direction is almost neutral under gradient
    descent.
  - **Different routes to it.** Without weight decay, gradient descent gets
    there by inflating the weights (norm 12 against 2.2 with ridge 1e-3) and
    saturating the tanh units (mean tanh' 0.16 against 0.59). In that
    saturated regime three two-step multipliers are 1 to four digits: neutral
    directions. Training drifts slowly along a family of near-2-cycles with no
    rotation, so there are no tongues and no shrimps.
  - **What weight decay adds.** It bounds this escape: attractor norm about
    3.3, 2.6 and 2.1 at ridge 1e-4, 3e-4 and 1e-3. It also turns the neutral
    directions into contracting ones, by a factor close to (1 - eta lambda)^2
    per bounce (0.984 measured against 0.986 at 1e-4).
  - **Refuted alternative.** Saturation does not make the curvatures at the
    two bounce points commute (relative commutator 0.32 without weight
    decay, 0.18 at 1e-3).
  - **Weight decay is a third control parameter.** It reorganizes which
    instability comes first. On the tested ray of the 13-parameter network,
    the 2-cycle's first instability is:
    - folds at very weak weight decay (1e-5, 3e-4);
    - Neimark-Sacker at 1e-4 and 1e-3;
    - recruitment of a unit that weight decay has switched off, at 3e-3.
    In a 12-unit network, strong weight decay makes all units identical
    copies; a Neimark-Sacker crossing then belongs to an effectively
    one-unit network.
  - **Scope and open questions.** One network, one rate point and one ray.
    The window itself still has to be continued in the weight-decay
    coefficient.
  - **Language-model link (hypothesis, untested).** With normalization
    layers, weights are scale-invariant, and without weight decay their norm
    grows and the effective step shrinks. Weight decay sets an equilibrium
    norm. If the same escape operates there, decoupled weight decay (as in
    AdamW) would be what keeps training in the bounded edge-of-stability
    regime where these bifurcations can occur.
- **Not yet shown:** any of this in large models or under minibatch noise, or
  any training benefit from knowing it.

## Open questions and the plan

1. **The precession, mechanistically.** The angle now follows from a
   four-mode curvature model. Still to do: an analytic version, and the
   modulus, which decides when the precession goes unstable, from the same
   geometry.
2. **A fully annotated shrimp map.** For every boundary curve, say what
   changes in training when you cross it, and test the predicted loss
   signatures. The fold edge is done (intermittency confirmed); the flip and
   Neimark-Sacker edges are next.
3. **Generality.**
   - Do other windows (1/5, period 10; others) show the same organization,
     including hidden tongues?
   - What organizes the odd-period windows? Locking on the period-2 circle
     only gives even periods.
4. **Weight decay.** Partly answered: without it, training escapes to
   saturated weights with neutral directions and no rotation (see the
   implications section). Still to do:
   - continue the window and the period-2 Neimark-Sacker curve in the ridge
     coefficient;
   - explain why no 2-cycle could be seeded at the flip without weight
     decay;
   - check whether the zero-loss set without weight decay is isolated.
     Zero-loss points of norm 1.55 and 2.01 were both found.
5. **Noise.** Do shrimps survive minibatch noise as metastable cycles (escape
   times, phase diffusion)?
6. **Certification** of the new organization claims (the hidden-tongue
   collapse point and the connecting family), once the numerics settle.
