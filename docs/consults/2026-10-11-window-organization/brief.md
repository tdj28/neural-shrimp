# Consult brief: how a period-18 resonance window is organized relative to a Neimark-Sacker curve

## System

Full-batch gradient descent F(theta) = theta - P grad L(theta) on a 13-parameter
network: 2 inputs, 3 tanh hidden units with biases, a scalar output with bias;
loss 0.5 * mean squared error on 24 fixed teacher-generated samples plus
5e-4 * |theta|^2. P = diag(eta_h I_9, eta_o I_4) holds the hidden-layer and
output-layer learning rates, so F is a real-analytic two-parameter family of
maps of R^13. Coordinates: x = log10 eta_o, y = log10 eta_h.

## Established facts (numerical unless marked proved)

1. **Period-2 Neimark-Sacker (NS) curve.** Beyond the edge of stability the
   period-2 orbit of F loses stability through a complex pair of DF^2 along a
   curve close to y = 1.668 - x for x in [-0.10, -0.06]. The first Lyapunov
   coefficient is negative there (supercritical), so an attracting invariant
   circle of F^2 (a pair of circles for F) exists just beyond the curve.
2. **Birth rotation along the curve** (critical multiplier angle / 2 pi) has a
   local maximum 0.2228 near x = -0.0725 and equals 2/9 at roots
   A = (-0.0769, 1.7447) and B = (-0.0681, 1.7369).
3. **Rotation field** (rotation number of F^2 on the attractor, 20,000 two-step
   samples projected on a principal plane, 1024^2 map). In every column
   x in [-0.083, -0.072] the rotation falls from the curve to an interior
   minimum rho_min(x) between 0.2178 and 0.2204, all below 2/9 = 0.22222, at
   y = 1.78-1.79 (about 0.04 decades above the curve). It then rises, reaches
   exactly 2/9 at y = 1.82-1.83, stays locked at 2/9 over a band up to
   0.045 decades thick (the window) and exceeds 2/9 above the window. Below
   y = 1.82 the attractor is a quasi-periodic circle (Lyapunov exponent near
   zero). In and above the window the surrounding dynamics is chaotic, with
   embedded periodic windows. Between A and B a thin near-locked arc
   (|rho - 2/9| < 2e-4) joins A and B within 0.006 decades of the curve.
4. **Window bottom.** Numerical continuation of the period-18 fold (multiplier
   +1) gives a smooth U-shaped curve. Its minimum in y is at (-0.0800, 1.8200).
   Node spacing is about 1e-3 decades and nothing is singular at the turn.
   Just below the U the attractor is the invariant circle, with rotation
   0.22148, 0.22189 and 0.22206 at 0.003, 0.001 and 0.0003 decades below. The
   nine even points of the fold orbit lie on that circle: distance grows in
   proportion to the offset and is about 1e-3 of the circle's extent at
   0.0003 decades. Just above the U the motion locks to the period-18 orbit
   with rotation 2/9.
5. **Proved by interval arithmetic** (Krawczyk-type, with independent replays),
   in and around the window:
   - an attracting period-18 interval;
   - a nondegenerate fold point;
   - a generic 1:1 resonance of the period-18 orbit on the upper fold at
     (-0.0824, 1.8657): double +1 multiplier, Jordan block, subcritical normal
     form;
   - two cusps at (-0.0803, 1.8632) and (-0.0806, 1.8563), on the
     non-attracting continuation of that fold;
   - an NS curve of the period-18 orbit running from the 1:1 point to two
     junctions with flip curves, at (-0.0792, 1.9264) and (-0.0732, 1.8587);
   - flip curves bounding the window on the left and right, and a period-36
     orbit after the left flip.

## Our interpretation (to be tested)

The 2/9 resonance region of the circle born on the NS curve has at least two
components:

- **(i) An attached banana.** It is closed and attached to the curve at A and
  B. This is the parametric degeneracy of Peckham, Frouzakis and Kevrekidis
  (SIAM J. Math. Anal. 26 (1995) 190-217): the rotation decreases into the
  interior from a boundary maximum.
- **(ii) A detached component.** It is bounded below by a closed U-shaped
  saddle-node curve where the rotation increases with distance from the NS
  curve. This component is the certified window.

The two components are separated by a valley in which the rotation is below
2/9. The non-monotone rotation transverse to the curve would be a dissipative
analogue of nontwist (shearless) behavior. Inside (ii) the invariant circle
breaks down, as in Aronson, Chory, Hall and McGehee (Commun. Math. Phys. 83
(1982) 303-354), producing the cusps, the 1:1 point and the NS and flip curves.

## Questions

**Q1.** Is (i) plus (ii) a correct and standard reading?
- Are detached resonance components, bounded by closed saddle-node curves that
  never touch the NS curve, known in two-parameter families of maps or in
  forced oscillators?
- Under which names, and in which precise references?
- Does any theorem force every resonance region of the circle to reach the NS
  curve? For example, does Peckham (Nonlinearity 3 (1990) 261-280) apply?
- Does anything constrain where a detached component must end?

**Q2.** When a third parameter varies, what is the generic event at which the
attached banana and the detached component reconnect? Is it a saddle of the
rotation function at level p/q?
- What does the geometry of the boundary curves look like nearby: crossing
  saddle-node curves, a pair of cusps, lips or beaks?
- Is this the dissipative counterpart of separatrix reconnection of twin island
  chains in nontwist maps (del-Castillo-Negrete, Greene and Morrison, Physica D
  91 (1996) 1-23)?

**Q3.** Which cheap, decisive numerical tests would confirm or refute the
picture, and what would falsify it? Candidates:
- (a) For p/q between the minimum and maximum of rho_min along the valley, the
  two components of the p/q region should be connected; above the valley's
  maximum they should be detached. Which Farey neighbours of 2/9 should then be
  attached and which detached?
- (b) Locate the saddle points of rho.
- (c) Show that the banana is closed, with period-18 orbits on the circle
  between A and B.
- (d) Continue the U fold and the cusp branch to see how they join the upper
  fold and the region where the circle breaks down.

**Q4.** Within the detached component, is the observed skeleton consistent with
known scenarios of circle breakdown inside resonance regions? The skeleton is:
- the U fold below;
- an upper fold carrying a 1:1 point and two cusps;
- the period-18 NS curve between the 1:1 point and the flip-NS junctions;
- left and right flip curves.

What structure should exist between the cusps and the U?

**Q5.** Why might this resonance give the widest periodic window in the map? We
suspect it is because the rotation is nearly stationary over a large region: a
boundary maximum near 2/9 and a shallow valley. Is there a known quantitative
relation between resonance-region width and |grad rho| away from the NS curve?

Mark each statement as established theory (with a reference), plausible
inference or speculation. Do not invent references; if a citation is
uncertain, say so.

## Resources

Local CPU (32 threads), Python/NumPy tooling, carried continuation of fold, flip
and NS curves, rotation maps, and interval-arithmetic proofs for selected
points. A rented GPU is available for large rotation maps.
