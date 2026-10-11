# How the period-18 window is organized (S2-28011)

Status: exploratory numerics and one advisory consult; nothing here is
certified. Started 11 October 2026; update this note as results arrive.
Coordinates: x = log10 output-layer rate, y = log10 hidden-layer rate.

## Question

The certified period-18 window lies about 0.07 decades above the period-2
Neimark-Sacker curve, and the window is 2/9 locking (period 18 = 2 x 9). Is it
the 2/9 Arnold tongue of the edge-of-stability 2-cycle, reaching the curve
through a thin neck at a 2/9 root? Or is it a resonance region detached from
the curve?

## Answer so far

**Detached, at the level of the invariant circle's locking.**

- **Near the curve.** Rotation at 2/9 forms a closed arc joining the two 2/9
  roots A = (-0.0769, 1.7447) and B = (-0.0681, 1.7369). The birth rotation
  along the curve crosses 2/9 there, on either side of its local maximum
  0.2228. This is the closed-"banana" geometry of Peckham, Frouzakis and
  Kevrekidis (1995). Exact period-18 orbits on the arc have not yet been
  computed, so the arc is not yet a demonstrated locking region.
- **The window.** The window is a separate component of 2/9 locking. Its
  lower boundary is a smooth U-shaped fold with its bottom at
  (-0.0800, 1.8200). That fold is where the period-18 node-saddle pair is born
  on the invariant circle, with the rotation rising to 2/9 from below.
- **Between them.** In every mapped column the circle's rotation dips below
  2/9 (a "valley"). The highest column minimum is 0.22043 at
  (-0.0766, 1.792), 0.0018 below 2/9. So no path from the arc to the window
  keeps the circle's rotation at 2/9. The 7 October descent from the window
  ("narrows much faster than delta^3.5, then the tracker stalled") was
  meeting the bottom of the U, not a thin neck.

## What theory says

- **Detachment is allowed.** No theorem forces a locking region to reach a
  Neimark-Sacker curve. For example, the circle family
  `theta + p/q + d(s,t) + a sin(2 pi q theta)` with `d = a/2 + s^2 + t^2` has
  a closed locking disk bounded by folds and no Neimark-Sacker curve at all.
- **Closed bananas attached at two roots are known.** They appear where the
  birth rotation along the curve has an extremum (Peckham, Frouzakis and
  Kevrekidis 1995).
- **How the pieces would reconnect.** If a third parameter (for example the
  ridge coefficient) brought the components together, the generic event is a
  Morse saddle of the fold surface. The projected fold curves are hyperbolas
  that exchange connectivity, and no cusp is needed.
- **Nontwist is an analogy only.** Here the rotation is non-monotone in the
  distance from the curve. That resembles the shearless curves of nontwist
  maps (del-Castillo-Negrete, Greene and Morrison 1996), but it is not the
  same property.
- **Why the window may be wide (phase model).** With slowly varying detuning
  `delta` and resonant amplitude `a_q`, the locking width along a direction
  `n` is about `2|a_q| / |d_n delta|`. The rotation stays within about 0.005
  of 2/9 across this whole region. That may explain why this is the widest
  window, but the measurement has not been made.

## Evidence

| Item | Result | Where |
|---|---|---|
| Rotation map below and around the window (361 x 361, 20,000 samples, CPU) | The arc from A to B; a valley below 2/9 in every column; the window's 2/9 region; the carried U fold traces its lower edge | `../neural-shrimp-data/explore/window-organization/rotation-gap-361`, figure `anatomy-002/window-anatomy.png` |
| Earlier 1024 x 1024 GPU map (7 Oct) | Same column profiles (minimum 0.2178-0.2204 at y = 1.78-1.79) | `explore/a3-gpu-rotation/s2-28011-1024` |
| Circle check at the bottom of the U | Below the fold: a quasi-periodic circle with rotation 0.22148, 0.22189, 0.22206 at 0.003, 0.001, 0.0003 decades below. The fold orbit's points lie on the circle, at distances proportional to the offset. Above the fold: locked period 18 | `explore/window-organization/circle-check-001` |
| Carried fold and flip curves (Codex, 9-10 Oct) | The U turns smoothly at its bottom (node spacing about 1e-3 decades) | `explore/s2-period18-boundary-curves/combined-summary005` |
| Certified pieces in the window | Attracting interval, fold, 1:1 point, two cusps, period-18 Neimark-Sacker curve, two flip/Neimark-Sacker junctions | paper, Section 5 |
| Pro consult and local adjudication | References verified; cautions adopted | `docs/consults/2026-10-11-window-organization/` |

## Not yet established

1. **Orbit-level detachment.** Unstable period-18 branches off the circle
   could still connect the window to the near-curve structure. A continued
   branch joining them would falsify detachment at the orbit level.
2. **The arc as a locking region.** We still need exact period-18 orbits on
   the arc and its two fold boundaries meeting the curve at A and B.
3. **The right-hand side.** The curve has two more 2/9 roots, at
   (+0.0112, 1.6809) and (-0.4267, 1.8709). The window's right flip curve
   heads toward the first of them. A map of that side is running
   (`explore/window-organization/rotation-right-441`).
4. **Certification.** None of the separation is certified. The certified
   pieces are local to the window.

## Next steps

1. Shoot for exact period-18 orbits on the arc, seeded from circle points
   with the hidden rate free, and continue its two folds toward A and B.
2. Continue the U fold independently, and match branches by full-state
   orbit identity.
3. Inside the window, compute the unstable period-18 sheet and its saddle-to-
   node arcs, to see where the locked circle breaks.
4. Conditional predictions for later: 15/68 (just above the saddle value)
   should split like 2/9; 7/32, 9/41, 11/50 and 13/59 can connect through the
   valley.

## References

- B. B. Peckham, C. E. Frouzakis, I. G. Kevrekidis, "Bananas and banana
  splits: a parametric degeneracy in the Hopf bifurcation for maps", SIAM J.
  Math. Anal. 26 (1995) 190-217.
- B. B. Peckham, "The necessity of the Hopf bifurcation in families of
  periodically forced oscillators", Nonlinearity 3 (1990) 261-280.
- D. G. Aronson, M. A. Chory, G. R. Hall, R. P. McGehee, "Bifurcations from an
  invariant circle for two-parameter families of maps of the plane: a
  computer-assisted study", Commun. Math. Phys. 83 (1982) 303-354.
- D. del-Castillo-Negrete, J. M. Greene, P. J. Morrison, "Area preserving
  nontwist maps: periodic orbits and transition to chaos", Physica D 91 (1996)
  1-23.
- Y. A. Kuznetsov, *Elements of Applied Bifurcation Theory*, 3rd ed.,
  Springer (2004).
