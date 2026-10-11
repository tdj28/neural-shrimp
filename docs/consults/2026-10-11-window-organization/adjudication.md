# Adjudication: window-organization consult, 11 October 2026

Request: `brief.md` in this folder; response and receipts in
`../neural-shrimp-data/consults/2026-10-11-window-organization/execute-001/`
(GPT-6 Astra Pro, xhigh effort, completed, 42,853 input and 39,365 output
tokens; cost in the private ledger). Advisory review, not independent
validation.

## References checked at their sources

| Ref | Status |
|---|---|
| Peckham, Frouzakis and Kevrekidis, "Bananas and banana splits: a parametric degeneracy in the Hopf bifurcation for maps", SIAM J. Math. Anal. 26 (1995) 190-217 | verified (SIAM record; author's abstract page) |
| Peckham, "The necessity of the Hopf bifurcation in families of periodically forced oscillators", Nonlinearity 3 (1990) 261-280 | verified on the author's abstract page; the consult's recalled title was slightly off; applicability to our map not established |
| Aronson, Chory, Hall and McGehee, "Bifurcations from an invariant circle for two-parameter families of maps of the plane: a computer-assisted study", Commun. Math. Phys. 83 (1982) 303-354, doi:10.1007/BF01213607 | verified |
| del-Castillo-Negrete, Greene and Morrison, "Area preserving nontwist maps: periodic orbits and transition to chaos", Physica D 91 (1996) 1-23 | verified |
| Kuznetsov, *Elements of Applied Bifurcation Theory*, 3rd ed., Springer (2004) | standard text |

## Accepted

1. **Detachment is possible in principle.** No general theorem forces every
   locking region to reach a Neimark-Sacker curve. The consult's example is
   the circle family `theta + p/q + d(s,t) + a sin(2 pi q theta)` with
   `d = a/2 + s^2 + t^2`. Its first-order locking region is the closed disk
   `s^2 + t^2 <= a/2`, bounded by folds, and the family has no
   Neimark-Sacker curve at all.
2. **Precise wording.** What the rotation maps show is that locking of the
   invariant circle at 2/9 has two components on the mapped sections. A
   component of *all* period-18 solutions is a different object: unstable
   period-18 branches off the circle could still connect the window to the
   near-curve structure. We claim only the first until orbits are continued.
3. **The near-curve arc is not yet a resonance region.** An arc of
   near-2/9 rotation (`|rho - 2/9| < 5e-5`) is not a demonstrated locking
   region. Exact period-18 orbits there, with two fold boundaries meeting the
   curve at A and B, are needed before we call it a banana.
4. **Generic reconnection is a fold-surface saddle.** A third parameter would
   generically reconnect the components through a Morse saddle of a regular
   fold surface (local form `u^2 = lambda + s^2 - t^2`): the projected fold
   curves are hyperbolas that exchange connectivity. No cusp is needed.
   Stationary points of the measured rotation number are not the right
   classifier, because the rotation is a devil's staircase.
5. **Nontwist is an analogy only.** Our rotation is non-monotone with respect
   to a parameter. Failure of twist with respect to an action coordinate
   inside one map is a different thing, and we will use the analogy only as
   an analogy.
6. **Phase-model width estimate.** With slowly varying detuning `delta(mu)`
   and resonant amplitude `a_q(mu)`, the locking width along a direction `n`
   is about `2|a_q|/|d_n delta|`. Near a detuning extremum it is about
   `sqrt(|a_q|/|d_n^2 delta|)`. This is a local phase-model statement, not a
   law far from the curve.
7. **Next step.** Find exact period-18 orbits on the near-curve arc by
   shooting from circle points with the parameter free. Continue its two folds
   toward A and B, continue the window's U fold, and match branches by
   full-state orbit identity, not by parameter proximity. A continued branch
   joining the two structures would falsify orbit-level detachment.
8. **Unstable skeleton.** The most informative missing object inside the
   window is the unstable period-18 sheet with its one-dimensional unstable
   manifolds, the saddle-to-node arcs that form the locked circle. We should
   also check the spectra at the two flip/Neimark-Sacker junctions.

## Not adopted, or adopted only in part

- The held-out-section protocol (six development and twelve held-out
  sections, endpoint `C`) is more ceremony than this geometric question
  needs. We adopt its substance: report every section as supported,
  contradicted or unresolved, and fix tolerances before the wider sweep.
- The Farey-neighbour predictions (7/32, 9/41, 11/50 and 13/59 can connect
  through the valley; 15/68 = 0.220588 sits just above the measured saddle
  value 0.22043 and should split like 2/9) are recorded as conditional
  predictions. Their periods (64-136) make them a low priority.
- The third-parameter (ridge) reconnection study is deferred until the orbit
  topology at the current ridge is settled.
