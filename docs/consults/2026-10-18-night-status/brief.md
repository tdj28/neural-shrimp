# Consult 15: everything in motion — the shrimp and its head, the hunt for a one-dimensional head, the norm bounds, and the certificate ladder

## The system (for orientation)

A soft-token feedback loop of a released language model: the next input embedding is the probability-weighted average
of the token embeddings, F(x) = E^T softmax(z(x)/T), with z the logits of Qwen2.5-0.5B-Instruct (float64), steered by
adding alpha times a concept vector to the residual stream at one or two blocks. The paper (arXiv draft, cs.CL) maps
chaos and bifurcations of this deterministic map: folds with 1/N^2 window laws, Feigenbaum ratios, located cusps, an
interval-arithmetic proof of an attracting period-8 orbit, and shrimp-shaped windows. Two parameter planes matter here:
(i) one block, (alpha, log10 T), "philosophy" at block 12 (unimodal band maps; strongly one-dimensional periodic orbits,
second singular value of the 8-step cycle 1e-6 to 1e-4 of the first); (ii) two blocks, (alpha1, alpha2) at
log10 T = -0.1 (gratitude@4 + hostility@16, and hostility@4 + gratitude@16), where band maps have two or more turning
points.

## 1. The period-5 shrimp (gratitude@4, hostility@16, log10 T = -0.1): what is verified

- Orbits: one five-token text throughout (". I..你" repeated); float64 Newton residuals ~1e-13 on a 31x31 continuation
  grid (608 cells, 447 attracting) and along all curves.
- Boundary: fold and flip curves continued in float64 (critical multiplier at +1/-1 within 1.2e-7; residual 1e-10;
  no other multiplier above 0.4 in modulus along them). Each leg lies between a fold and a flip curve; four located,
  nondegenerate cusps (cubic coefficients of opposite sign in pairs) in two "dovetails" (two cusps sharing a segment of
  the fold curve plus a self-crossing of the fold curve), which is a plane section of a swallowtail; an orbit census at
  13 points fixed in advance matched the swallowtail's predicted counts at 9 (four orbits in the pocket, two
  attracting = kite-like bistability).
- New tonight — the domain itself. The fixed-start maps hide the legs (the fixed start reaches a coexisting period-4
  attractor there). We followed the period-5 attractor cell by cell (each cell starts from the final state of a member
  neighbour; member = period a multiple of 5 and the body's token word in >= 4 of 5 phases, or no period with exactly
  repeating tokens and exponent < 0.005; chaotic cells with the same repeating text are recorded but do not
  propagate; a first, looser rule leaked across the fold through intermittent chaos and is preserved). Grid spacing
  1.25e-4: 2173 cells computed, 1720 members (822 period 5, 371 period 10, 82 period 20, 11 period 40; 240 chaotic
  with the same text; 178 slowly converging). Shape: body against the fold curve, three long thin legs, period-10/20/40
  shells beyond the flips, chaos outside; a fourth leg is thinner than the grid.
- Boundary gate: of 292 neighbouring (period 5, period 10) pairs, 251 straddle a computed flip curve within one grid
  step (median 0.24 steps); of 754 (period 5, attractor lost) pairs, 504 lie within one step of a fold curve; nearly all
  others lie along a leg beyond where an earlier curve continuation was clipped by its parameter bounds (a rerun with
  wider bounds is finishing).
- Second temperature: at log10 T = -0.105 the same anatomy (1760 members); the period-5 body's centroid moves
  (+0.00085, +0.00236), the head (below) by about (+0.00105, +0.00247) per the same step.
- Our claim (now in the paper, abstract and both companions): "a numerically verified shrimp of a two-dimensional
  map" — it meets the criteria we required before calling a window a shrimp (primitive full-state periodic orbits,
  their stability, boundary organization by continued bifurcation curves). Not proved by interval arithmetic; tail,
  symbolic sequences (other than the token loop), periodicity hub not tested.

## 2. Its head

- On the 31x31 grid the leading multiplier never vanishes (|mu1| >= 0.0139). The trace-zero curve closes around an
  island of negative trace (Type A, a classification derived for fixed-sign determinants; here D = mu1 mu2 < 0 at 41
  of 447 attracting cells).
- Near-nilpotent point (Newton on: sum of the 6 leading Ritz values = 0, sum of pairwise products = 0) at
  (-0.1184836, 0.1017217): three leading moduli 3.58e-3 (real) and 3.36e-3 (complex pair at +-117 degrees), then 5.2e-4,
  1.6e-5, 7e-7 — numerically (0.32)^5, the per-step transverse rate over five steps.
- Continued in temperature (steps of 0.001; gates: orbit residual < 1e-9, no divisor-period orbit, invariants < 1e-7):
  eleven passed temperatures, log10 T in [-0.104, -0.094]; the point moves on a nearly straight line; the three leading
  moduli stay 2.4e-3 to 5.6e-3, and a fourth approaches them twice (1.05e-3 at -0.104, 1.81e-3 at -0.096; 9.6e-5 at
  -0.094). The track stopped at -0.105 (invariant gate) and at -0.093 (orbit lost). Our reading: over the window's
  temperature range the head stays a near-nilpotent block of three comparable directions — no temperature makes it
  one-dimensional.
- At log10 T = -0.105, grids over the tip of the band where mu1 > 0 show an avoided crossing: the band narrows between
  two strips of complex leading pairs (modulus 0.03-0.1) that come within one cell (5e-5) without crossing.

## 3. Hunting a head of the one-dimensional kind (two superstable curves crossing)

- Rationale we used: a 1-D head needs (a) two turning points (two "critical" steps) and (b) strong anisotropy so the
  one-dimensional reduction is valid near the crossing. One-block planes have (b) but not (a); the two-block P5 window
  has (a) but not (b) near its head.
- Candidate: a period-11 window of the hostility@4/gratitude@16 plane (the plane where every phase of a 20-band
  attractor is N-shaped), text " To solve the question:\nLet now the ...:". Followed domain (spacing 5e-5): 1813
  members (1789 period 11, 8 period 22, 2 period 33): a sinuous band with two thin "wings" rising from it; almost no
  doubled-period shells at this resolution (fold/flip curves being computed now).
- Near-nilpotent point at (-0.0333026, -0.0522112): three leading moduli 3.9e-7, 3.9e-7 (pair, +-69 degrees), 3.4e-7;
  then 1.35e-11 (2.9e4 smaller), 4e-14, 5e-16.
- Trace/multiplier grid around it (17x17, spacing 1.75e-5, still filling, 131 cells): |mu1| from 3.9e-7 to 0.19,
  |mu1|/|mu2| median 6156 (10-90 %: 30 to 4.8e4) — strongly one-dimensional away from the point. The leading real
  multiplier changes sign along curves where the leading pair turns complex in a strip one cell wide with
  |mu| ~ 1e-6: superstable curves avoided only at the 1e-6 level. Two such curves bound a strip of negative mu1 two
  to four cells wide; they run nearly parallel through the box; the near-nilpotent point lies ON one of them; no
  crossing in the box. A Newton search on the signed step factors (reduction with references) did not converge:
  reference cosines fell to 0.01-0.2 (the leading eigenvector is not unique at the point). No single step factor
  vanishes on the grid (the steps that contract most do so everywhere; one step's stretching varies 0.05-13.6).

## 4. Norm bounds of the factorized derivative (tests proposed by a collaborator)

DF = (1/T) E^T Sigma_p J_h, Sigma_p = diag(p) - p p^T. Chain on each orbit: lambda <= (1/k)<log||DF^k||> <=
<log||DF||> <= <log(||Sigma_p||/T)> + <log||J_h||>; worst case contraction for T > T* = diam(E)^2 L / 4.
- Test A: diam(E) = 1.21092 (exact, all pairs, float64); largest ||J_h|| met along 7936 trajectories = 1.54e5 (a
  lower estimate of L = sup); so T* >= 5.7e4 (log10 T* >= 4.75), while the unsteered map already has a single
  attracting fixed point from log10 T = 0.3 to 1.5.
- Test B (7936 cells: every 4th of the stripes map, every 8th of the crossing window; float32; 300 transient + 60
  measured steps; power iteration, 6 iterations per step, warm-started — estimates can only fall short):
  0 chain violations at any k. One-step bound positive in every cell (medians 2.98 chaotic / 2.57 periodic, stripes;
  2.93 / 2.62, crossing) against median exponents 0.30 / 0.17 of the chaotic cells. Precision = chaotic base rate for
  k <= 3; at k = 5 the bound is negative in 27 % / 29 % of periodic cells (medians 0.91 / 0.80 chaotic, 0.25 / 0.22
  periodic). 4 chaotic-labelled cells with negative k = 5 estimates have negative exponents on their own (shorter)
  trajectories. Only ~5 % of rows have one-step power iterations converged to 1e-3 (mostly periodic cells).
- Paper text now: "The inequalities of the factorization hold on the model's orbits, but they do not locate the chaos;
  the composed cycle does."

## 5. Certificate ladder (another agent owns the rigorous pipeline)

Arb ball arithmetic at 1024 bits (Krawczyk), ~6.6 CPU-hours per orbit point at 896 state dimensions. A one-shard remote
calibration reproduced every signed endpoint of the original shard exactly. Next targets: a coexisting period-2 and
period-3 pair at one cell (finite candidates settled with closure ~1e-16), then a ladder (P3, P6, P12, P24 of a
cascade; a band P6; a stripe P10; stretch: the shrimp's P5 head). The critical path is the period-parametric
scientific preparation (derivative qualification, centre refinement, domains), not compute. The human wanted the
ladder within ~12 hours; that will not be met soundly.

## 6. Microtransects (selection versus existence at single switches)

Eight held-out switches, bracketed to 1/64 of the spacing, with cross-continuation and multipliers: 2 selection
(both families persist and attract on both sides), 2 intrinsic (a family ends), 4 unresolved (chaos or slow
convergence); the classifier fixed before the run (comparing opposite continuation arms) classed 0 of 7 as selection
and was replaced before any cross-test data; both results reported.

## Questions (please rank your advice by expected value for a world-class paper)

1. Is "numerically verified shrimp of a two-dimensional map" defensible as stated? What, concretely, would a
   dynamical-systems referee still demand (tail, symbolic dynamics, hub, a reduced 2-D model, a different gate)? Is our
   boundary gate (agreement of the followed domain's edge with continued curves) an appropriate and sufficient
   verification of "boundary organization"? What wording would you use?
2. The head: is calling it Type A (closed trace-zero loop) legitimate when D changes sign and three multipliers are
   comparable at the near-nilpotent point? Is there a better invariant classification for a near-nilpotent 3x3 block
   (e.g. characteristic-polynomial invariants, the codimension-two "nilpotent" point of the leading block), and what is
   the right interpretation of the temperature track (three comparable moduli; a fourth approaching twice)?
3. One-dimensional head: is our operational criterion right (two superstable curves of the reduction crossing, with a
   complex strip much smaller than the window)? Given the period-11 data (curves avoided only at 1e-6 but parallel;
   the near-nilpotent point on one curve), where should two superstable curves cross, and what is the cheapest decisive
   computation to find it (wider multiplier grid, continuation of each superstable curve as a curve of the full map
   via "real eigenvalue = 0 of the leading pair" or "trace = 0", following both until they meet)? Or is the period-11
   object not a shrimp head at all? Which plane/parameters maximize the chance of a 1-D head (strong anisotropy + two
   turning points)?
4. Norm bounds: is the honest conclusion "valid but not localizing" correct and well framed? Is there a cheap,
   *sharper* computable bound worth trying (e.g. a weighted/adapted norm along orbits, cone fields, Lyapunov-function
   style estimates, or the k-step ladder to larger k with converged power iterations), and is test A's
   lower-estimate-of-L framing correct?
5. Given limited compute (rented GPUs/CPUs) and roughly a day, what should the next 12 hours of
   compute go to, among: the 1-D head search; the period-11 skeleton; tracing the period-5 shrimp's tail; the
   certificate ladder support; anything we are missing? What should we NOT claim?
