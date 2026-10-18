# Forward research strategy: a computer-assisted proof of chaos in a language model's feedback map

Consult 14, 18 October 2026. Nothing in the proof study has been run yet; this brief asks how to do it and how to
estimate its cost before committing.

## 1. The map

We study the soft-token map of Qwen2.5-0.5B-Instruct:
- 24 decoder blocks, hidden size 896, 14 query and 2 key-value heads, SwiGLU MLP of width 4864, RMSNorm with eps 1e-6,
  rotary embeddings (base 1e6), and tied embeddings (vocabulary 151,936).
- A fixed 37-token chat prefix (system and user turns) is followed by a single input slot.
- Each step reads the slot x and computes the next-token logits z_alpha(x), with a steering vector alpha*v added to the
  residual stream after block 12. It then writes the expected embedding F(x) = E^T softmax(z_alpha(x)/T) back into
  the slot.
- The prefix's key-value cache is constant, so F: R^896 -> R^896 is one fixed real-analytic map per (alpha, T). Its
  image lies in the convex hull of the token embeddings.

**Numerical evidence of chaos** (float64, with norms and rotary tables kept in float64):
- Many cells have a positive leading Lyapunov exponent. Twin-trajectory exponents agree with exact-tangent (JVP)
  exponents across precisions and horizons.
- Each spectrum has one positive exponent and three strongly negative ones (Kaplan-Yorke dimension 1.13-1.43).
- At the chaotic "crescent" cell (alpha 0.0652, log10 T -0.156, lambda_1 about 0.098 per step), the composed 12-step
  Jacobian is nearly rank one: sigma_2/sigma_1 is about 3e-4. The return map in a scalar log-odds coordinate (the
  logit difference of two tokens divided by T) is a single hump.
- A second candidate expands faster: alpha 0.07484, log10 T -0.1226, lambda_1 about 0.437 per step.

The paper currently says the chaotic cells are numerically validated, not proved.

## 2. What exists

**An attracting period-8 orbit of this map is proved.**
- Method: Arb ball arithmetic through all 24 blocks and the full 151,936-token readout. The weights, alpha, log10 T
  and the RoPE base are exact stored binary rationals; temperature and RoPE frequencies are enclosed as reals.
- A Krawczyk-type inclusion for the multiple-shooting system on boxes of radius 2^-640, then attraction from a
  monodromy-power bound.
- About 56 CPU-hours were projected for all 8 x 896 signed derivative columns over those boxes.

**Measured limits of plain interval and ball enclosures:**
- At 384 bits, even the point evaluation of the prefix failed: radii grew from 1.7e-107 after block 0 to 0.054 after
  block 15, and block 16 could not prove the softmax denominator positive. At 1024 bits the prefix and one point
  step succeeded, with radius 1.2e-142. The prefix and one point step together took about 900 s on one core.
- **Wrapping.** A state box of radius 2^-80 with tangent cube [-1,1]^896 reached primal radius 1.7 by block 2, and
  RMSNorm positivity failed entering block 3. In a box-derivative pilot, one map step blew a 2^-800 box up to a
  radius of about 5e-79.
- **Affine products.** At 128 bits, a 4096 x 896 weight block applied to two columns took 0.47 s on one core
  (synthetic).

**Resources.** One machine with 32 cores and 91 GB RAM, and a GPU with weak float64. Paid CPU or GPU pods are
possible with a costed plan. Arb is CPU-only here.

## 3. Our draft route

**Covering relations with cone conditions** (Zgliczynski and Gidea's h-sets):
- Two h-sets N_0 and N_1 in a chart on a section of the attractor. Each is an interval along the unstable direction
  u times a ball of radius rho in the 895 stable directions.
- Prove that G = F^k (k about 3-12) maps each N_i across both, stretching over each along u and landing inside each
  in the stable directions.
- Prove cone conditions for the full shift: positive topological entropy and periodic orbits of all periods for G.

**Engineering:**
- Never enclose F over a fat box. Subdivide the h-sets only along u into thin segments.
- On each segment, use a first-order (mean-value, affine or Taylor-model) form: G(c) + DG(segment)(x - c) + remainder.
- Carry the stable directions linearly, as in the period-8 Krawczyk step, rather than boxing them.

**Draft milestones:**
- M0: method choice (days).
- M1: a float64 horseshoe search: find intervals I_0, I_1 on the unstable manifold with G(I_i) covering both, and
  measure margins, cone angles and the needed stable thickness (days).
- M2: a rigorous one-step enclosure over a thin segment, with an O(s^2) remainder through RMSNorm, rotary, attention
  against the fixed cache, SwiGLU and the 151,936-token log-sum-exp (1-3 weeks).
- M3: composition and cone conditions (1-2 weeks).
- M4: the full covering run (embarrassingly parallel).
- M5: an exact-rational recheck and an outside audit.

**Our cost scenarios for M4 are guesses**, for two sets and k = 12:

| scenario | segments per set | CPU-minutes per segment-step | CPU-hours |
|---|---|---|---|
| optimistic | 1,000 | 5 | 2,000 |
| middle | 10,000 | 10 | 40,000 |
| pessimistic | 100,000 | 20 | 800,000 |

Two further unknowns may dominate:
- **The stable directions.** A full Jacobian enclosure per segment and step costs CPU-hours (896 columns); a norm
  bound may be far cheaper but looser.
- **The iterate needed.** The crescent's weak expansion may force a high iterate before a full two-branch horseshoe
  appears.

## 4. Proposed feasibility trial (to replace the guesses)

**Part 1: geometry, float64 (local, 1-2 days).** At both candidate cells:
- build the section and the return map along u;
- find the lowest iterate with a two-branch horseshoe;
- measure the covering margins, the second derivative along u (which sets the segment length the first-order
  remainder allows), the stable contraction, and the stable thickness the images need.

This yields the number of segments and k, or an early no-go.

**Part 2: the cost of one rigorous step (about 1 week to build, a few CPU-hours to run).**
- Extend the existing ball code to affine forms with one noise symbol along u, and push one thin segment through one
  map step at 128-256 bits.
- Measure the time and how the remainder scales with the segment length s.
- Also time a few full derivative columns at 256 bits, to price the full-Jacobian treatment of the stable directions.

**Then** the cost of M4 is the number of segments times k times the measured cost per segment-step, plus the cost of
the stable directions.

## 5. Questions

1. **Route.** Is a full covering of the attractor's unstable extent the cheapest rigorous route here? Alternatives we
   see:
   - a transversal homoclinic orbit of one saddle periodic orbit (Smale-Birkhoff), via a short chain of covering
     relations along that orbit, or via validated one-dimensional unstable manifolds;
   - a discrete Conley index computation;
   - a rigorous reduction to an interval map, using an invariant cone field and strong contraction, followed by a
     one-dimensional entropy argument.

   Which needs the fewest and thinnest enclosures for a map with one unstable direction and 895 strongly contracting
   ones?
2. **Stable directions.** How should we bound them cheaply but rigorously? Options include norm bounds of DG on an
   approximate stable subspace, Lohner-type or QR coordinates, logarithmic norms, or very thin h-sets. What does that
   cost per step?
3. **Arithmetic.** Arb balls with hand-written affine forms, or Taylor models (CAPD, or a port)? What precision should
   we expect to need for thin segments? Is a GPU interval implementation with directed rounding worth building?
4. **Cell and iterate.** Should we prefer the weakly or the strongly expanding cell? How should we choose k, and full
   or non-full transition matrices?
5. **Estimate.** Is the trial in section 4 the right one, or what else must be measured for a reliable estimate? What
   effort and compute range would you expect, and what are the most likely failure modes?
6. **Claim.** What would a referee in rigorous numerics require, for example the trusted base and the scope (the
   real-number map of the stored constants, not float64 runs)? What should we not claim?

Please give:
- a recommended route with reasons;
- a staged plan with go/no-go gates;
- the trial design that most improves the estimate;
- a reasoned range for the effort and compute;
- the main pitfalls specific to transformer layers (RMSNorm lower bounds, attention, the large softmax).
