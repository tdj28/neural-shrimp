# Consult 18 brief: why the superstable curves of the language-model shrimps avoid crossing

## The system

A pretrained transformer language model, Qwen2.5-0.5B-Instruct, is fed back its own expected next-token embedding.
This defines a smooth map of R^896:

    F(x) = E^T softmax(z_alpha(x) / T),

with E the tied embedding matrix (151,936 tokens), z_alpha the logits of the full network for the fixed prompt prefix
and the fed-back state x, a steering vector of strength alpha added at layer 12, and temperature T. The parameter
plane is (alpha, log10 T); a second plane steers at two layers with strengths (alpha1, alpha2) at fixed T.

Everything below is computed in float64:
- periodic orbits by Newton's method on F^p(x) = x, with residuals around 1e-14;
- multipliers from the dense or Ritz-reduced monodromy M = DF^p;
- fold and flip curves by two-parameter continuation, with every point gated: critical multiplier real and within
  1e-7 of +1 or -1;
- cusps located by the oriented quadratic normal-form coefficient, the cubic coefficient c and the unfolding
  determinant det J.

Chaos is the numerically observed kind (positive finite-time Lyapunov exponents); periodic orbits are verified
numerically, and three are certified by interval arithmetic.

## Shrimp-shaped windows we have found

Gallas's shrimps (Hénon map) and the shrimps of one-dimensional maps have a body bounded by fold and flip curves,
legs between a fold and a flip curve, doubled shells beyond the flips, and a center where two superstable curves
(zero curves of the multiplier) cross: a doubly superstable point. In two-dimensional maps with a determinant of
fixed sign no multiplier can vanish. Façanha, Oldeman and Glass (Phys. Lett. A 377, 1264, 2013) use the trace-zero
curves as the skeleton instead. These either close around the head (Type A) or form two arcs that approach and turn
away (Type B).

Three windows of the language-model map are relevant.

1. **Period-5 window, two-depth plane.** This is a numerically verified shrimp: body, four legs, four located cusps
   and doubled shells.
   - On a 31 x 31 grid of the continued orbit the leading multiplier never vanishes (modulus at least 0.0139). Where
     a one-dimensional map would be superstable, the two leading multipliers form a complex pair or pass each other
     as real numbers of opposite sign.
   - A two-parameter Newton search for the point where the sum of the three leading multipliers and the sum of
     their pairwise products both vanish converges to a "near-nilpotent" point. There the three leading multipliers
     have nearly equal moduli, 3.4-3.6e-3, and the return map is nearly rank one: second singular value about 1.8e-4
     of the first.
   - The trace of the two leading multipliers vanishes on a closed curve around an island of negative trace,
     resembling Type A.
2. **Hostile period-7 window**, with a different steering direction. The curves where the sign of the leading real
   multiplier changes between cells, with the modulus small along them, form two arcs that approach near one point
   without crossing. In the closest-approach region the two leading multipliers form a complex pair, resembling
   Type B.
   - Caveat (from an internal review): a sign change between cells can also come from two real multipliers of
     opposite sign exchanging dominance. These are therefore called candidate superstable curves.
3. **Period-6 window, (alpha, log10 T) plane: the new shrimp of tonight.**
   - Body near (0.0731, -0.152). A float64 follow of the attractor gives 4224 member cells: period 6 in 1946, its
     doublings 12/24/48 in 945, and chaos behind the repeating text in 811.
   - Gated fold and flip curves bound the body. A located cusp lies at the tip of a smaller period-6 window below
     the body.
   - Multipliers of the body's orbit:
     - most contracting cell (0.073097, -0.15204): complex pair 0.01008 +/- 0.01249i (modulus 0.0161), then
       -0.0010, 2.3e-5, ...;
     - (0.073200, -0.15204), toward the fold: real 0.0746, next modulus 0.0235;
     - (0.072900, -0.15204), toward the flip: real -0.767, next 0.0022.
   - The superstable grid (zero curves of per-step factors of the leading multiplier, read only from cells with
     |mu1| > 10 |mu2|): SEE "DSS6 RESULT" BELOW.

DSS6 RESULT: (to be filled in when the run returns; it states whether the zero curves cross or avoid, where, and the
multipliers at the closest approach)

## The proposed explanation (to be checked)

> **Why the superstable curves avoid each other.** In a one-dimensional map the multiplier of a period-p orbit is a
> product, mu = prod_i f'(x_i). Its zero set is therefore the union of the curves x_i = c on which one orbit point
> meets the critical point c. Two of them, for different i, cross transversally at a doubly superstable point.
>
> In a map of higher dimension the leading multipliers are eigenvalues of the return map's Jacobian M, not a product
> of scalars. Near a would-be crossing the two leading multipliers mu1, mu2 are well separated from the rest (here by
> a factor of ten or more). They are then the eigenvalues of M on a two-dimensional invariant subspace, with trace
> tau = mu1 + mu2 and determinant delta = mu1 mu2. A real multiplier vanishes exactly where delta = 0.
>
> In the decoupled one-dimensional limit delta factorizes like the one-dimensional multiplier, delta ~ eps s_a s_b,
> where eps is the transverse multiplier and s_a, s_b vanish on the two would-be superstable curves. In general
> delta = eps s_a s_b - kappa, where kappa collects the coupling of the two critical directions through the
> transverse ones. A crossing needs kappa = 0, one extra condition that nothing in a higher-dimensional map
> enforces. For kappa != 0 the zero set is a hyperbola: two arcs that approach each other and turn away. Wherever
> tau^2 < 4 delta between them, the two leading multipliers form a complex pair. This is a Type-B-like geometry; if
> higher-order terms join the arcs, they form a closed loop, Type-A-like.
>
> Because the language model contracts strongly but finitely in the transverse directions, kappa is small and the
> arcs come close without meeting. At the period-6 body's most contracting cell the leading pair is complex, as
> required in the gap between the arcs when the saddle value of delta is positive.

## Questions

1. Is the mechanism correct as stated?
   - In particular: is the crossing of two zero curves of a real leading multiplier non-generic for a map of
     dimension at least two?
   - Is the reduction to the leading 2 x 2 invariant block (trace, determinant, "a real multiplier vanishes where
     delta = 0") sound, and under what spectral-separation condition?
   - Is the hyperbola unfolding (a Morse saddle of delta with a nonzero saddle value) the right local picture?
2. Is the decomposition delta = eps s_a s_b - kappa a fair heuristic, and what exactly is kappa in terms of the
   monodromy (for example, off-diagonal entries of M in the basis of the two critical directions)? Is there a cleaner
   statement?
3. How does this relate to Façanha-Oldeman-Glass, who assume a two-dimensional map with a determinant of fixed sign?
   Here delta is the determinant of a block of an 896-dimensional map and can change sign. Is "Type-B-like" and
   "Type-A-like" acceptable language under this mechanism? What should we avoid saying?
4. What quantitative test on our grid data would best confirm or refute the mechanism?
   - Per cell we have: the leading multiplier (real and imaginary parts), the next modulus, per-step factors of the
     leading multiplier (they multiply to it exactly), a clean flag (|mu1| > 10 |mu2|) and the token loop.
   - For example, fitting delta near the closest approach and comparing the predicted gap or the complex-pair region
     with the observed arcs. What should a careful paper report, given the project rule that a heuristic must be
     compared quantitatively with the full computation and its discrepancy stated?
5. Exact wording. Which sentences can go into an arXiv paper tonight as explanation, and which must stay labeled as
   interpretation? Please propose replacement text for the explanation paragraph above, at most about 180 words, for
   readers of Chaos.
