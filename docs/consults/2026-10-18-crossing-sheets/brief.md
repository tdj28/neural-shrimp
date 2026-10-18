# Consult 13: why two families of windows cross in both orders (research strategy)

## System

A language model's soft-token loop as a smooth map of R^896. The model is Qwen2.5-0.5B-Instruct and the computation
is float64.

- **The map:** F(x) = E^T softmax(z_alpha(x)/T).
  - E is the input embedding matrix (151,936 tokens).
  - z_alpha(x) are the next-token logits when the model reads a fixed chat prefix followed by the single input
    embedding x. The prefix is a system message, then the user turn "Hello!".
  - A steering vector alpha·v (a contrastive "philosophy" direction) is added to the residual stream after layer 12
    at every position.
  - The window is one slot, which is replaced each step, so the map is autonomous.
- **Maps over (alpha, log10 T):** every cell starts from one fixed state x0, the embedding of the prefix's last token
  (a newline). Each cell is labeled by its attractor: the period, the loop of most probable tokens over one period,
  and the largest Lyapunov exponent from a renormalized twin.

## What the map looks like (window alpha 0.0625-0.0775, log10 T -0.17 to -0.125, 512 x 512)

- **The " is it" 2-cycle** fills the left. It is bounded on the right by its fold curve (saddle-node; normal form and
  passage times verified).
- **Stripes:** periodic windows of period 2N+6 accumulate on that fold with distances near 1/N^2 (type-I
  intermittency). Their texts are ". This is there (is it)^N is a is a". They run nearly vertically and tilt
  slightly with T. Each window is bounded by a fold and a flip; the side the fold is on is exchanged between periods
  12 and 16.
- **A crescent** of the period-3 loop "\n\nThe existence" (a paragraph loop). It contains a period-doubling cascade
  with a chaotic core. It lies inside the 2-cycle's region, so the two coexist there.
- **The band.** The first doubling of the crescent's period-3 orbit, continued to larger alpha, is a period-6 band.
  At alpha 0.070 it lies between flips at log10 T -0.1393 and -0.1414. It runs almost perpendicular to the stripes
  and crosses them.
- **Chaos** below the band (the band's own cascade) interrupts the stripes there. Period-3 "\n\nThe existence" cells
  also appear above and right of the band (up to alpha 0.0721) between stripes.

## The question (from the human)

The two "arcs", the crescent and band (3-family) and the 2-cycle with its stripes (2-family), look like two surfaces
crossing in 3D. In some places the 3-arc lies over the 2-arc, in others the 2-arc over the 3-arc, and on either side
of a crossing the order can differ. Examples:
- The crescent covers the 2-cycle's region.
- Along the band, most stripe cells are replaced by the band, but some stripe windows show through near the band's
  edges.
- In the accumulation region, the crescent family's period-6 cells alternate with stripe windows along one row.
- Along the column alpha 0.0708, going up in T: a period-10 stripe (below the band), then the band's cascade (12,
  then 6), then the period-3 loop (above the band).

We want to explain all of it, so that a reader finds it obvious.

## Evidence so far

1. **The map shows only one attractor per cell, chosen by x0.** Coexistence is real:
   - At one point a start on the 2-cycle stays there, while x0 reaches the period-3 loop.
   - A census found the band 6-cycle and the period-10 stripe coexisting at (0.0708, -0.1396).
2. **Basins at three crescent cells** (log10 T -0.156; Codex Task 28). Of 128 random single-token embeddings, 107
   reach the 2-cycle and 20 the period-3 orbit (alpha 0.0642). Top-64 eight-token mixtures give similar fractions.
   - x0 and its first 16 iterates lie nearer the crescent's "\n\n" phase. The "\n\n" token has probability 0.74 at
     that phase.
   - Nearest-phase proximity does not predict where random starts go (26 of 248 correct, against 222 for "always
     2-cycle").
   - Along segments between attractor states, the destination changes several times (5, 7 and 8 coarse changes).
     Brackets refine to width 1/16384. This is not a single threshold, and not proven fractal.
3. **The crescent's left edge at log10 T -0.148 is not its fold.**
   - Natural continuation of the period-3 orbit in alpha reaches a fold at alpha 0.064444 (multiplier 0.997 at the
     last point).
   - x0 reaches the 2-cycle up to alpha 0.064675, and the period-3 loop from 0.0647.
   - At 0.0647 the reached period-3 orbit has a twin exponent of -0.45 per step (|mu| about 0.26 per period). Settle
     times of x0 stay at 20-42 steps on both sides of the switch on a grid of spacing 2.5e-5. A bisection and a
     search for the orbit on the basin boundary are running.
4. **Running now (6 GPUs; 256 alpha x 192 T over the same window):**
   - seeded maps: every cell starts from x0, from the crescent's period-3 state, from the band's 6-cycle state, or
     from the period-10 stripe state;
   - adiabatic sweeps (each column starts from the previous column's final states) in alpha+, alpha-, T+ and T-;
   - per cell: the settle step and the mean probabilities of 9 key tokens (a fingerprint that separates the two
     families, since the 3-family puts about 0.2-0.25 on "\n\n" and the 2-family about 0).

## Working explanation

Each family is a sheet of attractors over the plane. The sheets overlap far more than the fixed-start map shows.
Visible edges are of two kinds:
- (a) bifurcation curves, where the visible family itself is born or dies;
- (b) switch curves inside overlaps, where both exist and the basin boundary moves across x0.

The 3-family often wins in the overlap because x0 starts near its "\n\n" phase. The 2-family wins where the
3-family's basin shrinks, for example next to its folds, where a newborn node has a tiny basin. The crossing pattern
then traces where basin boundaries (presumably stable manifolds of saddle orbits) pass through x0.

## What we ask

1. **Is this explanation right and complete for the "both orders" pattern?** Which mechanisms might we be missing?
   Candidates: basin-boundary metamorphoses, boundary crises of the chaotic parts, saddle-node bifurcations of
   chaotic attractors, fractal or Wada boundaries, transient chaos, and riddling (unlikely here). Which signatures
   would distinguish them in our data?
2. **The minimal set of decisive tests and figures** that makes it obvious to a broad reader, for example:
   - 3D sheet renderings;
   - cross-sections along one row with all branches;
   - transient-length ridges (log divergence at basin boundaries, against inverse-square-root passage near fold
     ghosts);
   - saddle orbits located on the boundary;
   - a map from a second, different fixed start.
3. **Quantitative predictions** we could test, such as how the switch curve's offset from the fold scales with
   distance along the fold, or how basin fractions scale near a fold.
4. **Pitfalls:** finite transients mislabeling cells, the twin direction, the label changes where tokens tie, the
   sweep direction, and claims we should not make.

Please be concrete and prioritize. We have one day before submission, local CPUs, and a modest GPU budget.
