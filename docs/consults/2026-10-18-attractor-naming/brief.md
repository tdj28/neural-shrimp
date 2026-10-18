# Consult 16: may we call these sets attractors (strange or chaotic), and is "return map" the right term? (terminology)

## System

A language model's soft-token loop as a smooth map F: R^896 -> R^896, iterated in discrete time (Qwen2.5-0.5B-Instruct,
float64 unless stated).
- F(x) = E^T softmax(z_alpha(x)/T): E is the input embedding matrix (151,936 tokens); z_alpha(x) are the next-token
  logits when the model reads a fixed chat prefix followed by the single input embedding x; a steering vector alpha v
  is added to the residual stream after layer 12 at every position; T is the temperature. The one-slot window is
  replaced each step, so the map is autonomous. One iteration = one full forward pass that "writes" one soft token.
- The cells below lie on the row log10 T = -0.148 of a 512 x 512 float64 map over (alpha, log10 T).

## Evidence

**Strongly chaotic cell** (alpha 0.07248; the row's largest map exponent, 0.355 over 300 + 300 steps):
- 512 starts (the map's fixed start displaced by 5 % of its norm in random directions), 400 transient steps, 600
  recorded: 498 never show a period up to 64 (relative recurrence tolerance 1e-9); 14 settle on a coexisting
  period-4 orbit.
- Lyapunov spectrum by QR (4 vectors, forward-mode JVPs, 3000 steps after 1000): 0.274, -1.052, -1.486, -1.856 per
  step; Kaplan-Yorke dimension 1.26 (running estimates 1.25-1.28).
- Principal axes of the recorded states: 3 axes carry 77 % of the variance, 12 carry 99.4 %. The states lie on thin
  folded filaments; magnifications by 12.5 and 125 show layers within layers.
- Correlation sum over 8 principal coordinates (8000 states): the local slope wanders between 0.42 and 0.91 over
  r in [7e-6, 1e-1] and is not constant (within 10 %) over any decade; we report no correlation dimension.
- A single step moves the state by a median 0.30 (full 896 dimensions), more than the mean distance of the states
  from their centroid (0.21); the 3 shown coordinates carry a median 93 % of each jump's length.
- Takens check: a scalar of the output distribution (log of the largest next-token probability) in delay
  coordinates: false nearest neighbours 70.8 % (m = 1), 10.4 % (2), 3.4 % (3), 2.3 % (4), 1.9 % (5, 6); points that are
  nearest neighbours in the 3-delay reconstruction are a median 0.00037 apart in the full state, random pairs 0.30.

**Banded chaos past a period-doubling cascade** (alpha 0.06626): exponents 0.036, -1.208, -1.283, -1.684 (D_KY 1.030);
each trajectory visits three thin bands in turn. The period-3 orbit that lost stability in the cascade, continued by
Newton (residual 4.2e-13), is a saddle with real multiplier -1.772 (next 0.040). A fundamental domain of its unstable
manifold, iterated under F^6, lies within 9.8e-6 (median 3.9e-7) of all 100,600 recorded states of one band, in 12
principal coordinates; the band is 0.028 wide.

**Periodic cells:** at alpha 0.0734, 509 of 512 starts converge to one period-8 orbit. At alpha 0.06468 an interval
proof (Krawczyk, 2048-bit balls) shows a primitive attracting 2-cycle and 3-cycle coexisting, each with a certified
basin neighbourhood (radii 2^-681 and 2^-700); on a plane of 9216 starts all settle on one of the two (float32).

All chaos labels are finite-horizon numerical labels (stated as such in the paper).

## How we use the words now

- Paper paragraph "What the attractors look like"; figure title "the strange attractor of the soft-token loop";
  "its states lie on thin folded filaments"; "the band is the closure of the saddle's unstable manifold, to the
  resolution of the record"; "the loop is a map, not a flow ... a single trajectory hops between filaments".
- Figure panel (d): "return map: stretching and folding", a plot of s_{n+1} against s_n for the first principal
  coordinate s.
- Public videos: "A strange attractor of a language model's feedback loop"; "Each color is the word the loop writes
  next".
- The author asks: "if it's really just an 896-dimensional return map, maybe we shouldn't call it an attractor?"

## Questions (answer briefly and concretely; replacement sentences welcome)

1. **Attractor.** For a smooth map of R^896 iterated in discrete time, with this evidence, is "attractor" the right word
   for the observed set? Which definition should the paper name (Milnor's, a topological attractor with an attracting
   neighbourhood, the support of a physical/SRB measure), and what may numerics alone support ("numerically observed
   attractor", "chaotic attractor", finite-time qualifiers)? Does the coexisting period-4 attractor (14 of 512 starts)
   change the wording?
2. **Strange or chaotic.** Is "strange attractor" justified by a non-integer Kaplan-Yorke dimension (1.26) and layered
   filaments, without a correlation-dimension scaling range? Or should we write "chaotic attractor" (positive exponent)
   and reserve "strange" (fractal geometry)? What wording for the paper, and for public videos?
3. **"Return map."** Our map is natively discrete, not a Poincare section of a flow. Is calling the loop itself "a return
   map" wrong? Is the panel label "return map" for s_{n+1} against s_n acceptable, or should it be "next-iterate map",
   "first-return plot of one coordinate", "delay plot"?
4. **Projections.** May a principal-coordinate projection be captioned "the attractor", or must it say "a projection of
   the attractor"?
5. **The manifold sentence.** With the numbers above, is "the band is the closure of the saddle's unstable manifold, to
   the resolution of the record" adequate, or an overclaim? Better phrasing?
6. **Anything else** a referee in nonlinear dynamics would object to in this framing.
