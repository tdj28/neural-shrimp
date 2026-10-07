# Manuscript under review (rendered text of the working draft)

This is the compiled PDF of the private working draft converted to plain text, so every generated number is expanded and citations appear as [n] with the reference list at the end. Figures are represented by their captions. Placeholders mark sections still in progress.

Beyond the edge of stability: periodic windows, quasi-periodicity
and phase locking in the learning-rate plane of gradient descent
Authorship to be finalized by the human researcher
Private working draft, not for distribution

AI-use disclosure. Generative-AI tools helped implement, audit, execute, interpret, visualize, review,
and draft this study. The author selected the research question, authorized the compute, has inspected
the artifacts, and is responsible for the final text and claims. This is an independent, non-peer-reviewed
preprint. Verify numbers against the released receipts before relying on them.
Draft note: every clause above must be confirmed by the author before release.
Abstract
Full-batch gradient descent with a large learning rate rarely converges to a minimum: it
oscillates at the edge of stability, cycles, or behaves chaotically. Maps of trainability over
learning-rate planes summarize this as a binary label, convergent or divergent, and so do not say
which of these behaviors occurs where. We resolve the dynamics behind such maps in two small
tanh networks trained with separate learning rates for their two layers.
(1) In a prospectively designed and independently audited study, the learning-rate planes
contain 61 numerically certified attracting periodic orbits of the full parameter-space update
map, with periods from 4 to 58, whose domains are surrounded mainly by rerun-confirmed chaos.
Across six coarse maps, 87.1% to 99.97% of the cells that a published binary convergence score
labels convergent are not fixed points after 20,000 updates.
(2) In exploratory analyses, the windows follow a quasi-periodic route to chaos. On all 35
traced slices whose starting point is the edge-of-stability period-2 orbit, that orbit loses stability
through a Neimark–Sacker bifurcation of the two-step map. On one slice the crossing multiplier
predicts a rotation number of 0.2180, and the first quasi-periodic cell rotates at 0.2180. Of the 56
testable windows, 35 are well-ordered circle orbits with rotation m/q and period 2q or a doubling
of it, against 3.58 expected for random angles. Only 2 of 61 windows meet draft criteria for
classical shrimp anatomy.
Takeaway: in these networks, the region that a binary map calls trainable holds a structured
transition from oscillation to chaos, including quasi-periodic training and phase-locked windows.
The mechanism is exploratory and still needs a test on fresh networks and seeds; nothing here
concerns stochastic or large-scale training or establishes a training benefit.

Study status. The map study (Section 4) was designed and pushed before any outcome and
checked by an independent audit; the mechanism analyses (Section 5) are exploratory. Provenance
for every figure and quoted number is described in Section 9.

1

Introduction

Training with a learning rate beyond the classical stability limit is common practice. At the edge of
stability, the sharpness of the loss hovers near 2/η while the loss still decreases over long horizons
[5], and simplified models show period doubling and chaos as the step size grows [4, 11, 16]. Over
1

two-dimensional hyperparameter planes, the boundary between trainable and divergent settings is
fractal [19, 20, 25, 29]. These maps record whether training stays bounded and settles, not what it
does.
We ask what the dynamics inside such maps are. We treat full-batch gradient descent with
separate learning rates for the hidden and output layers as a two-parameter family of maps on
parameter space. For every cell of a learning-rate grid we classify the long-run behavior (fixed point,
periodic orbit with its period, chaos, drift, divergence), certify periodic orbits with Newton’s method
on the exact Jacobian, and follow selected orbits and their stability boundaries through the plane.
Prior work. Periodic windows in one-dimensional learning-rate sweeps date back at least to
van der Maas et al. [30]. Period-colored maps over optimizer hyperparameters exist without full-state
verification [28]. Continuation of flip and catapult thresholds has been carried out for one-dimensional
reductions of gradient descent [11], which also derives complex unstable multipliers of a 2-cycle
in a two-mode linear network. Riddled basins and learning-rate-dependent reproducibility are
documented by Ly and Gong [21]. Arnold tongues and Neimark–Sacker bifurcations are reported
for learning dynamics in games [22], and heavy-ball momentum has cycles that rotate by 1/K [10].
This paper’s contribution.
• A period-resolved description of the learning-rate plane of full-batch gradient descent in two
small networks, with 61 certified attracting windows embedded in chaos, confirmed by an
independent audit.
• A direct comparison with the published binary convergence score on the same cells, showing
that most cells it calls convergent do not converge.
• Exploratory evidence that the windows are organized by a quasi-periodic route: Neimark–
Sacker bifurcation of the period-2 orbit, invariant circles with measured rotation numbers,
phase-locked orbits with period 2q at rotation m/q, and rotation-number staircases.
• Open code, receipts and a verification script that re-derives every figure and quoted number.
Not claimed.
• No claim about stochastic, minibatch or large-scale training, other architectures or real data
(a real-data test is in progress; Section 5).
• No claim that periodic or quasi-periodic training helps or hurts generalization or speed.
• The mechanism in Section 5 is exploratory; it has not yet been tested on fresh networks and
seeds.
• Classical shrimp organization is shown for a small minority of windows only.

2

Gradient descent as a family of maps

Let L(θ) be the full-batch training loss of a network with parameters θ ∈ Rn , split into hidden-layer
and output-layer blocks. Gradient descent with per-layer learning rates is the map
θt+1 = F (θt ) = θt − P ∇L(θt ),

P = diag(ηh 1hidden , ηo 1output ),
2

(1)

a family of maps indexed by the two rates (ηh , ηo ). Its Jacobian is DF (θ) = I − P H(θ) with H the
Hessian of L. An orbit of period p is stable when all eigenvalues (multipliers) of DF (θp−1 ) · · · DF (θ0 )
lie inside the unit circle, and it loses stability when a multiplier crosses the circle: at +1 (fold), at
−1 (flip, which doubles the period) or as a complex pair e±2πiω (Neimark–Sacker, which creates an
invariant closed curve) [18].
Why the route starts at period 2. DF (θ) is similar to the symmetric matrix I − P 1/2 HP 1/2 ,
so at a fixed point all multipliers are real and a fixed point can only lose stability by fold or flip.
The edge-of-stability oscillation is the period-2 orbit born at that flip. For a period-2 orbit {θa , θb },
the two-step Jacobian is similar to Sb Sa with Sj = I − P 1/2 H(θj )P 1/2 . A product of two symmetric
matrices has real eigenvalues when one factor is positive definite, but at the edge of stability each
Sj has a negative eigenvalue in the oscillating direction, so complex multipliers, and with them a
Neimark–Sacker bifurcation of the two-step map, become possible. (This elementary argument is
ours; the same observation appears in a literature check summarized in Section 8.)
Circle maps and phase locking. On an invariant circle the dynamics has a rotation number
ρ, the mean fraction of a turn per iterate. When ρ = m/q is rational the motion can lock onto a
periodic orbit; the locked parameter sets are Arnold tongues, rotation numbers along a parameter
path form a devil’s staircase, and the widest plateau between two locked rationals is typically their
Farey mediant [1, 12, 13]. Breakdown of the circle leads to chaos [2, 7, 23, 24]. Because our circle
belongs to the two-step map, a locked orbit at rotation m/q has period 2q for F .
Shrimps. In two-parameter families of dissipative maps, periodic windows inside chaos often take
a characteristic “shrimp” shape bounded by fold and flip curves [8, 9, 15, 27]. We test for this
anatomy explicitly rather than assume it.

3

Methods

Networks and data. S1 is the published mean-field tanh configuration of the fractal-trainability
study [25]: width 8, inputs of dimension 8, no biases, as many standard-normal training points as
parameters, mean squared error. S1R is S1 with weight decay 10−3 . S2 is a 13-parameter tanh
regressor (two inputs, three hidden units, biases, weight decay 10−3 ) fitted to 24 points from a
random teacher. We refer to a network and seed together, for example S2-28011. Training is
deterministic full-batch gradient descent in float64.
Maps. Each map is a 256 × 256 grid of logarithmically spaced rate pairs (10 maps of 65,536 cells,
coarse and zoomed). Every cell runs 20,000 updates from the same initial parameters. The last 384
updates are screened for the smallest full-state recurrence period up to 64; the largest finite-time
Lyapunov exponent (FTLE) is estimated with one tangent vector; cells are labeled fixed point,
periodic, chaotic (FTLE above 0.005 per update), drift (parameter norm still growing), unresolved
or divergent. The published binary score marks a run convergent when its normalized loss, averaged
over the last 20 of the first 1,000 updates, is below one [25].
Certification and embedded windows. A connected component of equal-period cells qualifies
as an embedded window when its period is at least 3, no cell of its exterior ring is a fixed point or
has a proper divisor period, and at least half of the ring is chaotic on reruns. Its representative
orbit is refined by damped Newton iteration on the exact Jacobian and must close, be primitive and
3

be attracting with margin 10−7 . The design, thresholds and analysis were pushed before outcomes
were seen, and an independent audit repeated the checks.
Continuation and boundary type. From the center of a window, the certified orbit is continued
along 64 rays in the rate plane until it loses stability, located by bisection, and the boundary is
typed by the crossing multiplier: fold (+1), flip (−1) or Neimark–Sacker (complex pair). The same
procedure traces where the period-2 orbit loses stability along slices of fixed output rate.
Rotation numbers and well-ordering. For a non-periodic cell we take the two-step samples
zk = θ2k , project them on their first two principal axes, and define the rotation number as the mean
unwrapped angle increment in turns per two steps. For a certified orbit with q two-step points we
sort the points by angle and call the orbit well-ordered (a Birkhoff orbit of rotation m/q) when
every step advances the angular rank by the same shift m with gcd(m, q) = 1. Orbits of 2 or 3
points pass trivially; only tests on at least 5 points count as evidence. The null is the pass rate of
the same test on uniformly random angles. A refined version also orders points along the shortest
k+1
closed tour through them and tests the points of F 2
(k = 1, 2, 3), which detects orbits born by
period doubling inside a tongue; its null sums the per-test rates.

4

Periodic windows inside chaotic training

Evidence label: prospectively frozen, audited.
The learning-rate planes of S1R and S2 contain 61 embedded windows that qualify on both
the producer’s and the auditor’s evidence (62 and 63 respectively; the disagreements all sit at the
50% ring threshold). All 154 attempted orbit certifications verified. Confirmed periods range from
4 to 58 and include the odd periods 5, 11, 13, 17 and 29. Their exterior rings were checked with
2,801 reruns, and 157 of 160 randomly chosen chaotic cells stayed chaotic on a rerun with a different
tangent vector.
Figure 1 compares these dynamics with the published binary score. In every coarse map,
most of the cells the score calls convergent are not fixed points: 87.1% to 99.97%. The published
configuration without weight decay (S1) has 0 periodic cells in both seeds: its runs diverge, stay
chaotic, or drift toward large weights while the loss slowly falls. Adding weight decay makes the
windows appear.

4

S2-28011 (coarse)
log10 hidden-layer rate

A Published score

published score: divergent (and class divergent)

B Dynamical class

C Period of periodic cells

3.0

3.0

3.0

2.5

2.5

2.5

2.0

2.0

2.0

1.5

1.5

1.5

1.0

1.0

1.0

0.5

0.5

0.5

42.2% scored
convergent

−1

0

64

32

16
1

−1

0

1

−1

0

1

log10 output-layer rate log10 output-layer rate log10 output-layer rate

S2-28011-zoom (zoom)
log10 hidden-layer rate

D Published score

E Dynamical class

2.05

2.05

2.05

2.00

2.00

2.00

1.95

1.95

1.95

1.90

1.90

1.90

1.85

1.85

1.85

1.80

1.80

1.80

1.75

1.75

1.75
1.70

99.98% scored
convergent

−0.4

−0.2

1.70

F Period of periodic cells
8

4

1.70
−0.4

−0.2

screened period

published score: convergent

2
−0.4

−0.2

log10 output-layer rate log10 output-layer rate log10 output-layer rate

G What the score-convergent cells of every coarse map are after 20,000 updates
S1-28001

29,697 cells; 99.97% not fixed points

S1-28002

31,192 cells; 99.82% not fixed points

S1R-28001

25,627 cells; 89.7% not fixed points

S1R-28002

23,697 cells; 87.1% not fixed points

S2-28011

27,659 cells; 94.0% not fixed points

S2-28012

27,623 cells; 91.1% not fixed points

0.0

0.2

0.4

0.6

0.8

1.0

fraction of cells the published score calls convergent (1,000 updates)
fixed point

periodic (p >= 2)

chaotic

drift (norm growth)

unresolved

late divergence

Figure 1: The published binary score calls almost everything in these planes convergent, while the same cells hold fixed points, periodic orbits of many periods, chaos and
drift. Panels A to F: one network (S2-28011), coarse and zoomed maps; the rectangle marks the
zoom. Panel G: composition of the cells the score calls convergent, for all six coarse maps; only
the blue segment converges to a fixed point. The comparison is cell by cell on the same runs; the
score uses the first 1,000 updates and the classes use 20,000. Labels other than certified orbits are
finite-time screening labels. Generated by paper/gd-route/figures/f1_binary_vs_period.py; receipt
figures/f1_binary_vs_period.provenance.json.

5

5

An exploratory quasi-periodic route

Evidence label: exploratory. The analyses in this section were chosen while exploring and used
data that shaped the hypothesis.

5.1

The period-2 orbit loses stability through a Neimark–Sacker bifurcation
A Periodic cells, Neimark-Sacker curve, predicted roots

B Birth rotation number
64

2.05

Neimark-Sacker curve of the period-2 orbit

0.24

1.95

1.90

16

3/13

1.85

1.80

2/9

8
1/5

1.75
2/11

4

1.70

screened period (periodic cells)

log10 hidden-layer learning rate

32

crossing angle theta / 2 pi (turns per two steps)

2.00

3/13 (period 26)

0.23

2/9 (period 18)

0.22

0.21

1/5 (period 10)

0.20

0.19

2/11 (period 22)

0.18

1/5

0.17
1.65

2/11

not mapped

−0.40

2
−0.35

−0.30

−0.25

−0.20

−0.15

−0.10

log10 output-layer learning rate

−0.4

−0.3

−0.2

−0.1

0.0

log10 output-layer learning rate

Figure 2: The edge-of-stability period-2 orbit loses stability through a complex pair of
multipliers along the whole traced curve. A: periodic cells of a 1,024 × 1,024 map (shade: period) with the traced Neimark–Sacker curve (black) and the points where its crossing angle equals a
rational m/q with q ≤ 13 (orange). B: crossing angle along the curve. On 5 of 40 slices the starting
point was not on the period-2 orbit; those slices break the curve. Predicted tongue roots are interpolated between traced slices, and tongues near their roots are narrower than one map cell, so the map
alone cannot confirm them. Generated by paper/gd-route/figures/f2_ns_tongue_roots.py; receipt
figures/f2_ns_tongue_roots.provenance.json.

We continued the period-2 orbit of S2-28011 toward larger hidden-layer rates along 40 slices of
fixed output rate. On all 35 slices that started on it, the orbit loses stability with a complex pair
of two-step multipliers crossing the unit circle (Figure 2). The curve is U-shaped, with its lowest
point at log10 ηh = 1.650, and the crossing angle ω ranges from 0.1670 to 0.2401 turns per two steps,
passing 2/11, 1/5, 2/9 and 3/13.
The crossing angle predicts the rotation of the training that follows. On the slice through the
period-18 window, the curve interpolated to that slice places the bifurcation at log10 ηh = 1.738
with ω = 0.2180; the first quasi-periodic cell of an independent staircase computation lies at 1.738
and rotates at 0.2180.
6

A Period 2: inward spiral

B Quasi-periodic circle

C Locked 2/9, period 18

D Chaotic: broken circle

light: early, dark: late

log10 hidden rate 1.7375
rotation 0.2178, FTLE -0.0002

log10 hidden rate 1.7811
rotation 0.2159, FTLE 0.0001

log10 hidden rate 1.8564
rotation 0.2222, FTLE -0.0072

log10 hidden rate 1.8942
rotation 0.2298, FTLE 0.0504

Figure 3: Four stages of the route along one slice: an inward spiral to the 2-cycle, a closed
invariant curve, nine locked points, and a broken curve. Each panel shows every second parameter
vector of one run projected on its own two principal axes. The rates are chosen by a fixed rule from the staircase
receipt (the period-2 cell closest to onset, the median quasi-periodic cell, the middle of the longest 2/9 plateau,
the median chaotic cell between the 2/9 and 1/4 plateaus), not by eye. Projections onto two axes can hide
structure in the remaining dimensions. Generated by paper/gd-route/figures/f6_route_portraits.py;
receipt figures/f6_route_portraits.provenance.json.

Figure 3 shows what training looks like along that slice. Beyond the bifurcation, the two-step
samples lie on a closed curve (rotation 0.2159, FTLE 0.0001); inside the period-18 window they
are nine points (rotation 0.2222, FTLE -0.0072); beyond it the curve thickens and breaks (FTLE
0.0504).
Placeholder. Rotation-number map of the S2-28011 plane overlaid with the Neimark–Sacker curve, testing
whether the rotation just beyond the curve equals the crossing angle everywhere along it (computation in
progress).

5.2

Rotation numbers form staircases

Along the slice through the period-18 window of S2-28011, the rotation number rises from the
bifurcation value, locks at 2/9 for log10 ηh from 1.833 to 1.879, and locks again at 1/4 before chaos
(Figure 4). The labeled plateaus are 5/23, 7/32, 2/9 and 1/4 in S2 and 1/9, 2/17, 1/8, 3/23 and 2/15
in S1R. Consecutive plateaus are Farey neighbors (|m1 q2 − m2 q1 | = 1) in 3 of 3 pairs in S2, against
49% per pair for randomly drawn fractions in the same range, and in 4 of 4 pairs in S1R, where the
null is 100% because every fraction with a small enough denominator in that range appears. The
Farey pattern is therefore weak evidence in S2 and uninformative in S1R.

5.3

Most windows are phase-locked circle orbits

Across the 61 windows, the 3,904 rays end at 1,316 folds, 1,012 flips and 1,395 Neimark–Sacker
crossings (181 rays reached the end of their range or stayed unresolved; Figure 5A). Tongue edges
are folds of the locked orbit on the circle, while flips and secondary Neimark–Sacker crossings are
expected where a locked orbit loses stability off the circle.
Among all certified orbits with at least 5 two-step points, 65 of 115 are well-ordered in their
principal plane, against 1.57 expected for random angles. For the windows themselves, 35 of 56
testable windows pass the refined test, against a multiplicity-corrected null of 3.58; 7 of them pass
first at a period-doubled level, for example period-20 orbits that are well-ordered under four steps
7

B S1R-28002: rotation-number staircase

1/4

0.25

rotation number of the two-step map

rotation number of the two-step map

A S2-28011: rotation-number staircase

0.24
0.23
2/9
5/23

0.22

7/32

0.21
0.20
0.19

0.15

0.14
3/23

0.13

2/15

1/8

0.12
1/9

2/17

0.11

0.10

0.09

1.70

1.75

1.80

1.85

1.90

1.95

2.00

2.05

1.250 1.275 1.300 1.325 1.350 1.375 1.400 1.425 1.450

log10 hidden-layer rate (log10 output rate -0.0856)
period 2 (decaying-spiral rotation)

log10 hidden-layer rate (log10 output rate 0.9385)

periodic, period >= 4

quasi-periodic candidate

chaotic candidate

Figure 4: Along slices, quasi-periodic cells trace continuous rotation numbers and periodic
cells sit on rational plateaus. Plateaus are detected automatically (locked cells with period 2q or a
doubling of it, rotation within 6 × 10−4 of m/q) and labeled when they hold at least 8 cells. In S1R
two rotation branches coexist over the same rates, a sign of coexisting attractors. Chaotic cells outside
the plotted range are not shown. Generated by paper/gd-route/figures/f3_staircases.py; receipt
figures/f3_staircases.provenance.json.

with rotation 2/5, twice 1/5. Every passing window has period 2q or 4q for its tongue denominator
q (Figure 5C). Of the 5 odd-period windows, 0 are well-ordered (null 0.17), so they need a different
explanation.

5.4

Classical shrimp anatomy is the exception

Draft criteria for classical shrimp anatomy require fold and flip boundaries each on at least 8 of 64
rays, at least half of the flip rays opening onto period 2p in a local map, and agreement between ray
and map edges on at least three quarters of rays. 15 windows meet the boundary-count part and 2
meet all of it. The clearest, the period-18 window of S2-28011 (Figure 6), has 46 fold, 15 flip and 3
Neimark–Sacker rays; ray and map edges agree within two cells on 61 of 64 rays, and 10 of its flip
rays open onto period 36. Neimark–Sacker boundaries, absent from classical shrimps, are common:
41 of 61 windows have at least 8 such rays.
Placeholder. Minimal two-parameter model (scalar factorization with weight decay and two learning
rates), testing whether the same route appears where invariant circles can be drawn in the full state space
(computation in progress).
Placeholder. Real-data network trained with a single learning rate, testing whether the period-2 orbit
loses stability by Neimark–Sacker and whether loss spectra show the rotation (stream in progress).
Placeholder. Consequence for training: whether rotation number or locking predicts reproducibility or loss
beyond the Lyapunov exponent and sharpness, against periodic and chaotic controls (stream in progress).

8

A How each window's periodic orbit loses stability (64 rays per window)
S1R (72 parameters)

share of 64 rays

1.0

S2 (13 parameters)

0.8

0.6

0.4

0.2

0.0
8 30 30 58 16 38 30 20 44 16 14 34 14 14 34 12 46 30 44 42 20 4 50 20 16 20 22 30 34 46 22 20 20 4 6 18 34 12 16 32 26 11 10 10 29 17 26 36 24 14 14 13 6 5 50 24 26 22 26 22 20

confirmed windows, labeled by period (sorted by system, then fold share)
flip

Neimark-Sacker

B Well-ordering versus random-angle null

no loss within range / unresolved

C Period versus tongue denominator (35 windows)

1.0
observed
null expectation
0.8
35/56

0.6

65/115
31/61

0.4

0.2

2q x 4

60

certified window period

share well-ordered (Birkhoff circle orbit)

fold

0.0
certified orbits,
>= 5 points

confirmed
windows

confirmed,
refined

2q

50
40
30

x4
x4

20

x4

x6
x3
x5
x2

10
0/5

2q x 2

period = 2q
period = 2q x 2

0
0

confirmed,
odd period

5

10

15

20

25

30

tongue denominator q (rotation m/q of the two-step map)

Figure 5: Across all 61 windows, stability boundaries mix fold, flip and Neimark–Sacker,
and most testable windows are well-ordered circle orbits whose period is twice the tongue
denominator or a doubling of it. A: boundary type along 64 rays per window. B: observed share
of well-ordered orbits (gray) against the random-angle null (orange) on the same test. C: certified period
against the tongue denominator for windows passing the refined test; the multiplicity of coincident points
is printed. Orbits of 2 or 3 points pass trivially and are excluded from the refined share. Generated by
paper/gd-route/figures/f5_window_census.py; receipt figures/f5_window_census.provenance.json.

9

A Window S2-28011-zoom:587 (period 18): typed stability boundaries
B Crossing multipliers (unit circle)
1.0

Im of dominant multiplier

log10 hidden-layer learning rate

1.90

ray and map edges within 2 cells: 61/64 rays
flip rays opening onto period 36: 10/15
(local map at 256 x 256)

1.88

1.86

1.84

1.82

0.5

0.0

−0.5

−1.0
−1.0

1.80

−0.5

0.0

0.5

1.0

Re of dominant multiplier
−0.105 −0.100 −0.095 −0.090 −0.085 −0.080 −0.075 −0.070

log10 output-layer learning rate
fold (+1): 46 rays
flip (-1): 15 rays

Neimark-Sacker (complex pair): 3 rays
period 18

period 36
other periodic

Figure 6: The period-18 window of S2-28011 has a fold-bounded body with flip edges opening
onto period-36 bands. A: local map from the shared initial parameters with the window’s orbit continued
along 64 rays; markers show where and how it loses stability. B: crossing multipliers on the unit circle.
This window was selected as the clearest example and is one of only 2 meeting the draft shrimp criteria; it
is not representative (Figure 5). Generated by paper/gd-route/figures/f4_window_anatomy.py; receipt
figures/f4_window_anatomy.provenance.json.

10

6

Discussion

Everything in this section is interpretation. The audited results in Section 4 stand on their own,
and the hypotheses here need their own prespecified tests.
The picture that fits the exploratory evidence is the quasi-periodic route familiar from forced
oscillators and circle maps [2, 7, 13]: the edge-of-stability oscillation is a period-2 orbit; past a
Neimark–Sacker bifurcation of the two-step map, training moves on an invariant circle with a
rotation number set by the crossing angle; rational rotations lock into Arnold tongues; and the circle
breaks into chaos, with period doubling inside tongues along the way. The argument in Section 2
explains why this route needs at least period 2 in gradient descent, and why per-layer rates are
not required for it in principle. Under this reading, the windows of Section 4 are mostly tongue
windows, and classical shrimps are a special case.
Three observations do not fit yet. The odd-period windows are not circle orbits; a one-dimensional
period-doubling route of the kind organized by shrimp hubs [15] is a candidate. Windows also
coexist with other attractors, so maps started from one initial point show which attractor that point
reaches rather than the full existence region of a window. And the evidence comes from two small
networks.
A confirmatory test on fresh seeds and at least one new network would fix in advance: (a) a
quasi-periodic band between the period-2 region and chaos, with rotation numbers that start at
the crossing angle; (b) locked windows with period 2q at rotation m/q; (c) Neimark–Sacker loss of
stability of the period-2 orbit at the band’s onset; and (d) an explicit null for any ordering claim
about plateaus.

7

Limitations

The networks have 13 and 72 parameters and are trained on synthetic data with deterministic
full-batch updates in float64. Screening labels are finite-time; chaotic labels rest on a single tangent
vector and reruns. Rotation numbers and well-ordering use two-dimensional projections of curves
that live in 13 or 72 dimensions. The mechanism analyses were chosen after seeing related outcomes,
and the staircase and Neimark–Sacker analyses cover two slices and one curve. No result addresses
generalization, training speed, or modern training practice.

8

Related work

Edge-of-stability dynamics and their simplified models include Cohen et al. [5], Kalra et al. [16],
Chen and Bruna [3], Song and Yun [26], Chen et al. [4], Cohen et al. [6] and Hofmann [11]; chaos in
deterministic training is discussed by Kong and Tao [17] and Jiménez-González et al. [14]. Fractal
trainability boundaries are studied by Sohl-Dickstein [25], Torkamandi [29], Liu [20] and Liang and
Montúfar [19], and riddled basins and reproducibility by Ly and Gong [21]. Shrimp-shaped windows
and their organization are reviewed in Gallas [8, 9], Jones [15], Stoop et al. [27]; circle maps, Arnold
tongues, devil’s staircases and torus breakdown in Arnold [1], Aronson et al. [2], Curry and Yorke
[7], Jensen et al. [12, 13], Ostlund et al. [23], Ruelle and Takens [24]. A literature check using generic
queries found no report of quasi-periodic training, Arnold tongues or rotation-number staircases for
optimizers acting on losses; Hofmann [11] derives complex multipliers of a 2-cycle without rotation
numbers, and tongues are documented in learning dynamics for games [22]. Every citation in this
draft still needs checking at its primary source.

11

9

Reproducibility

Code, plans and compact receipts live in the project repository; bulky raw maps live in a separate data
store referenced by SHA-256. Every figure is generated by a script in paper/gd-route/figures/
that reads saved evidence and writes a receipt with input hashes, the generator’s hash, output
hashes, alt text and the values it displays. Every number in the text is a macro generated by
paper/gd-route/scripts/build_numbers.py; data/numbers.json records each macro’s source
file, hash and computation, and provenance.json indexes all receipts. paper/gd-route/scripts/
verify.py regenerates every figure and number and fails on any difference. The map study
corresponds to the repository’s EXP-028 plan and audit; its plan, receipts and audit are linked from
docs/EXP-028-results.md.
Placeholder. Author contributions, acknowledgments, license choice and public code and data links are
human decisions to be made before release.

References
[1] Vladimir I. Arnold. Small denominators. I. Mappings of the circumference onto itself. American
Mathematical Society Translations, Series 2, 46:213–284, 1965. Translation of the 1961 Izvestiya
article; details to verify.
[2] D. G. Aronson, M. A. Chory, G. R. Hall, and R. P. McGehee. Bifurcations from an invariant circle
for two-parameter families of maps of the plane: a computer-assisted study. Communications
in Mathematical Physics, 83:303–354, 1982. doi: 10.1007/BF01213607. verified at Crossref
2026-10-07.
[3] Lei Chen and Joan Bruna. Beyond the edge of stability via two-step gradient updates. In
International Conference on Machine Learning, 2023.
[4] Xuxing Chen, Krishnakumar Balasubramanian, Promit Ghosal, and Bhavya Agrawalla. From
stability to chaos: Analyzing gradient descent dynamics in quadratic regression, 2023. Version
1 (journal version, TMLR 2024, to verify); verified at arXiv 2026-10-07.
[5] Jeremy M. Cohen, Simran Kaur, Yuanzhi Li, J. Zico Kolter, and Ameet Talwalkar. Gradient
descent on neural networks typically occurs at the edge of stability. In International Conference
on Learning Representations, 2021.
[6] Jeremy M. Cohen, Alex Damian, Ameet Talwalkar, J. Zico Kolter, and Jason D. Lee. Understanding optimization in deep learning with central flows. In International Conference on
Learning Representations, 2025. arXiv version 2 verified 2026-10-07; venue to verify.
[7] James H. Curry and James A. Yorke. A transition from Hopf bifurcation to chaos: computer
experiments with maps on R2 . In The Structure of Attractors in Dynamical Systems, volume
668 of Lecture Notes in Mathematics, pages 48–66. Springer, 1978.
[8] Jason A. C. Gallas. Structure of the parameter space of the Hénon map. Physical Review
Letters, 70:2714–2717, 1993. doi: 10.1103/PhysRevLett.70.2714. verified at Crossref 2026-10-07.
[9] Jason A. C. Gallas. Dissecting shrimps: results for some one-dimensional physical models.
Physica A, 202:196–223, 1994. doi: 10.1016/0378-4371(94)90174-0. verified at Crossref 2026-1007.
12

[10] Baptiste Goujaud, Adrien Taylor, and Aymeric Dieuleveut. Provable non-accelerations of the
heavy-ball method. Mathematical Programming, 2025. doi: 10.1007/s10107-025-02269-2. verified
at Crossref 2026-10-07; related open problem arXiv:2502.19916 (Two Riddles in Heavy-Ball
Dynamics).
[11] Thomas Hofmann. The map behind the flow: Finite-step gradient descent as a dynamical
system, 2026. Version 1; verified at arXiv 2026-10-07.
[12] M. Høgh Jensen, Per Bak, and Tomas Bohr. Complete devil’s staircase, fractal dimension, and
universality of mode-locking structure in the circle map. Physical Review Letters, 50:1637–1639,
1983.
[13] M. Høgh Jensen, Per Bak, and Tomas Bohr. Transition to chaos by interaction of resonances
in dissipative systems. I. Circle maps. Physical Review A, 30:1960–1969, 1984.
[14] Pedro Jiménez-González, Miguel C. Soriano, and Lucas Lacasa. Leveraging chaotic transients
in the training of artificial neural networks. Physical Review Research, 8:L022032, 2026. doi:
10.1103/t5p9-kv5w. verified at Crossref 2026-10-07.
[15] Timothy D. Jones. Topological origins of a bi-parameter periodicity hub for the rössler attractor,
2012. Version 1; verified at arXiv 2026-10-07.
[16] Dayal Singh Kalra, Tianyu He, and Maissam Barkeshli. Universal sharpness dynamics in neural
network training: Fixed point analysis, edge of stability, and route to chaos. In International
Conference on Learning Representations, 2025.
[17] Lingkai Kong and Molei Tao. Stochasticity of deterministic gradient descent: Large learning
rate for multiscale objective function. In Advances in Neural Information Processing Systems,
2020.
[18] Yuri A. Kuznetsov and Hil G. E. Meijer. Numerical Bifurcation Analysis of Maps: From Theory
to Software. Cambridge University Press, 2019.
[19] Shuang Liang and Guido Montúfar. Gradient descent with large step sizes: Chaos and fractal
convergence region. In International Conference on Learning Representations, 2026. arXiv
version 3 verified 2026-10-07; venue to verify.
[20] Yizhou Liu. Complex fractal trainability boundary can arise from trivial non-convexity, 2024.
Version 1; verified at arXiv 2026-10-07.
[21] Andrew Ly and Pulin Gong. Fractal basin geometry and the limits to predictability and
reproducibility in deep learning, 2025. Version 2; verified at arXiv 2026-10-07.
[22] Konstantinos Metaxas and Themistoklis P. Sapsis. Risk-sensitive learning in population games
under extreme events: Bifurcations and chaotic dynamics, 2026. Version 2; verified at arXiv
2026-10-07; content claim (tongues) to verify.
[23] Stellan Ostlund, David Rand, James Sethna, and Eric Siggia. Universal properties of the
transition from quasi-periodicity to chaos in dissipative systems. Physica D, 8:303–342, 1983.
[24] David Ruelle and Floris Takens. On the nature of turbulence. Communications in Mathematical
Physics, 20:167–192, 1971.
13

[25] Jascha Sohl-Dickstein. The boundary of neural network trainability is fractal, 2024. Version 1;
verified at arXiv 2026-10-07.
[26] Minhak Song and Chulhee Yun. Trajectory alignment: Understanding the edge of stability
phenomenon via bifurcation theory. In Advances in Neural Information Processing Systems,
2023.
[27] Ruedi Stoop, Philipp Benner, and Yoko Uwate. Real-world existence and origins of the spiral
organization of shrimp-shaped domains. Physical Review Letters, 105:074102, 2010. doi:
10.1103/PhysRevLett.105.074102. verified at Crossref 2026-10-07.
[28] Sergiy Sveleba, Ivan Katerynchuk, Yaroslav Shmyhelskyy, Lucjan Pelc, Volodymyr Brygilevych,
and Nazar Sveleba. Multi-parameter dynamic learning maps of neural networks trained with
Adam optimiser. Electronics and Information Technologies, 34, 2026. doi: 10.30970/eli.34.2.
verified at Crossref 2026-10-07.
[29] Bahman Torkamandi. Mapping the edge of chaos: Fractal-like boundaries in the trainability of
decoder-only transformer models, 2025. Version 2; verified at arXiv 2026-10-07.
[30] Han L. J. van der Maas, Paul F. M. J. Verschure, and Peter C. M. Molenaar. A note
on chaotic behavior in simple neural networks. Neural Networks, 3:119–122, 1990. doi:
10.1016/0893-6080(90)90050-U. verified at Crossref 2026-10-07.

14

