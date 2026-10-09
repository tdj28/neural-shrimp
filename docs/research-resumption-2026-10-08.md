# Research progress — 9 October 2026

A small nonlinear neural network can settle into an attracting training cycle
that repeats only after **36 gradient steps**. We now have a computer-assisted
proof for all 13 trainable parameters, with every shorter possible period
excluded. Nearby, a separate proved period-doubling mechanism predicts stable
36-step cycles emerging as an 18-step cycle loses stability. At a different
selected boundary, the proved doubling mechanism creates unstable cycles.
This begins to explain behavior inside and around a candidate shrimp; it does
not yet certify the entire shrimp or establish a training benefit.

## What is now proved

| Result | Evidence | Scope |
|---|---|---|
| Attracting primitive 18-cycle at the candidate's center | Computer-assisted proof, independently reviewed | One full-state orbit of the exact-real 13-parameter tanh GD map |
| Primitive 18-cycle saddle-node on ray 0 | Computer-assisted proof, independently reviewed | Local meeting of an attracting and a saddle cycle; no validated connection to the center |
| Subcritical flip on ray 13 | Computer-assisted proof, independently reviewed | Nearby unstable primitive 36-cycles on the side with an attracting parent 18-cycle |
| Supercritical flip on ray 51 | Computer-assisted proof, independently reviewed | Nearby attracting primitive 36-cycles after the parent 18-cycle loses stability |
| Attracting finite primitive 36-cycle | Computer-assisted proof, independently reviewed | One explicitly specified fixed-rate full-state orbit; all eight proper divisors excluded |
| Initial continuous attracting 18-cycle interval | Computer-assisted proof, independently reviewed | Width 2^-19 in actual hidden rate, anchored to the original center |
| Wider connected attracting primitive 18-cycle interval | Computer-assisted proof, independently reviewed | Width 1/1024, with the original center and the shared chart endpoint identified exactly |
| Further connected primitive 18-cycle interval | Computer-assisted proof with independent saved-arithmetic review | Width 1/256; existence and connection proved, uniform attraction unresolved |

The supercritical flip occurs near ray parameter 0.14049631414519. Its total
multiplier crossing is negative and cubic normal-form coefficient positive;
the noncritical modes are attracting. The finite 36-cycle separately has a
rigorous monodromy power bound `||M^512||inf < 0.117942`. It fixes the exact rates
`5107950665142887/70368744177664` and
`7317034811349323/9007199254740992`. The flip instead uses an exact exponential
rate family. Their numerical association and simultaneous proof do **not**
establish a continuous connection between those exact maps.

The new independent review reconstructs the exact inputs, replays both proofs,
checks saved interval arithmetic and succeeds again with higher precision in
strictly nested boxes. These checks share the analytic derivative kernels and
Arb/FLINT. The flip reviewer authored that kernel; root separately reviewed its
mathematics. This is not a separate higher-derivative implementation or formal
proof-assistant verification. See the [new producer results](reviews/s2-ray51-certificates-outcome.md)
and the canonical [36-cycle receipt](../certificates/s2-period36-receipt.json).

There are now five individually certified attracting-orbit instances: two
13-dimensional period-10 cycles, one 13-dimensional period-18 cycle, one
13-dimensional period-36 cycle, and one 72-dimensional period-2 cycle. Neutral
fold/flip points and an interval through an already counted orbit do not add
instances. The older 13-dimensional examples are our own private results.
We make no largest-network or bibliographic priority claim.

## What changed in the shrimp picture

The cross-checked census still contains 61 periodic windows with periods 4–58
and two exploratory classical-shrimp anatomy candidates. The strongest,
S2-28011-zoom:587, has historical coarse labels of 46 fold, 15 flip and 3
complex-pair rays. These are not a certified census of the first boundary:
finer continuation on ray 13 found a complex instability pocket followed by
re-entry before the later flip. The coarse radial samples skipped that pocket.
The complex crossings remain numerical candidates, not proved nondegenerate
Neimark–Sacker bifurcations.

Likewise, the historical 10/15 adjacent coarse period-36 labels do not identify
which local doubled branch was observed. Ray 13 and ray 16 produced unstable
small-amplitude doubled cycles in the numerical follow-up. Ray 51 produced
attracting ones and now has the rigorous results above. A visual island or an
adjacent period label is insufficient to establish the local mechanism.

## Work continuing at this checkpoint

The main gap is a validated connection from the certified center to the proved
boundaries. Two joined multiple-shooting charts prove a continuous primitive
18-cycle on an interval 512 times wider than the first interval. Root-image
narrowing and six matrix-product bounds did not establish uniform attraction.
A periodic quadratic metric also remained inconclusive on each entire chart:
midpoint metrics were well resolved, but interval uncertainty prevented
positivity in some phases. Fixed parameter subdivision resolved this obstacle.
All 32 closed slices passed their periodic quadratic stability inequalities
without changing the parent existence proof. Independent review replayed both
charts, checked exact root joins, and reevaluated all 576 native Jacobians at
higher precision; every slice passed again. See the [combined connection
receipt](../certificates/s2-period18-connection-receipt.json). Earlier failed
bounds remain retained and do not imply actual instability.

The mean-value gradient enclosure then extended the connected primitive branch
to width **1/256**, four times wider again. Both root and all primitive-period,
original-anchor and shared-endpoint gates passed. An independent reviewer
rebuilt the 234-dimensional shooting matrices and repeated the saved Krawczyk
and join arithmetic at 384 bits. This review shares the source-reviewed
derivative enclosures; it did not reacquire them. Attraction on this larger
interval remains unresolved. See the [wider-branch review](reviews/s2-period18-meanvalue-independent-review.md).

The next numerical obstacle is reducing the uncertainty in those wider root
enclosures before spending on another full stability cover. The current root
image is about 1.7e-6, roughly seven times the preceding image; its effect on
stability floors has not yet been measured. Localize or refine the root
enclosures on parameter slices, measure the resulting stability floor, and
only then size a covering run. All completed attempts are preserved; no
larger slice run has been launched.

After a useful connection method is established, continue toward the boundary
and its organizing curves. Opposite criticality at two isolated flips alone
neither proves they belong to one flip curve nor establishes a generalized
flip point. Complete shrimp anatomy, cusps, ancestry and useful behavioral
control remain separate research questions.

## Scaling, manuscript and preservation

No practical training, inference, LLM or nanoGPT speed benefit has been shown.
The prior LM pilots were negative, the CIFAR correction remains unresolved,
and the six-reference curvature approximation failed. A useful scaling target
is identifying actual multi-step dynamical directions through matrix-vector
actions, with an explicit cost/accuracy comparison. Protected holdouts remain
untouched.

The [current reading draft](../paper/gd-route/gd-route-working-draft.pdf) has
36 pages, 347 verified source-derived entries and ten figures. It includes both flip
types, the finite 36-cycle, a six-point numerical comparison and the corrected
coarse-boundary interpretation. The independently reviewed attracting interval
is included on page 11. Clean compilation and visual inspection passed; root
rechecked all source, proof, figure and PDF hashes and all 347 number entries.
The latest wider primitive-only extension is documented above and in its
review; the paper makes no additional attraction claim for it. Local CPU
suffices for this work;
no new paid run, GPU experiment, external message or submission occurred.
Regular private Git pushes preserve source and compact evidence. Raw proofs,
failed attempts and reviews have additional hash-verified copies on a separate
physical disk on this machine; see [backup receipts](backups/).

The completed research units and manuscript refresh are preserved. No owned
scientific process remains running. The broader shrimp and scaling research
is unfinished; the next concrete step is the enclosure refinement above.
