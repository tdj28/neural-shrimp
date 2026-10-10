# Full native tangent-gauge reanchor — adapted exploratory pilot, 10 October 2026

TLDR: The proved period-18 boundary can continue in state and critical-mode space
when its projection into learning rates turns. This pilot changes the coordinate
used to follow it; it does not assert a cusp or join the distant numerical curve.
The original ten proved charts and their evidence remain unchanged.

## Inputs and rationale

Use only the completed unit002 endpoint at old exact output gauge 19/32768,
execution `692943933a061c718bfb8ccdb8d17c8a5c99feb8`, raw
`/data2/neural-shrimp-data/explore/eigenpair-locus-prototype/attempt-002-march-target`.
Outer manifest `6984dff87e855a4fa43d036358688ebf536d9feac88bbe1a0fcdc2d0fc9a2715`,
run manifest `f1bb8a6acc3b6b50e53d71ef8fa57be1ea1c9035f3dfbec8d5d6cacc1c8a2cad`.
The full final chart, its durable compact acceptance, complete unit summary,
exact native map/data, all source files and the original fold ancestry must join.
Partial or failed tails are ineligible. No new data or trajectory initialization.

The fixed output coordinate need not be regular at a cusp: the two-rate
projection can have zero derivative while the lifted state/mode locus remains
regular. A second +1 multiplier is a different possible obstruction. Existing
midpoint spectral diagnostics remain descriptive and do not certify simplicity.
Neither finding would by itself connect this locus to the distant numerical
curve or prove a cusp. This change is disclosed after unit002's passing outcome.

## New containment/proposal adapter; unchanged proof kernel

At the old exact endpoint, retain its entire 470-dimensional physical root box.
Fix m=mid(X_initial) and the parent's exact dyadic preconditioner C. Evaluate
H(m,s_old) once. For exactly three iterations, enclose DH over
hull(m,X), form K=m−C H(m)+(I−C DH(hull(m,X)))(X−m), and intersect X with K.
The known parent root remains inside without a contraction assumption. Empty or
nonfinite returns fail; every returned image survives before intersection.
This inherits existence, primitive period, phase and mode sign from the parent;
it is not a new root existence argument. Rounding intersections outward is safe.

Take the saved endpoint predictor derivative v=S+T*delta, rounded to an exact
Arb midpoint. Normalize in the raw Euclidean metric of all native state, mode,
and actual-rate coordinates, then round the normal n to exact dyadics. Preserve
n·v>0 for forward orientation. This metric keeps existing coordinate units; it is
not an intrinsic measure. Record the rounding, norm, and interval transport
vector t=v/(n·v), including n·t−1. Record ||v_rates||/||v|| separately from the
existing complementary +1 distance; these are predictor diagnostics only.

Choose the largest dyadic halfwidth no larger than delta_old*||v||, capped at
1/16 and required to be at least 2^-30. The center is the localized midpoint and
new parameter center is zero. For conditioning, propose coordinate radii from
(17/16) max[r_old + |t| sum |n_i|r_old_i,
2*sup|X−mid(X)−t*n·(X−mid(X))|] + 2^-100, rounded upward to exact dyadics and
encoded by a covering power-of-two base and positive weights. No clipping,
refitting or proof on the old derivative domain. The output-coordinate error
is no longer artificially constrained to its old fixed-output-gauge radius.

Run at most two fresh calls of the unchanged quadratic kernel (256 bits,
two midpoint Newton proposals), conditional second on first passing. All 470
strict coordinate inclusions, weighted contraction, primitive divisors and
whole correlated anchor tests remain required. The first anchor uses the full
localized physical box and its projected new gauge interval, never a midpoint
or the old exact-parameter shortcut. The second has the unchanged exact native
endpoint join. No rebox or fallback is included in this first reanchor attempt.

## Resources and evidence

The localization takes four native assemblies =72 calls. Two quadratic charts
take254 calls including preconditioner ticks; total326, with a hard counter360.
CPU/BLAS1, nice10, internal180 seconds, external200+5 seconds, monitored owned
RSS2GiB (not a kernel-enforced resident-memory ceiling), virtual address4GiB,
worker512MiB raw (504 ordinary +8 finalization), launcher1MiB/log128KiB.
All returned phase arrays, localization matrices/images/intersections, proposal,
full chart returns and final failure/partial evidence are immutable and hashed.
No target acquisition until the separate bounded runner and synthetic cost
profile are committed. Profile is source-only synthetic470D containment algebra,
with no trained-map calls; measured original two-chart timing and profile cost
must pass the target's declared conservative cost gate. Exact profile pins and
command will be appended before target execution. No paid/GPU/holdout work.

## Source-only controls

Both critical signs on a coupled native period-two map preserve full phase/mode
identity and fresh anchor gates. Additional controls cover a cusp with zero rate
projection, nonzero raw tangent; coupled affine known-root containment when m is
outside X; singular exact C; nonexact C rejection; positive exact radius encoding
and carried orientation; zero tangent rejection; image retention on empty
intersection; and no subsequent assembly after durability failure. Nine controls
passed in0.19 seconds before the adapter freeze. Independent source review and
empirical reanchor acquisition remain pending.
