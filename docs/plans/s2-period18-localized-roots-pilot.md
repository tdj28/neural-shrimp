# 9 October 2026 — localize the proved period18 branch before more stability coverage

Exploratory adaptation of the reviewed width1/256 primitive branch, not a new
orbit search, confirmation set or shrimp verification. Existing root, primitive
period, anchor and join receipts remain immutable. The question is whether the
uniform correction image (about1.7e-6) can be tightened locally enough to make
periodic quadratic stability certificates inexpensive. No connection to the
fold at hidden rate75.764 or any flip follows from these probes.

## Mathematical operation and dependencies

The frozen parent is execution916c76d35c1786b906991bebcc16406864e62c25, with
worker manifestdbb6ea5e57606788cbf1a7659ebcff2b4e91ebad2d11ca7779b88c32e2dcc9b7,
at `/data2/neural-shrimp-data/explore/certificates/s2-period18-connection-meanvalue-2026-10-09/attempt-001-target/run`.
Independent saved384-bit reviewe3cfd534c90abd5200b7b59ef7458830717b9d09 proves
both root/primitive/anchor/join gates. Exact original consumed model, data, native
18 phases, fixed output1848862009810323/2251799813685248 and actual hidden-rate
semantics are reconstructed from the unchanged C1 receipt. The loader requires
all original parent source hashes and the complete immutable parent inventory.

For a selected closed subinterval I, the parent proves a root in p(I)+E.
Recenter the affine predictor at the exact midpoint hc using a dyadic center;
retain every rounding displacement in E. Evaluate H(center,hc), and acquire
A=H_theta and B=H_h on p(I)+hull(0,E). Then

`R = H(center,hc) + (A*s+B)*(I-hc)`

encloses H(p(h),h). For the known root, its error e satisfies
`e = -C*H(p(h),h) + (I-C*A_h)*e`, with A_h inside the saved enclosure. Intersect
E with the resulting image. Repeat exactly three times, without new inverses,
Newton proposals, search or stopping upon an attractive result. This refines
containment of an existing root; it is not a standalone existence theorem.
Hull(0,E) is essential when a correction box excludes zero. Arb intersection
rounding may expand a last bit; the saved ball is still a sound enclosure.

Acquire the18 true native phase Jacobians on the refined root boxes, then use
the unchanged exact-dyadic periodic Lyapunov proposal and Arb positivity gates.
All matrices, phase domains, intersections, metric witnesses and failed gates
are retained. A passed probe proves attraction only on its own selected closed
interval, conditional on the parent existence/identity proof. Parent primitive
period and joins persist. No new individual-orbit count or full wider coverage.

## Fixed sequential stages

1. One resource profile, no target input: original24-sample synthetic zero-label
   ridge1 tanh model, zero exact fixed point repeated18 times (explicitly not
   primitive18). One analytic18-phase center/J acquisition proposes exact C,
   then the original-image metric plus three refinements and localized metric.
   Exactly126 wrapped/252 unit calls if complete. Zero-root existence is exact;
   this only exercises the engineering path. Full profile must finish with a
   passing localized metric, exit0, no live owned PIDs, matching sources,
   libraries, threads and finite measured resources. Projection3*time+5<55s and
   3*profilebytes<120MiB is required for either two-probe stage.
2. Two center floors, one per parent chart, exact h=chart center. Keep the
   original full correction-image metric result as a diagnostic before the
   three refinements. Exactly108 wrapped/216 unit calls per chart; retain both
   even if a sufficient metric gate fails. No full interval claim from a floor.
3. Only if both refined floors pass, one outer finite slice per chart of exact
   width2^-15: chart0's leftmost and chart1's rightmost slice. This is the same
   slice width as the previous passing narrow-branch coverage, at a different
   parameter location, with newly localized roots. Exactly90 wrapped/180 unit
   calls per chart. This is a deliberately sparse diagnostic, not128-slice
   coverage or an automatic partition search. Retain both outcomes.

Profile/floor/probe internal limits20/60/60s, external25/75/75s plus5s cleanup;
ticks300/560/400; wrapped calls126/216/180 and unit calls252/432/360. Each
worker2GiB RSS at operation checkpoints, CPU/BLAS1, nice>=10, worker128MiB with
120MiB ordinary allowance plus1MiB launcher. Each compressed member has a
decompressed256MiB cap and exact uncompressed checksum. Full source, stages,
partial counters, failures and immutable manifests are saved; owned group is
killed/checked at completion. A source-fixed single launch per stage uses an
exclusive output. Resource safety is also enforced by bounded parent launch;
RSS checks are cooperative, not a kernel memory cgroup.

Current host is shared and recently had only5.9GiB available RAM. Recheck before
each stage; require>=3GiB available and coordinate with root to avoid overlap
with another>1GiB proof. Do not touch foreign jobs or use GPU/paid/holdout data.

## Invocation and interpretation

Run `tools/certify_s2_period18_localized_roots.py` with mode profile/floor/probe,
under `/data2/neural-shrimp/.venv/bin/python`, `PYTHONPATH=src`, CPU/BLAS1 and
nice10. Base output:
`/data2/neural-shrimp-data/explore/certificates/s2-period18-localized-roots-2026-10-09/`.
Use exclusive `attempt-001-profile`, `attempt-001-floor`, `attempt-001-probe`.
Floor/probe must pass `--profile <profile/run> --profile-manifest <exact SHA>`;
probe also requires `--floor <floor/run> --floor-manifest <exact SHA>`. Commit
measured profile/floor receipts and exact next-stage pins before each launch.
Parent performs brief mathematical/source review before first acquisition.

If original floors fail but refined floors pass, the root-image hypothesis is
supported. If refined floors fail, inspect metric conditioning and irreducible
derivative bounds before any wider partition. If finite slices fail despite
floors passing, parameter variation or the fixed tangent predictor is limiting;
do not automatically spend on full coverage. A pass justifies designing a
coverage implementation with measured cost and independent review.

## Scaling beyond this pilot

The fold is about4.43 hidden-rate units away; success on a1/256 interval is not
a scalable boundary connection. Tighter local roots may permit cheap verified
endpoint handoffs and better predictor centers. First-order residual estimates
still grow quadratically in chart width; a higher-order predictor needs a
rigorous higher-order residual enclosure, not only a floating quadratic fit.
Near a fold fixed-rate A becomes singular, so any full connection will need a
validated augmented chart (for example pseudo-arclength) and containment of
the separately certified fold root in an exact common-parameter uniqueness
domain. Overlapping boxes or separately plausible endpoint spectra do not
establish that connection. This card does not launch that extension.
