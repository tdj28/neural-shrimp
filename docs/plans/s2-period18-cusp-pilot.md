# Two local cusp candidates on the carried period18 fold, 10 October UTC

Exploratory local certificate attempt; no public release, priority claim or
complete shrimp claim. Root owns this unit. Existing proof kernels are unchanged.

## Evidence and fixed choice

Saved-only combined-summary003 (manifest245f74ec72a67c9893e196444687dedab271924a6be9d47a4b700054e37293fb)
reduces448 retained boundary nodes and their returned analytic Jacobians. There
is a simple-mode fold coefficient sign bracket across target003 plus031 to
fold-plus004 plus000 (c+.030409 to-.043778; critical gap>.43). A second lies
between004 plus026/027 (c-.041418 to+.008340; gap>.34). The first crosses a
panel boundary, so it is not listed in summary003's within-panel sign-change
array; raw endpoint continuity is explicitly retained. These are different
from the earlier extra+1 crossing/pole at003 plus012/013. Other modes are
numerically unstable (radius roughly2.1 and3.68); neither candidate is an
attracting orbit.

Target cusp-a starts at003 plus031; cusp-b starts at004 plus027. Full parent
inventories and exact saved extended dyadic coordinates are bound in the driver.
Use the returned mixed Jacobian to propose h from the floating bordered solve
[A q;q^T 0](h,beta)=(-B(q,q),0). This is only a seed. Do not change seed, radius,
precision, Newton count or acceptance gates automatically after failure.
Preserve both attempts whether either passes. No new scan, orbit initialization,
neuron symmetry quotient or protected data.

## Mathematical claim and checks

For R=F^18(theta;hidden,output)-theta, A=R_theta, solve the41-dimensional square
system (R,Aq,Ah+B(q,q),(q.q-1)/2,q.h)=0. The native state remains13-dimensional.
CuspJet differentiates the analytic gradient map along theta+tq+t²h/2, retaining
full base derivatives of f,d1,d2 with respect to all41 unknowns. Thus the second
and third equation blocks are d1-q and d2-h; their gradients include exact third
derivatives of F^18. Independent scalar polynomial, multistep tanh finite-difference
and unchanged GD mixed-jet comparisons are synthetic controls, not evidence
about the target.

At256bits take eight midpoint Newton proposals, fixed radius2^-80, then true
whole-box augmented Jacobian. Strict Krawczyk inclusion and infinity norm<1
prove a locally unique normalized augmented root. Full-box bordered inverse
[A q;q^T 0] proves simple+1 and supplies w with w^Tq=1. Every proper divisor
1,2,3,6,9 must be excluded on native phase boxes. Both actual rates must be
positive throughout. The full-box cubic coefficient is
w^T(C(q,q,q)+3B(q,h))/6; using the retained gradients this is
w^T(g2_theta*q+2*g1_theta*h)/6. Require it excludes0. For each parameter j,
[A q;q^T0](y_j,beta_j)=(-R_j,0) gives beta_j=-a_j,
a_j=w^TR_j, b_j=w^T(R_theta,j*q+B(q,y_j)). Require determinant of the2x2
matrix with rows a,b excludes0. This is a local versal unfolding; there is
no global uniqueness claim.

Deflate Q=M-qw^T. Floating eigenvectors propose an EXACT dyadic complex basis;
Acb validates its inverse and S^-1 Q S. Outward Gershgorin discs must each lie
strictly inside or strictly outside the unit circle. This permits unstable
modes but excludes other neutral modes, as required for a generic dynamical
cusp with one center direction. A largest spectral radius>1 alone would not
satisfy this gate. No claim of attraction is made for the neutral cusp.

The Lyapunov–Schmidt equations/signs were independently checked by
optimizer_claim_check before target use; full kernel/driver review precedes
acquisition, outcome replay/nested proof review precedes manuscript claims.
The MatContM manual section5.1 describes the discrete-map cusp and its local
fold wedge: https://webspace.science.uu.nl/~kouzn101/NBA/ManualMatcontMNov2017.pdf.
Lessard–Pugliese2025 supplies related computer-assisted cusp methodology for
ODEs, not a ready-made theorem for our map:
https://arxiv.org/html/2404.00535v2 (sections1 and3),
https://doi.org/10.3934/dcdsb.2024181. Our unit-circle gate is specifically for
the discrete return map, not the ODE imaginary-axis condition. The cubic and
unfolding formulas above are explicitly derived/reviewed for R.

## Resource and preservation contract

First two synthetic18-step13D evaluations of the41D augmented system, using
24 artificial samples and no target inputs. Profile worker15s, existing tested
supervisor20+5s; require20*max(evaluation_seconds)+5<85s and exact sources,
settings/libraries/CPU threads. This is twice the ten target evaluations plus
5s overhead, not an outcome guarantee. A committed measured receipt pins the
profile before either target. Each target worker90s, external200+5s through
the existing owned-process supervisor;180 native steps maximum (8 Newton plus
center and whole-box passes),CPU/BLAS1,nice10,512MiB RSS/768MiB address,
32MiB worker (30MiB ordinary,2MiB finalization reserve). Check resource headroom.
Every returned Newton residual/Jacobian and proposal, center evaluation, root
gate and final full witness is written exclusively/fsynced. All models/inputs,
source, versions, PIDs/exits/failed or interrupted attempts receive immutable
complete inventories. No target restarts or adaptive settings in this card.

Every proposal and the final full theta/rates/q box must remain within0.1 in
fixed coordinate scales(theta1,hidden10,output.1,q1) from its chosen seed.
Curvature h is not used for that proximity check. This is a proposal-neighborhood
constraint, not a validated branch connection. Native arithmetic/cusp proof
must pass independently of that floating-parent proximity.

Exclusive base /data2/neural-shrimp-data/explore/certificates/s2-cusps-2026-10-10/,
profile attempt-001-profile, targets cusp-a-attempt-001 andcusp-b-attempt-001.
Driver tools/certify_s2_cusps.py --mode profile/target --panel cusp-a/cusp-b
--out <exclusive path> --profile <profile/run> --profile-manifest <SHA>.
No new proof will be called a complete shrimp until boundary curves, interior
and stability are connected rigorously under the separately stated target.

Pre-target source amendment: independent audit found the driver used restore_ball from rigorous_connection.py without listing that source in its archived pins. Add the exact dependency, preserve profile001, and use fresh source-bound profile002. Also strengthen receipt eligibility to require explicit successful exit/nice and finite nonnegative RSS/elapsed, positive evaluation times, exact recomputed projection. Focused bad-receipt controls cover negative/NaN/infinite resources, wrong source, counters and exit. No target has been acquired, and no mathematical kernel or scientific settings change.
