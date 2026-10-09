# Completing evidence for S2-28011-zoom:587 — 9 October 2026

## Current amendment: after both criticalities and a finite 36-cycle proof

The original targets below have advanced. Ray 13 is rigorously subcritical,
with a numerically detected earlier complex instability pocket and re-entry.
Ray 51 is rigorously supercritical, and a separate fixed-dyadic primitive
36-cycle is rigorously attracting; both passed independent replay and stronger
nested proofs. Ray 16 has numerical unstable doubled cycles. The exact ray
family and finite fixed-rate map remain distinct; no germ-to-finite-point
connection has been proved.

The center now anchors an independently reviewed attracting interval of width
2^-19 in actual hidden rate. A producer has proved two joined primitive charts
of total width 1/1024; independent outcome review is in progress. Their uniform
stability remains unresolved. Native, framed and root-image-narrowed product
bounds failed, and cyclic quadratic metric bounds failed on some interval
phases despite well-resolved midpoint metrics. Fixed parameter subdivision
will reduce the diagnosed uncertainty while retaining the parent root,
primitive-period and exact common-parameter join proofs.

Continue from these artifacts; do not rerun the completed flip/36-cycle units.
The next scientific result is a useful validated continuous branch, then
boundary connection and organization. Subcritical and supercritical signs at
isolated points do not establish a generalized flip or a shared flip curve.
Read `docs/research-resumption-2026-10-08.md` and the new source-bound receipts
for the current evidence. The earlier note below is retained as dated design
history; its statements that no flip or connecting run had started are obsolete.

## Original design note


The next useful result would explain how this particular period-18 training
pattern loses stability to period36, while proving that our interior orbit and
boundary points belong to the same cycle family. We now have an attracting
interior18-cycle certificate and a local18-cycle fold certificate (both independently reviewed; see
`docs/reviews/s2-period18-fold-certificate-independent-review.json`). Neither proximity nor these two isolated proofs
establishes their branch connection. This is a scientific note, not an
executable freeze, resource allocation, or launch authorization; no new target
calculations were performed.

**A concrete flip target.** The lowest-index historical flip ray is13, with
saved log-rate direction `[0.29028467725446233,0.9569403357322089]`, bracket
`[0.2540544368966948,0.25405443768249825]`, and dominant multiplier approximately
−0.999999883. Start there, carrying the complete18-phase orbit from certified
row111. The reported10/15 “flip→36” statistic is only the aggregate first
adjacent coarse-map cell label in `explore_a2_survey.edge_metrics`; it does not
identify a saved primitive36 orbit for ray13. A specific18→36 association still
needs full-state correction, closure, proper-divisor exclusions, phase matching,
and stability qualification on the doubled branch. Both rates vary on ray13:
state an exact one-parameter rate family and differentiate both rates; do not
inherit ray0's fixed-output assumption or silently replace the historical curve.

**What a rigorous flip needs.** Write `G=F_lambda^18`, `R=G-theta`, `M=G_theta`.
Enclose a root of `(R,(M+I)q,(q^Tq-1)/2)`, prove primitive18 and algebraically
simple−1, and normalize `M^T p=-p`, `p^Tq=1`. Enclose the other modes away from
the unit circle; a power bound for `M+qp^T` can establish an attracting
complement. Since `M-I` is invertible, the orbit-branch derivative is
`theta'=-(M-I)^(-1)G_lambda`. Require the **total** multiplier crossing
`p^T(G_theta,lambda q + G_theta,theta[theta',q])` to exclude zero.
For `B=D_theta^2G`, `C=D_theta^3G`, also require

```
b = p^T C(q,q,q)/6 - p^T B(q,(M-I)^(-1)B(q,q))/2 != 0.
```

Here b multiplies the scalar cubic term after quadratic elimination. Its sign
and the crossing direction determine the doubled branch's side/stability;
a −1 eigenvalue alone is insufficient. This convention follows
[Kuznetsov's normal-form derivation, pp.11–12](https://webspace.science.uu.nl/~kouzn101/NBA/UvA_lect3.pdf).
The present fold jets give D²G, not D³G. The new proof would need validated
third directional/mixed derivatives of the GD update—hence fourth loss
derivatives—plus validated linear solves, with no floating derivatives in
proof gates. A finite-detuning primitive36 certificate must exclude divisors
1,2,3,4,6,9,12,18. The local flip theorem predicts nearby doubled cycles;
a separate enclosure is needed for a stated finite parameter value.

**Prove the connection, not just more endpoints.** Use the consumed exact
model/data and the center's exact binary rates as the starting anchor. First
carry all phases numerically along a short declared path, preserving failures
and reverse/overlap witnesses. Then validate a continuous tube of roots using
parameter-dependent interval Newton/Krawczyk charts, with uniform closure and
primitive-period exclusions. Adjacent charts must identify their shared root
at a common exact parameter through a common uniqueness domain; overlapping
boxes alone are insufficient. Fix or explicitly transport the18-phase label.
Validate stability along the tube if calling it an attracting branch. At the
fold, switch from fixed-rate to a nonsingular pseudo-arclength chart and prove
that the certified augmented fold root lies on that chart. At a simple flip,
`M-I` remains invertible, so the18-cycle branch itself is locally regular.

These two local boundary theorems plus a validated connecting tube would
materially strengthen this candidate. They still would not establish the
complete two-parameter shrimp anatomy, cusp/fold-curve organization, global
ancestry, or a shrimp-specific training benefit. Those require separate
boundary continuation and any claimed codimension-two nondegeneracy proofs.
No new seed, holdout, broad map, or protected outcome is needed for this step.

Metadata source: consumed `explore/a2-survey/rays.json`, SHA256
`a184d3f4fbdeed82f9933515fc61f4a74809936f2693728bbcb80323340c1db4`;
aggregate scope checked against saved `survey.json` and existing
`tools/explore_a2_survey.py`, without opening map arrays or evaluating models.
