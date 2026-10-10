# Native eigenpair-locus continuation prototype — 10 October 2026

TLDR: this prepares a way to prove that separated local bifurcation certificates
belong to the same branch of training cycles. It does not yet acquire a target
chart, establish stable window geometry, or connect the certified cusps and fold.
The first implementation deliberately uses an affine predictor; a quadratic
predictor for these equations would require fourth derivatives of the update
map. The mean-value alternative needs only second derivatives of that map.

## Mathematical object and implementation

For dimension n and period p, order z by native phases (x_i,q_i), then the two
actual learning rates. The first 2pn equations are

- F(x_i,lambda)-x_(i+1)=0;
- DF(x_i,lambda)q_i-q_(i+1)=0, with the last q successor replaced by sigma*q_0.

Add (q_0.q_0-1)/2=0 and a fixed affine gauge normal.(z-origin)-s=0. For n=13,
p=18 this is 470 square equations. sigma=+1 or -1 specifies the critical mode.
No dynamic eigenvector renormalization or phase exchange occurs. Original
rate coordinates, exact model constants and native order must remain bound.

`rigorous_eigenpair_locus.py` differentiates each local phase in only n+2 base
variables using the already reviewed MixedJet. Holding q fixed gives J and
D_x(Jq); the q columns are J itself. Both rate columns include D_lambda(Jq).
The reviewed TwoRateCuspMap adapter also accepts these lower-order jets.
Construct that adapter under the intended Arb precision before parsing rational
model constants. The name of this reused adapter implies no cusp theorem.

For a proposed midpoint Jacobian, solve the local recurrence
u_(i+1)=T_i*u_i+E_i*d_lambda-b_i, with terminal
u_p=diag(I,sigma*I)u_0. This condenses to 2n+2=28 unknowns. No native J_i is
inverted; controls explicitly include singular native Jacobians. The full
preconditioner is rounded to exact dyadic entries. Its quality is established
only by the complete interval defect I-C*DG over the 470D correction domain.
Midpoint condensation and solve accuracy are never proof assumptions.

With P(s)=U+S*(s-sc), the gauge-aware residual enclosure is
H(P(s),s) in H(U,sc)+(DG(P(I))*S-e_last)*(s-sc).
This retains cancellation in the tangent equation. The derivative domain is
the convex affine predictor box, and the correction Jacobian is acquired on
P(I)+E. Strict componentwise inclusion and a weighted norm below one establish
one normalized root in the correction box for every s. The complete matrices,
phase returns, preconditioner, residual and endpoint boxes are exposed.

A passed low-level chart does not authenticate its prior root: a future target
adapter must bind the prior certificate, model, native order, critical sign
and gauge. Entire anchor containment at its correlated gauge, followed by
common exact-s endpoint containment, identifies the branch. An anchor overlap
or a midpoint-only gauge is insufficient. The prototype stops after a failed
chart and does not advance its anchor. Proper-divisor separation is separate.

This is a *lifted normalized eigenpair locus*. It need not project regularly
to the rate plane. Full bordered rank alone is not asserted to prove a simple
critical multiplier. Generic fold/flip coefficients, simplicity, complementary
spectrum, attraction and rate-tangent projection need separate classification.
Other-mode unit-circle crossings and c=0 are allowed; a cusp can have zero
rate-plane tangent while its lifted locus is regular. Augmented rank loss is
reported as an obstruction, never skipped. No complete shrimp follows.

## Source and synthetic checks

New source: `src/neural_shrimp/rigorous_eigenpair_locus.py`.
Controls: `tests/test_rigorous_eigenpair_locus.py`.
The 14 focused tests pass in 0.15 s: independently specified coupled primitive2
fold/flip maps, every Jacobian column, dense inverse versus condensation,
singular native J, two-chart exact endpoint joins, a cusp with vanishing rate
tangent, complementary -1 crossing, rank obstruction at an additional +1,
wrong anchor sign/phase, false primitive18/repeated-cycle rejection, returned
nonfinite data retention, interruption and the nonunit-scale biased GD adapter.
The primary finite-difference checks are numerical controls, not proof gates.

## Authorized synthetic-only resource profile

No target path or target mode exists in the profile runner. Before this run,
commit source, tests, this card and the append-only log. Output is exclusive:
`/data2/neural-shrimp-data/explore/eigenpair-locus-prototype/attempt-001-profile`.
Run with CPU/BLAS1, nice10 using `tools/launch_eigenpair_profile.py`.

The profile executes two full 470D polynomial charts on an exact repeated
primitive2 (so the primitive18 test must fail) and 12 independent local GD
MixedJet evaluations on synthetic 13D/24-example data. Each chart exercises
two midpoint proposals plus center, predictor and correction-box evaluations.
The GD model is constructed at 256 bits. The full observer serialization is
included in the chart timing. Retain all return records, sources and inputs,
including any failed control or resource screen.

Bounds: internal120 s, external135 s plus5 s kill grace, one owned process
group, 225 native/guard ticks, worker256 MiB plus launcher1 MiB output,
monitored worker RSS1 GiB, hard virtual-address limit3 GiB. RSS is a cooperative
sampled cap, not a kernel hard resident-memory limit. No GPU. The launcher
keeps all failures and freezes the complete inventory. A short synthetic
profile may fail; that is not an eigenpair-locus theorem failure.

## First two target charts: proposal, not authorized acquisition

Start only from the independently reviewed ray0 primitive18 fold certificate,
not from an unqualified numerical curve node. Reconstruct its whole 27D root
box, fixed exact output rate and same consumed exact dataset. Eighteen native
mixed-jet steps propagate both state and mode enclosures to form the complete
470D prior root. Retain original source/manifest hashes and every propagated
return. The ± mode sign and phase0 must remain exactly those of the certificate.
No existing target arrays were opened for this prototype.

Use the fixed gauge s=eta_o-eta_o(anchor), where eta_o(anchor) is the exact
dyadic certificate constant. This keeps rate units explicit and is regular at
the anchor because its existing one-rate augmented Jacobian is nonsingular.
Tentative fixed first profile-screened settings are 256 bits, two midpoint
proposals/chart, all coordinate weights1, halfwidth2^-20, radius2^-24. The
first interval is [-2^-20,2^-20], the second [2^-20,3*2^-20]. This is a local
feasibility test, not a meaningful fraction of the distance to either cusp.
Root review may revise these proposed settings before target acquisition;
any later adaptation is separately disclosed with failed evidence retained.

Operation budget:18 native anchor steps plus2*5*18 chart steps =198 native
MixedJet evaluations; two additional center-preconditioner guard ticks make
200. The proposed hard count ceiling is216. There are six 28D condensed solves
(two proposal and one final per chart); two full 470D defect products; no eig,
SVD, third/fourth-F derivative, mode search, or complementary stability test.

The profile screen reports 2*(sum(two full synthetic chart times)+198*maximum
synthetic GD phase time)+10 seconds. This double-counts cheap polynomial local
work and adds a factor2 plus10 s reserve; it is a cost heuristic, not a runtime
guarantee. Require projection<110 s before proposing an internal120/external
135+5 s target. RSS projection is2*profile_peak+128 MiB, required<2 GiB.
Compressed-output projection is4*profile_bytes+8 MiB, required<256 MiB;
compression can differ, so a strict aggregate cap/finalization reserve remains
necessary. Source/profile/model identity and exact target adapter still need
review. There is deliberately no executable target launcher in this unit.

If the affine residual bound fails, preserve its forcing/propagation split.
Only then compare a smaller disclosed step with a higher-order path jet or
component weights. A bivariate jet degrees(t<=3,u<=1) could supply the needed
fourth-F terms for a later quadratic predictor, but is not implemented here.
