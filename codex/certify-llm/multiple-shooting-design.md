# A streamed multiple-shooting certificate design

Date: 2026-10-13. Method design only; no new target computation or proof result.
Let `p=8`, `n=896`. Every matrix below is `n by n`, never `7168 by 7168`.

## Evidence constraints and scope

The 384-bit prefix failed at layer 16. The 1024-bit attempt obtained a sharp
point map but the natural state box of radius `2^-80` failed at layer 3. The
reported point-output rounding radius is about `1.2e-142`. Thus neither the
current binary64 center defect nor its rigorous evaluation width permits an
arbitrarily tiny proof box. A higher-precision constants run (possibly 2048
bits) and high-precision center refinement must precede choosing smaller boxes.
The needed precision/radius pair must come from measured preconditioned
residuals and derivative bounds, not from a desired answer.

For phase centers `c_i`, positive radii vectors `r_i`, and boxes
`X_i=c_i+diag(r_i)[-1,1]^n`, solve the cyclic system

`H_i(x_0,...,x_7)=F(x_i)-x_(i+1 mod 8)=0`.

An accepted contraction proves one unique root tuple in the **full product
box** `X_0 × ... × X_7`. It does not automatically prove that every other
`F^8` fixed point in `X_0` is excluded if its intermediate phases leave the
other boxes. This distinction must be explicit in any receipt or theorem.

## Exact candidate matrices and a small inverse proposal

Choose exact dyadic matrices `A_i` approximating the one-step Jacobian at each
center. Binary64 matrices are admissible proposals; higher-precision dyadic
midpoints can be used later if necessary. Define exact real products

`P_0=I`, `P_i=A_(i-1)...A_0`,

`Q_j=A_(p-1)...A_(j+1)`, with `Q_(p-1)=I`.

Choose another exact dyadic matrix `B` approximating `(I-P_p)^(-1)`. It may be
proposed by ordinary floating linear algebra. No unverified inverse identity
is assumed. The rigorously enclosed inverse defect is

`D = I - B(I-P_p)`.

All products defining `P`, `Q`, `D` and later operator blocks are mathematical
products of the stored exact constants. Arb may enclose their rounding.
**A rounded product midpoint must not be silently substituted as the exact
product.** If a midpoint is used as a different approximation, its discarded
product defect must be explicitly accounted for.

Let `L` denote the cyclic block matrix with row `A_i z_i-z_(i+1)`.
Define a fixed linear preconditioner `R` implicitly, without forming it:

1. For a right-hand side `b`, set `v_0=0` and
   `v_(i+1)=A_i v_i-b_i` through all eight phases.
2. Set `z_0=B v_p`.
3. Set `z_(i+1)=A_i z_i-b_i` for the remaining seven output phases.

The returned tuple is `R b`. If `B` were the exact inverse above, this would
solve `L z=b`. The proof instead retains `D`, so approximate `B` is allowed.
The entries of this fixed operator are exactly

`R_ij = -P_i B Q_j - 1[j<i] A_(i-1)...A_(j+1)`.

The second product is `I` when `i=j+1`. Its minus sign matters. These two
terms can cancel strongly; sum their signed enclosures before taking norms.

## Full-box derivative defect without losing the useful cancellation

Obtain genuine full-box enclosures `J_i(X_i)` for `DF` at every point of `X_i`.
Point derivatives or eight signed JVPs do not satisfy this requirement. Set

`Delta_i = J_i(X_i)-A_i`.

The true shooting Jacobian is `DH=L+diag(Delta_i)`. The exact cancellation
identity is

`(I-R DH)_ij = 1[j=0] P_i D - R_ij Delta_j`.

This is the central computational advantage. The candidate cyclic linear
system cancels analytically; only its inverse defect and the actual
Jacobian approximation/remainder errors remain. Form the signed products
and differences on the right in Arb before any absolute-value majorant.
Replacing them by products of per-phase operator norms would discard the
nonnormal cancellation that this formulation is intended to retain.

Point-Jacobian enclosures plus a proved centered derivative remainder may
supply `J_i(X_i)`. Alternatively, validated column-batched AD on sufficiently
small whole boxes may do so. Which is feasible remains an evaluator question;
this algebra does not manufacture missing derivative bounds.

## Streaming order and sufficient inequalities

Process block columns `j` in descending order. Maintain just one suffix matrix:
start `Q_(p-1)=I`, then update `Q_(j-1)=Q_j A_j`.
For each `j`, obtain the blocks of `R` by the signed recurrence

`R_0j=-B Q_j`,

`R_(i+1),j=A_i R_ij-1[i=j] I`.

Only for column zero, advance `K_0=D`, `K_(i+1)=A_i K_i`, so `K_i=P_i D`.
For each current block form

`E_ij=1[j=0] K_i-R_ij Delta_j`.

Immediately accumulate its outward absolute row contribution
`abs(E_ij) r_j`, and discard that block when finished. Also accumulate
`-R_ij h_j`, where `h_j` rigorously encloses `F(c_j)-c_(j+1)`.
This retains all columns and all transverse directions without storing `R`
or `E` as full 7168-square matrices.

Let `s_ia` be the accumulated outward upper bound on the row contribution of
`E`, and let `eta_ia` bound the absolute preconditioned center residual.
Require, for every phase/coordinate,

`eta_ia + s_ia < r_ia`,

and record `q=max_ia s_ia/r_ia < 1` with strict outward comparisons.
These prove strict self-mapping and contraction of `x-RH(x)` on the entire
product box. Since `||I-R DH||<1` at an actual point, square `R` is nonsingular;
no separate assumption that its floating inverse proposal was correct is
needed. The unique fixed point of the preconditioned map is therefore a zero
of the shooting equations.

A potential implementation stores the eight `A_i`, `B`, the current `Delta_j`,
a suffix, a current `R` block, the product/defect temporaries and small phase
vectors. It can load `Delta_j` one phase at a time. Counts are of order `p*n^2`
stored coefficients, with no `p^2*n^2` storage. Native allocation, precision
and temporary lifetimes still need measurement; scalar counts are not RAM
claims.

## Center refinement and honest radius selection

The same implicit `R` can propose high-precision corrections using separate
rigorous one-phase residual evaluations. A chosen new center may be the exact
dyadic midpoint of `c-Rh`, saved as a new proposal with the old attempt retained.
All residual/domain-dependent bounds must then be recomputed at the new center
and boxes. `A_i,B` may be reused as fixed proposals; validity comes from the
newly checked defects, not from their age or origin.

Compute both center-defect magnitudes and enclosure radii, then `eta=abs(Rh)`.
No radius below the currently available preconditioned residual bound can pass
these inclusion inequalities. In particular, a 1024-bit point enclosure of
width around `1e-142` cannot justify a proposed `1e-250` inclusion through this
calculation. A 2048-bit run may reduce arithmetic uncertainty, but that must be
measured and the centers must actually be refined; changing the precision does
not remove the existing roughly `1e-14` binary64 orbit defect by itself.

## Primitive period and attraction stay separate

Once a root tuple has been proved, its coordinates satisfy
`x_d=F^d(x_0)`. Strict coordinate separation of `X_0` from `X_1`, `X_2`, and
`X_4` therefore excludes the proper divisors of eight, without propagating an
entire box through those iterates. This requires a dedicated shooting-root
primitive gate. **Do not feed `X_d` into the existing primitive gate as though
it enclosed all of `F^d(X_0)`**; that stronger statement has not been proved.

For attraction, rigorously enclose the signed matrix product

`M=J_7(X_7)...J_0(X_0)`.

It contains the actual monodromy at the proved orbit. Bound a declared power
of this matrix in a positive weighted induced norm and require it to be
strictly below one. Retain the point/interval matrix products before taking
norms; multiplying eight separate norm bounds can destroy the useful signed
cancellation. This proves local attraction, not that the entire shooting box
is a dynamical basin. A numerical spectral radius is still only a proposal.

## Concrete checks and cost evidence

A CPU1 synthetic test at 256 bits checked the displayed block-defect identity
exactly against an explicitly assembled system with `p=4,n=2` and signed
dyadic matrices. It also checked that the implicit residual solve equals the
explicit block action. Both identities held exactly. No target data was used.

Small dense, full-mantissa point-matrix products at 1024 bits measured about
0.131 seconds for 128-square and 0.679 seconds for 256-square on one CPU thread.
These are kernel measurements only. Before a full proof pass, measure a small
number of actual-size 896-square products under a separate bounded local plan;
do not infer guaranteed runtime or memory from these two small sizes.

Root reports a completed one-phase, eight-direction Arb target step of
261.808 seconds on CPU1, peak RSS 6.209 GiB, with maximum derivative radius
`2.526e-141`. Its separate eight-direction binary64 comparison took 15.786
seconds including other setup. At the current block size, eight full 896-column
phase Jacobians entail 896 eight-column forward evaluations; multiplying the
measured Arb duration gives roughly 65 CPU-hours as a scheduling warning, not
an execution forecast. Larger streamed batches or bounded parallelism need
actual memory/throughput measurements before authorization. The floating
`A_i,B` proposal acquisition is a different, cheaper operation and supplies
no rigorous derivative enclosure.

The next useful artifacts are therefore: finite numerical `A_i,B` proposals
with all phase/closure diagnostics preserved; a measured higher-precision
point residual floor; a short center-refinement feasibility run; and a bounded
measurement of full-box derivative/remainder acquisition. None of these alone
is a certificate.
