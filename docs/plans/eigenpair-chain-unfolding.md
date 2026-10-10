# Conditional normal-form coefficients and physical-rate transversality

10 October2026. Source preparation only; no target coefficient evaluation.
The independently proved root satisfies R=0,Aq=0,Ar=q, q.q=1,q.r=0, with
A=D_xR and R=F^18-x. Its complementary spectrum has no unit multipliers.
Use exactly this same root box and its complete augmented MixedJet Jacobian.
The n×n blocks of rows n:2n and2n:3n, columns0:n, are B(.,q) and B(.,r),
where B=D_x²R=D_x²F^18. Their two rate columns are A_rate q and A_rate r.
The first n rows supply A and R_rate. No new derivatives are required.

Validate an interval inverse of V=[A^T,q;r^T,0]. Solve V(p0,γ0)=(0,1)
and V(p1,γ1)=(p0,0). At the actual proved root the auxiliary scalars vanish:
left multiplication by q^T uses Aq=0,q.q=1, then p0.q=p0.Ar=0.
Consequently A^T p0=0,A^T p1=p0,p0.r=p1.q=1,p0.q=p1.r=0.
All computations retain root-box uncertainty; enclosure of these identities
alone is never substituted for their supplied exact root proof.

Define a=p0.B(q,q), b=p0.B(q,r)+p1.B(q,q). In this note the explicit critical
normal form is (u,v)→(u+v,v+a*u²/2+b*u*v)+O(3). These definitions agree with
Kuznetsov–Meijer, SIAM J.Sci.Comput.26(6),1932–1954(2005),§4.4, equations
(4.8)–(4.9), DOI[10.1137/030601508](https://doi.org/10.1137/030601508).
The [authors' primary preprint](https://www.math.uu.nl/publications/preprints/1290.pdf)
was independently checked by the Clark reviewer. Do not silently mix them with sources using an unfactored quadratic coefficient.
The helper retains a,b,b−a and(a)(b−a) with explicit conventions; a physical
stability-side conclusion also needs the validated parameter unfolding below.

Here is the parameter-rank derivation, independent of a copied normal-form
unfolding formula. For each physical rate j, validate K=[A,r;q^T,0] and solve
K(h_j,γ_j)=(-R_j,0). At the exact root let c_j=p0.R_j, so γ_j=−c_j and
A h_j=−R_j+c_j r. When c.δα=0, h.δα solves the linear fixed-point equation.
The remaining free state displacement is s q. Linear differentiation of the
fold equation has obstruction d.δα+a*s, where

    d_j = p0.[B(h_j,q)+A_j q].

If a≠0, the fold displacement is H_j=h_j−q*d_j/a. The spectral projector onto
the isolated two-dimensional critical cluster is P=q*p1^T+r*p0^T. Derivative
of its eigenvalue sum is tr(P M'), since the off-diagonal derivative of P has
zero trace against block-diagonal M. Thus along the fold the derivative of
its second multiplier minus1 equals

    tau_j = p1.[B(H_j,q)+A_j q] + p0.[B(H_j,r)+A_j r].

The extension tau off ker(c) is immaterial. Strict nonzero determinant of
[c_hidden,c_output;tau_hidden,tau_output] proves that the physical parameters
independently unfold fixed-point existence and the remaining critical
multiplier. This explicit determinant avoids assigning all needed genericity
to the nonsingular41D isolated-root Jacobian. It still requires independently
proved exact chain/spectral assumptions and a≠0. b−a≠0 is an additional
resonant quadratic nondegeneracy condition; none follows merely from a visually
shrimp-shaped island. Native branch connection and full boundary topology are
separate and remain mandatory.

Seven synthetic controls use exact polynomial maps, a nonorthogonal stable
complement coupling, outward root-box uncertainty, a state shift depending on
both physical rates and an independently known determinant5, singular
quadratic/resonant/parameter cases, and incorrect chain/shape rejection.
First fixture used unsupported MixedJet division; corrected to exact half
multiplication before any scientific run. No target run/card yet.
