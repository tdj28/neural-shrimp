# What a memory intervention can establish about an avoided trace crossing

**Status, 21 October 2026:** repository theory draft; numerical continuation
and mechanism results are pending audit. The counterexample below is an exact
construction, not a fitted or measured model of the neural network.

A stable periodic network can retain strong damping while its parameter-space
curves change their connections. A crossing of the curves studied here means
that two return multipliers balance along intersecting parameter curves. It
does not by itself mean that a multiplier vanishes or that perturbations are
erased exactly.

This note applies the existing Morse lemma with parameters, stated in
[Appendix B, Proposition 16 of Plantinga's thesis](https://cs.nyu.edu/~exact/pap/mesh/vegter/plantingaThesis.pdf).
It derives diagnostics for the memory intervention and gives an algebraic
counterexample to the claim that nonzero coupling necessarily causes
avoidance. It does not identify a native network pathway responsible for the
observed geometry or supply a two-dimensional nonlinear reduction.

## The local statement and its assumptions

Let \(p=(\alpha,\log_{10}T)\) denote the original controls and let
\(s=\beta\) denote the memory parameter. For a continued periodic orbit,
write \(M(p,s)\) for the full return Jacobian and
\(\tau=\mu_1+\mu_2\), \(\Delta=\mu_1\mu_2\) for the trace and
determinant of its selected multiplier pair. These are pair quantities, not
the trace and determinant of the full high-dimensional matrix.

Assume a smooth map, a continued primitive orbit with
\(1\notin\sigma(M)\), and a real or complex-conjugate pair isolated from
the remaining spectrum.
Then the orbit and the pair's spectral projector vary smoothly locally.
The sum \(\tau\) remains smooth when the two multipliers exchange order by
modulus. Suppose also that

\[
\nabla_p\tau(c(s),s)=0,\qquad
H(s)=\partial_p^2\tau(c(s),s)
\]

is nonsingular with one positive and one negative eigenvalue. The implicit
function theorem continues the saddle center \(c(s)\). The parameterized
Morse lemma then supplies smooth local coordinates \((U,V)\), depending on
\(s\), such that

\[
\tau(p,s)=h(s)-U^2+V^2,\qquad h(s)=\tau(c(s),s).
\]

Under these assumptions this is an exact local normal form. A fitted
quadratic approximation alone does not verify all its hypotheses for a
particular computation.

For \(h>0\), the zero contour is \(U^2-V^2=h\). For \(h<0\), its
opening direction switches. At \(h=0\), it consists locally of the two
transverse lines \(V=\pm U\). A sign change with \(h'(s_c)\ne0\)
therefore produces a generic trace-zero reconnection. Equivalently, the
Jacobian of \((\nabla_p\tau,\tau)\) with respect to \((p,s)\) has
determinant \(\det(H)h'(s_c)\) at that point. If \(h'=0\), higher-order
analysis is needed; an even-order touch of zero need not reconnect the arms
across the parameter value.

If \(\Delta<0\), the multipliers on a trace-zero curve are

\[
\mu_\pm=\pm\sqrt{-\Delta}\ne0.
\]

Thus a verified trace crossing is compatible with two nonzero contracting
multipliers. A zero of \(\Delta\) would establish a zero within this pair;
even that would not establish that all multipliers of the full return map
vanish. A trace reconnection is also distinct from a stability bifurcation.

## Separating translation from a change of saddle height

Differentiate the stationary-point equation. With derivatives evaluated at
\((c(s),s)\),

\[
c'=-H^{-1}\tau_{ps},\qquad
h'=\tau_s,\qquad
h''=\tau_{ss}-\tau_{sp}H^{-1}\tau_{ps}
    =\tau_{ss}-c'^THc'.
\]

The first identity describes motion of the center. The last separates the
fixed-location curvature from the curvature of the moving saddle's height.
All derivatives must use the same parameter-coordinate scales. Residual
gradients at an approximate numerical saddle can contaminate the height
derivative when the center moves rapidly.

For the intervention
\(F_\beta(x)=(1-\beta)F(x)+\beta x\), let \(x_i\) be the original
periodic orbit, \(J_i=DF(x_i)\), and \(\eta_i=\partial_\beta x_i\).
At fixed original controls,

\[
\eta_{i+1}=J_i\eta_i+x_i-x_{i+1},\qquad
\eta_m=\eta_0,
\]

where \(m\) is the period. This gives \((I-M)\eta_0=b\), with \(b\)
obtained by propagating the forcing from zero initial tangent. The total
one-step Jacobian derivative is

\[
\dot J_i=(I-J_i)+D^2F(x_i)[\eta_i,\,\cdot\,].
\]

For \(M=J_{m-1}\cdots J_0\), the product rule gives \(\dot M\).
The pair-trace derivative is \(\dot\tau=\operatorname{tr}(P\dot M)\),
where \(P\) is the pair's spectral projector. Propagating the two displayed
terms separately distinguishes the explicit identity feedback path from
orbit motion. It does not identify native internal mode coupling.

If memory merely relabels the original controls, so that
\(\tau(p,s)=\tau_0(\psi_s(p))\) for a local parameter diffeomorphism,
the distinguished saddle height stays constant. A resolved change in
\(h\) rules out that explanation for the trace field. Constant \(h\)
with a moving center does not establish equivalence of the full maps. The
pair determinant at the moving saddle supplies another invariant to check.

## An exact counterexample: nonzero coupling can preserve a crossing

The following autonomous polynomial map is chosen analytically as a
counterexample. Its parameters have not been estimated from neural data:

\[
\begin{aligned}
q(x)&=1+ux+(-3-2u-v)x^2+(2+u+v)x^3,\\
F_1(x,y)&=q(x)+k[1+(r-1)x]y,\\
F_2(x,y)&=k(x^2-x).
\end{aligned}
\]

Direct substitution gives \(q(0)=1\), \(q(1)=0\),
\(q'(0)=u\), and \(q'(1)=v\). Consequently
\((0,0)\mapsto(1,0)\mapsto(0,0)\) is a primitive period-two orbit.
Differentiating the displayed map at these two distinct points gives

\[
A=DF(0,0)=\begin{pmatrix}u&k\\-k&0\end{pmatrix},\qquad
B=DF(1,0)=\begin{pmatrix}v&rk\\k&0\end{pmatrix}.
\]

The exact return matrix and its invariants are therefore

\[
M=BA=\begin{pmatrix}uv-rk^2&vk\\ku&k^2\end{pmatrix},\qquad
\tau=uv+(1-r)k^2,\qquad \Delta=-rk^4.
\]

This multiplication verifies the claimed algebra without a numerical fit.
The parameter-plane Hessian of \(\tau\) is
\(\begin{pmatrix}0&1\\1&0\end{pmatrix}\), so its saddle is at
\(u=v=0\), with height \(h=(1-r)k^2\). At that saddle the two
multipliers are \(-rk^2\) and \(k^2\). For \(r>0\) and
\(k^2\max(1,r)<1\), this primitive cycle is locally attracting.

With \(k=0\), the trace factorizes as \(uv\). With nonzero \(k\),
two cross-channel paths contribute \(+k^2\) and \(-rk^2\) to the
return trace. Their imbalance gives an additive offset and avoidance when
\(r\ne1\). But at \(r=1\), all four off-diagonal one-step entries
remain nonzero and their contributions cancel: the trace-zero curves cross
while \(\Delta=-k^4<0\). Nonzero coupling alone is therefore insufficient
to force avoided trace crossings.

## Discussion: what a native mechanism would require

The following concerns possible interpretation; the counterexample and
conditional theorem do not establish a network mechanism. Those hypotheses
need separate, disclosed tests against the full computation.

Off-diagonal entries of a single two-by-two return matrix do not directly
change its trace. Moreover, off-diagonal entries depend on the chosen basis;
diagonalizing the return matrix removes them without changing the spectrum.
The counterexample instead concerns complete return paths through a
composition of maps, expressed in explicitly defined state channels.

To connect a similar offset to the neural network, a proposed pathway must
be defined independently of the fitted saddle height. Its derivative
contribution must predict the full trace, determinant and local geometry
over additional parameter points, with projection and nonnormality errors
measured. A controlled intervention on that pathway must then continue the
same primitive orbit and its moving saddle. Calling the measured height
itself a coupling coefficient would only rename the observation.

**Experiment results:** pending audited continuation and derivative
qualification. No crossing, noncrossing interval, or causal pathway result
is asserted here.
