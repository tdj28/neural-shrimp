# What a memory intervention can establish about an avoided trace crossing

**Status, 21 October 2026:** fixed-location derivatives and the completed local
saddle continuation have passed saved-data audits. The continuation provides
numerical evidence for a trace-contour reconnection under added memory, with
an explicit residual-qualified crossing estimate and the limitations below.
The counterexample below is an exact
construction, not a fitted or measured model of the neural network.

In the pinned Qwen feedback loop, the sampled saddle centers retain a
primitive period-18 state orbit and the same repeating six-token readout
while added memory changes the trace geometry. The cycle remains strongly
damped at these sampled centers as its parameter-space curves change their
connections. A crossing of the curves studied here means that two return
multipliers balance along intersecting parameter curves. It does not by
itself mean that a multiplier vanishes or that perturbations are erased
exactly.

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

The crossing imposes three scalar conditions: zero trace and zero trace
gradient in the two original controls. When those conditions are independent,
two controls generically do not suffice; a third control can produce an
isolated crossing. An exact factorization or symmetry can make the conditions
dependent. This explains why avoidance is generic without identifying a
particular native network pathway, and does not guarantee that the chosen
memory intervention reaches a crossing within its tested range.

If \(\Delta<0\), the multipliers on a trace-zero curve are

\[
\mu_\pm=\pm\sqrt{-\Delta}\ne0.
\]

More generally, in this opposite-sign regime the pair's spectral radius is

\[
\rho_{\rm pair}=\frac{\sqrt{\tau^2-4\Delta}+|\tau|}{2}.
\]

At fixed negative \(\Delta\), this is smallest at \(\tau=0\): the two
oppositely signed modes have equal absolute multiplier values. They both
contract there if additionally \(-1<\Delta<0\). This gives a
spectral meaning to the traced curves without assigning a basis-dependent
coupling coefficient. In the network, \(\Delta\) also varies with the
controls; consequently a trace-zero curve is not automatically a ridge of
maximum contraction or the exact minimum of a finite-time image.

More precisely, at a regular trace-zero point take a unit direction \(n\),
with \(a=\partial_n\tau\ne0\), \(b=\partial_n\Delta\), and
\(r=\sqrt{-\Delta}\). Along that direction,

\[
\rho_{\rm pair}(t)=r-\frac{b}{2r}t+\frac{|a|}{2}|t|+O(t^2).
\]

The strict inequality \(|b|<r|a|\) is sufficient for a strict transverse
cusp minimum. If \(|b|>r|a|\), the point is not a minimum along that
direction. Equality is inconclusive: higher-order terms can still produce a
strict minimum. For the normal \(n=\nabla\tau/\|\nabla\tau\|\), the
strict sufficient condition is
\(|\partial_n\Delta|<\sqrt{-\Delta}\,\|\nabla\tau\|\).
The direction and all derivatives must use the same declared coordinate
scales; this is a local pair-radius condition, not a finite-time-image test.

At a trace-saddle crossing, \(\tau=0\) and \(\nabla\tau=0\), so
the absolute-trace contribution starts quadratically. Instead,

\[
\rho_{\rm pair}(q)=\sqrt{-\Delta_0}
-\frac{\nabla\Delta_0\cdot q}{2\sqrt{-\Delta_0}}+O(\|q\|^2).
\]

A nonzero determinant gradient therefore gives a linear tilt and excludes a
local minimum at the trace crossing. Trace-contour reconnection alone does
not establish reconnection of the strongest-contraction curves. Saved
determinant stencils can assess this distinction numerically; applying the
exact statement to a computed candidate also requires reporting its finite
trace and stationarity residuals. The completed continuation below includes
this saved-stencil diagnostic.

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
one-step Jacobian derivative, evaluated at \(\beta=0\), is

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
pair determinant at the moving saddle supplies another quantity to check:
relabeling the full map or the joint spectral field preserves it, whereas
relabeling only the trace field need not.

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

## Qualified derivative and continuation results

At the original child orbit's alpha and temperature, the added memory path
contributes approximately -0.032696 to the pair-trace derivative, orbit
motion contributes -0.106638, and their sum is -0.139334. These contributions
refer to this specified intervention, not to shares of native mode coupling.
The total derivative was evaluated at one cyclic phase. Three-phase
finite-offset orbit controls agree with it; the primary phase's finest
central slope differs by less than 0.008%, with approximately quadratic
convergence across decreasing offsets. Independent composed-Hessian checks
also pass. Exact values, scope and hashes are in the
[sensitivity receipt](memory-sensitivity-receipt.json).

These fixed-location derivatives are not automatically the moving saddle's
height derivative: the original saved center is only an approximate saddle.
The first accurate continuation accepted eleven saddles through beta=0.003.
Their height falls from 0.0004647042445 at beta=0 to 0.0002753879880,
a 40.7% decrease, while the primitive-period and cyclic readout gates remain
unchanged. The largest observed empirical height sensitivity among these
centers is below 1.554e-11; this is not a rigorous error bound. The resolved
height change is inconsistent with a mere parameter relabeling of the same
trace field. It does not identify a native coupling pathway.

The next predictor at beta=0.005 exceeded the unchanged corrector trust
radius, so no saddle at that parameter was accepted. The preserved partial
attempt, source/array/archive checks, exact resource termination and
visually inspected continuation figure are bound by the
[continuation receipt](memory-continuation-receipt.json). Its archived source
omitted the final return-state array, so closure replay retains the explicit
recorded-summary limitation.

The completed local extension replayed both prior endpoints, then accepted
seven new saddle centers using 273 qualified point evaluations. Every local
point passed the saved-data gates, including independent closure replay from
the newly saved return-state arrays; the largest relative closure residual
was 4.3386485564e-13. The quadratic-corrected saddle heights are
\(+3.9471972323\times10^{-5}\) at \(\beta=0.006\) and
\(-4.9987419184\times10^{-5}\) at \(\beta=0.007\). Their empirical
height-sensitivity scales are respectively \(2.8699527363\times10^{-12}\)
and \(2.5973422469\times10^{-12}\). These are observed numerical
sensitivities, not rigorous error bounds; the sign change is nevertheless
well resolved by the saved checks.

The root search then met its declared \(|h|\le10^{-8}\) stopping rule at
\(\beta\approx0.00644853\), with
\(\alpha=0.07331776448598727\),
\(\log_{10}T=-0.15804614858293237\), and a remaining positive trace
\(1.6365300102\times10^{-10}\). The candidate's empirical height
sensitivity is \(8.7720409293\times10^{-12}\); its height remains positive
across the recorded estimates. It is therefore a numerical crossing estimate,
not an exact zero. The solver's retained sign bracket is
\([0.0064484162589609016,\,0.007]\), not a narrow certified parameter
interval. In the declared scaled coordinates about the original saved anchor
\(u=(p-p_0)/(2.5\times10^{-6},7.5\times10^{-6})\), the final finite-difference
trace-gradient norm is \(8.2218453695\times10^{-11}\), and the Hessian
eigenvalues are \(-0.04261232777\) and \(+0.01330814211\). Thus the
computed saddle remains nondegenerate and indefinite. Coarse and fine final
stencils have different centers: their comparison includes quadratic center
transport, and direct Richardson extrapolation is not applied. Their
transported trace-gradient discrepancy has norm \(2.6044333702\times10^{-7}\):
the much smaller final solver residual alone is not a derivative-accuracy
estimate.

The corrected-height secant from \(\beta=0.006\) to \(0.007\) is
\(-0.08945939151\). The last two positive root iterates give a secant of
\(-0.08920083185\), with an empirical endpoint-sensitivity scale of
\(0.00010016156\) after division by their parameter separation. These
negative finite secants support a transverse unfolding numerically; they do
not certify the derivative at an exact root. Under the smooth-family and
Morse assumptions above, the resolved sign change means that the local
trace-zero arms exchange their connections. It rules out mere translation
or parameter relabeling of the original trace field.

Full-orbit alignment provides additional evidence that the same period-18
family was followed despite the moving controls. Across all 18 accepted
historical and local centers, all 17 neighboring comparisons prefer the
same cyclic phase and retain the same token cycle. The largest relative
full-orbit change is 0.005599786683418215; the smallest mismatch for a second-best
cyclic alignment is 0.10712282855144359. This uses the full 18-by-896 state arrays,
not token equality alone. It supports continuation of the same family but
does not prove uninterrupted branch identity between sampled points.

At the candidate, \(\Delta=-6.7591540467\times10^{-9}\), with pair
multipliers approximately \(+8.2214156331\times10^{-5}\) and
\(-8.2213992679\times10^{-5}\). Both remain nonzero and strongly
contracting. The saved determinant stencils give
\(\nabla_u\Delta=(-4.5480903166\times10^{-10},\,
8.7524285017\times10^{-10})\); the coarse-to-fine gradient discrepancy
after center transport has norm \(2.6276693849\times10^{-16}\). If this
nonzero gradient persists at the exact trace-saddle crossing, the conditional
linear pair-radius tilt is approximately
\((2.7660046920\times10^{-6},-5.3229502092\times10^{-6})\).
The finite trace and stationarity residuals prevent treating this as a formal
nonzero-gradient certificate at an exact root. It gives concrete numerical
reason to distinguish trace-contour reconnection from reconnection of exact
strongest-contraction ridges or the bright curves in a finite-time image.

The [local continuation receipt](memory-local-extension-receipt.json) binds
the completed audit and the combined full-orbit and determinant diagnostics.
The spectral isolation checks remain conditional on the computed Ritz
subspace, and no interval proof is claimed. This result demonstrates a
controllable change of trace geometry under added memory; it establishes
neither exact superstability nor a native internal coupling mechanism.

The exact polynomial example has a visually inspected three-panel figure;
its source, external PNG/PDF and review are bound by the
[figure receipt](memory-theory-figure-receipt.json). It illustrates the
counterexample only and contains no neural scan data.
