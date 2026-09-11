# An exact neural training cycle with a prediction consequence

**Status: 2026-09-11 theoretical derivation for a constructed toy model.**
This is algebra, not an additional empirical pilot or a reproduction result.
It demonstrates a neural training cycle and observable consequences; a
one-parameter branch does not establish a two-parameter shrimp geometry.

## Model and loss convention

Consider a depth-two scalar linear network whose two weights are tied to the
same scalar parameter `theta`:

\[
\widehat y=\theta^2x,\qquad (x,y)=(1,1),\qquad
L(\theta)=\tfrac12(\theta^2-1)^2.
\]

Differentiating with respect to the single tied parameter gives

\[
L'(\theta)=2\theta(\theta^2-1),\qquad
F_\eta(\theta)=\theta-\eta L'(\theta)
=(1+2\eta)\theta-2\eta\theta^3.
\]

The factor of two comes from tying both layers to one parameter. Untied
weights with independent updates define a different map. Throughout this
note, **MSE** is squared prediction error at the sole input `x=1`; the
training loss is one half of that MSE. There is no held-out task population or
generalization claim.

The fixed points are `0` and `±1`. The useful parameter states `±1` both yield
prediction `1`. Since

\[
F_\eta'(\theta)=1+2\eta-6\eta\theta^2,\qquad
F_\eta'(\pm1)=1-4\eta,
\]

the two correct fixed points are locally attracting for `0<eta<1/2` and
reach a flip at `eta=1/2`. The zero fixed point is repelling for positive
`eta`.

## Exact nontrivial same-sign period-two branch

Write `a=1+2 eta`, `b=2 eta`, and suppose `F(u)=v`, `F(v)=u` with
`u != v` and `S=u+v != 0`. Set `P=uv`. Adding and subtracting the orbit
equations gives

\[
1=a-b(S^2-3P),\qquad -1=a-b(S^2-P).
\]

Consequently,

\[
P=\frac1b=\frac1{2\eta},\qquad
S^2=\frac{a+2}{b}=\frac{3+2\eta}{2\eta},\qquad
(u-v)^2=\frac{2\eta-1}{2\eta}.
\]

Define `c=1/(2 eta)` to keep this coefficient separate from the primitive
period, which is two. The positive branch is

\[
u=\frac{\sqrt{1+3c}-\sqrt{1-c}}2,\qquad
v=\frac{\sqrt{1+3c}+\sqrt{1-c}}2.
\]

Here `u<v`. The orbit exists with distinct positive states for `eta>1/2`. The negative branch
is `(-u,-v)` and has the same predictions. Direct substitution verifies
`F(u)=v`, `F(v)=u`; the nonzero separation verifies primitive period two.
At `eta=1/2`, the two states merge and the primitive period becomes one.

The multiplier of one complete period is

\[
\begin{aligned}
\mu_2&=F_\eta'(u)F_\eta'(v)\\
&=(a-3bu^2)(a-3bv^2)\\
&=a^2-3ab(u^2+v^2)+9b^2P^2\\
&=9-2a^2=9-2(1+2\eta)^2,
\end{aligned}
\]

where `u²+v²=a/b` and `P=1/b` were used. Therefore the orbit is linearly
attracting with a strict multiplier margin precisely when

\[
|\mu_2|<1
\quad\Longleftrightarrow\quad
\frac12<\eta<\frac{\sqrt5-1}{2}.
\]

The endpoints have multipliers `+1` and `-1` and are excluded from this strict
linear-stability statement. This calculation certifies local attraction in
the open interval. It
does not identify the entire basin, guarantee convergence from arbitrary
initialization, or establish what follows the second endpoint.

## Phase prediction, gradients and averaging

At `x=1`, the two phase predictions are

\[
z_-=u^2=\frac{1+c-\sqrt{(1+3c)(1-c)}}2,
\qquad
z_+=v^2=\frac{1+c+\sqrt{(1+3c)(1-c)}}2.
\]

For the attracting branch, `z_+>1` and `z_-<1`: the output visibly
overshoots and undershoots the target. The state and prediction phase
variances are

\[
A_\theta^2=\frac{1-c}{4},\qquad
A_z^2=\frac{(1+3c)(1-c)}4.
\]

The two training gradients are opposite:

\[
L'(u)=\frac{u-v}{\eta},\qquad
L'(v)=-\frac{u-v}{\eta}.
\]

Zero mean gradient along the cycle therefore does not imply either state is
a stationary point. The readout retains the cycle even though temporal
averages suppress the oscillation.

The quantities a user might actually deploy differ:

| Readout policy | Prediction or definition | MSE at `x=1` |
|---|---|---|
| Uniform random phase / average phase error | Select `u` or `v` with equal probability | `(1-c²)/2` |
| Average the two predictions | `(u²+v²)/2 = (1+c)/2` | `(1-c)²/4` |
| Average the weights, then predict | `((u+v)/2)² = (1+3c)/4` | `9(1-c)²/16` |
| Exact solution checkpoint | `theta=1` or `theta=-1` | `0` |

The table concerns readouts of the exact orbit. It does not count the work of
finding or reaching it. The solution checkpoint row is an available exact
solution, not a claim that the same unstable learning-rate setting will
converge to it.

To derive the phase MSE, use `u²+v²=1+c` and `u²v²=c²`:

\[
\begin{aligned}
\tfrac12[(u^2-1)^2+(v^2-1)^2]
&=\tfrac12[(1+c)^2-2c^2-2(1+c)+2]\\
&=\frac{1-c^2}{2}.
\end{aligned}
\]

The averaged predictions have bias `(c-1)/2`, while the averaged-weight
prediction has bias `3(c-1)/4`. Squaring gives the other two MSEs. In
particular,

\[
\operatorname{MSE}_{\rm phase}
-\operatorname{MSE}_{\rm prediction\ average}=A_z^2.
\]

Prediction averaging removes phase variation but retains a nonzero bias
everywhere on the strictly attracting two-cycle. Weight averaging has a
different, larger bias in this model. None of these statements shows that a
cycle-based method beats an ordinary optimizer or an equally funded generic
averager on a learning task.

## Exact readout aliasing, with the stability qualification visible

Sign symmetry gives `R(q;x)=R(-q;x)` for every `x`. For this gradient map,
the sign-alternating states form an exact period-two orbit when

\[
q^2=1+\frac1\eta,\qquad F_\eta(q)=-q.
\]

It is prediction-invisible across its two phases, but is repelling:

\[
F_\eta'(q)=F_\eta'(-q)=-5-4\eta,\qquad
\mu_{\rm sign}=(5+4\eta)^2>1\quad(\eta>0).
\]

This is an exact counterexample to the implication that nonconstant state
motion must appear in the chosen readout. It is **not** a stable hidden cycle
or a useful training regime. A conceptual sign-symmetry diagram can instead
show two equivalent parameter states without asserting any trajectory or
stability at all.

## What this contributes to the research question

The branch supplies a concrete manifestation: rate-induced nonconvergence
produces prediction oscillation and a biased average. It also supplies an
honest candidate benefit, removing phase error by prediction averaging, and
its limit, residual bias plus extra readout cost. The exact solution remains
better for this toy fitting objective.

The next empirical question is whether a verified boundary in a richer neural
model helps select an averaging, stopping or escape action better than loss,
recurrence, local stability and ordinary performance search. That question
requires the separate matched-control experiments in
[next-steps.md](next-steps.md). The analytic example answers neither the
existence of shrimp geometry in that model nor the incremental utility of its
shape.
