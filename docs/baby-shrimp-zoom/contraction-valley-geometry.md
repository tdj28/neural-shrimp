# Trace junctions and contraction valleys

This note gives conditional local algebra for the BSC5 investigation. It is
not a numerical result, an interval certificate, or a classification of the
whole shrimp. Its practical question is whether two parameter paths on which
the continued periodic orbit strongly damps perturbations actually meet.
The coordinate metric is the scaled Euclidean metric declared in the
experiment plan. An anisotropic rescaling can change a valley definition.

## The contraction objective

Suppose a smooth, isolated pair of real Floquet multipliers has opposite
signs, with trace $\tau$ and product $\Delta<0$. Its eigenvalues are

$$
\lambda_\pm=\frac{\tau\pm\sqrt{\tau^2-4\Delta}}2.
$$

The larger modulus is therefore exactly

$$
\rho=\frac{\sqrt{\tau^2-4\Delta}+|\tau|}2.
$$

If this pair dominates the entire spectrum, $\rho$ is the asymptotic
linear growth or decay factor per full period; it describes contraction when
$\rho<1$. It is not a singular-value bound
on transient amplification, a finite-time image intensity, or a measure of
language quality. Numerical Ritz separation supports the dominance
assumption; it does not supply a full-spectrum proof.

Trace zero balances the moduli of the two multipliers. At trace zero,
$\rho=\sqrt{-\Delta}$, which need not vanish or be locally minimal as the
parameters change. Exact superstability requires a separate definition and
zero-multiplier evidence; a trace-zero intersection alone supplies neither.

## A regular trace-zero crease

At a regular point of $\tau=0$, let $n=\nabla\tau/\|\nabla\tau\|$ point
toward positive trace. Differentiating on the two sides along $u+s n$ gives

$$
\partial_{s,-}\rho=-\frac{n\cdot\nabla\Delta}{2\sqrt{-\Delta}}
                       -\frac{\|\nabla\tau\|}{2},\qquad
\partial_{s,+}\rho=-\frac{n\cdot\nabla\Delta}{2\sqrt{-\Delta}}
                       +\frac{\|\nabla\tau\|}{2}.
$$

A strict first-order transverse minimum requires the left derivative to be
negative and the right derivative positive. Equivalently,

$$
|n\cdot\nabla\Delta|<\sqrt{-\Delta}\,\|\nabla\tau\|.
$$

If the strict inequality fails with a nonzero sign margin, trace balance is
not a normal minimum there. Equality is a degenerate case requiring
higher-order information. A finite numerical comparison must also resolve
its sign against measured errors. This criterion concerns the geometric
normal of the regular trace-zero curve, not an arbitrary plotting section.

## Why a trace X need not be a contraction X

Consider a stationary trace zero: $\tau=0$, $\nabla\tau=0$, with a
nondegenerate indefinite trace Hessian. A change of local coordinates puts
the trace into a saddle form, so its zero set has crossing branches. Since
$\tau(u+q)=O(\|q\|^2)$ at this point, the absolute-value term contributes
no first derivative. The radius is differentiable there and

$$
\boxed{\nabla\rho=-\frac{\nabla\Delta}{2\sqrt{-\Delta}}.}
$$

Thus a nonzero determinant gradient leaves a direction in which contraction
improves even at the trace junction. The radius need not possess a Hessian
there: differentiability does not remove the absolute value of the
quadratic trace term.

There is a precise necessary condition for a stronger crossing claim.
Suppose two continuously differentiable valley branches meet with distinct
tangents and, at their shared point, $\rho$ has a two-sided local minimum
on each branch's geometric normal line. Their two independent normals must
both annihilate $\nabla\rho$. Consequently $\nabla\rho=0$, and at a
stationary trace zero this requires $\nabla\Delta=0$. This argument does
not assume a full two-dimensional local minimum, and the necessary
condition is not sufficient for a crossing.

The trace-junction conditions impose three scalar equations on steering,
temperature and memory: $\tau=0$ and two components of $\nabla\tau=0$.
The additional contraction condition adds two components of
$\nabla\Delta=0$. Generically these five equations cannot all be tuned
independently with three controls. Symmetries or identities can reduce the
number of independent equations, so this is a generic explanation, not an
impossibility theorem for this network. It also does not exclude tangential
mergers, mixed nonsmooth junctions, different orbit families, or features
defined from finite-time images.

Even if both gradients vanish, the leading directional change is

$$
\rho(u+q)-\rho(u)=\frac14\left(
 |q^T H_\tau q|-\frac{q^T H_\Delta q}{\sqrt{-\Delta}}
\right)+o(\|q\|^2).
$$

Its directional signs and degeneracies still need examination.

## This does not establish mode coupling

An explicit diagonal example shows why the geometry alone does not prove
coupling. Take $r_0>0$, a nonzero $\varepsilon$, and a small neighborhood
where $r_0+\varepsilon x>0$. Set

$$
\tau=h+x^2-y^2,\qquad \Delta=-(r_0+\varepsilon x)^2,
\qquad M=\operatorname{diag}(\lambda_+,\lambda_-),
$$

with the eigenvalues defined above. The two linear modes have no off-diagonal
coupling. At $h=0$ the trace-zero lines cross at the origin, but
$\nabla\rho=(\varepsilon,0)$ there. They cannot also be two transverse
geometric-normal minimum branches through that point. This is an algebraic
counterexample to inferring coupling from the mismatch; it is not a fitted
model of the neural experiment. Explaining why the neural map has its
measured determinant variation remains a separate dynamical question.

## Defining and testing a valley

An independently minimized finite section is evidence about that section.
Joining such minima does not automatically identify a unique valley curve.
For a smooth branch of $\rho$, one possible explicit convention uses the
unit eigenvector $e_+$ of its largest Hessian eigenvalue and solves

$$
f=e_+\cdot\nabla\rho=0,
$$

requiring positive curvature and a simple eigenvalue. The eigenvector's sign
must be transported continuously, and the active trace sign must remain
valid. Because the eigenvector can rotate, $e_+$ need not equal the
geometric normal of the resulting $f=0$ locus. The measured curve tangent
and actual normal-minimum test must be checked separately.

On a smooth $C^3$ radius branch, at a stationary radius point with a simple,
positive largest Hessian eigenvalue $\kappa_+$,
$\nabla f=H_\rho e_++(D e_+)^T\nabla\rho=\kappa_+ e_+\ne0$.
The implicit-function theorem then gives one locally regular curve under
this convention. Two transverse smooth branches there would require loss of a
precondition or a degeneracy. Regular trace-zero creases use the one-sided
criterion instead. A meeting between a smooth feature and a crease is a
mixed junction and must be classified as such; proximity does not establish
that their definitions glue into a single crossing valley.

The numerical helper `tools/contraction_valley_geometry.py` implements these
local identities, refuses incompatible regimes and reports curvature,
eigenvalue gaps and orientation diagnostics without supplying invented
uncertainty thresholds. Exact-zero algebra is kept separate from near-zero
measured inputs. Its regular-crease extension limits are explicitly one-sided
derivatives of the two smooth formulas, not a single radius Hessian at the
crease. A numerical driver must retain any measured nonzero trace and record
a conditional zero projection separately. Actual curve continuation, pair dominance, orbit identity,
two-scale error checks and numerical or rigorous uncertainty bounds remain
the experiment's responsibility.
