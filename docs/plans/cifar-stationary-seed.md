# Bounded stationary/eigenpair CIFAR seed

11 October2026 UTC. Source-only preparation; no new native evaluation yet.
The current fixed-rate shooting correction and fixed4D span are complete
negative controls. This new formulation targets a stationary point of the
same original subsetA512 regularized loss from the exact midpoint of the two
retained current phases. It does not transfer Jacobian actions between states.

Choose at most four Newton units for g(m)=0. Each solves the symmetric actual
Hessian equation H(m)p=−g(m) with SciPy MINRES, implicit exact-zero initial
vector, rtol1e−12 and max256 actions. Recompute H(m)p independently after the
solve. Require actual forcing norm(g+Hp)/norm(g)<=0.5 and negative derivative
g.Hp of gradient-norm merit before any trial. Thirteen fixed step lengths
2^(-j),j=0..12 are tested in order, with Armijo1e−4 on half squared gradient
norm. Enforce total distance from the initial midpoint <=5% of
max(1,norm(initial midpoint)); reject outside-radius trials before native work.
Stop at the first failed forcing/backtracking unit. No automatic iteration,
span, trust radius or step-count extension. A passing numerical stationary
candidate requires norm(g)/max(1,norm(m))<=1e−10.

Only after this stationarity gate, perform32 ordinary Hessian power steps
starting from the normalized original nonzero phase displacement, followed
by one fresh final action. Require positive Rayleigh value tau and actual
relative residual norm(Hq−tau*q)/max(1,abs(tau))<=1e−8. This is a numerical
positive eigenpair, not a proof that it is the largest eigenvalue. Its candidate
learning rate2/tau changes the physical optimizer; no cycle or attraction
follows automatically. An amplitude-constrained branch launch is separate.

The forthcoming wrapper must authenticate original state/data/objective/native
source, preserve all four failed predecessor cases, meter complete actual
state/action inputs and returns, retain intermediate accepted points, qualify
HVP by finite differences and symmetry before solving, and admit work with a
source-matched measured cost and exact local-resource ownership. Proposed
hard envelope is1063 HVP and59 direct gradients,400s/2GiB RSS/8GiB address
space/3GiB raw, CPU/BLAS/Torch1 nice10; native input source is unchanged and
prior measured operation maxima are available. This preparatory plan is not
a target run card. A committed driver/card and independent source review
must precede any target evaluation.

Seven combined stationary/amplitude controls check exact nonlinear Jacobian
identities, full quadratic stationary closure, actual nonlinear merit and
radius behavior, a deliberately corrupted fresh action rejected before trials,
and positive/negative actual Hessian-mode qualification. All passed0.27s.
