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

## Fixed attempt001 card after helper review

Clark independently cleared f967ff94 helper/math with seven controls0.19s.
One source-reviewed wrapper now fixes
`/data2/neural-shrimp-data/explore/cifar/stationary-seed/attempt001`.
The unchanged initial pair is exactly the completed projected001 pair;
outer89eb660c4f1ba722381db36eda561c8c826a642efa704b5c0c7ba9f7e9c4803e,
executiond9c74b034913e7891def3919548ae276f9d92944, unchanged subsetA512,
MLP3072–32–10tanh/ridge0.01. The preceding scaled correction, saved geometry
and phase diagnostic are also authenticated and preserved. Every original
executed source byte/archive and Python/numpy/scipy/Torch environment matches
the measured parent before target arrays. The new midpoint is literal(x+y)/2.

Full unit has at most1063 HVP and59 gradients: two Hessian qualification
vectors; one initial and six centered-difference gradients at scales1e−3,
1e−4,1e−5 times max(1,norm(midpoint)); at most four256-action solves plus
fresh action checks and four13-trial gradient searches; conditional32+1
Hessian mode actions. Final FD relative error<=5e−6 and symmetric bilinear
error<=1e−10 are mandatory before solving. No LSMR, old Jacobian transfer,
fixed-rate shooting step, protected subset or new data acquisition occurs.

All native state inputs are retained once and referred to by full hash in
per-attempt records before each call; every HVP direction, gradient/HVP return
and operation counter is durably retained. Complete exact source archive,
actual SciPy MINRES source, every accepted/intermediate native state and final
candidate remain available. Isolated copied-archive import test checks all
executed repository modules including package initialization.

Fixed400s worker/420s launcher+5 cleanup, CPU/BLAS/Torch1 nice10,2GiB RSS,
8GiB address space,3GiB worker+3MiB launcher raw,10GiB host headroom. Same-native
projected001 maximum timings0.0849204881s/HVP and0.9516282950s/gradient give
~374.28s bound including2× all operation maxima,20s overhead and61.44s for
full raw-cap writes at50MiB/s. Repeat that exact admission with the fresh
qualification timings before solving. No conditional budget expansion.
Eleven combined controls passed2.50s; independent wrapper review still needed
before launch. A completed negative stationary attempt remains a result,
not authorization to expand iterations or trust radius.
