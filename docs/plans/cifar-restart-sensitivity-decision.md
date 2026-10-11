# CIFAR linear solve: one restart-sensitivity diagnostic

8 October 2026 UTC. **Design only; no implementation or launch authorized.**
Recommendation: compare one unpreconditioned GMRES cycle of restart200 with
the already retained restart50/four-cycle result at the identical state and
operator. Reuse the saved baseline; do not rerun it or start a full corrector.
This is a small test of restart sensitivity, not an assertion that restarting
caused the stall or that a larger basis will converge.

The new input-span result reports complementary squared-residual fractions
1.74894e-8 initially and6.35536e-9 for the true linear residual. Its fixed >=0.5
reduction-priority gate failed; its independent outcome audit is pending at
this decision. Projection is therefore not proposed as a convergence repair.
The algebraic reduction can remain valid without explaining this residual.

## Why this test, and what the old history does not show

The independently audited old solve used200 inner iterations and205 operator
calls, with restart-end estimated relative residuals0.8921721,0.8885642,
0.8862027 and0.8843136. The independently checked final true forcing ratio was
0.8843136324964805, failing0.5. Progress is slow but not zero; the history alone
cannot distinguish discarded Krylov information, a difficult active operator,
finite-precision effects or other causes. It establishes neither a condition
number nor inconsistency of the Newton equation.

In exact arithmetic, unrestarted GMRES200 minimizes over the full degree200
Krylov space and can achieve residual no worse than four restarted50 cycles
from the same zero guess: the restarted residual is also a degree-at-most200
residual polynomial with constant term1. That inclusion is a reason to test,
not a floating-point guarantee. The installed SciPy implementation uses one-pass
modified Gram--Schmidt; a longer basis can lose orthogonality. The final true
residual, not only its inner estimate or return code, is decisive.

A new projection implementation, preconditioner or solver family would change
more than one mechanism. Repeating the old solve or paying for faster products
would not answer restart sensitivity. A second profile is unnecessary for
this small bounded test given the saved target-specific timings. No condition
number, spectrum or Krylov-eigenvalue estimation is part of this unit.

## Exact prospective comparison

Use the same98,666-coordinate full problem, subset-A normalized float64 x/y,
binary64 eta0.00827753 and ridge0.01, theta, R=F²(theta)-theta and
A=D(F²)(theta)-I. No projection, preconditioning, initial-state perturbation,
new batch or optimizer change. Initial Krylov guess is explicitly zero, not
the old returned direction. Set restart200,maxiter1,rtol1e-10,atol0 and
callback_type='pr_norm'. Stop early only under the existing SciPy convergence
or breakdown rules; do not spend unused calls on a second cycle.

Input authority is the original reference bundle manifest414e8f2b…,
corrector-only manifestd0239383…, and audited linear-diagnostic manifest
d4798379…. Their full hashes/paths are in the companion decision receipt.
The original state NPY hash is36066b4b…; exact subset identities come from the
original reference receipt, not newly loaded CIFAR archives. Keep the old
producer executiona9830e71 and new execution identities distinct.

Before solving, validate clean committed source including indirect imports,
exact data/state/model/rates and all original inventories. Require the same
recorded Torch2.14.1+cu130,NumPy2.5.3,SciPy1.18.1 and GMRES source-function hash;
a mismatch requires review, not automatic substitution. Replay the initial
map1/map2/residual and require their bytes to match the saved CPU baseline.
Retain the existing gradient/HVP/J2VP reference-direction parity gates
(atol1e-11+rtol1e-9*abs(reference)). Any failure blocks the solve.

Save every inner residual as it arrives. The first50 estimated residuals must
match the old first50 with rtol1e-10,atol1e-12; check sequentially and abort on
mismatch. This predeclared control isolates the restart change. Earlier true
convergence/breakdown remains visible and cannot be silently interpreted as a
matched200-step comparison. Save p before the independent A*p check. Compute
true q=||A*p+R||/||R|| and d=R^T A*p, retaining info even if nonzero. The existing
forcing q<=0.5 and strict d<0 rules remain unchanged.

Evaluate the same four independent trial states theta+alpha*p,
alpha=(1,1/2,1/4,1/8), from the original theta. Keep absolute merit||R||²/2,
Armijo1e-4, closure1e-8 and primitive separation1e-7. All four are diagnostic
trials: accepted_orbit=false, applied_to_continuation=false, spectrum=false.
Passing the inexact forcing gate does not change the existing corrector's
rtol1e-10 or Newton1e-11 stopping rule.

## Counts, measured cost and caps

With zero initial guess and explicit LinearOperator dtype, the installed SciPy
path uses at most200 inner A products plus one end-of-cycle true-residual
product; the independent final product makes **202**, versus the old205.
The one reference J2VP makes203 total two-step JVP calls. Since each uses two
HVPs, with one separate reference HVP the cap is407 HVP calls. Initial two map
updates, one reference gradient and eight trial map updates add11 direct
gradient calls, apart from gradient work internal to HVPs. If fewer inner
steps run, record the actual count; unused calls are not reassigned.

Saved event timings sum to6.780299s for205 A products (33.074628ms mean), while
the old solve took8.011272s and its whole diagnostic9.188636s. Scaling product
work by202/205 gives6.681075s. Modified Gram--Schmidt dot/axpy pairs increase
from4*(50*51/2)=5100 to200*201/2=20100, factor3.941176. Conservatively scaling
*all*1.230974s of old nonoperator solve overhead by that factor, then adding
1.177364s of other work, gives **12.709923s**. This is a heuristic based on saved
measurements, not a new benchmark or a runtime upper bound. Allow roughly15–25s
under comparable host conditions; enforce internal45s/external60s+5s.

The dense basis changes from51*98666*8=40,255,728 bytes to
201*98666*8=158,654,928 bytes; extra118,399,200 bytes. The200x201 Hessenberg array
adds321,600 bytes. Old peak RSS was787,500KiB; a2GiB RSS cap leaves margin for
Torch and workspace and is small relative to the approximately41GiB available
reported by root. Recheck capacity and foreign processes immediately before
any later launch. CPU/BLAS/Torch intra/inter-op1,nice>=10; no GPU, paid compute,
subsetB, reserved seeds or protected outcomes.

New retained artifacts: **32MiB total**, partitioned31MiB scientific outputs
plus1MiB launcher/log evidence (log<=128KiB). The prior output occupies18,218,537
bytes with the same23 full saved vectors and200 history values, so this cap is
supported without a new storage profile. Preserve source, settings, exact
inputs, initial states/images/R, every callback/count/timing, returned p before
validation, A*p/e, all four states/images/residuals, every decision and failure,
and immutable final manifests. Internal Arnoldi basis/Hessenberg vectors are
not archived; this limitation prevents a later claim of an independently
replayed orthogonalization history. A deadline inside GMRES can retain history
without a returned p and must remain incomplete; do not manufacture an iterate.

Use one exclusive prospective root
`/data2/neural-shrimp-data/explore/cifar/restart-sensitivity/attempt-001` and
sibling`attempt-001-launcher`. A task-owned process-group wrapper must retain
source/resources/argv/PIDs/exit/full hashes, terminate only its owned group,
and bound inherited pipes even after leader exit. Preserve all failed/partial
attempts, without retry. No directory is created by this design.

## Decisions and preparation required

1. Parity/prefix/control/resource failure: comparison unqualified; stop and
   retain evidence. Do not change tolerances, libraries or the state and retry.
2. True q>0.5: the fixed inexact-direction gate still fails. Report any measured
   reduction descriptively; no larger restart/cycle sweep or corrector follows.
3. q<=0.5,d<0 but all Armijo trials fail: linear progress is insufficient for
   this declared nonlinear trial set; stop with a globalization-design question.
4. q<=0.5,d<0 and an Armijo trial passes: a useful local inexact direction exists
   under the declared rule. This may justify designing a separately reviewed
   inexact corrector; it authorizes no update or continuation automatically.
5. Estimated/true residual disagreement or misleading info0: reject solver
   convergence, retain both, and investigate arithmetic/operator consistency
   before any scientific follow-on. Even strict linear convergence establishes
   neither nonlinear orbit closure nor stability.

Before execution, implement one dedicated fixed-settings driver using the
reviewed core (do not silently change old global defaults). Independently test
one-cycle semantics/product cap/zero guess, prefix mismatch, false convergence,
true-forcing/Armijo separation, parity failures, persistence before later
failure, immutable inputs, all source imports, byte and wall limits, and
launcher ownership. Commit/push source, synthetic controls and an exact run
card; obtain independent source review and root's separate one-launch approval.
Prepare a blind saved-outcome audit before reading new results. No target,
profile, model, JVP or solve was executed in preparing this decision.
