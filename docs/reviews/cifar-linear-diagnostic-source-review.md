# CIFAR bounded linear diagnostic: independent source review

**Verdict: pass for the declared one-state exploratory diagnostic.** No target
model evaluation, target derivative, Krylov solve or trial was run by this
review. This approval concerns source correctness and the bounded pilot scope,
not a periodic orbit or a successful correction.

Producer source: `a9830e714d259597a53cee6ef7be41a9f07bb205`, branch
`codex/cifar-linear-diagnostic`. Independent review branch:
`codex/review-cifar-linear-diagnostic`, owned worktree
`/home/d7082791602/.codex/worktrees/scalar-ns-independent-review/neural-shrimp`.
Reviewed the new core, driver, all sixteen synthetic controls, the dated compute
plan amendment and final prospective research-log card. Producer files remain
unchanged.

## Findings

- With `R=F^2(x)-x`, `A=D(F^2)(x)-I`, the code solves `A p=-R` at the same
  original state. It independently evaluates `e=A p+R` after GMRES returns.
  The true forcing ratio `q=||e||/||R||` is distinct from the callback estimate
  and solver return code; unsuccessful GMRES does not become declared success.
- The merit derivative is correctly `d=R^T A p`. Since
  `d=-||R||^2+R^T e <= -(1-q)||R||^2`, `q<=0.5` implies descent for exact
  linear algebra. The implementation independently checks the actual numerical
  negative sign and tests absolute merit `||R||^2/2`, so a larger parameter
  norm cannot alone produce a passing Armijo test.
- The four fixed damping factors each start from the original x. Candidate
  status requires finite output, the true forcing gate, descent and actual
  Armijo decrease. Closure and primitive-separation screens are unchanged and
  diagnostic only. Every trial explicitly has `accepted_orbit=false` and
  `applied_to_continuation=false`; no trial becomes another trial's initial
  state, and no spectrum/continuation routine is called.
- The installed SciPy implementation with `callback_type='pr_norm'` counts
  four restart cycles. With its default zero initial guess it uses at most
  200 inner operator products and four restarted true-residual products.
  The independent final product makes the declared cap **205**. A separate
  synthetic ill-conditioned 80-dimensional diagonal control exercised all
  200 inner iterations, `info=4`, and exactly 205 calls, with no fifth cycle.
  The pre-solve stored-direction parity check is separate from this solve
  budget and is disclosed in the plan.
- Source identity is checked against committed HEAD before output creation.
  The two input manifests are pinned by digest, and their entire current
  saved-file inventories/hashes were independently checked read-only in this
  review (2 reference files and 6 prior-attempt files). Model, shape, initial
  state and exact normalized x/y identity are additionally checked by the
  driver. Initial residual/separation and gradient/HVP/J2VP parity precede
  the solve. No raw CIFAR archive, subset B or fresh seed is loaded.
- Returned p is fsynced before the independent product can fail or time out.
  Trial states are saved before evaluation; completed trial outcomes are
  fsynced to the event stream, even when later trials interrupt before the
  aggregate `trials` field is assigned. The exclusive directory, failure
  status and final inventory retain unsuccessful attempts. No fallback or
  automatic retry is present.

## Verification and limits

Eighteen synthetic tests passed in 1.02 seconds: all sixteen producer controls
plus two independent controls in
`tests/test_cifar_linear_diagnostic_review.py`. The latter check the full
50-by-4 operator count and demonstrate that true forcing/descent still require
an independently passing nonlinear Armijo trial. All tests ran CPU-only, one
thread, nice 10. Existing target artifact verification was hashing/read-only.

The planned command has CPU/BLAS/Torch one thread, nice 10, internal 100 seconds
and external SIGTERM at 115 seconds plus 5 seconds kill grace. Internal checks
are cooperative between native operations; the external timeout is necessary
for an operation that stalls. The driver records nice/thread settings, while
the approved CLI and launch command enforce them; calling `run()` directly
outside that command is not the reviewed launch procedure.

The fixed completed path saves 23 vectors of length 98,666 float64 entries:
18,154,544 data bytes before small NPY headers and bounded JSON/events, safely
below 32 MiB. Each array write checks the existing folder bytes against that
cap. JSON/event writes do not use a general-purpose quota guard; the fixed
205-product/200-callback/four-trial structure supplies their small bound here.
This is not approval for arbitrarily enlarged diagnostics under the same cap.
SIGKILL, storage failure, or interruption during finalization can leave a
truncated file/manifest; these remain incomplete evidence, not successful runs.

A syntactically valid existing review commit is recorded, but the CLI does not
interpret that commit as an authorization decision. Root's actual signoff and
exact approved command remain required. A partial direction that passes these
checks would justify only a local inexact-step candidate; no general Newton
convergence, conditioning diagnosis, primitive orbit, stability or orbit
nonexistence follows.
