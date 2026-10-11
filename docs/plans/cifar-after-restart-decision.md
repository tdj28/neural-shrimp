# CIFAR after span and restart: stop the current escalation

8 October 2026 UTC. **Design only. No implementation, profile, new derivative,
solve, damping trial or launch is authorized by this note.** Based on integration
`e9ecb9ba6356206af36b2e0da01915f35d545f34` and the independently audited saved
subset-A evidence below. No arrays were reduced or operators evaluated while
preparing it; timing arithmetic uses recorded event durations only.

**Decision.** Keep the current CIFAR candidate unresolved and stop this sequence
of correction attempts now. The smallest useful decision is to retain a
negative, matched restart result, rather than automatically enlarge the budget.
If obtaining a CIFAR periodic orbit becomes an explicit renewed priority, a
single declared larger-space test has more direct decision value than a reduced
implementation or more damping of the same failed direction. That conditional
ranking is not a request to launch it and does not relax any scientific gate.

## What the saved evidence actually changes

At the identical98,666-dimensional state, restart50/four cycles and
restart200/one cycle both spend200 inner iterations. The first50 callbacks and
sampled derivatives agree. True forcing improves0.8843136325 ->0.6619885191,
but remains above0.5; SciPy info1 and independently tested1e-10 linear tolerance
remain failures. The new direction has negative R dot Ap, but all four
alphas1,1/2,1/4,1/8 increase absolute residual merit (ratios665.1137,58.4069,
4.09947,1.06083). No Armijo, closure or candidate-step acceptance follows.
The direction is saved evidence, not an applied update.

The longer cycle's estimated residuals at iterations100/150/190/200 are
0.818611/0.775488/0.682763/0.661989. Thus there is **continued linear progress**,
not evidence of a plateau or an inconsistent Newton equation. These samples
do not predict the iteration count needed for forcing0.5 or nonlinear closure.
They do not identify conditioning, nonnormality or orthogonality loss.

The span audit retains16,746 active coordinates, but only1.7489397669e-8 of the
original squared residual lies in the81,920-dimensional complement. Its fixed
>=0.5 priority test fails. This rejects dominance of those directions in this
residual; it does **not** rule out conditioning or engineering benefits from
eliminating their simple scalar block. The quoted complementary fraction of
the true linear residual6.3553641512e-9 belongs to the **old50x4 direction**;
no decomposition of the new200-space direction is being implied.

## Options and outcome consequences

| Option | Information a new outcome could add | Cost and present decision |
|---|---|---|
| Stop this fixed-state escalation | Preserves a qualified negative result; makes no orbit-absence claim. | Zero new model/solver work. Recommended now; no gate or evidence is discarded. |
| One larger Krylov space | q>0.5 still rejects the direction; q<=0.5 with no fixed Armijo trial passing isolates a remaining globalization problem; q<=0.5 plus a passing fixed trial would justify designing an inexact corrector. None alone establishes an orbit. | This has real conditional decision value. A400-space illustration costs about29.68s by the heuristic below, with a larger basis; not a proposed run. The unsaved old basis prevents resuming it, so a fresh run necessarily repeats prefix work. Do not separately repeat the200 solve or perform a length sweep. |
| Explicit reduced coordinates | Successful parity would qualify an equivalent implementation. Measured lower memory or time would be an engineering gain; changed linear progress would need its own matched control. | The200-space basis falls158,654,928 ->26,927,568 bytes. It is not justified as a residual repair by the failed span-priority test. Current observed RSS is below1GiB, so memory is not the demonstrated bottleneck. Defer. |
| More damping / a trust-region method | A new trust-region direction might meet both forcing and nonlinear merit gates; merely finding a smaller merit-decreasing step along the current direction adds little. | The current direction fails the fixed forcing rule independently of its trial alpha. Smooth-model local descent already predicts sufficiently small decreasing steps. No smaller trial to confirm that prediction. A trust-region solver is a new algorithm requiring direction/adjoint/constraint controls and a separate costed design; it is not a parameter-only repair. |

The exact-coordinate reduction preserves the training-input map under the
stated plain-GD/isotropic-ridge assumptions. It can reduce Krylov storage and
dot/axpy work by the coordinate ratio5.8919. A genuine reduced model with
precomputed XQ might also reduce derivative arithmetic; a wrapper that only
lifts/projects full-model products does not provide that benefit automatically.
Neither speed nor active-operator conditioning is established by a smaller
coordinate count. Deleting complementary weights can change unseen-input
predictions, so this is not a claimed training/inference optimization.

## Cost basis, not permission for a follow-on

Recorded restart200 solver time is11.474315s; whole worker12.686797s and
launcher14.160857s. Its202 saved matvec timings sum to8.658276s, leaving
2.816039s of solver overhead and1.212481s outside it. Those categories include
instrumentation/I/O and are not pure derivative versus orthogonalization
benchmarks. For comparison, old205 matvecs summed6.780299s and the old solver
took8.011272s; the protocols and observed timings do not isolate a speed cause.

For a hypothetical single400 cycle with402 total A products, linear scaling
of recorded products and quadratic Gram--Schmidt work gives
`8.658276*(402/202) + 2.816039*(80200/20100) + 1.212481 = 29.679445 seconds`.
It would allocate316,520,528 bytes for the401-vector full basis, versus
158,654,928 for201 vectors. These are arithmetic estimates, not a measured
profile, runtime bound, predicted residual or approval. The previous actual
attempt fit CPU/BLAS/Torch1, nice>=10, internal45s/external60+5s, monitored2GiB,
and31+1MiB raw (20,750,563 bytes retained including its launcher). No new cap
or output directory is allocated by this stop decision.

Reopen only through an explicit scientific-priority decision: the purpose must
be to find a direction meeting **both** forcing<=0.5 and the unchanged finite
trial merit rule, not merely a prettier linear residual curve. Any future
unit needs committed code/card, independent source review, meaningful true-
residual/parity/failure controls, exact resource ownership and a fresh capacity
check, and a separately authorized fixed attempt with a prospective saved-data
auditor. A passing direction would still require a separately reviewed nonlinear
corrector, closure<1e-8, primitive separation>=1e-7 and subsequent stability work.
No automatic continuation, reduced-solver implementation, arbitrary smaller
trial, repeat200 solve or threshold change follows from this note.

## Exact retained authority

All artifact paths are under `/data2/neural-shrimp-data/`; the same original
state is `explore/cifar/corrector-only/attempt-001/`'s saved NPY, SHA256
`36066b4b1d657924bd6efdce83dc3fd2d06a7ab135ef78afb03fb87649dd9674`.
The original binary64 eta0.00827753, ridge0.01 and normalized subset-A identity
remain fixed; subsetB and all protected outcomes remain untouched.

- Old diagnostic `explore/cifar/linear-diagnostic/attempt-001`, execution
  `a9830e714d259597a53cee6ef7be41a9f07bb205`; manifest
  `d47983793f58b033389fca5fc4aedc9ad78798b38cef06b2f3a4cf0f731534a3`;
  event timing file SHA256
  `fb9721f73b383cc1b074d2a61d701a0b87e185474378da097b0d72144fdafd62`.
- New diagnostic `explore/cifar/restart-sensitivity/attempt-001`, execution
  `6fea29cda810ebb72c234dbd0fc1533bbca4e5cb`; manifest
  `5f9f77bb2926581c828e5528e02e475858ca626fc9df49be7a6f234eae4b1e96`;
  event timing file SHA256
  `4df1d1630d1c99d2a20ffb18541d14b5065749722a1c45bb52fea4f71f29cbf8`.
- New independent audit `audits/cifar-restart-independent/attempt-001`,
  execution `4de644d379884ba85ce71bf39611c7be6ab90d10`; manifest
  `0a6e42f5bd926504cbe6ed3f7f81f9fe4d33c6e634f17a4a193ed8d3c9bc9c1c`.
  Compact receipt `docs/reviews/cifar-restart-independent-outcome-receipt.json`
  SHA256 `40d61236d5692baf6aa7c378b7af0e583f0992fcdb6542c317f1bce80c62365d`.
- Independent span audit `audits/cifar-input-span-independent/attempt-001`,
  execution `b8f074b8539d5a1ae7e3d4f629b8dcb6fb659bdd`; manifest
  `0221596b14e01b6630f0d4014d5f34d29d23aa6011e9397ab7b9ee1d1097d82d`.
  Compact receipt `docs/reviews/cifar-input-span-independent-outcome-receipt.json`
  SHA256 `20950658d64ee3a8c18bb17a124907f2a4e564a523ef0b8f7136a7b7dd28c785`.

These are exploratory floating-point development diagnostics. Saved maps and
products remain source-bound, not independently regenerated here; no existence,
attraction, general solver advantage or language-model utility follows.
