# Huginn draft v3: independent numerical pre-outcome review

The proposed held-out association is worth testing, but the current implementation
cannot be described as a same-map spectral prediction of step-32 convergence.
Resolve the outcome/map definition and spectral-feature scope before freezing.
This is a finite source review, not a request for a full-spectrum certificate.

**Reviewed authority.** Claude's 13:38Z assignment; commit
`576b0b86c4051fe226b8c8cc4e6bd5e1e0f792b3`. Inference exported these exact sources
and verified that all ten files also match then-current private main `af09e666`.
Export manifest SHA256:
`dffc8118fd3edc20aa2079632b2824a2d3204e57f437a76534f9f377f73334b9`.
The review read source, tests, the draft and its disclosed development summaries.
No Huginn test item, outcome, numerical array or checkpoint was opened; no model,
producer analysis command or test suite was executed. Part B inference and Part A
coverage/degenerate-bootstrap handling are reviewed in the separate statistical
lane; the findings below concern map, spectrum and finite-precision semantics.

## 1. The predictor and residual currently describe different systems

`huginn_relax_eval.py:359–361` builds the spectrum prompt from the question and
**gold choice alone**. `HuginnLoop.setup` tokenizes that combined string at
`explore_latent_loop.py:197`. In contrast, `arc_batch` tokenizes the question/context
once and each continuation separately, then concatenates IDs (`eval:377–379`).
Equal displayed strings do not guarantee equal token IDs or special-token
handling. No source assertion establishes equality of the actual gold-sequence
IDs between these paths.

More decisively, `Relaxed.trace` computes one norm over the entire masked batch
of **all choices**, not the gold sequence (`eval:160–163`, assigned mask at464).
Ignoring rounding, the recorded squared relative residual is
`sum_c ||F_c(s_c)-s_c||² / sum_c ||s_c||²`: a state-norm-weighted aggregate of the
choice residuals. A gold-prefill Jacobian is not this aggregate's same-system
Jacobian. The all-choice map is a product map, whose spectral radius would depend
on all the choice blocks, even if tokenization were aligned.

Gold conditioning is not automatically invalid leakage for a declared
*gold-conditioned mechanistic probe*. It does prevent claiming a label-free
predictor available when the correct answer is unknown. The current sentence
“no outcome below depends on correctness” does not resolve this: it is the
predictor's input selection that uses the label.

**Finite correction before freeze:** choose one estimand. For a same-prefill
probe, share the exact token-ID builder, retain per-sequence residuals, select
the same gold sequence on both sides, and label the analysis gold-conditioned.
The all-choice answer-change endpoint can remain a separate secondary transfer
question. Alternatively keep the current aggregate, explicitly call it a
cross-system empirical predictor, and remove the same-map contraction claim.
Identical token IDs should be recorded/checked before computing each test
predictor; no test outcome is needed for that check. The gold-selection decision
must be fixed now, not chosen after correlations are known.

## 2. “Residual at step 32” is presently residual at the input to call 32

In the wrapper, `out=Core(x)` is computed, the trace appends `||out-x||/||x||`, and
then the readout receives `new` with label `step+1` (`eval:153–165`). Under the
usual `s0` initial-state convention, the last of 32 trace entries is
`R(s31)=||F(s31)-s31||/||s31||`; the step 32 likelihood readout uses `s32`.
`item_table` simply takes `native_residual[-1]` (`analysis:270`).

This is a naming/definition mismatch, not an invalid residual. Either freeze it
explicitly as the native residual **entering the 32nd update**, or perform the
separate native evaluation at `s32` if the intended endpoint is `R(s32)`.
Do not silently shift the endpoint after outcomes. Validate the expected trace
length and exact 24–32 readout indices, rather than accepting whatever last
entry exists. The present item-level synthetic test supplies 32 fabricated
entries; it does not test the actual wrapper's indexing.

## 3. Partial ARPACK is a usable feature, not established spectral completeness

`spectrum_at` applies float32 JVPs and converts their outputs to binary64
(`latent:925–935`). A float64 `LinearOperator` container does not make its map or
JVP float64. It requests 8 LM/SR/LR eigenvalues with tolerance 1e-5, saves no
vectors, and retains whatever ARPACK converged when it raises
`ArpackNoConvergence` (`latent:937–946`). Those flags are saved at 963, but
`items_cmd` does not inspect or include them in its question-level output
(`analysis:295–305`). Nor does it require a nonempty finite spectrum before
computing maxima. The design's promise to report flags is not implemented there.

There is a deeper issue even if all three sets converge. LM/SR/LR of J need not
contain the largest-modulus eigenvalue of the **shifted** matrix
`(1-lambda)I+lambda J`. For an exact real matrix with eigenvalues
`+.991,…,+.998`, `−.890,…,−.897`, and `−.88±.45i`, all exact k=8 LM/LR and SR
sets omit the complex pair. At lambda 1.05 all selected transformed moduli are
below 1 (maximum 0.9979), whereas the omitted pair has squared modulus
`0.974²+0.4725²=1.17193225>1`. This is a mathematical counterexample, not a Huginn
measurement. Thus “all ARPACK sets converged” alone cannot make the union's
maximum the true relaxed spectral radius. A good Ritz residual verifies an
approximate eigenpair, not the absence of omitted modes.

**Acceptable finite paths:**

- Keep the fixed-budget union as a predeclared **estimated Ritz-spectrum feature**
  for A1. Require finite nonempty sets, record every count/convergence flag and
  fixed-point residual, freeze how failures count toward the setup ceiling, and
  report feature reliability. In A3/A4, distinguish detected numerical instability
  from “no unstable mode detected”; do not certify stability by a missed mode.
- If a leading-radius interpretation is essential, target LM of the relaxed
  linear operator for each frozen arm and freeze a bounded reliability policy
  before outcomes: retained eigenvectors, explicit relative Ritz residuals,
  iteration/count flags and an agreement/error-band check suited to float32.
  Unresolved near-unit values are indeterminate. This is ordinary numerical
  qualification; neither a dense spectrum nor an interval proof is required.

No exact numerical threshold for float32 JVP or bf16 errors can be inferred from
this source review. Do not invent one. Any added reliability calibration should
use development data and be frozen before test outcomes.

## 4. bf16 step residual is a finite-budget endpoint, not a rate estimate

Both `x` and `out` follow the bf16 recurrence in Part A. The code subtracts them
**before** casting the difference to float32 (`eval:163`); scalar norm evaluation
in float32 does not undo recurrence quantization. Computing
`out.float()-x.float()` would avoid any extra subtraction rounding, but would not
remove the underlying bf16 floor. Zero or floor-limited residuals and ties need
an explicit policy; `log(max(residual,1e-30))` is an arbitrary zero clamp, not a
floor measurement. Negative/nonfinite residuals should fail admission rather
than be clipped or silently disappear from correlations.

Step 32 can be a sensible practical, predeclared finite-budget outcome. It is not
the asymptotic decay rate: transient amplitude, nonnormality, finite distance to
the fixed point, selected basin and bf16 quantization also affect it. Log is
monotone, so Spearman ranks are unchanged for positive residuals; it cannot cure
floor ties. Do not move this endpoint based on test results. Prefer frozen
reporting of zeros/ties and development-calibrated floor diagnostics, with any
float32 matched-state residual as a clearly separate diagnostic. A correlation
with the bf16 endpoint remains an empirical cross-precision association.

## 5. State the exact relaxation theorem and actual numerical authority

For a smooth fixed map and lambda>0, fixed points are shared and
`D G_lambda=(1-lambda)I+lambda D F`. Strict stability at small positive lambda
requires **every Re(mu)<1**. For each such eigenvalue the upper boundary is
`2(1-Re(mu))/|1-mu|²`; a mode with Re(mu)>=1 prevents any strictly stable positive
interval. The `stability_interval` docstring's “always contracts” statement is
false in general (`analysis:178–180`). Its grid implementation can return no
interval; that part does not assume contraction. Its reported boundaries are
grid approximations on [0.02,2] and may be clipped, so label them accordingly.
The older `lam_critical` helper filters crossings to lambda>1 and cannot report a
native-unstable upper boundary; importantly, **items uses stability_interval,
not that older helper**, so this is a legacy-documentation distinction rather
than an extra primary-path failure.

The theorem is about an exact smooth map; its use for finite-precision inference
is a numerical prediction. Fixed-point residual<=1e-4 at a computed centre does
not itself bound the Jacobian error or establish convergence to that centre for
every test trajectory. These limits do not invalidate a frozen held-out
association, provided its claims use the measured numerical scope.

Finally both loaders call `from_pretrained` by repository name without an explicit
`revision` (`latent:163–165`, `eval:212–214`), while the draft names a fixed snapshot.
Make the resolved weights/remote-code/tokenizer identity match the promised
snapshot in the actual frozen execution (explicit revision/path or verified
resolved manifest). This review did not instantiate a tokenizer/model or inspect
its cache, and therefore cannot certify actual resolved identity.

## Verdict and minimum regression checks

**Revise Part A before freeze.** Resolve gold-sequence versus all-choice residual,
exact token IDs, residual clock, and partial-spectrum language/admission. Keep
Part A's four-arm cohort and missingness rules consistent with the separate
statistical review. The algebraic intervention and the idea of testing predictive
association remain sound; no general certificate machinery is needed.

Small synthetic checks should cover: per-choice norm versus batched norm; wrapper
input/output step labels; a tokenizer stub whose joined/separate encodings differ;
empty/nonfinite/partial spectra with retained flags; the shifted-spectrum
counterexample; floor/zero/nonfinite residuals and an indeterminate near-unit
classification if binary stability is retained. The source's existing tests do
not cover these contracts. No new target evaluation is requested by this note.

Key source SHA256s: design `ed5d38475c654b76dba0080c3c3a2d6a1adad18605d49f0aeb817307cb33e299`;
analysis `79a078ce77c9235116a72c76858a93dd970191c15f306062ffa2be9df06aaa04`;
eval `62e3cc12782a29bd6ac1bb52adf417087d753decff8ae45b1d85fde94d313d52`;
latent implementation `4f4fbd0962221ee784ca0175f5efa7fe1720bfeb2c181b99ab20e43e04b8201c`.
All line references above refer to those pinned files, not later edits.
