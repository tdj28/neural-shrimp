# Pre-freeze review of the Huginn causal-test draft

13 October 2026 UTC. Read-only review by Codex while Task 1 runs. Claude owns
this experiment. No model run, development outcome or protected test outcome
was inspected for this review. This is a review of a **draft**, not approval
of a final frozen design.

Reviewed design SHA-256: `f9d860290af77ce4ba12994922afa8086b1aaf5348c794aee7e2d2673e3a551b`.
Reviewed evaluation source SHA-256: `600489b4bad007c1df6fe8e934d7935fe3f2aa0be4c890db628ebc986379c2e2`.

The question is worth pursuing: can a dynamical prediction choose an intervention
that improves native recurrent-model task performance at a stated budget? The
paired comparison, untouched test problems and explicit negative-result policy
are appropriate. I would resolve the following before freezing.

## Separate convergence from the intervention's step-size factor

For the relaxed map T_lam(s)=(1-lam)s+lam F(s), the identity
T_lam(s)-s=lam(F(s)-s) holds. The current trace records the norm of the left-hand
side divided by the new-state norm. H3 therefore gives smaller lam a mechanical
advantage at a fixed step count, even if the state is no closer to a fixed point.
Record the native residual norm ||F(s)-s|| with a common normalization, available
from `out-x` before blending in the existing wrapper. Keep the relaxed step norm
as a separate diagnostic. Where feasible, also compare against an independently
converged reference with a recorded residual and the same input/cache semantics.
Do not claim smaller state error solely from the relaxed update length.

A fixed lam factor cancels from an ideal asymptotic ratio, so this observation
does not alone invalidate every tail-rate estimate. However, the current endpoint
fit, native bfloat16 arithmetic and filtering of zero values need a defined
precision-floor gate and retention of original step indices. A rounded plateau
or omitted zero cannot count as measured asymptotic contraction. Freeze the
usable fitting window, minimum signal and failure handling before test outcomes.

## Test the added value of the selection rule

The coordination agreement proposed native, random-relaxation and untargeted-
damping controls. The draft currently contains native, chosen-fast and chosen-
slow arms only. A gain over native could demonstrate a useful relaxation effect;
it would not by itself show that spectral selection improves on ordinary damping.
Add the feasible agreed controls, with their sampling/tuning budget fixed, or
explicitly narrow the conclusion. Keep a spectral prediction distinct from a
parameter selected by scanning measured convergence rates. For one isolated
pair a +/- ib, the relaxed modulus squared is
(1-lam(1-a))^2+(lam b)^2, whose minimizer is
(1-a)/((1-a)^2+b^2). Other modes and nonnormal finite-time behavior can change
the optimum: check the full relevant spectrum and the measured discrepancy.
The quoted one-prompt pair is a hypothesis for broader prompts, not their
validated spectrum. Freeze how prompt-dependent spectra are aggregated.

## Make the practical comparison reproducible

Equal recurrence steps per token match one computational budget. The extra
blend and possibly different generated lengths mean total work and latency
are not automatically equal. Retain generated token counts, actual core-call
counts and elapsed time; state the matched quantity precisely. Compare accuracy
and efficiency without quietly changing the primary budget after outcomes.

Pin model, tokenizer and remote-code revisions as well as the dataset and source
commit. Greedy token selection does not alone establish deterministic latent
initialization; check the native model's initialization and pair/control its
randomness across arms if present. Freeze cache/initialization semantics and
record their effect on the claimed recurrence map. Retain complete generated
answers or a hash-bound raw artifact, rather than only the last 400 characters.
Use an exclusive output directory so an accidental repeated invocation cannot
append duplicate paired rows to a prior attempt.

Finally, freeze the minimum effect of interest, interval estimates, treatment
of incomplete/truncated generations and the primary/secondary multiplicity
policy. A null result from 300 paired problems should be interpreted using its
uncertainty, not as proof that every practically small effect is absent.

These recommendations do not authorize Codex to run or modify Claude's study.
Task 1 remains Codex's primary assignment. A final frozen-design review should
identify the final immutable source/design versions separately from this draft.
