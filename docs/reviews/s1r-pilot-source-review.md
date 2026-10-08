# S1R bounded local branch pilot: independent source review

Reviewed 8 October 2026 UTC by the root agent, before any target execution.
Producer source: `db10bd1e1b25b0e24b1f0f8f97cb84035742c776` on
`codex/s1r-branch-pilot`. Decision: pass for the exact first pilot described
in `docs/s1r-branch-association-plan.json`, with the limitations below.

Read all three numerical/IO/driver modules, saved-state checker, plan and dated
amendment, and all synthetic tests. Independently ran the sixteen tests on
one CPU thread at nice 10: 16 passed in 0.81 seconds. Pytest could not write
its optional cache in the sandbox; this did not affect test results. Tests
use substituted polynomial maps and do not evaluate the S1R target anchor.

The corrector uses the analytic derivative of F²-I, retains the undamped
correction, and applies full-state closure and primitive-period gates after
refinement. Damping cannot by itself satisfy convergence. Both phases are
refined; phase labels are carried explicitly in continuation predictors.
Same-parameter spectra use both full monodromies and the conjugate-adjoint
left-vector convention. Residual and phase-multiset checks are numerical
consistency diagnostics, not eigenvalue forward-error enclosures.

Accepted output links require parent proximity and an unambiguous cyclic
phase assignment, direct-versus-two-half-step agreement, and reverse replay.
Rejected links remain rejected if smaller steps later succeed. Four cumulative
step reductions per requested endpoint are allowed; an unresolved gap stops
that direction. The four hidden-backoff steps have the separately disclosed
closure, spectral and phase-link gates, without the output-link replay tests.
The first anchor timing screen estimates twenty remaining pair evaluations;
it does not replace the hard time/corrector/storage caps.

Execution remains CPU-only, one thread, nice 10, at most 120 internal seconds,
150 external seconds plus five seconds of kill grace, 160 phase corrections
and 64 MiB raw evidence. No trajectory fallback or extra parameter search is
allowed. The append-only journal persists state before spectral work and
reserves finalization space. Native-library calls are only interruptible at
Python boundaries; a hard kill may leave an explicitly incomplete checkpoint
without a final manifest. The checker is a saved-state numerical replay using
shared map primitives, not an independent interval proof or adversarial
provenance authentication. Its source/input inventory is hash-bound; the
producer commit and private branch should also be checked externally.

This pilot can establish a small numerically consistent local period-two
branch. It cannot establish uniqueness, historical start-family association,
global shrimp ancestry, an invariant circle, irrationality or utility. Those
claims remain outside the approved unit regardless of the outcome.
