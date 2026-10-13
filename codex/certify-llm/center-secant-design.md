# One-candidate inverse-secant setup

Source: `center_secant.py`, SHA-256
`c828ebc744eb95795e9051d5dbe0b8361828ebdfb13688afae5c1f60f7694132`.
Tests: `test_codex_llm_center_secant.py`, SHA-256
`48f6919a28450d0954127915f72a9209e8ac7aa8648fb9dcb2ff24a02fb0955b`.
**30 focused tests and 102 combined secant/resume/correction tests passed.**
Only synthetic fixtures were used; no target computation or current partial
parent output was inspected during implementation.

A new `codex-llm-center-secant-v1` plan pins `parent_admission_plan`, current
`center_resume.py` and `inverse_secant.py` hashes in `method_source_sha256`,
`precision=2048`, an exact `history_tuple_count`, `max_candidates=1`, the caps
below, and the resource envelope. The planned first history count is four,
conditional on the two-correction parent completing and passing its independent
audit. The nested admission plan is the unchanged existing resume reader
contract; its one-hour envelope does not authorize a one-hour secant run.

The effective new envelope is **CPU1, nice 10, 18 GiB, 1800 seconds, one candidate,
one complete fresh eight-phase pass**. Caps are `max_history=4`,
`signal_ratio=128`, `update_norm_limit=1024`, `step_ratio_limit=4`. A separately
reviewed committed launcher/plan/run card/private push must precede target use.
The measured eight-phase worst-case reference is approximately 1527 seconds;
remaining time covers admission, exact-matrix conversion/copy, history replay,
checkpoint load, prefix restore and persistence. The launcher owns the watchdog;
there is no retry, second candidate, new prefix computation or budget extension.

The unchanged `center_refine.load_bound_inputs` and
`center_resume.load_completed_parent` authenticate the seed, prefix, complete
parent lineage, source hashes, plan, backend and every phase artifact. Only
completed `center_refine` and fixed-R `center_resume` parents are supported.
The new extractor then rereads receipt-bound files and enumerates the initial
before/after tuples followed by every resumed after tuple in chronological
order. Reused before tuples are aliases, not new evaluations. Exact centre
matching, final-image provenance and cumulative phase counts are checked.
Every selected history tuple must be present; there is no improvement-based
selection. Images are decoded outward and full residuals are recomputed using
the corresponding complete successor tuple, with no binary64 centre conversion.

`history-tuples.json` retains all exact centres, original and decoded image
balls, phase inputs/hashes/outcomes and recomputed residuals. Every chronological
secant attempt and its accepted/rejected/reset decision is saved. Exact Arb
preconditioners are copied before the mutable kernel receives them; native
exact equality checks after every action detect modification before model use.
The original proposal arrays are read-only. The chosen correction and exact new
centres are persisted before the first fresh phase evaluation. Existing prefix
restoration uses the unchanged receiver construction checks and evaluator.

Each new phase input and full image is saved immediately. A failed pass keeps
its completed phases, failed-phase input, all history and candidate; it is not
admitted as a future completed parent by this first version. A completed pass
retains either improvement or regression, with both upper-bound comparison and
strict residual-norm-decrease diagnostic. Zero/noisy/capped secants can clear
history and produce a baseline candidate; that is reported, not retried.

This is a setup accelerator, not an existence, uniqueness, primitive-period or
attraction certificate. Midpoint inverse updates need not satisfy an exact
secant equation or be nonsingular. A smaller residual does not replace any
whole-box Jacobian/inclusion/stability gate. All statuses retain
`certificate_passed=false`; no public release, paper claim or target launch is
authorized by this design note.
