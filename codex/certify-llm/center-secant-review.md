# Independent review: one-candidate inverse-secant driver

Reviewed 13 October 2026, before any target run by this driver. No actual parent
attempt, model weights, or target map was loaded for this review.

Reviewed source SHA256:
`c828ebc744eb95795e9051d5dbe0b8361828ebdfb13688afae5c1f60f7694132`.
Test SHA256: `48f6919a28450d0954127915f72a9209e8ac7aa8648fb9dcb2ff24a02fb0955b`.
Design SHA256: `f0dd2c819df829a5748f24ed9614c8887c23ebe53727462b4804d4bbebc9463d`.

**No blocking issue found for the stated one-candidate setup experiment.** The
reviewed program remains a proposal generator followed by point residual
validation. It cannot issue a root, primitive-period or attraction certificate.
Launch still requires the separately reviewed concrete plan, completed-parent
result audit, committed run card, private push and bounded launcher.

## Admission and complete history

The source reuses the unchanged complete-parent importer. It checks externally
pinned receipts and every inventoried source/artifact, current importer/kernel
hashes, original fixed-R seed and prefix plans, exact assets and parameters,
2048-bit arithmetic backend, restored prefix receiver binding, source-bound
phase input/output hashes and successful completion. Recursive ancestry must
start with a complete `center_refine` attempt; only complete fixed-R
`center_resume` descendants are accepted. Partial, failed and secant parents are
refused. The source files that define the map and input reader remain unchanged.

The extractor includes both original before/after tuples and every resumed
after tuple, chronologically and without selection by outcome. A reused before
tuple is an exact-checked alias, not a duplicate evaluation. The frozen history
count and cumulative phase count must agree, and the final tuple's image
provenance must exactly match the admitted final parent. Thus the proposed first
four-tuple history means all 32 old phase evaluations, including any unimproved
outcome; this review does not assert that the still-pending parent has completed.

Full packed centres remain exact dyadics. Original image records and conservatively
decoded image balls are retained. Each residual is recomputed as the entire
image ball minus the exact successor in that same tuple. Neither image radii nor
small high-precision centre differences are replaced by binary64 values.

## One proposal and one fresh pass

Every consecutive secant is replayed with fixed caps: history 4, signal/radius
ratio 128, update norm 1024, and accelerated/baseline step ratio 4. Each running,
accepted, rejected or reset decision is persisted, and failure stops the attempt.
No alternative history subset or second proposal is selected after observing
results. Rejected updates can produce the deterministic baseline proposal.

Exact A/B matrices are copied for the mutable kernel. Native equality and
non-alias checks after each secant and the proposal detect changed entries before
model evaluation, and the original NumPy proposals are read-only. The reviewed
kernel keeps its A-list length fixed; a future generalized kernel should also
add an explicit length check before the current zip-based comparison.

All history and the one chosen candidate are saved before model construction.
The existing strict prefix restore is reused. Exactly one fresh complete phase
pass is attempted: eight phases in the actual scope. Each input and complete
image is saved as it is produced. The final comparison retains improvement or
regression and distinguishes a smaller upper bound from a strict enclosed norm
decrease. No retry or extra candidate follows a regression.

A Python failure retains history, candidate, completed images and the failed
phase input/status. A hard launcher timeout can bypass the exception handler;
its incomplete artifacts and timeout receipt, rather than a fabricated final
status, must be retained and rejected by future complete-parent readers. The
CPU1/nice10/18 GiB/1800-second cap is the launcher's obligation. The nested old
one-hour admission plan does not extend this driver envelope.

## Independent checks

Ran the new driver, complete-parent importer, inverse-secant kernel and shooting
correction suites with CPU1 environment settings, no bytecode/cache writes:
**102 tests passed in 3.45 seconds**, exit 0. These include four-tuple extraction,
1700-bit small differences, all replay decisions, exact A/B mutation detection,
incomplete/tampered parent refusal, interrupted pass retention, regression
without retry, one fresh pass in mocked end-to-end runs, immutable parent paths,
strict resource/cap types and isolated `python -I --help`.

The synthetic tests establish implementation behavior, not target convergence
or an acceleration gain. Every driver status keeps `certificate_passed=false`.
