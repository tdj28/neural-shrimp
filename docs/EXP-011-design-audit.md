# EXP-011 pre-fit design and implementation audit

14 September 2026 (Pacific; final preparation on 15 September UTC). This audit
precedes all real-data fitting for EXP-011. The experiment is retrospective
development on the existing 24 physical origins; it cannot confirm geometry
utility on fresh origins or establish end-to-end compute savings.

## Scope and frozen evidence

The complete plan contains 24 outer folds, 23 inner folds per outer recipe,
192 contexts, 3,456 candidate entries and 96 model recipes. Its 6,944 label
record bindings cover the 6,912 native EXP-010 records and all 32 independent
records. It binds 21 source/dependency files, 281 label-free feature input
files, the qualification certificate, and six external qualification,
artifact and independent audit files. The plan is 4,598,731 bytes with SHA-256
`a32c5283a2674ddaec01ba4cb508cf473f0be95f5b1724fa4ef7468a3c9b3f08`.
It records metadata, identity and hashes; bulky trajectories, labels and
model packets remain in the external data store.

The feature adapter was reviewed and pushed separately at
`a554cfa96b1c7177b76459e09eddb893a7d736ed`; see the
[feature audit](EXP-011-feature-design-audit.md). The complete protocol, model,
runner, tests, qualification certificate and plan must be committed and pushed
before the first fit. The actual source commit belongs in the run receipt,
not a self-referential plan field. The original EXP-008 strict Linux replay
failure remains unchanged. EXP-010 qualification concerns the measured
75-unit preview endpoint at the tested resolutions.

## Independent review and closed findings

Two independent reviews covered the model/statistical recipe and the runner's
data access, seals, resource ownership and verification. They found no
remaining execution blocker after the following corrections:

- Preflight now decodes only the metadata qualification certificate and
  inventory. It hashes label-bearing EXP-010 artifacts without decoding their
  outcomes, checks their receipt membership and binds the certificate itself.
- The feature input access test uses an independent exact allowlist. It does
  not accept the loader's own claimed bindings as the sole source of truth.
- Fixed controls and learned decisions require the complete context census,
  with no duplicate contexts replacing missing contexts.
- Each selected winner is sealed using only the four revealed shortlist
  labels before any full-18 oracle evaluation. Prediction seals, winner seals,
  access logs and chronology are verified and replayed.
- The full-18 reference is explicitly charged the sensitivity prerequisite
  inherited through candidate IDs 16 and 17. Reused data do not establish
  a measured reduction in total runtime.

The model review checked physical-origin isolation, train-only weighted
scaling, equal-origin inner validation and aggregation, exact penalty tie
rules, independent SVD/QR predictions and ordered-shortlist numerical gates.
An independent synthetic replication check preserved the selected penalty
and predictions to 2.6645e-15 when one training origin was duplicated with
the corresponding equal-origin weights. Train/held-origin overlap is rejected.
No real-data fits were used for these reviews.

## Validation and resource budget

The integrated repository suite passed **385 tests in 25.58 seconds**. The two
warnings concern optional PyYAML formatting of NumPy configuration output.
The implementation/feature subset passed 37 tests; independent review passed
65 tests including the unchanged EXP-009 process lifecycle checks. No-fit plan
reconstruction and all source/input binding checks passed. `git diff --check`
was clean before staging.

A generated-noise benchmark, with 43 columns, 3,312 training rows and 23 inner
folds, took 0.457817647 seconds of wall time for one complete SVD/QR recipe.
Four recipes per outer fold and 96 recipes total give about 43.95 seconds of
summed recipe wall time for one pass; a serial model replay adds a similar
amount (about 87.90 seconds combined) before I/O and other
overhead. This synthetic proxy supports a four-worker, one-thread-per-library
CPU run with 600 seconds per fold, 3,600 seconds for execution and built-in
replay after preflight, and a 1 GiB external output reserve. The standalone replay is also
bounded. Actual stage times and all terminal statuses must be retained.

Use an exclusive attempt under `/data2/neural-shrimp-data/raw/EXP-011/`.
Hold the repository clean throughout execution, including the postflight
source guard. Preserve failures and partial packets without resuming or
changing the frozen recipe. After execution, verify the complete artifact
inventory and replay, obtain independent numerical/aggregate audits, and push
compact results and receipts. Paid compute and paid review are unnecessary.

GitHub SSH pushes are working. GitHub API authentication remains unavailable,
so the locally reviewed branch is not merged and PR creation is still pending.
This does not prevent the authorized private pre-fit freeze and local study.
