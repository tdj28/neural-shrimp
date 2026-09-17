# EXP-013 operational recovery, 17 September 2026

Attempt 001 is retained unchanged and is **incomplete and unqualified**.
Its source is `9955567894cc89cf7fee60e44663931eedc142d9`, and its plan hash is
`370a98075eadd3c948b9cbcdc9ef5a6bd19e7864ac3321a630cbc62c7e34236c`.
The [compact receipt](EXP-013-attempt-001-receipt.json) binds the external
receipt and all 21 artifacts (64,269,805 bytes). All four old-data recipes
finished; no fresh reference, sensitivity or preview trajectory started.
All 10,681 planned task statuses and 6,144 unavailable readouts remain present.
No cohort means or intervals were produced.

Two operational defects were observed before any fresh dynamics:

1. The model-seal logging call supplies an artifact field named `path`, which
   collided with the log helper's positional argument also named `path`.
   The recovery helper names its positional argument `log_path`; the logged
   artifact field and chronology contract remain unchanged.
2. The original saved-evidence verifier refits the model and directly compares
   in-memory NumPy scalar strings with Python strings decoded from JSON.
   Its strict category check rejects identical string values of these distinct
   scalar types. Recovery applies the **same existing `safe()` serialization
   conversion used when writing the model packet** before comparison. Numerical
   values, packet hashes, floating tolerances and category contents are unchanged.
   The original failed verification is preserved separately under
   `../neural-shrimp-data/audits/EXP-013/failed-attempt-replay-001/`.

The frozen original runner, protocol, plan, tests and raw evidence are not
edited. [run_fresh_preview_recovery.py](../tools/run_fresh_preview_recovery.py)
is an explicit source copy with the two fixes and separate plan/review paths.
This deliberate copy preserves the original source bindings and replay.
The new plan is `experiments/hopfield/EXP-013-recovery-plan.json`; its exact
committed review is `docs/EXP-013-recovery-review.json`. The original protocol
continues to define every scientific rule; this document supersedes only the
entry-point and review-path instructions for a new attempt.

The source inventory adds the recovery runner, its synthetic regression tests
and this amendment. Selection, all 24 origins, candidate/sentinel identities,
old-data training recipes, all controls, costs, statistics, numerical gates,
resource caps, chronological access rules and failure denominators remain
identical. No threshold, reference, phase, radius, direction, label or policy
is selected using a fresh result. The old-data fits from attempt 001 are
acknowledged observed development outputs; the fixed recipes are refitted
unchanged in a new exclusive attempt, with no resume or artifact overwrite.

Targeted regression tests cover the actual model-seal event with a payload
`path`, mocked successful model sealing followed by a reference-stage failure,
serialization-compatible replay and the unchanged scientific definitions.
The independent reviewer audits attempt 001, the narrow source diff and the
new plan's identities before a new private freeze/push. An original failed
replay is not retroactively called a successful original verifier run.

After exact review, commit and verified private push:

```bash
.venv/bin/python tools/run_fresh_preview_recovery.py prepare
.venv/bin/python tools/run_fresh_preview_recovery.py validate
# Preparation precedes the commit/review/push required by the following run.
.venv/bin/python tools/run_fresh_preview_recovery.py run --data-root /data2/neural-shrimp-data/raw --attempt attempt-002
.venv/bin/python tools/run_fresh_preview_recovery.py verify /data2/neural-shrimp-data/raw/EXP-013/attempt-002 --audit-output /data2/neural-shrimp-data/audits/EXP-013/replay-002
```

No paid request, public release, PR creation or main-branch merge is added.
The Pro consultation remains not run, with USD 0 incremental spend.
