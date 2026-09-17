# Independent saved-array refinement audit

The four Python modules here are byte-identical backups of the independently
written auditor used for EXP-014-refinement attempt 001. They do not import the
producer, integrate trajectories, fit a decoder or adjust tolerances.

Run from the repository with a new external report path:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python tools/phase_memory_refinement_audit/audit_refinement.py --repo /data2/neural-shrimp --raw /data2/neural-shrimp-data/raw/EXP-014-refinement/attempt-001 --output /data2/neural-shrimp-data/audits/EXP-014-refinement/new-replay/report.json
```

The process exit code reflects evidence and scientific-decision checks.
**Read `strict_arithmetic_replay_passed` separately.** Attempt 001 passes the
former but fails the latter; neither status is a numerical qualification of
the original EXP-014 setting. Both check lists remain in the report.

`selection.py` preserves the superseded 61-case proposal used by the independent
selection derivation; `selection_v2.py` completes both signs to the frozen
85-case panel. `snapshots/` archives the exact separate preflight and
saved-summary interpretation sources, which require their original companion
evidence files and are not this general CLI.
