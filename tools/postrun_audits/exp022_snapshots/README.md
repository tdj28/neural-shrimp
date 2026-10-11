# EXP-022 audit source snapshots

These are exact source snapshots of the already executed external audits bound in `docs/EXP-022-summary.json`. They are archived in Git to protect the audit code as well as the selector and adapters. The original execution paths and outputs remain immutable outside Git.

These one-off scripts use their own directory for output and refuse to overwrite named results. For a new diagnostic replay, copy the selected source to a fresh external audit directory under `/data2/neural-shrimp-data/audits/`, inspect its pinned inputs and run there. Do not run them in this repository or into their original consumed attempt directories. This snapshot is not a source freeze for new neural integrations.
