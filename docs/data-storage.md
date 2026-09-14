# External research data storage

Created 14 September 2026 at the user's request. The server data root is
`/data2/neural-shrimp-data`, adjacent to `/data2/neural-shrimp`.

- `raw/<experiment>/<attempt>/`: new raw experimental trajectories and arrays.
- `audits/<audit>/<attempt>/`: full diagnostic reports, reconstructed inputs,
  model packets and logs.
- `checkpoints/`: bulky model/checkpoint state when needed.

This folder is outside the Git worktree. Source, tests, plans, compact
summaries and artifact receipts remain in the private repository. Existing
committed evidence is retained at its original paths; this policy applies
to new bulky outputs and does not authorize rewriting history.

Every executed attempt must retain its frozen source SHA, plan SHA-256,
environment, command, start/end status, all failures and an artifact ledger
of relative path, byte count and SHA-256. Store a compact receipt and the
external location in Git and verify file hashes before downstream analysis.
Output directories are exclusive and immutable after completion. Never
reuse an attempt ID to hide or replace a failed run.

The data root has been created locally; no independent backup destination
has been configured by this change. Never describe a Git push as a backup
of these external raw files. Preserve them on the server and record any
later verified additional copy, including its location and checksum result.

The initiating numerical follow-up uses a branch derived from the existing
EXP-008 branch because its frozen source and evidence are required inputs.
Its review must retain that dependency; it must not overwrite or bypass the
unresolved original replay gate.
