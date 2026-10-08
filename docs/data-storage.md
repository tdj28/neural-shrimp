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

## Verified separate-disk copy, 8 October 2026

The user reiterated regular private Git pushes as protection against disk
failure. The existing cadence in `Skills.md` remains: each coherent work unit,
approximately every 30 minutes of sustained work, and before long runs or
handoff, with the remote commit verified after pushing.

A private local snapshot is now verified at
`/archive/neural-shrimp-backups/20261008T065521Z`. Its `raw-data/` directory contains
75,831 regular files / 18,011,529,623 bytes from
`/data2/neural-shrimp-data`. Its `private-handoff/` directory also preserves the
three Git-ignored handoff/cost/checkpoint files as they existed during the copy.
Every regular file passed a source/copy SHA-256 comparison, and the source
file size/mtime inventory remained unchanged during copying and verification.
No symlink targets needed separate handling. The complete manifest lives with
the copy; its hash and a compact receipt are in
`docs/backups/2026-10-08-local-archive.json`.

The destination has a private parent directory (mode 0700). Host block-device
inspection identifies source `/data2` on `nvme1n1p1` and destination `/archive`
on `sda1`: distinct physical disks. This is a verified local snapshot, not an
offsite copy or a continuously updated mirror. Later raw outputs and later edits
to ignored files need a new verified snapshot; a Git push alone does not update
this archive. No cloud service, paid storage or data publication was used.

The first copy was interrupted to remove a slow per-file filesystem flush.
The second resumed retained files. Its first inventory check found one extra
temporary file left by the interrupted copy: no source files were missing and
no common file had a size/mtime mismatch. The fragment was retained separately
under `copy-interrupted-fragments/`; the complete inventory/hash verification
was then repeated and the destination filesystem flushed at completion. Both
attempt statuses are retained with the snapshot. Original evidence and
unrelated processes were untouched.

A subsequent four-file private snapshot at `/archive/neural-shrimp-backups/20261008T070802Z-handoff` preserves the
updated handoff and the original checkout's pre-existing uncommitted
`docs/research-log.md` as well as its cost ledger and older checkpoint. Its
checksums also passed; see `docs/backups/2026-10-08-private-handoff.json`.
The original working files were not changed by this backup.

The next completed research unit has a separately verified incremental snapshot
at `/archive/neural-shrimp-backups/20261008T082704Z-research-increment`.
It covers all newly selected S1R/CIFAR/LM producer and audit evidence, their logs,
and literature attempts002--004:1,037 files/361,396,978 bytes, complete inventory
and SHA-256 checks. It complements the full snapshot and is neither offsite nor
continuous. Exact roots and manifest are in
`docs/backups/2026-10-08-research-increment.json`.

A subsequent four-file private snapshot at `/archive/neural-shrimp-backups/20261008T083741Z-handoff-current` preserves the
updated handoff and the original checkout's pre-existing uncommitted
`docs/research-log.md` as well as its cost ledger and older checkpoint. Its
checksums also passed; see `docs/backups/2026-10-08-private-handoff-current.json`.
The original working files were not changed by this backup.
