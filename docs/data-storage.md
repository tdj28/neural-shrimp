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

## Huginn transfer protection after the 23 October 2026 failure

HCM2 completed its numerical scan but lost most of its full-state evidence when
the watchdog treated final archive creation as numerical stagnation. It launched
a second archive, timed out, and terminated during transfer. HCM3 suffered the
same legacy archive failure, but its complete scientific dataset had already
been saved by the incremental collector. Both failed archives remain preserved;
neither is described as hash-verified recovery.

Future Huginn map launches use the protected worker path:

- `runpod_scan_group.py` binds the worker to a live controller and watchdog.
  Standalone Huginn launches and pinned runners with the old archive protocol
  are refused before resource creation.
- `huginn_incremental_retrieval.py` starts before scientific execution. It copies
  completed compact chunks first, then immutable full-state tails, with remote
  and local hashes. Files are published atomically only after verification;
  failed transfers retain their uniquely named partials. Completed files and
  directory entries are flushed to disk.
- The remote Huginn runner no longer builds a monolithic tail archive. Numerical
  completion and verified retrieval have separate states and inactivity checks.
  Progress means new bytes, verified files or completed final hash checks;
  repeated timestamps and duplicate transfer attempts do not reset timers.
- Normal success requires numerical exit zero and the entire final manifest
  verified locally by the collector. The worker records the manifest, source,
  collection mode and exact resource identity. Termination is checked by exact
  lookup and inventory, with a receipt even if the watchdog deleted first.
- Computation stops before the original retrieval reserve begins. Retrieval
  cannot extend the original approved hard deadline. If a failure or deadline
  wins, retain partial evidence, record incomplete recovery, and terminate the
  exact owned resource. No automatic new rental or scientific retry follows.

The revised integration is tested locally with synthetic data and simulated
provider responses, including an actual local collector process killed during
transfer. This does not claim a new live RunPod qualification. Before the next
approved scan, its costed plan must also check local/remote disk capacity and
measure representative transfer throughput against its output volume and
retrieval reserve. Keep source and deployment hashes with the run. Completed
scientific files also need a verified copy on the separate archive disk; a
successful transfer onto `/data2` alone is not an independent backup.

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


The 8 October S1R proof/association increment is preserved at
`/archive/neural-shrimp-backups/20261008T095251Z-s1r-certificates-increment`:
2,122 files / 52,354,182 bytes. The complete selected source/copy inventory and
SHA-256 values agree, including a second source hash pass. This includes all
new S1R certificate, bootstrap and independent-audit artifacts/logs and primary
reference archives005-007. Receipt:
`docs/backups/2026-10-08-s1r-certificates-increment.json`. This is a local
separate-physical-disk snapshot, not offsite or continuous protection.

A subsequent four-file private snapshot at `/archive/neural-shrimp-backups/20261008T100723Z-handoff-s1r` preserves the
updated handoff and the original checkout's pre-existing uncommitted
`docs/research-log.md` as well as its cost ledger and older checkpoint. Its
checksums also passed; see `docs/backups/2026-10-08-private-handoff-s1r.json`.
The original working files were not changed by this backup.

Additional lower-branch, utility and CIFAR diagnostic evidence, audits and the Curry preview archive are verified at `/archive/neural-shrimp-backups/20261008T113630Z-lower-span-increment` (10,746 files/163,313,975 bytes). Full inventory/hash checks plus a second source pass agree; receipt `docs/backups/2026-10-08-lower-span-increment.json`. The destination is the previously verified separate physical disk, not offsite storage.

A subsequent four-file private snapshot at `/archive/neural-shrimp-backups/20261008T113919Z-handoff-lower-span` preserves the
updated handoff and the original checkout's pre-existing uncommitted
`docs/research-log.md` as well as its cost ledger and older checkpoint. Its
checksums also passed; see `docs/backups/2026-10-08-private-handoff-lower-span.json`.
The original working files were not changed by this backup.

### 8 October 2026,12:26UTC — restart and reference increment

Verified60 files/21,993,233bytes at `/archive/neural-shrimp-backups/20261008T122628Z-restart-increment`, manifest214c7c4a8b7a9bd709cab00d72d62e37e76fdb160b19749cf784e7d49783044b. Includes complete CIFAR restart producer/audit and launchers, Arnold preview009 and Jensen full-text010. Exact source/copy inventory and all hashes, second source hash pass, and stable metadata agree. Receipt `docs/backups/2026-10-08-restart-increment.json`. Separate physical disk on the same machine; complements prior snapshots, not offsite or continuous.

A subsequent four-file private snapshot at `/archive/neural-shrimp-backups/20261008T123940Z-handoff-restart` preserves the
updated handoff and the original checkout's pre-existing uncommitted
`docs/research-log.md` as well as its cost ledger and older checkpoint. Its
checksums also passed; see `docs/backups/2026-10-08-private-handoff-restart.json`.
The original working files were not changed by this backup.

A subsequent four-file private snapshot at `/archive/neural-shrimp-backups/20261008T132925Z-handoff-gap` preserves the
updated handoff and the original checkout's pre-existing uncommitted
`docs/research-log.md` as well as its cost ledger and older checkpoint. Its
checksums also passed; see `docs/backups/2026-10-08-private-handoff-gap.json`.
The original working files were not changed by this backup.

The final morning checkpoint at `/archive/neural-shrimp-backups/20261008T142721Z-morning-checkpoint` preserves the current tracked
paper tree/PDF, human report and selected evidence/review companions, together
with the original checkout's four private handoff/ledger/checkpoint/log files.
Complete inventories and two source/copy hash passes agree. See
`docs/backups/2026-10-08-morning-checkpoint.json`. Private contents were not
staged in Git and the foreign log was not edited.


## 2026-10-09 UTC — resumed curvature evidence backup

Verified separate-disk copy at `/archive/neural-shrimp-backups/20261009T022241Z-curvature-resumption`: 343 files / 14543690 bytes, including the resource profile, fixed negative panel, adapted saved-plane explanation and all launcher evidence. Full source/copy inventories, every SHA256 and a second source hash pass agree. Manifest `a24bc32efa37cc73e1eadda087af112d4aeab627a79bc27ed17d63b8e5f9546c`. Receipt: `docs/backups/2026-10-09-curvature-resumption.json`. This supplements private Git backups; it is a point-in-time copy on the same machine, not offsite storage.


## 2026-10-09T02:27:07.413532+00:00 — resumption paper and private handoff snapshot

Verified 64 files / 3784568 bytes at `/archive/neural-shrimp-backups/20261009T022654Z-resumption-checkpoint`, including the32-page PDF, current report/reviews and four private original-checkout files. Exact inventories, complete SHA256, second source hash pass and stable source metadata agree. Private ledger and foreign log stay outside Git. See `docs/backups/2026-10-09-resumption-checkpoint.json`. Sources were not altered; same-machine separate-disk snapshot supplements private Git.

## Verified separate-disk snapshot, 13 October 2026

At the user's request (protection against a failure of the /data2 drive), a full snapshot of the data root, minus
the re-downloadable model caches (`models/`) and rebuildable environments (`envs/`), is at
`/archive/neural-shrimp-backups/20261013T155602Z-llm-dynamics-13oct` (tools/backup_snapshot.py; receipt
`docs/backups/20261013T155602Z-llm-dynamics-13oct.json`, manifest SHA-256 82fd0b33...). Counts:
- 924,084 regular files (134.6 GB) and 11 symlinks, copied by rsync;
- 924,073 files verified by a complete source/copy SHA-256 comparison;
- no mismatches and no errors.

The 11 unverified files were changing during the copy, all live logs and state of the running certificate campaign;
they are listed in the receipt. 1,548 files created during the copy belong to the next snapshot. The Git-ignored
private files (cost ledger, checkpoint, /data2/Handoff.md) are in `private-handoff/`. `.env` is excluded on
purpose. This is a same-machine copy on a separate physical disk (sda), not offsite and not continuous; run
tools/backup_snapshot.py again after substantial new raw data.

## Verified Huginn evidence snapshot, 23 October 2026

The completed HCM1 map, surviving HCM2 evidence, complete HCM3 scientific dataset, their audits/provenance and failed archives are preserved at `/archive/neural-shrimp-backups/20261023T165845Z-huginn-hcm-evidence`. All 389 files (40,963,108,908 bytes) passed independent destination and second-source SHA-256 checks; exact inventories and stable source metadata also passed. Source physical disk nvme1n1 and archive disk sda were checked again. The private full manifest stays with the backup; its SHA-256 and exact selected roots are in [the compact receipt](backups/huginn-hcm-evidence-20261023.json). This is a same-machine snapshot on another disk, not offsite or continuous backup. It preserves the authentic failed HCM2/HCM3 archives alongside the recovered evidence; it does not repair missing HCM2 data.
