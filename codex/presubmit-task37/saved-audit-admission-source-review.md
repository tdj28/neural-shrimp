# Task37 saved-audit admission preparation

**HELD. No new network behavior or numerical result has been measured.** This packet proposes one saved-value replay after the original R pair closes. It neither admits the replay nor the eventual science. The existing checker and supervisor are unchanged; the original active output, arrays, admission, startup and receipt were deliberately not opened. No model, checker, control suite, process query/control, capacity measurement or Git mutation was performed.

## Source and identity

- Original target source: `8b56522882743d6c84afd2eccb6ec09616b7e9be`.
- Original plan: 5873 bytes, SHA256 `6dd23aa38be6c464f87649399eac59cf3b7ecc0e08e9475c9df2ee9c7df77a41`.
- Original card: 7848 bytes, SHA256 `8222df4af6ed721cc79e54616c82b3a9376fda00a62921f3dee4d3a4cb0f3db7`.
- Unchanged checker: `codex/presubmit-task37/audit_saved_numerics.py`, 20218 bytes, SHA256 `e840f06a91501a81dd396d3e20126c020e64c8698043e78a8dd0149274b40215`, introduced at `36bb7b5416a86131053357c8b1f4dd65ecb90343`.
- Unchanged supervisor: `codex/certify-chaos/supervise.py`, 8727 bytes, SHA256 `306db32a2c8ada9beb67547bbaafc6d0d7a597b5861b0e629ca9a07801223c44`.

Plan/card hashes match the original Git snapshot; the checker hash matches its introducing Git snapshot. These source checks do not authenticate the still-active target data. All source and closed compact reference files read are pinned in `source-input-pins.json`.

## Interface blocker retained

At supervisor line 63 only `out.parent` is created. Checker lines 307–316 perform all saved replay before exclusively opening the requested report, without making its parent directory. Direct dispatch to `r-pair-001-audit-001/result.json` is not ready and would fail at serialization. The held plan therefore names a future minimal wrapper with `--root` and `--out` directory arguments and fixed `result.json` basename, whose final source pin is deliberately deferred; `wrapper-design.md` specifies the remaining work. No wrapper or checker modification is supplied or implicitly approved.

The plan uses the supervisor's actual keys: `status`, `repository`, `raw_root`, `python`, `cpu_threads`, `cpu_ids`, `startup_ram_gib`, `source_files`, and one `audit` lane with entry/arguments/output/supervision/preflight pins/input inventory/prior receipts/limits. No fictional status field was added to an admission record. The supervisor currently refuses `status: held`; incomplete null pins also cannot verify. Root must not promote status alone. Card metadata is not automatically enforced by this supervisor; Root's final source/interface review must establish its requirements.

## Original closure and authentication to fill

Both draft files explicitly retain null pins for the original final receipt, original admission, original worker-startup record, a new compact host-closure record, and a complete original raw-file inventory. Admission/startup may already exist but were intentionally not read; the completed receipt and closure are not available for this preparation.

After the original worker exits, Root must establish all of the following:

1. Original receipt `returncode == 0`, `failure is null`, worker reaped, identity absent, group/session absent, plus independent original host closure tied to the recorded PID/starttime/group/session. The supervisor's `prior_receipts` loop checks only `worker_reaped` and `worker_group_session_absent`; it does not check successful return or scientific qualification. Restricted `/proc` absence is insufficient independent host evidence.
2. Original source `8b565228…`, exact plan/card/admission hashes, lane `r_pair`, argv, source blobs, input/seed/model/tokenizer inventory, actual resource limits and effective environment. Preserve original failures and every original row's gate. A closed process is not a qualified orbit.
3. Derive the full original raw inventory from its receipt, not from the new checker's read list. Require exact canonical regular-file membership and bytes/hashes, no symlinks (including path components), aliases, escapes, missing or extra files, before and after saved replay. `Reader` resolves symlinks (lines 43/47); its containment check alone does not authenticate original membership. Supervisor `input_inventory.files` verifies listed content, not set completeness or path topology. Keep these distinctions explicit.
4. Store the inventory as a pinned JSON object with a `files` list of `{path, bytes, sha256}` entries, bind it to the original receipt and closure, and put its own pin in `preflight_pins` as well as `lane.input_inventory`. Add the completed closure record as a preflight pin. Null draft placeholders are not evidence.

Root must independently check these again after the saved worker closes, retain the actual exit/failure, output inventory and host closure, and inspect the saved result. A successful saved audit still leaves `scientific_verdict_admitted` and `certificate_passed` false. No switch admits the science. Unchecked items remain left-eigenvector condition estimates, decoded tokenizer text, expected embedding readouts, complete Newton/GMRES decisions and counters, original operator authenticity and effective internal precision.

## Fresh allocation and conservative resource proposal

Proposed new output: `/data2/neural-shrimp-data/scratch/audit-codex-1019/r-pair-001-audit-001`, with `result.json` inside and sibling `r-pair-001-audit-001-supervision`. These paths were not created or probed. Root must confirm both absent at admission. Do not put results inside the original output or overwrite prior evidence.

Proposal: one serial CPU (CPU 0 only if Root confirms it is allowed), one thread for all four numerical thread variables, nice 10, 4 GiB address-space cap, 300 CPU seconds, 360 wall seconds, 16 MiB total sampled output limit, 24 GiB startup available RAM, unchanged 16 GiB runtime available-RAM and disk floors. The supervisor itself retains its existing 1 GiB soft address-space limit. This is a finite stop proposal, not a duration prediction or a newly measured workload. `capacity_window` stays null; original target coordination is not a new saved-audit reservation.

The only timing reference used is the closed Task34 cascade-P6 saved audit: `1.0203935001045465` seconds externally, source `f13a1aad926c5018827f406a2afc530e1b482999`, recorded in `dense-cascade-p6-001-completion.json:saved_audit_execution`; receipt SHA256 `2bba1dee9ece04c6bdecb395bd1246eccf139981e9d6d56130c0df670749d568`. That audit did not perform a fresh eigensolve. Task37 checks two 896-square fresh spectra/assignments and two complete journals; its cost is not established by that reference. No model-run time or manufactured small-fixture runtime is treated as a saved-replay forecast. If the proposed cap proves insufficient, retain the failed attempt and reassess separately; no retry or extension is authorized.

## Final integration owned by Root

Implement/review the minimal wrapper and fill its exact pin; fill original closure/authentication records; review and freeze the final card/plan/source with private checkpoint equality and a clean owned worktree; obtain a new capacity reservation covering the full wall cap and perform fresh startup checks. The supervisor CLI `--source` must use that new audit checkpoint, not the old target source. The card's draft location pins name their intended owned-worktree destination, so Root must regenerate hashes after any content edits and before committing. The wrapper filename is provisional pending the separately prepared final source; no final wrapper pin is claimed.

Only then may Root consider changing plan/card status to reviewed and launching the single lane. Root must supply `HF_HUB_OFFLINE=1` to the supervisor environment, because the current supervisor does not set it itself. Do not precreate the audit output to hide the wrapper defect. No automatic retry, floor weakening, checker replacement, body-point execution or original rerun is included.

## 2026-10-19T11:01:58.554663+00:00 — Root integration amendment

The preceding source-only design remains retained. The minimal launcher now exists as run_saved_numerics.py with SHA256 091c70ce93616586611ad6005512999f073900b9c8c7bd516b5b23f92a292cd3. Root read the entire launcher, eight new mocked controls and frozen checker entry point. It uses --root/--out and fixed result.json; no existing checker or original source/card/plan was edited. The held card/plan now pins this wrapper. New mock controls remain pending; original closure and full authentication remain deferred. Repository companion filenames are saved-audit-admission-source-pins.json, saved-audit-wrapper-design.md and saved-audit-wrapper-preparation.md. The initial design does not authorize dispatch.
