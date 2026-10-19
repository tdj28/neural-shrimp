# Held first-tube001 packet review

Disposition: new read-only packet/interface review completed; one concrete input-list mismatch identified. No target/raw/model/backend binary or process was read, no numerical/control suite was run, and no repository/Git edits occurred. This is not runtime admission.

Reviewed held packet snapshots:

- driver002 `2c8e6ae4cb86b5d06b4e6a534d2e6f8a518d19f55fcab92b9b5d0b25210befc9`;
- card `af148062f060f0238ca845667a81a968c5791283b751a1bccf475dc54b74b6d9`;
- plan `393b26fcfbd258a0cad2ba8ecc812d88eb5ad1639e326ae9eb4d1233910598ab`;
- regularity review `7dedd72959ece740abac8c9e7e016cba3e02cc706a5eedf66416b6b5c5104282`;
- preparation `d4a3c13b95fca68bdc6230b5a5c91f86e51dc5ea37e8c4a9b2d01d9dbefc51cb`.

## Concrete mismatch

`codex/coexistence-task33/tube_target_driver_002.py:623` constructs its loader map solely from `card.input_files`. Lines639–642 require both original prefix receipt and admission paths in that map. The reviewed card contains7 input files (initial,step1,canonical model,canonical config,vector,prefix metadata,prefix cache) and omits:

- `/data2/neural-shrimp-data/codex/coexistence/prefix-001-supervision/receipt.json`,5559 bytes,SHA `0e787d0ebd6fb505a202c96fb3cc2429803aa8cc50c5271f728dc3da59af6f12`;
- `/data2/neural-shrimp-data/codex/coexistence/prefix-001-supervision/admission.json`,4466 bytes,SHA `84be8d159949dc60aa0f3909c6a16ee5a967dab0e40cd717dda2738331bcfa28`.

They are already present as exact pins in prefix_execution and the plan's preflight list, but those separate locations do not satisfy the main() input map. The held packet would therefore refuse before model loading. Add these two existing exact pins to card.input_files and refresh card hash in plan argv/sourceinventory; no scientific source or threshold change is needed. Root acknowledged this fix. This review has not re-read a repaired packet.

## Other packet joins checked

All16 source-file pins in the reviewed plan match current owned repository bytes and the card's12 source/hash declarations. No duplicate source, input or preflight paths occur. The21 preflight pins cover every declared audit result, regularity review, current input and thin/audit/prefix original card/plan/receipt/admission authority. The source card pin, driver002 path, complete argv, output and sibling supervision paths agree. The original thin/audit002/prefix repository card and plan byte pins and exact historical argv agree; audit002's original_execution matches the packet's thin_execution exactly. Raw receipt/admission/output hashes were treated as declared pins only and were not re-authenticated here.

The canonical checkpoint and config destination pins correctly match the narrow002 loader adapter. No snapshot symlinks are misrepresented as regular input files. All asset content hashes retain the accepted map. The full37-token prefix, binary64 alpha .06468 and log10T -.148 integer ratios match the source; the map JSON canonical hash recomputes exactly. This is the Task33 cell, not the Task34 cascade cell. The serialized backend hash recomputes exactly, including the full `2.14.1+cu130` torch version and38 extension records. The live backend was not reimported or inspected; actual runtime equality remains mandatory.

Limits agree between held card and plan:18GiB AS,3600 CPU seconds,3900 wall seconds,64MiB output, CPU1/affinity0,32GiB startup floor. The unchanged supervisor enforces16GiB runtime RAM/disk floors and sampled cumulative output cap in addition to per-file limit. The worker checks its actual external limits, nice10/one CPU and thread settings before use. Two auxiliary slots include the supervisor. Resource references are honestly labeled incomparable scale evidence, not a divided per-direction estimate or a20-tube forecast. A first tube may fail its conservative domain/resource bound and must remain preserved.

The mathematical scope is appropriate: one2048-bit uncertain-primal call on the exact selected c1 cube with an independently varying[-1,1]^896 tangent cube, producing a conditional full infinity-operator bound after all24 layers and both149-chunk head passes. All native full-image/signed derivative radii and scope flags are retained. The exact first image supplies only e1<=delta0<=radius. Neither point residuals nor centre-only Jacobians are represented as a later trajectory proof. An independent saved operator/endpoints/coverage/source audit and successful actual execution closure are required before using L. No basin/reachability/certificate result follows from this packet.

## Revised closure/capacity prerequisite

During review Root reported the original P6 phase closed with a RAM-floor failure and exit-15, retaining only phase0 residual output. This supersedes any wording that assumes a successful P6 phase/matrix audit. No such target data were read here. The held status must remain until Root authenticates the actual failed-phase receipt, retained original outputs and exact host closure, discloses the failed outcome, and supplies coordinated protected/stable fresh capacity for a new serial unit. No failed P6 result is an input to this Task33 tube mathematics.

The unchanged supervisor's prior_receipts gate requires reaped worker/group closure, not scientific success, so an authenticated failed-phase closure can be used for serial resource release without relabeling the phase. Its exact pin must be in preflight and its path in prior_receipts. Do not fabricate a successful P6 audit or run one on nonexistent matrices. Final admission must explicitly change both held status fields, update all changed card/review/source hashes, privately checkpoint the exact plan and source, verify fresh capacity and original task ownership, and retain all existing hard limits. These remain Root-owned steps.

No additional concrete packet/interface blocker was found beyond the two missing input_files entries and the now-superseded successful-phase prerequisite wording. The final revised packet has not been runtime admitted by this review.

## Root correction and failed-phase closure, 2026-10-19T03:16:50.481497+00:00

Both missing original prefix receipt/admission pins were added verbatim from prefix_execution into card.input_files. No driver or mathematical change; card/argv/source inventory hashes refreshed. Root authenticated the actual P6 failure in phase-cascade-p6-001-completion.json (SHAa83c999c1e692b323f259acdff9d7758cf2c86a69487feb81f6c214d078df1cb): source36366f7f, worker-15, original12source blobs and sole residual retained, both original host identities/groups absent. The real failed receipt is now included in preflight/prior_receipts. No nonexistent successful phase audit is required or inferred.

Card and plan remain held, with new coordinated stable/protected local capacity and final exact reviewed private source admission required. The failed attempt does not release authority for a retry, and no whole-tube execution has occurred. All existing resource floors and hard limits remain unchanged. Only new packet consistency checks are performed; earlier controls and completed scientific audits are not repeated.
