# Task37 r-pair-001 source and execution-record closure review

2026-10-19. Scope: original supervisor admission/receipt/startup metadata, original plan/card and seed inventory at `8b56522882743d6c84afd2eccb6ec09616b7e9be`, current small source files and the corresponding Git blobs. **No source/input/argv/resource-limit mismatch was found in these joins. This is not raw-output authentication or scientific admission.**

No numerical checker, control, model or producer computation was executed; no output array was decoded. No model/tokenizer/seed asset was rehashed. No Git mutation or process-management action occurred. The metadata comparisons used a small stdlib-only script with nice 10 and one-thread environment settings. Root owns original asset/output authentication and the subsequent saved-numerical admission.

## Verified joins

The original admission, final receipt, worker's `startup.json`, and Root's retained post-exec startup record all name source `8b56522882743d6c84afd2eccb6ec09616b7e9be`. Admission's plan hash equals the original Git blob and current plan. The worker-startup card and admission hashes match the original card and actual admission bytes. Root's retained post-exec record repeats the same admission hash. Target metadata agrees exactly: R, alpha `0.072912`, log10-temperature `-0.154104`, period 6.

All five admission source-blob records match the original plan pins, original Git blobs and current local bytes. The four code records also match the card's complete source inventory. The plan/card and seed inventory remain byte-identical to their original blobs:

| Relative path | Bytes | SHA256 |
| --- | ---: | --- |
| `codex/presubmit-task37/p6_kite.py` | 22400 | `370c05f86334814848fdeb64ba323ab4d89bd50e7a549ac10e83ca3d322d9d45` |
| `codex/presubmit-task37/periodic_audit.py` | 20436 | `667ea2d366e6f49f48793d769a8c231079bce359315d4499a84a60b316b0d245` |
| `codex/crosscheck/qwen_map.py` | 14540 | `493a3d1d57fec227f5ba2e3f8f9cfcfdd02d22495d1d44e6449ede901310ae6e` |
| `codex/certify-chaos/supervise.py` | 8727 | `306db32a2c8ada9beb67547bbaafc6d0d7a597b5861b0e629ca9a07801223c44` |
| `codex/presubmit-task37/r-pair-001-card.json` | 7848 | `8222df4af6ed721cc79e54616c82b3a9376fda00a62921f3dee4d3a4cb0f3db7` |
| `codex/presubmit-task37/r-pair-001-plan.json` | 5873 | `6dd23aa38be6c464f87649399eac59cf3b7ecc0e08e9475c9df2ee9c7df77a41` |
| `codex/presubmit-task37/p6-seed-inventory.json` | 3214 | `a10ed66f8b9621c4373955a13daaace4244b1d54734c78f808be723e9c8da6d8` |

The ten original card input-pin records equal the plan's ten preflight records: seed inventory; two states arrays and their two curve records; model, config and tokenizer; steering vectors; prior closed receipt. Inventory names seed rows 6 and 2 and records their original file and row hashes. These are matching metadata references, not a fresh check of the seed rows, asset contents or prior raw evidence. Inventory's additional body targets do not authorize them: the card/preflight restrict execution to the R pair and require a separate conditional follow-up admission.

Exact worker argv agrees between plan, admission, final receipt, pre-exec `planned_argv`, final receipt's observed post-exec cmdline and Root's post-exec cmdline:

```text
/data2/neural-shrimp/.venv/bin/python -I -B /home/d7082791602/.codex/worktrees/llm-hostile-fold-crosscheck/neural-shrimp/codex/presubmit-task37/p6_kite.py --card /home/d7082791602/.codex/worktrees/llm-hostile-fold-crosscheck/neural-shrimp/codex/presubmit-task37/r-pair-001-card.json --out /data2/neural-shrimp-data/scratch/audit-codex-1019/r-pair-001 --admission /data2/neural-shrimp-data/scratch/audit-codex-1019/r-pair-001-supervision/admission.json
```

The plan, card, admission and worker limits agree: 18 GiB address space (`19327352832` bytes), 15000 CPU seconds, 15300 wall seconds and 512 MiB output/file cap (`536870912` bytes). Worker-startup soft/hard AS, CPU and file-size limits equal those values; affinity is CPU 0. Post-exec identity records show UID 1000, nice 10 and one thread. Admission records all four numerical thread variables as `1`, available memory `35638579200` bytes (above 32 GiB), and disk free `2661459795968` bytes (above 16 GiB). The dated card permits the first unused R pair within 09:02–15:00Z; the retained post-exec observation is `2026-10-19T09:05:45.266108+00:00`.

## Status and limits that must remain explicit

1. **Nonzero worker exit, no supervisor-declared failure.** Final receipt reports `returncode: 1`, `failure: null`, `worker_reaped: true`, elapsed `7763.898525267839` seconds. This is consistent with supervisor source: a nonzero worker return produces supervisor exit 1 even when no supervisor budget/resource/check exception was recorded. Null supervisor failure must not be described as a successful scientific run. The reason for the worker's nonzero exit was not assessed in this source-only lane.

2. **Pre-exec versus post-exec cmdline.** `worker-startup.json.identity.cmdline` still names `supervise.py` because this record is written in `preexec_fn`, before replacing the child image. Its separate `planned_argv` is correct. The final receipt and retained Root post-exec record explicitly observe `p6_kite.py` with the same worker PID/start time. The pre-exec cmdline is therefore not a mismatched launch, and must not be presented as post-exec evidence.

3. **Original authority is bounded.** Pinned preflight source checks the actual parent/worker identities, exact argv, interpreter path, original commit blobs, local HEAD/origin equality, fresh allocation, all inputs, window, HF offline variable, one-thread environment, CPU affinity and limits before writing the worker `startup.json`. This provides a source-and-record chain, not an independent runtime trace of every instruction. The admission does not serialize HF_HUB_OFFLINE or interpreter/library binary hashes. Current source equality cannot prove that no transient change occurred during execution. Do not upgrade this review into cryptographic execution attestation.

4. **Resource observations are sampled.** The supervisor polls available RAM/disk/output and wall time; `failure: null` means no corresponding sampled-stop reason was recorded. It is not a continuous memory-floor trace. The observed one-thread state is likewise a snapshot. `children_ru_maxrss_kib_at_reap: 5804368` is the supervisor's child-resource aggregate, not an independently measured exact model-worker peak. No CPU-usage total was verified here.

5. **Host closure is supplied by Root separately.** Original identities are supervisor PID 112011/start 234729136 and worker PID 112195/start 234729229. Receipt records worker identity/group/session absent, but its supervisor record is captured while the supervisor itself is still alive. Root reports actual-host verification at **2026-10-19 11:33:39Z** that both original identities are absent and both groups/sessions empty, with original session 8013 exit 1. This reviewer did not repeat that process inspection; its authority is Root's external observation, not the receipt's live supervisor field.

6. **Raw and scientific authority remains open to the responsible lanes.** Receipt lists 1289 unique output paths, and its listed byte counts sum to its stated `96764991` bytes. These are internal inventory checks only: this reviewer did not hash or decode those output files, check complete filesystem coverage, authenticate the large assets, replay numerical gates or infer orbit qualification. Root must bind its actual raw/asset inventory and closure receipt before any saved-numerical admission. The original plan's `no_target_run: true` and seed inventory's zero-evaluation field are frozen preparation-state descriptions, not current execution results.

## Metadata evidence hashes

Base: `/data2/neural-shrimp-data/scratch/audit-codex-1019/`.

| Record | Bytes | SHA256 |
| --- | ---: | --- |
| `r-pair-001-supervision/admission.json` | 4367 | `fb8c140a1b33852ae4b58615b646433229b42e7102279522d8937780aaa0681e` |
| `r-pair-001-supervision/receipt.json` | 301960 | `f4ecd1e9e090a1fecdbba35b0296efb1cd12a582e45199083b1645deaf59e907` |
| `r-pair-001-supervision/worker-startup.json` | 2583 | `969c04582ba834ff552f4b5856da3a26313f9a4eae852f0c69cdf03d0ac1dce8` |
| `r-pair-001/startup.json` | 469 | `10d51c88e41826d249594d2f9b9996477c4081a0bf8e63f904aa359a381ab1ef` |
| Owned-worktree `codex/presubmit-task37/r-pair-001-startup.json` | 1773 | `202a7a263592546a120eb27f745e1af9857d68e0f69956fa3e5a9be0d3950b02` |

Read-attempt limitations retained: the first filename lookup used the nonexistent suffix `-supervisor` rather than `-supervision` and a nonexistent `codex/active-status.md`; the correct paths were then read. Initial login-shell commands emitted `Failed to create stream fd: Operation not permitted`; subsequent reads used a non-login shell. The initial full receipt display exceeded output limits; complete JSON parsing subsequently selected every non-inventory field and checked the inventory's unique paths and byte sum. No target-output content was decoded in response to those display failures.
