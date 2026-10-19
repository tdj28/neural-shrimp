# Task33 saved audit002: closed source-provenance refusal

Recorded 2026-10-19T14:23:48.983401+00:00. The separately admitted saved audit002 is **closed and failed**. No operator bound is admitted, and fixed-start reachability remains unproved. The completed target measurement and failed audit001 remain unchanged.

## Execution and closure

Exact source `834befae0ccb6838f686c6dff7259d96cdebc781` was privately pushed and verified before session83834. Root invoked the frozen supervisor with explicit `HF_HUB_OFFLINE=1`; the worker passed the unchanged offline/thread guard. It exited1 after at most **2.800237 seconds**. The supervisor recorded no resource stop reason. Its `failure:null` field does not mean the checker succeeded.

The sole output is the23531-byte `failure.json`, SHA256 `7f99ba3309b7d4b87608a50e9663257b6c3dae1f29997670aa9885446853bdc5`. The receipt is5382bytes, SHA256 `66077d263ab85c04445291eb1bdae890db163079c87a6a59a529c930b19fc7a0`.

Root authenticated all25 source blobs against the actual execution commit,68 preflight pins, plan/card/argv/limits/startup, sole output and supervision inventory. Root also rechecked all84 worker-authenticated pins and all8 original target outputs. Admission recorded44,233,879,552bytes available RAM. At14:21:50.199550Z in the actual host view, supervisor559098/start236618679 and worker559158/start236618771 were absent; their process groups and sessions were empty. The short audit finished before Root's separate live check, so there is no Root live-start observation. There are zero owned numerical reservations.

The immutable completion receipt is `tube-target-001-audit-002-completion.json`, SHA256 `2f6b59fe705fb497c343b4da84eee5d90cdc279cac9f95fe20f4257e2aebdb51`.

## What failed

The worker reached `join_upstream -> original_execution(thin) -> verify_pins` and refused **pinned byte length differs**. The offending11th preflight pin in the original thin plan is `tools/explore_llm_soft_loop.py`. The original thin commit `e8db350006098683ab170645bf7ee7c317ca3b56` contains exactly122549bytes with SHA256 `28418408507409f52df9217c71255131a1de8cfe7d059a35761971738883e120`. The current source at the audit002 commit has130247bytes and SHA256 `6d9440be7f11b305a2e2e7e96956e533a4cdff9ea1309d7f2b006c2d289baf4e`.

Both original source blobs retain the declared `SoftLoop` class source-segment hash `c590a4c0bb2ee3742333d66cfa204add121b63bcb43d6446f96e8cd047f3e50b`. The current class hash is `139fe8e3534873807375c326be13cc3f6161d28c4a9bdf2e33e0447759dc5032`. Therefore this is a historical-versus-current producer-source mismatch; it does not establish raw corruption, and current class equivalence is not asserted. Root and independent static diagnosis agree. The prior static review and mocked integration controls missed this real historical-source join.

The failure precedes the downstream asset/embedding and prefix joins and scalar evaluation. Earlier preflight asset/cache consistency checks had already run; the final complete rechecks did not finish. No L was emitted, no model was called, and no scientific/operator/trajectory/reachability/certificate flag was admitted. Earlier authenticated target records do not complete the interrupted audit. The independent compact-record review agrees with the source, pin counts, identity, exit and false-flag records; it did not repeat Root's actual-host or large-file checks.

## Recovery boundary

A future separate allocation may add an explicit adapter for this one literal historical producer-source pin, authenticated by the exact original Git commit, length, hash and class witness. It must cover both the original preflight and the later producer read, retaining historical Git provenance separately from current-file rechecks. A generic fallback from mismatched current files to Git is unacceptable. Ordinary raw files, cache, assets, original receipts, all coverage/scalar gates and original failures remain unchanged.

That adapter, focused controls, new card/private source checkpoint, failed002 closure reference, explicit offline launch and fresh coordinated capacity have **not** been prepared or executed. There is no audit003 or target rerun. The existing Task33 window ends15:00Z; later execution needs renewed coordination.
