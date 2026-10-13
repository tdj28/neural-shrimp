# Independent bounded command-admission review

13 October 2026 UTC. **Approved; no blocker found in this narrow startup-race delta.** No target run, campaign plan, source edit or Git operation was performed. Reviewer: Codex `literature_cascades`.

The exact diff is against the already reviewed nice10 dispatcher snapshot `5266776c…`. A spawned `Popen` handle still enters the owned table before fallible inspection. Parentage, UID, session/group leadership and birth-checked pidfd ownership are obtained immediately; transient `/proc` command content no longer prevents retaining that cleanup identity. The new flag explicitly separates this fact from command admission.

Until admission, each read revalidates the same birth identity and compares the entire observed argument list with the expected launcher command. Only exact equality observed strictly within the original registration interval admits it. The original spawn time is retained: a persistent mismatch fails, and an exact command first seen at or after the deadline also fails. A zombie or missing process cannot gain command admission. Distinct observations and the admission timestamp are persisted.

`register` returns before inspecting the worker manifest while command admission is pending. The scheduler's existing next-launch condition requires every active child to be fully registered, so neither a transient mismatch nor cleanup-only ownership permits the next dispatch. Existing worker command, direct parent/birth identity, priority, AS, source and plan gates are unchanged.

On failure, the birth-verified direct child remains owned for cleanup even if command admission never succeeded. The stopped-parent cancellation protocol can recover a matching worker for cancellation only. That does not change the original failure reason, allow another assignment, or promote the result to success. Unknown children and parent death before ownership recovery retain the earlier explicit unsealed uncertainty behavior.

I independently ran the four new targeted controls: **4 passed in3.00 seconds** (28 unrelated cases deselected), CPU/thread1 and nice10. They use tiny owned subprocesses and substituted command reads: transient empty, transient wrong, persistent wrong, and late exact. The transient cases wait for exact admission before subsequent dispatch; persistent/late cases fail with no next dispatch and verify owned-process cleanup. No target/checkpoint/model/network operation occurred. The author's larger63-case suite is reported in its amendment; this review does not present that as independently rerun.

The amendment candidly distinguishes the retained transcript of the original incidental failure from its auto-pruned pytest files. Its deterministic fixed-directory controls are separate evidence, not a reconstruction of those deleted files. No scientific raw evidence is implicated by this narrow review.

SHA-256 bindings:

- Dispatcher: `708eb618fa31afbab4e27ce0e834c16724a1d03985f1bfbef57bf8eb202bba03`.
- Dispatcher tests: `fbd351342c1df5274fd3a345ab3e653abfc895518bd2c43381a71164c60038a7`.
- Separate amendment: `0776715c42f1c5d5bf447a1b164a74827063132365257a2fbca64edc6f1441b6`.
- Before source: `5266776c8618a27d6c97772e4aa7018983b5bb1407872fbb315b9f7fc17542a9`; before tests: `194115a938806b6e53be0a4c9e94ae870cdff48c6861e5cc01cad09702e509a3`.

The nice-only launcher and mathematical evaluators are unchanged. Actual first-result audit, measured resource card and frozen execution checkout remain required before a campaign; this source signoff does not replace them.
