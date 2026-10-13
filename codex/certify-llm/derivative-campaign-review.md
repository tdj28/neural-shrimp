# Independent fixed-derivative dispatcher review

13 October 2026 UTC. Source/lifecycle review only; no target shard, campaign plan, model, checkpoint or Git operation was run. Reviewer: Codex `literature_cascades`.

**Verdict:** no unresolved implementation blocker in the released fixed-scope dispatcher. This permits source preparation and commit. It does not authorize a campaign: the first actual final-domain shard still requires its independent completed audit, followed by an explicitly frozen measured resource card and separate owned execution checkout.

Reviewed identities:

| File | SHA-256 |
|---|---|
| `derivative_campaign.py` | `0cc8a192737508b449472eb7d20562c6d84abf84b07565de7e0108d712cea161` |
| `tests/test_codex_llm_derivative_campaign.py` | `194115a938806b6e53be0a4c9e94ae870cdff48c6861e5cc01cad09702e509a3` |
| `derivative-campaign-design.md` | `d6f3d75848f0cdc3783685901d8e40be8b7d07ad92ff77ae24d3dffbcce11179` |

I read the complete dispatcher, tests and design, and the unchanged launcher's source-snapshot, resource and process-creation paths. I independently ran the released 28 tiny synthetic admission/process controls with one numerical thread and nice10: **28 passed in 6.56 seconds**. These tests substitute small owned processes and metadata fixtures; they neither establish an actual first-result authority nor audit real derivatives.

## Concrete lifecycle findings

- A successful `Popen` handle and expected command enter the child table before fallible birth/pidfd registration. Cleanup can therefore recover that unreaped direct child after an initial registration failure.
- Launcher admission verifies direct parentage, UID, birth time, session/group leadership and exact command. Worker registration additionally requires the source manifest, exact archived driver/plan command, direct launcher parent, actual nice10 and both actual 18 GiB address-space limits. Ordinary workers must be registered before another registration starts.
- Signals use fresh birth-checked pidfds. Group cleanup verifies the recorded session/group, UID and start ordering; a recycled numeric leader does not authorize signalling an unrelated process. A leader's exit alone does not hide surviving members of its recorded owned group. The reviewed derivative worker does not create subprocesses; unexpected surviving descendants are a failure, not successful completion.
- During cancellation before registration, the verified launcher is stopped before inspecting possible in-flight worker creation. The short handoff allows a forked child to reach its final command. A confirmed matching child can be recovered for cancellation only. Unknown children remain uncertain; a missing manifest is never equated with absence. Every issued STOP is paired with a finally-path CONT, including failed STOP confirmation.
- Registered workers receive TERM, then KILL if required, while the launcher remains available to save its receipt. One child's cleanup error does not skip peers. Failure-log errors also enter cleanup. No subsequent assignment is dispatched after the failure is observed.
- Unknown ownership, surviving workers or an unfinished receipt writer leave an unsealed result. The existing launcher's parent-death-before-manifest window is explicitly unresolved, not silently treated as terminated. SIGKILL of the controller/host failure likewise requires external ownership recovery. The owner must resolve any such event before more runs.

## Source and resource admission

The exact 111 remaining canonical assignments, distinct absent output paths and committed plan names are enforced. Common exact domain/map/backend/prefix/source identities and CPU1/18 GiB/3600-second envelopes must agree with the externally audited first shard. The independent first-audit object is a pinned external authority; its truth does not follow from its `passed` field. Full scientific derivative admission remains with the existing readers.

The dedicated execution checkout is fixed in the card. Startup requires clean pinned HEAD, exact committed source and all plan bytes, plus remote-main ancestry. Before each launch and at final completion, the controller rechecks cleanliness, HEAD and campaign bytes. The unchanged launcher snapshots committed files and performs its own remote admission. This isolates the execution source from the active research checkout's ordinary commits.

The explicit time/RSS/storage headroom factors are applied to the actual first measurement. Forecasts reserve controller storage/RSS, registration time and receipt grace. Concurrency reserves at least two additional CPU slots within the total limit of12. The controller must start at nice0 because the unchanged launcher adds10 to worker priority; numerical workers are checked at nice10. The controller's 1 GiB soft/18 GiB hard AS arrangement permits the unchanged child to establish its own18 GiB cap.

Aggregate RSS, MemAvailable, storage and deadline checks are sampled watchdogs, not cgroup or affinity guarantees. In particular a minimum-available-memory floor is reactive, not a reservation of all future workers' measured working sets. The concrete campaign card must choose concurrency against current shared-machine headroom and retain a margin; this source review supplies no actual capacity admission. The code's finite cleanup grace can extend beyond the stop-dispatch deadline to retain receipts and verify termination.

No source change is requested. No certificate flag becomes true, and no campaign-level success substitutes for the eventual full original derivative/evaluator audit or mathematical gates.
