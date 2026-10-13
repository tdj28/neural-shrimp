# Independent campaign003 recovery source review — 2026-10-13

**Verdict:** no remaining source blocker in the released narrow recovery amendment. This approves the reviewed implementation for the next concrete-plan review; it is not a campaign launch, a target-evidence audit, or a certificate.

Reviewed source SHA256 `9357d0aee5e5880b37c752a5c928c988480c63a1871db94f1fc3f8b3df70ee52` (`derivative_campaign.py`), against original `708eb618fa31afbab4e27ce0e834c16724a1d03985f1bfbef57bf8eb202bba03` retained at `/tmp/campaign003-before/derivative_campaign.py`. Exact reviewed diff `/tmp/campaign003-recovery.diff` has SHA256 `e43b08d30446ca8b3bac860d6c65dcd1bdd0503491a290d22e6ad52a1c438a3a`.

The recovery-specific test source is `dbf264850d64b8baca725d91586c99b3e3c9997b2dd6f8432d837315738eb534`; the unchanged lifecycle test source is `fbd351342c1df5274fd3a345ab3e653abfc895518bd2c43381a71164c60038a7`. The reviewed design note is `df7f78e5df42205c750013316b4a8ad88da0060f464a4e9d4af7b8c35cdbb625`.

## Findings

The Git amendment is restricted to the controller's `status` invocation with command-scoped `core.preloadIndex=false`. It retains the exact dirty/untracked status check, HEAD/blob checks and remote/source admission. It changes neither global Git configuration nor the launcher or evaluator. This addresses the optional threaded-preload operation; the underlying transient resource failure is not claimed to be diagnosed.

The new recovery schema authenticates the original failed controller receipt and complete artifact inventory, literal admitted plan and source, 21 dispatched records, externally pinned closure report/source, and all 43 recorded birth identities. The original first-shard authority and exact common domain, parameters, native evaluator bundle, precision, backend and prefix remain mandatory. Each of the 17 retained shards goes through the existing full receipt/source/status admission. Each interrupted attempt retains its original receipt, command, source, plan and termination evidence. A fresh read-only process-table check rejects still-live or reused original groups. The closure report is independently reviewed evidence; its truth is not manufactured by a Boolean or hash.

The assignment partition is exact and ordered: 17 retained successes are removed from the original 111; four interrupted assignments receive new explicit attempt/plan filenames with identical plan payloads; all 90 undispatched rows keep their original identities. Thus exactly 94 prospective rows remain, plus the first and 17 retained shards. Existing attempt paths are refused. Changed native source/domain, missing or duplicated retained history, retained reruns, renamed undispatched rows, and changed replacement payloads are rejected. Recovery is not recursive or automatic.

Ownership, command-registration deadlines, PID/pidfd checks, own-group cancellation and receipt collection remain unchanged. Only an authenticated existing-byte baseline is added to the dispatcher's storage accounting. Forecasts use 94 new rows and charge the first shard, original controller and all 21 prior shard roots to retained storage. All resource fields except the whole-campaign deadline must match the original card; that deadline can only decrease. The proposed 17-hour ceiling must still pass the concrete first-shard forecast and fresh resource admission. No greater concurrency, per-worker limit or expanded scientific scope is introduced.

During draft review I found a `pair` helper shadowed by a later loop binding in `validate`, which would have refused all v2 admission. The author renamed the loop variable before release and added an actual synthetic integrated v2 success path. This finding is resolved in the bound source.

## Independent controls and limits

I independently ran the released v1 lifecycle plus v2 recovery controls with CPU1 environment limits and nice10: **54 passed in 11.25 s**, exit0. Evidence remains at `/tmp/campaign003-independent-controls-001`. The integrated v2 control uses real complete synthetic controller/shard receipts, `validate`, `recovery_authority`, `first_authority`, and dispatch selection of all 94 rows. Only process spawning is substituted in that new selection test; the existing tiny owned-process lifecycle controls also passed unchanged. Resealed missing/duplicate history, changed domain/source, stale original artifacts, incomplete closure, a live prior identity, enlarged resources and storage charging are exercised.

No actual target receipt/derivative array was read, model evaluated, real shard selected, campaign launched, source changed, or Git mutation performed during this review. A concrete plan still needs its externally pinned original evidence and private source checkpoint. Final interval-derivative and orbit-proof audits remain separate.
