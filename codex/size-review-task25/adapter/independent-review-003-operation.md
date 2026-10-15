# Task25 manufactured operation compatibility

15 October 2026. **Accepted as a narrow source-only adaptation of the previously reviewed supervisor; no correction requested.** This supplements scientific/source review `/tmp/task25-adapter-independent-review-002.md` (`30ca3e2c6e50444d693b751dc5ab48d3a5040ea53fa211b168d8781ffacd8ea7`) and closes its pending operational-source item. It does not authorize execution or claim that controls have passed.

I read the complete concrete plan, inherited supervisor and exact parent diff:

| Artifact under `/tmp/task25-manufactured-operation-prepared` | SHA256 |
|---|---|
| release.json | `b490a2ce5a1848dd35320e90fd76f06b659ed90844d041aa538e861bd9f98532` |
| manufactured-001-supervise.py | `2ebc048e96dc4dc6e6a909fa1c094da887ec986f9ec05285cc4a91b1bc0cfae1` |
| manufactured-001-plan.json | `842d307eb881e56e967acc1e1cb2de99851b8a81ccb470e18538a23e62424b2e` |
| manufactured-001-supervise.py.diff | `7f40148eada7a828808db66e1f3f225fe5499820e8adfe80b4251de55abc3ea5` |
| parent/controls-001-supervise.py | `578921cdaf888f17bfd70c287436d4206b2ec0264f541d6cc4f9db77d58ce1eb` |

The adaptation changes the fixed Task25 code/output names, capture list, worker CLI, expected report and output allowance. It captures all ten reviewed Python files, the shape file, plan and supervisor (thirteen blobs). The scientific pins agree with release `c1cea5e1`, including run `196dc27b` and manufactured `3ddb960e`; the worker executes the captured `manufactured.py --out` with `-I -B`. Its expected report is exactly the manufactured schema, 38 passed groups and zero failed groups. Scientific claims remain unaccepted by the operational receipt.

The inherited admission checks require prepared status to be promoted explicitly to `reviewed_manufactured_only`, a clean checkout, the requested `--source` equal to HEAD, ancestry in private `origin/main`, and byte equality of all captured files with that commit. The optional null `source_expected_commit` field is not a refusal mechanism; the concrete source is bound through the launcher checks and capture. Both output and supervision directories must be fresh. Persistent streams are exclusive, the worker gets a new session, and the original process identity and owned group are retained.

The exact fixed limits match the plan: one numerical thread plus supervisor, absolute nice10, 1GiB worker address space, 55 CPU seconds, 60 seconds of externally sampled wall time, startup/runtime available-memory floors of 20/16GiB, a 16GiB startup free-disk floor, and the inherited 5+5-second termination cleanup. The source confirms hard worker AS/CPU limits after launch. The supervisor's own nice10 is required at entry; the coordinator retains launch/preflight ownership.

The 128MiB allowance covers the **whole manufactured directory plus stdout/stderr**, including generated inputs, expected-refusal artifacts and all serialized results. It is a sampled aggregate cap; the separate per-file hard limit is also 128MiB. Captured source and supervisor receipts are excluded and remain separate storage overhead. Thus the earlier output-scope reminder is resolved. The harness's nine-file scientific-output subtotal is not mislabelled as the total sampled workload. A bounded failure must remain a failed attempt; no automatic retry or expansion is introduced.

No imports, tests, manufactured or target execution, statistical work, model calls, process operations or Git changes were performed in this review. The five pure components were not re-reviewed. Inference retains admission/private checkpoint/fresh capacity, the sole manufactured run, all closure observations and any later actual-data plan. The proposed envelope is finite but still unmeasured until that separately authorized manufactured run completes.
