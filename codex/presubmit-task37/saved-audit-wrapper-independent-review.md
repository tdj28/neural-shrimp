# Independent Task37 saved-wrapper source review

**Source/interface disposition: suitable for Root's new mock-control run and held-packet integration. No blocking defect found in the inspected wrapper. This is not execution or scientific admission.** I read both new files and the held plan; I did not execute/import either file, run controls, read active target data or inspect/control a process.

Inspected files:

| File | Bytes | SHA256 |
|---|---:|---|
| `/tmp/task37-saved-wrapper-prep/run_saved_numerics.py` | 2987 | `091c70ce93616586611ad6005512999f073900b9c8c7bd516b5b23f92a292cd3` |
| `/tmp/task37-saved-wrapper-prep/test_run_saved_numerics.py` | 7382 | `2c7d47408afbceb51fdc8ca37181b965ae06c277625ca298056339df21205533` |

Plan reviewed: `/tmp/task37-saved-admission-prep/r-pair-001-audit-001-plan.json`, SHA256 `5d3395662d7ddc5565871c0edd196aa246eab70c97405dcf05bb8d53d8a6556f`.

## Interface and semantics

- Wrapper lines 15–19 name the exact unchanged checker, its `e840f06a…` SHA256, original `r-pair-001` directory, proposed `r-pair-001-audit-001` directory and fixed `result.json` basename. These agree with the held plan and previous checker pin. The plan's `--root ORIGINAL --out AUDIT_DIRECTORY` is accepted exactly; only the internal dispatch translates it to the checker's `--root ORIGINAL --report AUDIT_DIRECTORY/result.json`.
- `fixed_path` compares literal argument strings. Alternate roots, relative paths and textual aliases do not acquire authority. `plain_path` requires the existing input root, output parent and checker to be canonical existing paths of the expected kind before allocation; the checker hash is verified before directory creation.
- `out.mkdir()` has neither `parents=True` nor `exist_ok=True`. It creates only the allocated audit directory and refuses a prior directory, file or symlink. This closes the original report-parent gap without changing the existing supervisor or checker. The wrapper never removes an allocation or retries another name.
- `runpy.run_path(..., run_name='__main__')` executes the unchanged checker in the same worker process with its original CLI. There is no subprocess, model/helper import or formula implementation in the wrapper. The `finally` block restores the original `sys.argv` object but does not catch or rewrite `SystemExit`, other exceptions, reports or flags. Thus real checker exit 0/1 and an uncaught failure retain their meaning. A surprising normal return becomes an explicit failure rather than implicit success.
- The pinned checker opens its report exclusively after constructing its result and then raises `SystemExit`; the wrapper does not precreate the report. If the checker fails before reporting, the new directory remains. If serialization fails after opening, any partial report remains. Supervisor stdout/stderr/receipt retention remains external.

## New mock-control coverage, inspected but not run here

The eight tests exercise: fixed path and no checker override; checksum refusal before allocation; missing/wrong-kind/symlink path refusal; exclusive directory creation and no recursive parent creation; unchanged success exit and scientific-false report bytes; unchanged failure exit/report bytes; identity-preserving propagation of an uncaught error with partial-report retention; and refusal of an unexpected normal return. The invoked `runpy.run_path` is mocked, numerical libraries and the real checker are never imported, and fixtures remain under a temporary directory.

These controls are appropriately focused on the new wrapper. They do not revalidate old numerical arithmetic or authenticate any real original attempt. In particular, the success mock intentionally tests transparent propagation even when manufactured report flags are false; it is not a claim that the real checker emits that combination. No test-pass claim is made by this source review. Root owns the one new control invocation and its retained outcome.

## Remaining external gates

The wrapper does not authenticate descendant raw paths, full file membership, source/model/operator provenance, original successful exit, original scientific row gates, capacity reservation or original host closure. Its docstring correctly disclaims those responsibilities. Root must retain the complete canonical no-symlink raw inventory check before and after replay, exact original receipt/admission/startup and source/input bindings, and independent host closure; the supervisor's prior-receipt booleans alone are insufficient.

The wrapper also inherits one-thread/HF-offline/resource settings from the admitted supervisor environment; it does not create a new capacity gate. The held plan's null wrapper pin can be replaced by the reviewed source above, but every other deferred original/closure/inventory/capacity prerequisite remains. Final card edits require regenerating its plan pin and private checkpoint before any status promotion. No active-target or saved-arithmetic execution is authorized by this review.

## Final-control clarification — 2026-10-19 11:01:18 UTC

The initial review above read the earlier 7382-byte control snapshot (`2c7d47408afbceb51fdc8ca37181b965ae06c277625ca298056339df21205533`) while the source agent was finalizing additions. That timing disclosure and original hash are preserved; they must not be represented as the final control source.

I have now read the added exact canonical checker-path assertion in control 1 and the existing regular-file, live-symlink and dangling-symlink refusal subcases in control 4, including the checks that those occupied allocations remain intact. They agree with the wrapper's fixed-path and exclusive `mkdir()` behavior and introduce no interface or scope defect found in this inspection. The additions use only temporary fixture paths; no checker invocation or numerical import is added.

The final control file is **8367 bytes**, SHA256 **`a5f1a6c5bee9bc11c437b6a7dbbdcfbc7f7fe156ea207c67a045734082775e9a`**, independently verified at the stated time. The wrapper remains **2987 bytes**, SHA256 **`091c70ce93616586611ad6005512999f073900b9c8c7bd516b5b23f92a292cd3`**. The disposition remains suitable for Root's new mock-control run and held integration. No controls were executed by this reviewer, and all external admission gates above remain unchanged.
