# Task37 saved-checker wrapper preparation

Prepared 2026-10-19 for Root's source review. **No wrapper, checker or control was executed by this preparation agent.** The only original checker operation was a source read and SHA256 comparison. No original target arrays were read. No repository file was changed.

The wrapper fixes the narrow dispatch problem: the supervisor creates the output directory's parent, whereas the frozen checker opens its report with exclusive creation and expects the report's directory already to exist. `run_saved_numerics.py` accepts only the literal original input root and the single fresh output allocation below. It rejects path aliases, missing or symlink-resolved existing paths, a wrong checker hash, and any occupied allocation (including a file, directory or dangling symlink). It creates the output directory once, then calls the canonical checker in-process through `runpy.run_path(..., run_name='__main__')`. It starts no child process.

## Fixed interface and pins

- Checker: `/home/d7082791602/.codex/worktrees/llm-hostile-fold-crosscheck/neural-shrimp/codex/presubmit-task37/audit_saved_numerics.py`
- Checker SHA256: `e840f06a91501a81dd396d3e20126c020e64c8698043e78a8dd0149274b40215` (unchanged on final read)
- Required `--root`: `/data2/neural-shrimp-data/scratch/audit-codex-1019/r-pair-001`
- Required `--out`: `/data2/neural-shrimp-data/scratch/audit-codex-1019/r-pair-001-audit-001`
- Checker report: the fixed output directory plus `result.json`
- No CLI checker, hash, report, dimension or retry override exists.

Prepared source SHA256:

| File | SHA256 |
| --- | --- |
| `run_saved_numerics.py` | `091c70ce93616586611ad6005512999f073900b9c8c7bd516b5b23f92a292cd3` |
| `test_run_saved_numerics.py` | `a5f1a6c5bee9bc11c437b6a7dbbdcfbc7f7fe156ea207c67a045734082775e9a` |

## Eight prepared controls

All controls are NEW stdlib-only synthetic fixtures under temporary directories. They import only this wrapper; every dispatch to `runpy.run_path` is mocked. Authority constants are replaced only within fixture scopes. No numerical checker, NumPy, SciPy, model, producer module or original output is imported or executed.

1. Assert the production checker/root/output/report/hash constants; reject arbitrary checker override, aliased root, relative root and alternate allocation before dispatch.
2. Accept the exact pin and reject a wrong pin or changed synthetic checker bytes before allocation/dispatch.
3. Refuse missing, symlink-resolved and wrong-kind paths, including a symlinked checker.
4. Create the output once; refuse an existing directory, regular file, live symlink and dangling symlink without changing them; refuse a missing parent without creating it.
5. Preserve the identical `SystemExit(0)` object, exact report bytes and scientific false flags; check exact checker argv and restore the prior argv object.
6. Preserve the identical `SystemExit(1)` object and report bytes with the same dispatch/argv checks.
7. Preserve an uncaught checker `OSError` object and partial report bytes; retain the incomplete output and restore argv.
8. Refuse unexpected normal checker return rather than implicitly succeeding; retain the allocation and restore argv.

Control 5 intentionally supplies a manufactured exit/report combination to test noninterpretation: even an exit zero does not authorize changing false report fields. It is not a scientific validation. No Hugging Face environment assertion is present or required by these stdlib controls.

Root may run the controls after source review with this command (not executed here), capturing stdout, stderr and exit status in a new retained attempt:

```bash
HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 nice -n 10 /data2/neural-shrimp/.venv/bin/python -I -B /tmp/task37-saved-wrapper-prep/test_run_saved_numerics.py
```

## Limits and admission still required

The wrapper does not parse, modify or reinterpret checker output. It catches neither `SystemExit` nor other checker exceptions. A `finally` restores `sys.argv`; all failures otherwise propagate unchanged. Failed allocations/reports are retained, and this fixed allocation cannot be reused. The external supervisor captures stdout/stderr and the worker return code; its own failure exit may differ from that worker code.

The source hash is checked immediately before allocation and dispatch. `runpy` then reopens the canonical path: this is not an atomic defense against concurrent source replacement. Root must maintain the frozen source and bind the admitted execution externally. The wrapper checks only that the fixed input root is a canonical directory, not the input files, original run completion or their scientific validity.

Original supervisor receipt and exit, attempt identity, exact source/commits and source freeze, run card, model/tokenizer assets, full input/output inventory and hashes, original host/process closure, and scientific admission remain **Root's external gates**. No runtime admission or final scientific verdict follows from wrapper or mocked-control success. The checker retains `scientific_verdict_admitted: false` and `certificate_passed: false` as written; this wrapper supplies no new authentication for them.

Root's proposed saved-audit envelope is 4 GiB address space, 300 CPU seconds, 360 wall seconds, 16 MiB output, 24 GiB startup availability and 16 GiB runtime availability. These are proposed external supervisor/admission limits, not implemented or verified here. Execution remains held until original host closure and Root's complete admission check. No pinned active file, numerical behavior, data or process framework was changed.
