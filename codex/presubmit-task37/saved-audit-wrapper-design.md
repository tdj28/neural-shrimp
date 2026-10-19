# Minimal isolated wrapper design — not implemented or admitted

The proposed final source path is `codex/presubmit-task37/run_saved_numerics.py`. No such wrapper was created by this preparation. Its source pin remains null in both held drafts. Root must resolve this prerequisite before admitting any saved replay.

The existing supervisor constructs `python -I -B ENTRY ARGUMENTS`, requires the output directory to be absent, and creates only `lane.output.parent`. The unchanged checker accepts only `--root` and `--report`; its final `open(report, "x")` does not create the report's parent. A direct entry would therefore spend the replay work and then fail when writing `r-pair-001-audit-001/result.json`.

No suitable generic dispatcher was found in the inspected Task37 source set. The Task34 `audit_dense_p6.py` creates its parent directory exclusively, but also binds a different attempt, schema and arithmetic audit. It must not be reused as a dispatcher or silently relabeled for Task37.

Required minimal behavior, before any checker import or saved-array work:

1. Accept `--root` and `--out` (directory); require the exact original `r-pair-001` root and exact fresh `r-pair-001-audit-001` output directory specified in the card. Fix the report basename to `result.json` internally; do not expose a report-name override. Reject substitution, containment within original output/supervision, and pre-existing/symlink output paths. Do not infer a second attempt or make parent trees recursively.
2. Require the reviewed one-thread execution environment before importing numerical libraries; retain Python assertions. HF offline remains inherited launch metadata, with no new wrapper assertion required: the checker has no Hugging Face imports. Confirm the installed checker's literal SHA256 `e840f06a91501a81dd396d3e20126c020e64c8698043e78a8dd0149274b40215` before executing it.
3. Create only the exact new output directory with exclusive `mkdir(exist_ok=False)` after the source/admission preconditions are satisfied. Any failure remains a fresh failed allocation, never an instruction to remove the directory or retry it.
4. Dispatch the unchanged checker in the same worker process, for example via standard-library `runpy.run_path(..., run_name="__main__")` after translating `--out DIRECTORY` to the original checker interface `--root ORIGINAL --report DIRECTORY/result.json` in `sys.argv`. Do not spawn another worker, recreate numerical formulas, import the model/map/helper, change thresholds, supply test-only dimensions, or swallow/change its exit status.
5. Preserve any partial report, stdout/stderr and supervisor receipt. The output directory must still exist after failed replay so the supervisor can inventory what was written.

Authentication is additional to this dispatch design. Root must bind and inspect the original successful receipt, admission, source/input pins and independent host closure before launching. The existing supervisor can verify every file listed in a pinned `input_inventory` before and after, but it does not reject extra files, symlinks or duplicate canonical aliases. Complete no-symlink canonical inventory equality is required before and after replay in a separately reviewed wrapper check or Root's explicit admission/post-run review. The simple directory creation alone does not close that gate.

Root owns the implementation, any narrow new wrapper controls warranted by that new source, private checkpoint and execution decision. No existing controls were repeated here, and no target files were opened.
