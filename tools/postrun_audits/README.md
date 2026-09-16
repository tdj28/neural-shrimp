# Independent post-run audit source

These small scripts are preserved in private Git to protect audit code as
well as experiment code. They are byte-identical copies of the independently
executed post-run scripts identified in the corresponding results summaries.
Their original reports remain in the external audit directories identified
by each experiment's compact results summary. Original script paths and
hashes are recorded there as well.

They are forensic checks of named completed attempts, not experiment runners
or pre-outcome sources. Inputs use the explicit original checkout and sibling
data paths written in each script. They read raw evidence and write diagnostic
reports to their documented paths. Older scripts use temporary `/tmp/`
reports that a rerun can replace, without replacing the durable original
report or raw attempt. EXP-012 scripts accept a new exclusive external output
directory; use a different directory for each replay.
Receipt/source hashes distinguish the original executed copy. A later change
to a script requires its own audit provenance.

Use the project's locked numerical environment. Host process-cleanup checks
need host visibility; an empty process list inside a sandbox does not prove
that an owned host process has exited. These audits do not acquire new target
outcomes, modify the frozen experiment or replace its verifier. Read each
script's scope and the corresponding results audit before rerunning it.
Inspect the report's failure count and status, not just the process exit code:
some forensic scripts accumulate their findings in the report.

`exp012_independent_numerical_audit.py` accepts `--output` and optional `--raw`
and `--workers` arguments. It replays saved scoring through the older frozen
scorer and comparator, without importing the EXP-012 producer or CUDA backend,
and makes no host-cleanup claim. Its evidence checks should pass while the
recorded cross-device qualification fails; these are different result fields.

`audit_exp012_performance.py` accepts `--out`, optional `--raw`, and
`--host-scope-confirmed`. Supply the latter only with actual host PID/GPU
visibility. It uses the standard library to reconstruct schedule, timing,
cost, inventory and ownership checks without importing producer code,
integrating, scoring, fitting, loading CUDA or signaling processes. It does
not independently recompute numerical parity.
