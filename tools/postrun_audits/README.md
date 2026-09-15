# Independent post-run audit source

These small scripts are preserved in private Git to protect audit code as
well as experiment code. They are byte-identical copies of the independently
executed post-run scripts identified in the corresponding results summaries.
Their original reports and script copies remain in
`/data2/neural-shrimp-data/audits/<experiment>/post-run-audit/`.

They are forensic checks of named completed attempts, not experiment runners
or pre-outcome sources. Inputs use the explicit original checkout and sibling
data paths written in each script. They read raw evidence and write diagnostic
reports only to the stated `/tmp/` paths; rerunning can replace that temporary
report, but does not replace the durable original report or the raw attempt.
Receipt/source hashes distinguish the original executed copy. A later change
to a script requires its own audit provenance.

Use the project's locked numerical environment. Host process-cleanup checks
need host visibility; an empty process list inside a sandbox does not prove
that an owned host process has exited. These audits do not acquire new target
outcomes, modify the frozen experiment or replace its verifier. Read each
script's scope and the corresponding results audit before rerunning it.
Inspect the report's failure count and status, not just the process exit code:
some forensic scripts accumulate their findings in the report.
