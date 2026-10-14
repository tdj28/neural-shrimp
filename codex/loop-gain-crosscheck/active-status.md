# Task18 status

2026-10-14 07:20 UTC: run-001 failed operationally at the launcher registration
print with BrokenPipeError after outer setsid detached and the tool stdout pipe
closed. Original worker exit -15, no numerical result; all 14 artifacts verified
and both original PID/birth/groups/sessions absent. Original receipt353d1ad1
and closurec62624fc remain immutable; run-001-failure-closure.json records them.

Root and independent review accept the narrow recovery: literal002 paths only,
setsid --wait plus persistent outer stdout/stderr. New run-002 plan/card are
prepared, awaiting admission checkpoint/fresh preflight. No active reservation.
Original numerical source, parameters, gates and 3600s/12GiB envelope unchanged.
Task19 source preparation is accepted separately; no target run yet.
