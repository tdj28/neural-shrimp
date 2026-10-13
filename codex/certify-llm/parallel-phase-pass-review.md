# Independent review: bounded parallel point-pass scheduler

No blocker found for the stated Linux, immutable-source, no-worker-descendants
contract. Reviewed source `parallel_phase_pass.py` SHA-256
`2cdc3afd9e73d95857941034420d279db7f44b5177b82131cf5f1a44de6d2c43`,
tests `5c5a2686b05c1cfa160f5735bc32761a183a740078806e02b20d94a7df3500e1`,
and design note `3f33d9dc37191504d9afe77d4aea2898ad90c8c200aa8980f743f6f8bccbb466`.
The accompanying phase-worker source was
`e87f1efdffd17925d8ec971f7ce164cfd5c1eaa1f0ad046cbc02916994ec1b4a`.
This review performed only source inspection and tiny synthetic process tests;
no checkpoint, research request, target job or Git mutation was involved.

The controller admits one fresh owned candidate layout and exact request bytes,
with worker/source-commit/map/backend bindings. It rechecks request/worker/
controller hashes before each launch. Four fixed CPU1 slots schedule all eight
explicit global phase indices. Every launch needs the fixed 600-second shared
reserve; neither a late phase nor a failed worker can extend the absolute
deadline. Completion requires every worker to exit zero, pass its retained
status/source/request/output-hash checks, and have verified observed termination.
The final report orders all eight phase receipts regardless of completion order.
Failures retain completed and partial outputs plus explicit unstarted phases;
none are silently replaced by another phase or reported as complete.

The isolated bootstrap sets thread/GPU/offline controls before worker imports,
sets at most 18 GiB address space and bounded CPU time while preserving stricter
inherited limits, and records actual limits/niceness. Parent identity is checked
by PID and start time around arming Linux PDEATHSIG. SIGTERM/SIGINT are blocked
only across launch and ownership registration. Normal cleanup signals only
identity-checked launched groups, shares bounded TERM/KILL grace periods and
records any termination it cannot verify. Prior signal handlers are restored.
The reviewed worker must not spawn descendants: parent-death protection is for
the direct worker, not an arbitrary hostile tree that escapes its owned process
group. Unexpected observed live descendants invalidate successful completion.

The aggregate RSS watchdog includes controller/descendant/group usage and
checks available-memory headroom before launches and while workers run. Its
32 GiB ceiling and at least 12 GiB headroom are **sampled guards, not a hard
aggregate cgroup limit**. Individual workers retain hard address-space/CPU
limits. An outer resource-reviewed plan and launcher must pin the full source
snapshot, permit the four CPU1 workers, reserve cleanup grace, and retain a
failed-attempt receipt if the controller itself is killed before writing its
report. These are explicit integration obligations, not supplied by the pure
scheduler's successful return.

Independently ran all 62 scheduler/worker synthetic tests (4.79 seconds). Tiny
real subprocess cases covered reordered completion, worker crash, invalid
output hashes, timeout despite ignored SIGTERM, loss of memory headroom,
source/request mismatch and mid-pass mutation, stale output refusal, resource
parameter type/limit errors, parent identity mismatch, PID reuse refusal,
unexpected descendants, and controller SIGTERM/SIGKILL. All passed; no source
changes were required. Process success authenticates lifecycle and bytes only:
full exact-centre/image/residual mathematical checks remain external, and both
certificate and scientific-endpoint-audit flags remain false.
