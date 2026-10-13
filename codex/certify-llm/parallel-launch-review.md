# Independent review of continuation launcher admission

Reviewed `launch.py` SHA-256
`d991f4f58ffe6072e183bc8506079342c84506f48ae080d139537c573e2b9ee1`
and `tests/test_codex_llm_parallel_launch.py` SHA-256
`6b7cf01512be5f58d82e923b6026f17372bcac6653145b418e24a9e722a2acd4`.
No unresolved blocker found in the small continuation-specific delta. No source
edits, actual admissions, target jobs, model calls or Git mutations were made.

`center_continue` is added to the known driver list and mandatory committed-plan
list. It requires exactly an 1800-second outer watchdog, retains `--threads 1`,
and refuses skip-box or external/path-traversing plan names. The change does not
alter existing drivers' mathematical code or their accepted resource envelopes.
The launcher still verifies captured tracked source bytes against HEAD and
private-origin ancestry before starting a job, copies immutable Git blobs into
its attempt snapshot and executes that captured driver and plan with isolated
Python and CPU1/GPU-disabled/offline environment settings.

The manifest distinguishes the controller's 18 GiB address-space cap and one
thread from four CPU1 worker slots, a 32 GiB **sampled** RSS watchdog, at least
12 GiB available-memory headroom and 600-second per-worker allowance. Recording
those fields does not itself enforce the worker policy: enforcement remains
in the separately reviewed, source-pinned scheduler/worker and must be checked
when reviewing `center_continue`. The launcher starts its controller in an
owned process group. On its wall deadline it signals that controller group,
allows ten seconds for scheduler cleanup, then escalates if necessary. Worker
groups use the scheduler's separate ownership checks; its direct workers also
have the previously tested parent-death guard. No aggregate hard cgroup cap is
claimed.

The final immutable receipt retains all source/output bytes, exit status and
`timed_out`. Admission must continue to require both a successful exit and
`timed_out=False`, rather than inferring success from exit code alone. This
matters if a controller exits gracefully during the timeout cleanup interval.
All complete/failed attempt outcomes remain separate from scientific endpoint
validation and any orbit-proof claim.

Independently ran all 11 mocked launcher integration tests (0.10 seconds).
They verify the precise command/environment/manifest, receipt hashes and
read-only retention, timeout signalling, refusal of unpublished or dirty source
before launch, required plan, exact wall-time/thread limits and plan-path
boundaries. Git and process execution were mocked: these tests performed no
fetch, launch or target computation. Real tiny-process cleanup/PDEATHSIG tests
were covered separately in `parallel-phase-pass-review.md`.
