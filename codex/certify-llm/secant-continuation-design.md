# Complete secant-parent continuation and parallel point passes

13 October 2026. Design only: no implementation, target execution, real-data
admission or current secant-pilot outcome inspection. Based on the reviewed
`center_secant.py` (`c828ebc744eb95795e9051d5dbe0b8361828ebdfb13688afae5c1f60f7694132`)
and `inverse_secant.py` (`01065e31a70b7b7b0b394609e2afef78ea96669455a72fd4c17546b94d8d914b`).
Leave the existing strict fixed-R importer and evaluator/cache sources unchanged.

## Minimal scientific continuation contract

A separate reader/driver should accept only a completed, externally pinned
`center_secant` parent or its own reviewed continuation format. Authenticate the
entire immediate receipt/source inventory, same map/assets/prefix/backend and
2048-bit context, fixed A/B proposal identity, all eight fresh point outputs,
exact input/successor bindings, zero tangents, complete status and retained
comparison. Refuse partial/failed parents and mixed map or precision. Copy A/B
for the kernel and retain its existing immutability guard.

A reusable admitted-parent object needs only:

- The exact **before** and **after** centre tuples, full before/after image balls,
  raw source-image bindings and freshly reconstructed full residual balls.
- The active low-rank factors **after the parent's proposal**, their originating
  accepted-decision hashes, fixed caps, and global tuple/decision indices.
- Immutable links to every original centre, full residual, acceptance/rejection
  witness, proposal and completed evaluation; plus parent receipt/source pins.

The active operator is only the numerical proposal
`R = R0 + sum(u_k w_k^T)`, where `R0` is the existing implicit shooting recurrence.
Restore each u/w coordinate as an exact finite zero-radius dyadic, with complete
7168-coordinate shape and at most four pairs. Retain its full original arithmetic
witnesses. These factors are **arbitrary inverse proposals**, not certified
inverses or exact secant solutions. Future point-image and orbit-proof truth must
not depend on claiming otherwise.

The first adapter can recover the existing pilot's state from `secant-outcomes`,
their hash-bound decision files and `proposal.json`; it need not multiply A/B or
numerically replay every historical update. It must interpret all decisions in
order, check their before/after active counts, exact u/w midpoint encodings,
original enclosure/midpoint consistency and fixed cap metadata. An explicit
factor-state snapshot in future outputs makes this deterministic restoration
simpler. Do not claim to have recomputed historical numerical admission gates
when only restoring their authenticated proposal records.

## Restoration rules that must be exact

| Retained operation | Restored active state |
|---|---|
| Accepted midpoint rank-one update | Append that exact u/w pair; prior count must be below 4. |
| Any rejected update | Clear **all** active pairs, even if rejected u/w fields exist. |
| `history_cap_restart` at length 4 | Clear history and discard this new secant. Do **not** re-admit the same pair after clearing. |
| `accelerated` proposal | Keep all current active pairs. |
| `baseline_no_history` proposal | Require and retain empty history. |
| Step-cap or nonfinite-step baseline fallback | Clear history; the most recent accepted-update record is no longer the active state. |

The initial continuation operation is exactly
`update(parent.before, parent.before_residual, parent.after, parent.after_residual)`.
That pair produced the parent's fresh result but has not yet been used in its
saved inverse history. Then propose from `parent.after`, save the complete
candidate and state snapshot, and perform a fresh eight-phase pass. If two
candidates were frozen in the plan, append the newly completed adjacent secant
and repeat once. There is no history subset selection, best-centre jump, silent
re-admission of a discarded pair, line search or retry. A complete worsening
candidate is retained chronologically; any change of that policy needs an
explicit new plan.

For example, three retained pairs can become four on the next admitted secant;
the following secant triggers the specified baseline restart. If the parent's
proposal already cleared three accepted pairs via its step cap, continuation
starts empty instead. Snapshots must distinguish `history_before_update`,
`history_after_update` and `history_after_proposal`, and identify the one next
pending centre pair. These distinctions prevent double-consuming a secant.

## Avoid repeated ancestry replay without erasing evidence

A small generic **admitted continuation capsule** can unify completed fixed-R,
initial secant and later continuation parents without changing their old readers.
The first capsule is produced only after full source/receipt/history admission
and independent review. It copies the complete current before/after data and
active factors, contains a hash-linked archive index of all original evidence,
and is externally pinned by the next committed plan.

Subsequent jobs authenticate that capsule and the complete immediate parent,
recheck every current tuple and factor shape/encoding, and process only the new
local transition. They need not re-decode all old phase images or recompute all
old secant arithmetic. The new capsule includes the prior capsule hash and every
new decision/input/output hash; an independent linear audit can still traverse
the full archived history. Every original artifact remains retained.

This is an explicit trust boundary, not a cryptographic proof that a validation
claim is true. The capsule's producer/source and its admission audit are part of
the externally reviewed trust anchor. A plain self-reported “validated” flag is
insufficient. If every old file must be checked for current disk corruption on
every resume, hashing still grows with archive size; there is no honest O(1)
way to detect mutations in unread ancestor bytes. Separate full archival checks
from local numerical continuation, and disclose which was actually performed.
Even an arbitrary restored inverse proposal cannot establish an orbit: fresh
rigorous residuals and final whole-box gates remain mandatory.

Initially reject an attempt that failed to complete all its planned candidates
as a parent, even if its first candidate completed. Preserve those complete and
partial raw outputs for a later explicitly reviewed salvage contract.

## Preferred resource implementation: independent phase workers

Once a candidate tuple is fixed, its eight `F(c_i)` evaluations are independent.
A separate controller can dispatch up to **four isolated CPU1/nice10 workers**
and assemble outputs in canonical phase order. The controller has no loaded
model. Each worker uses the same immutable source snapshot, map/asset pins,
2048-bit prefix cache and exact phase input hash; it saves full input, output,
zero tangent, timing and failure data in an exclusive phase directory. Do not
fork a live Torch/Arb model or share mutable caches/threadpools: start an isolated
Python subprocess and set limits before importing/loading the model.

Freeze the exact candidate, its successor bindings and manifest before starting
workers. Require exactly phases 0 through 7, with no duplicate, stale or skipped
result and successful per-worker receipts. Residual assembly uses the original
candidate's cyclic successors, regardless of completion order. Only after all
eight pass may the next secant/candidate be generated. Two candidates are
sequential stages, not sixteen independent precomputable tasks.

The parent reports about 55 GiB available and roughly 5.5 GiB measured peak per
point worker. Four workers suggest about 22 GiB of worker RSS, but simultaneous
load/conversion peaks and controller state must be measured. Retain the 18 GiB
per-worker address-space limit and add a roughly **32 GiB aggregate owned-process
RSS watchdog**, counting controller and descendants. Recheck available-memory
headroom before each wave and keep total owned compute within the shared
12-thread limit. Sum RSS conservatively; shared pages may be double counted.
Use a delegated hard group-memory limit if available; a sampled RSS watchdog is
not a hard cap and may overshoot between samples. Do not describe it as one.

Every worker has its own owned process group and source/PID/start identity.
A worker failure, aggregate memory breach or shared deadline stops only these
owned workers, with SIGTERM and bounded SIGKILL escalation; verify exit and
retain every completed/partial artifact. One controller-owned deadline bounds
the entire attempt, including proposal preparation, startup, output collection
and termination. Worker starts cannot reset or extend that deadline. Do not
admit new work when the remaining time is insufficient under the frozen policy.

For an initial one-candidate parallel pilot, retain a maximum **1800-second**
controller allowance; a separately planned two-candidate attempt could use at
most **3600 seconds**. The prior largest serial phase was about 190.8 seconds;
two ideal four-worker waves would be about 381.6 seconds plus preparation,
startup and collection. That is an optimistic scheduling calculation, **not a
measured parallel runtime**. Memory bandwidth, simultaneous checkpoint loads
and CPU contention can erase the expected speedup. Measure a single candidate
first, recording every phase's wall time, controller overhead and aggregate peak
before choosing subsequent concurrency or a two-candidate budget. No launch or
new resource purchase follows from this note.

## Small verification set before implementation approval

Use synthetic capsules and tiny maps to test post-proposal clears, the exact
length-four restart/discard rule, rejected-update records that still contain u/w,
zero/noisy secants, full-radius reconstruction, exact centre continuity and no
double-consumption. Test tampered capsule/receipt/source bindings and retained
regressions. For the controller test reordered completion, duplicates, worker
crashes, timeouts, memory-triggered cancellation, stale phase outputs and complete
exit verification. Compare serial and independently executed parallel synthetic
phase enclosures without asserting identical radius encodings. No target data
or running-pilot outcome is needed to establish these control-flow properties.
