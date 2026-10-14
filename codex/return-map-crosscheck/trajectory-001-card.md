# Task19 trajectory001 operational run card

Reviewed 2026-10-14. Root accepted the complete scientific and operational
packet for one trajectory001 after private admission and exact fresh combined
preflight. The prepared history is preserved at9258b7b196be34b6ad063367240d4168bc7b7bda,
including prepared plan2c4f76d6 and cardcf0827c3. This transition changes only
admission/documentation; all scientific bytes and resource gates are unchanged.
The actual source commit and final plan/card hashes are bound by the immutable
preflight, capsule and startup. Saved001 remains separately prepared and may run
only after original trajectory closure and its own admission/capacity check.

## Fixed computation and resource basis

Exactly alpha .0655/log10T .1899492494639028, fixed embedding start,
1000 transient one-row steps then 2000 batch-three steps with base and absolute
1e-6/1e-9 renormalized twins, seed 0: 3000 native calls, 7000 row-forwards.
The steered prefix is included. Preserve every full-state input/return and
readout summary, both per-step growth/floor series, four disjoint 500-step
windows and twenty running 100-step means. Readout p(input) and saved F(input)
share the original clock. Primary fixed seven-way phase6 fit is reported with
explicit q==7 qualification; all actual phases remain retained. Compare actual
observed token IDs/cycle, not expected words inferred from period alone.
Sample cross-phase minimum distance stays distinct from centroid distance and
infinite-band separation. Every fit/candidate/root/derivative/rank disposition
and comparison delta is retained, including own finite-horizon disagreement.

CPU2 worker + one supervisor reserves 3 slots. Hard AS 12 GiB, exact nice10,
external 4200 s deadline + at most 10 s owned cleanup, output 512 MiB.
Startup MemAvailable>=36 GiB; runtime floor16 GiB. Root permits Task18 primary
3 slots plus separately admitted Task19 auxiliary3 slots (6 combined); two
12 GiB reservations plus12 GiB headroom motivate the startup floor. Current
combined availability still requires Inference's immediate preflight.

Measured basis is accepted prior Task14 CPU2 F7 forward 1.6170305 s, or
.23100436 s/row-step. 7000 scalar-equivalent rows predict1617.03 s; doubling
plus300 s gives3534.06 s, below4200 s. Batch-three throughput is unmeasured;
load/hash/IO/fits and different state costs are explicit uncertainty. Original
profile receipt75523fb2c0077fd5e6bb7835d430256be7b0084ade299816a8d0487650967dd2,
compact profile receipt8b94e265b463a5905722efd2cc749b11b47468289378ad23993033b035cbaf5b,
audit4b95510bdc5b3e5108fb6a2b43e7c2d94156ed4b1452aaf74b1fab00803cf791.
Prior AD peak5,859,160 KiB is a reference, not this batch's measured maximum.
About143.7 MB retained state/growth arrays plus small summaries/fit journals
are forecast below512 MiB. No extra profile or optional cell is included.
Model/config/vector hashes are checked before and after this attempt.

## Original scientific and input authority

This is an exploratory independent crosscheck after exposure to producer claims,
including the original 49/62 and outcome-informed amended 51/62 summaries.
The accepted design.md and five scientific files are unchanged. No producer
analysis/model wrapper is imported, and no supplemental cell is authorized.
The original input manifest is
/data2/neural-shrimp-data/codex/llm-return-map-crosscheck/input-authority-001/manifest.json,
SHA d3014365b8445f8508271959137b922f694f0519665e9dfcb8f459e49517d2e8.
All ten original files are hashed before and after computation. Snapshot
immutability does not establish historical producer writer closure; original
completion remains supported by the disclosed source/log authority.

| Captured scientific source | SHA-256 |
| --- | --- |
| codex/return-map-crosscheck/reduce.py | d87ec765cde3c0cc1145e6225b9b98b9a9c85468df85368c938c571aebda098b |
| codex/return-map-crosscheck/trajectory.py | 8da808f5413d4f7caca49b6c0cba4c8182ff4322c91e123513e9ccb3e9737524 |
| codex/return-map-crosscheck/runtime.py | 41f0f19f023f2bbd5f6842929bb07cad70b86ea8d0399eec717a22ae478d468a |
| codex/hostile-fold-crosscheck/assets.py | 06477ee36a78f15ee8f1e8ae196dc53ae230ff19368fbda638eb6a6e0ef758e0 |
| codex/crosscheck/qwen_map.py | 493a3d1d57fec227f5ba2e3f8f9cfcfdd02d22495d1d44e6449ede901310ae6e |

## Admission, original ownership and closure

Inference alone privately checkpoints the exact source/card/dispatch and the
admission-only reviewed plan, records the admitted plan hash and actual source
commit, and launches after Root's exact review and fresh combined preflight.
The prepared plan refuses launch. Before launch check a clean owned checkout,
private remote ancestry, all source/input pins, absent attempt and supervision
paths, at least 16 GiB free disk, current CPU reservations and current RAM.
Coordinate against Task18's exact live identities/command or authenticated
closure; never identify an owned process merely by a matching filename.

Use /data2/neural-shrimp/.venv/bin/python, -I -B -u, NumPy 2.5.3 and the
already admitted backend (trajectory also Torch 2.14.1+cu130), CPU only/offline.
The ten-file capsule is the five scientific files above, unchanged
codex/certify-llm/derivative_campaign.py SHA
9357d0aee5e5880b37c752a5c928c988480c63a1871db94f1fc3f8b3df70ee52,
launch.py, this card, the corresponding plan and corresponding dispatch.
Each byte is compared with the exact committed source. The dispatch copies
itself and external source/plan pins into a fresh exclusive supervision
folder; persistent exclusive launcher.stdout/stderr are installed before
setsid --wait. There is no additive outer nice: the launcher idempotently
establishes exact nice 10 before Git or child work, refuses inherited nice>10,
and checks actual worker nice and hard/soft address space at startup.

Launcher registration retains exact PID birth, parent, UID, process group,
session and pidfd ownership; command admission is bounded. The unchanged
birth-bound helper signals only its original worker group. External deadline
or failure stops that group, then allows at most 5 s TERM + 5 s KILL/wait
cleanup. The coordinator reaps the dispatch/launcher and verifies original
identities/groups are absent. Uncertain cleanup produces a pending record
and prohibits retry or success. Original stdout/stderr, native attempts,
source, every failure and all output inventories remain retained.

Worker AS is hard; runtime RSS/headroom/output checks are sampled and are not
hard guarantees. The outer monitor also checks headroom and the whole attempt's
output bytes. The 512 MiB output limit includes source capture; the worker
leaves 4 MiB for failure metadata. Final receipt/closing metadata and separate
supervision logs must be accounted for by the coordinator. No automatic retry,
cap extension or threshold change is authorized.

Computational completion is not scientific agreement. The worker must finish
its exact finite rows and status contract; negative, irregular, undefined and
producer-disagreeing scientific dispositions remain visible. All rows/claims
require subsequent saved review. certificate_passed and experimental_claims
remain false; no interval, exact-chaos or conjugacy theorem is asserted.

## Exact executable and output identity

Launcher SHA 259e056e695dbe3aa5b9112a793e9272360371d4efd5f1833d12cdaec56a546c.
Dispatch SHA 3f84bf1e005baec34b3f377fe098798a02d37812bfa5219fcafce9f6b78c3ecf.
Only attempt `trajectory-001`, plan `trajectory-001-plan.json`, and mode `trajectory`
are accepted together. Raw root is
/data2/neural-shrimp-data/codex/llm-return-map-crosscheck/trajectory-001;
original outer logs are in sibling trajectory-001-supervision. Both must be absent.
From the privately checkpointed owned worktree, the coordinator runs
`bash codex/return-map-crosscheck/trajectory-001-dispatch.sh SOURCE_COMMIT ADMITTED_PLAN_SHA256`.
The outer script calls the exact launcher with --attempt, --source, --plan and
--plan-sha256; the captured worker receives --plan and --output and exact
TASK19_SOURCE_COMMIT/TASK19_PLAN_SHA256 environment pins.

Twenty-one focused nonmodel controls passed in1.44 s. They cover both mode
caps/capsules/prepared refusal, exact nice0/10/above10 behavior, hard-limit and
headroom refusal, negative scientific outcomes, partial completion rejection,
real tiny success/failure/timeout closure, and forced setsid fork/wait with
persistent exclusive logs and exit propagation. No Task19 target, model,
producer-array or Git action occurred in these controls.

## Exact combined preflight

The coordinator uses /tmp/task19-trajectory001-preflight.py SHA-256
6478883f9c790df7d04d1099062814bb5857305ab238fe95303e5a9d0502bdfb under
CPU1/nice10/12GiB and an external60s limit. It binds all ten private capsule
files and ten immutable input-byte rows, without decoding numerical payloads.
It binds Task18 startup542fb95b and all three current birth/UID/parent/group/
session/command/AS/nice identities; worker OS threads1..3 and control threads1.
Only the exact known Torch deprecation warning e6ff90d4 is permitted and its
full bytes are retained; any other diagnostic or terminal phase refuses this
live-overlap admission. Fresh36GiB available memory,16GiB free disk, no duplicate
Task19 owner, six total/three auxiliary CPU slots, and read-only GPU inventory
are explicit. No pod action or scientific output is part of the preflight.
