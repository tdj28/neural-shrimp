# 13 October 2026: launcher address space before Git

This is a narrow prospective operational correction after campaign001 failed during launcher Git admission. It changes no mathematical evaluator, derivative domain, shard assignment, worker timeout, nice10 policy, controller limit or existing evidence. Inference retains all Git and failed-campaign closure ownership. No Git command, target evaluation, active-process action or target launch was performed while preparing this patch.

## Observed cause and preserved authority

Root identified campaign source `f599263f` and a1373682913-byte Git pack that could not be mapped under the launcher's inherited1GiB soft address-space limit. I read only the task-owned failure log:

`/data2/neural-shrimp-data/codex/llm-p8-certificate/derivative-campaign-001/run/launcher-000.stderr`

Its SHA-256 is `b59954f2d0facddef2a938c1e148be9cbca8bf21a1d8aa37eff9acfebd15c0c3`. It reports the pack mapping allocation failure and a `CalledProcessError` exit128 at `git fetch origin main` in `launch.py`. The pack size alone exceeds the inherited1GiB virtual-address allowance. Root reported no c064 attempt directory or target worker was created; inference is separately verifying/sealing the entire original campaign failure. This amendment does not replace or modify that evidence or claim an independently inspected process closure.

The controller deliberately keeps a1GiB soft/18GiB hard AS envelope. The old launcher established its18GiB bound only in the worker pre-exec callback, after Git admission. Git therefore inherited the controller's small soft limit. The remedy applies the already declared18GiB envelope to the launcher before Git rather than expanding the numerical worker's cap.

## Exact source delta

`launch.py` gains one small `ensure_launcher_address_space` function and one call immediately after the unchanged exact nice10 establishment, before argument parsing or any Git/source-snapshot subprocess. The helper reads the inherited hard AS limit. A finite hard limit below18GiB causes an explicit refusal without trying to raise it. An inherited hard limit of18GiB or larger, or infinity, admits `setrlimit(AS,(18GiB,18GiB))`. An inherited soft limit of1GiB or infinity therefore becomes18GiB, and a larger or infinite inherited hard limit is reduced to18GiB. A failed OS limit operation propagates as failure; there is no fallback with weaker bounds.

The worker's existing pre-exec callback still applies exact nice10 and the same18GiB soft/hard AS limit. The dispatcher is unchanged (`708eb618fa31afbab4e27ce0e834c16724a1d03985f1bfbef57bf8eb202bba03`), including its1GiB soft/18GiB hard controller envelope, sampled aggregate RSS and minimum-available watchdogs, sequential registration and ownership checks.18GiB AS is an address-space cap, not a memory reservation or RSS guarantee. Mathematical driver SHA remains `3fe44a583176e0551bb209093fec08cd059d055fcb1a57702ac8799c7cb7cc50`.

| File | Before | Prospective SHA-256 |
|---|---|---|
| `codex/certify-llm/launch.py` | `7c6fd77462e856f8f05753f0eb8b8c5c831a4a2ba43ac2daa692de3558371e81` | `0dc5bf5aef3b0351f5d65cbb389f668de0a4a11d52fa97c486fd3747b1ef4396` |
| `tests/test_codex_llm_nice_policy.py` | `aec6bf48968780d46a5e0f723fa85718430979e80167cf97bf626d3e41baea68` | `aa59a046c1bfbba202b6ca84b07a68c417cab6350850fee414813676175879e8` |
| `tests/test_codex_llm_launcher_as.py` | new | `4d6ac3b226ecc64f93d4d1c637c395cc35fdfe5ae4401c6598eb23fd5f314739` |

The existing nice-policy test changes only its expected limit-call count from one to two: launcher admission, then worker pre-exec. Before snapshots for non-Git delta inspection are `/tmp/llm-launch-before-pre-git-as.py` and `/tmp/llm-nice-tests-before-pre-git-as.py`. Historical reviews/amendments remain unchanged.

## Validation and release boundary

Six new controls cover inherited soft1GiB/hard18GiB, unlimited soft/hard, soft1GiB/unlimited hard, insufficient hard-cap refusal without `setrlimit`, refusal before any mocked Git or attempt creation, and actual launcher control flow giving every mocked Git operation18GiB before the worker receives the unchanged18GiB/nice10 envelope. The source tests use mocked Git/limit boundaries; no real fetch or model runs occur.

All69 launcher-AS, nice-policy, existing launcher and dispatcher controls passed in9.34 seconds, CPU/thread1 and nice10. Their tiny synthetic outputs are retained under fixed `/tmp/codex-llm-launcher-as-controls-20261013`; do not reuse this `--basetemp` path. No numerical gates changed. Root and independent source review, exact updated committed source/card admission, original-failure preservation and fresh task-owned execution remain prerequisites for another campaign. This amendment does not authorize a retry by itself.
