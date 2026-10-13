# 13 October 2026: bounded launcher command admission

This separate operational amendment follows the nice10 correction. Root requested it before preparing the remaining111 campaign because a transient `/proc` command read at synthetic launcher startup had caused a safe but unnecessary campaign abort. No Git, target/model evaluation, real-shard artifact or existing scientific process was touched.

## Narrow control-flow change

A `Popen` child enters the ownership table immediately, as before. Direct parentage, UID, session/group leadership and a birth-checked pidfd are established before considering its command. That identity is retained for cleanup; it does **not** mean the command has been admitted. While `launcher_command_admitted` is false, the dispatcher records distinct observed command strings and waits only within the original `registration_seconds` interval measured from the original spawn. Exact list equality with the expected command is unchanged. Both a persistent mismatch and an exact command first observed after that deadline fail.

`register` cannot read/admit a worker manifest until exact launcher command admission succeeds. Consequently no next launcher is dispatched before both exact command admission and full existing worker registration. PID birth, parent, UID, session/group, worker command, nice10 and actual AS gates remain intact. No deadline is reset, no retry is introduced and no source/plan/domain comparison is relaxed.

Cancellation distinguishes command admission from cleanup ownership. An immediate direct child whose birth identity and pidfd are known remains ours to inspect/stop for safe cleanup even if its command never matched. The existing stopped-parent/child recovery protocol still applies; unknown children and absent identity remain explicit uncertainty, not accepted ownership. A cancellation-only recovered worker does not turn a command mismatch into a successful shard or permit future dispatch.

## Exact identities

| File | Before this amendment | After |
|---|---|---|
| `codex/certify-llm/derivative_campaign.py` | `5266776c8618a27d6c97772e4aa7018983b5bb1407872fbb315b9f7fc17542a9` | `708eb618fa31afbab4e27ce0e834c16724a1d03985f1bfbef57bf8eb202bba03` |
| `tests/test_codex_llm_derivative_campaign.py` | `194115a938806b6e53be0a4c9e94ae870cdff48c6861e5cc01cad09702e509a3` | `fbd351342c1df5274fd3a345ab3e653abfc895518bd2c43381a71164c60038a7` |

The nice-corrected launcher remains `7c6fd77462e856f8f05753f0eb8b8c5c831a4a2ba43ac2daa692de3558371e81`; its focused tests remain `aec6bf48968780d46a5e0f723fa85718430979e80167cf97bf626d3e41baea68`. Mathematical evaluators and historical reviews remain unchanged. Non-Git before snapshots for exact diff inspection are `/tmp/llm-campaign-before-command-race.py` and `/tmp/llm-campaign-tests-before-command-race.py`.

## Controls and evidence disclosure

Four additional deterministic real-process controls substitute only the first launcher's observed `/proc` command while keeping actual birth/session/UID/PID checks: transient empty→exact, transient mismatch→exact, persistent mismatch, and exact command arriving after the registration deadline. The transient cases admit only the eventual exact command and delay the next dispatch; the failure cases stop future work and verify cleanup. Existing failure, signal, memory, source, PID reuse, pre-registration, dead-leader descendant and receipt controls remain.

All63 combined nice/launcher/dispatcher controls passed in9.67 seconds. The unchanged test source was then run once in a fixed non-rotating base directory to retain the control evidence: **63 passed in9.56 seconds**, CPU1/nice10. Those synthetic outputs remain at `/tmp/codex-derivative-command-admission-controls-20261013`; do not reuse that `--basetemp` for another run. No model, target artifacts or Git/network operations were used.

The original incidental failure was reported before this fix and read into tool output: `ValueError('actual launcher command differs')` at initial `ensure_leader`, result `campaign_failed`, completed1/started3/undispatched0, no uncertain ownership. The subsequent ownership snapshot recorded the exact expected third-launcher command and all owned processes were cleaned up. Its original auto-managed directory was `/tmp/pytest-of-d7082791602/pytest-1190/test_real_owned_overlap_sequen0/campaign`. **Pytest's automatic temporary-directory retention later deleted that directory during subsequent test runs.** The retrieved transcript records remain, but the original files are not claimed present or reconstructed. Root was informed immediately when this was detected. The fixed-path controls above preserve deterministic reproductions of admission/refusal behavior and are not substituted for that historical original. No scientific raw evidence was lost or changed.

Release requires Root and independent source review. No campaign card, source commit or target launch is supplied by this amendment; inference remains the sole Git owner. The actual first-shard audit and frozen measured campaign admission remain prerequisites.
