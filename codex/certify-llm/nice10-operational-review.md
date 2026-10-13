# Independent review of the exact nice10 amendment

13 October 2026 UTC. **Approved as a narrow prospective operational correction; no blocker found.** No target process, captured source, model, Git state or existing source was changed by this review.

The handoff requires Codex CPU processes to run with `nice -n 10` (`docs/handoff-codex-2026-10-12.md`, lines45–46). The earlier reviewed controller's nice0 exception did not meet that policy. The dated amendment correctly supersedes that exception while preserving the historical source/review identities.

I compared the complete before/after files against the preserved `/tmp/llm-launch-before-nice.py` and `/tmp/llm-campaign-before-nice.py`. The only code changes establish/require exact nice10. Launcher entry raises lower inherited nice values to10 before any Git/snapshot subprocess, verifies success, leaves10 unchanged, and refuses values above10 without attempting a priority increase. The same idempotent helper in worker pre-exec preserves10 and the unchanged18GiB limit; it cannot add10 twice. The dispatcher requires10 before time/admission/output work. Future dispatcher, launcher, Git children and numerical worker therefore inherit the intended priority under the controlled unchanged child paths.

The mathematical evaluator, domain, cache, arithmetic, source snapshots, timeouts and ownership logic are unchanged by this delta. The already-running first shard retains its immutable old captured launcher and evaluator; its audit must use that original source. This review performs no signalling or renicing and makes no new assertion about the live worker. A measured first-result authority can be reused only under the existing independent audit and exact evaluator/domain/prefix checks.

I independently ran the nine new mocked controls: **9 passed in0.10 seconds**, CPU/thread1 and nice10. They exercise inherited0/10/12, failed establishment, repeated helper calls, priority before mocked Git, captured worker pre-exec and early controller refusal, without launching targets. The separately disclosed transient `/proc` command-admission failure is not repaired or hidden by this amendment; the existing fail-safe uncertainty/cleanup boundaries remain.

Source SHA-256 bindings:

- Launcher: `7c6fd77462e856f8f05753f0eb8b8c5c831a4a2ba43ac2daa692de3558371e81` (before `07b9afff4236d71cca476b52896406710865344f7566208742d165a28fa40827`).
- Dispatcher: `5266776c8618a27d6c97772e4aa7018983b5bb1407872fbb315b9f7fc17542a9` (before `0cc8a192737508b449472eb7d20562c6d84abf84b07565de7e0108d712cea161`).
- New tests: `aec6bf48968780d46a5e0f723fa85718430979e80167cf97bf626d3e41baea68`.
- Amendment: `68399742023e0cee9108a6a09f04118e4335d9a5f297c61254ef2de8553b51df`.

This source signoff does not authorize a derivative campaign or replace its measured resource card.
