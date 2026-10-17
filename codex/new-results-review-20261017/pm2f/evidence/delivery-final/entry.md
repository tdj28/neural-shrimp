## 2026-10-17T01:03:02Z | from Codex | Task 27 item 1 — PM2F saved-scalar audit complete

The endpoint arithmetic reproduces exactly: **42/42 published numerical model fields**. For R1-Qwen, R1-Llama and QwQ respectively, endpoints above/below nominal `eps0=1e-10` are **40/216, 145/111 and 34/30**; answer flips are **6/256, 47/256 and 7/64**. All 288 base correctness and 576 twin flip/switch flags reconstruct exactly. Full report: `codex/new-results-review-20261017/pm2f/scientific-review.md`, SHA256 `0be172018c5ca40943787234fb3ea4b759fa0719d30a1c72842b5701846bf461`; private evidence commit `1589fb98799a937ffb461bb5279018488ca54e19`.

The definition matters. Positive fitted slopes occur for 115/256, 200/256 and 50/64 twins; **76, 57 and 16** of those finish below nominal eps0. The separately declared fitted-sign projection therefore retains six definition differences; it was not selected by agreement. QwQ/R1-Llama fitted medians approximately **0.009695/0.008202** describe all **53/229 defined fits**, not just the endpoint-growing subgroup. The later published `pi_f64_reason_summary.py` (`5a507a2a`, SHA `bc8f4b90…`) explicitly confirms endpoint semantics, but does not authenticate the older historical invocation. The original missing-source prose remains preserved with this dated clarification.

Please qualify the frozen main-paper lines 47–50 and 930–937, and the current Chaos abstract lines 51–52:

- Say **nominal relative injection** and **endpoint below nominal eps0**, rather than a measured initial separation or “closer than they started.” Realized initial distances were not saved.
- Keep endpoint counts separate from slopes fitted only in `10 eps0 <= d <= 1e-3`. Native distance histories and fits were not independently recomputed; scalar classifications and answer-value flags were.
- Fourteen R1-Llama and seven QwQ twins switch before their base, so the base-defined window can mix inference phases. These are finite pipeline-sensitivity observations, not an asymptotic chaos finding or exclusive causal attribution to rounding/near ties.
- For the two larger flip rates, name the models and use approximately **10.9% and 18.4%**, not an unqualified 11–18% extremal range. The twins of a problem are dependent; no new population interval is claimed.

The successful own run002 is source `8f4e96a1`, report `85e11fa9…`, receipt `b8c4967d…`, original closure `b42a05bd…` under `/data2/neural-shrimp-data/codex/new-results-review-20261017/run-002/`. Ten manufactured groups passed. Run001's duplicate-history alias failure happened before target analysis and is retained; the separately reviewed one-line snapshot correction preceded run002. All eleven original producer attempts, including eight C startup failures, remain associated with their original source/retrieval/termination records. Loaded weights/kernels and missing native trajectories are inherited limitations.

Root accepted the full independent note and its finite scope. All original outputs and both full comparison lists remain immutable. The supervisor measured **0.340606 s external elapsed**; the worker recorded **0.174160 s through posthash before final report** and **38,508 KiB RSS then**, not an independent whole-worker peak. Exact original owned processes/groups are closed. Raw coverage and all-row table remain hash-bound outside Git.

Task26's frozen number-candidate scan and E5 are separate bounded assignments; later Task27 units remain conditional on completed receipts. No recurring watch has been restarted. Producer paper/tools/log files were not edited.
