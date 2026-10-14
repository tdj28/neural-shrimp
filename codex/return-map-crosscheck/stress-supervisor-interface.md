# Task19 optional stress supervisor delta — 2026-10-14

Prospective operational source only. No optional trajectory/model/profile/target data or Git operation was executed. Inference owns concrete plans/cards, private checkpoint, current resource/combined preflight, adoption and launch. The accepted mandatory sources and original attempts are unchanged.

New proposed repository files are `codex/return-map-crosscheck/stress_launch.py`, `stress-cell04-001-dispatch.sh`, `stress-cell13-001-dispatch.sh`, plus focused `tests/test_codex_return_map_stress_launch.py` if adopted. `tests.py` is that test source in this packet. `operational.diff` is against accepted launch259e056e and trajectory dispatch3f84bf1e, not an instruction to overwrite them.

## Exact interface

`stress_launch.py --attempt stress-cell04-001 --source <40hex> --plan stress-cell04-001-plan.json --plan-sha256 <64hex>` and the corresponding cell13 form. No other plan/attempt/retry index is accepted.

`specification(plan_name)` returns schema `codex-return-map-stress-v1`, mode `stress`, exact cell_index4 or13, fixed attempt, worker `stress_runtime.py`, six-source inventory, `stress-cellNN-001-card.md`, `stress-cellNN-001-dispatch.sh`, resource_probe_only=false. Plan/card bytes are supplied and privately frozen by Inference later at those exact basenames; this preparation neither creates nor promotes a card.

`snapshot(repo,args,deadline,stopped)` keeps the existing signature and returns `(plan, blobs, remote_main)`. It captures exactly **11** files: six scientific files, unchanged derivative_campaign ownership helper, stress supervisor, selected plan/card/dispatch. It checks original committed bytes, full source hashes, private remote ancestry and fresh headroom as before. Added admission binds exact cell_index, stress schema/mode and all six literal scientific pins, including adapter **b125e6b5c09a83aee18eb1550611b2b719d10d1063f846286ee528cb6c88effe**. Prepared plans refuse. Runtime still performs its complete fixed experiment/input/source admission before native work; the launcher does not substitute a scientific result for that check.

Caps remain CPU2+one supervisor, exact nice10,12GiB hard AS,4200s absolute launcher deadline plus at most10s owned cleanup,512MiB sampled total output,36GiB startup/16GiB runtime MemAvailable. The existing idempotent priority/AS setup remains before Git/worker. Native worker environment/source/plan bindings are unchanged. RSS/headroom/output watchdogs remain sampled rather than hard claims; AS is a hard bound.

`validate_completion(status,result,mode,source,index)` preserves the existing computational contract/counts, both offsets/post-asset check and all four windows/20 running means. It additionally binds status/result/optional_identity to the exact cell and original comparison parameters. It deliberately does **not** require actual period10/28, positive exponents, a fitted map or any producer claim. q0/period mismatch/negative scientific findings remain valid completed outcomes. Outer status keeps `return_map_completed` or `return_map_failed` and now records exact cell_index; original failure/uncertain ownership semantics are unchanged.

Each dispatch takes exactly source commit and plan SHA arguments. It exclusively creates its fixed `stress-cellNN-001-supervision` directory, preserves executed-dispatch bytes and pins, uses persistent exclusive stdout/stderr, and invokes **setsid --wait** on `stress_launch.py` with the exact plan/attempt. No additive `nice` is introduced. It therefore waits through a forced setsid fork and returns the owned launcher's status as in the accepted original dispatch. Both fixed raw attempt paths must remain absent before their separate preflights; no launch or retry is implied.

## Narrow controls

Eight controls pass in0.04s, CPU1/nice10, without model/raw data/Git/subprocess execution. They cover both exact specifications, computational completion despite q0/negative science, identity/count/window corruption, cross-cell/arbitrary retry refusal, mocked exact11-file snapshot/private source path, prepared/wrong-cell/wrong-adapter-pin refusals, and exact dispatch substitutions. AST equality confirms unchanged limit/priority/ownership/cleanup/watchdog/CLI functions; execute differs only by passing/recording cell_index. Existing real-process lifecycle controls are not repeated because their code is unchanged.
