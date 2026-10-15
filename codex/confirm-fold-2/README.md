# Task 5b (PT2): a fresh prospective fold-passage test with the continuous predictor as primary

Adapted from `codex/confirm-fold/` (Task 5). The map, checkpoint, prompt, steering vector, independent float64
implementation (`codex/crosscheck/qwen_map.py`), fold gates, coefficient definitions, entrances and endpoint are those
of Task 5. Three things differ.

- **Primary predictor.** The continuous passage integral of the frozen normal form,
  T_arc = integral from u_in to U of du / (a eps + b u^2), unrounded. Primary success: |N_measured - T_arc| <= 1 for
  every row. The discrete count of u' = u + a eps + b u^2 (Task 5's primary) is the secondary, with its own verdict,
  never substituted. Task 5 found the two forms 2.4 iterates apart at its failing entrance before any outcome
  (`codex/confirm-fold/arctangent-scope.md`).
- **Distances.** Per target, eps = q^2/(a b) for q in (0.08, 0.05, 0.03, 0.018), chosen from the scalar predictions
  only. Every frozen full-channel forecast is at least 16 iterates and every midpoint forecast at least 8 (Task 5 had
  seven forecasts of at most 5 iterates). `build_design.py` refuses a row below the floors.
- **Fresh targets**, never used for passage tests (not the mechanism fold at log10 T = -0.115, Task 5's A or B, or
  Task 7's hostile fold):
  - **C:** the 2-cycle ("is it") fold at log10 T = -0.200, active parameter alpha. Seed: index 12 of
    `explore/llm-soft-loop/pod64/curve-p2-fold-plus/states.npy` (alpha 0.071332, log10 T -0.19918, second modulus
    0.091 on the producer curve). The fold curve runs nearly along log10 T there (d alpha / d log10 T ~ -0.05), so the
    alpha line crosses it transversally.
  - **D:** the period-6 window fold (G = F^6), a different orbit family, at log10 T = -0.0874, active parameter
    alpha. Seed: index 45 of `explore/llm-soft-loop/pod64/p6z3/curve-foldD-plus/states.npy` (alpha 0.075633,
    log10 T -0.08744, second modulus 0.0037 on the producer curve; d alpha / d log10 T ~ 0.15), far from the located
    cusps.

A secondary sign hypothesis is frozen with the plan: measured minus discrete count <= 0 for every midpoint and
transverse-kick row (the referee's observation on Task 5). It is reported, not a criterion.

Protocol: setup (`fold_setup.py` through `launch.py --stage setup`) -> `build_design.py` writes `design.json` with every
frozen prediction -> design committed and pushed -> passages (`launch.py --stage passage --freeze <commit>`) -> audit.
Runs execute from a git worktree pinned at the commit, so other commits to `main` cannot move HEAD under the launcher.
Owned output: `/data2/neural-shrimp-data/codex/task5b-fresh-prediction/`. CPU only, nice 10, at most two workers of
four threads, 18 GiB address space each. No paid compute.
