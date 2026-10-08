# Matched-pulse input qualification — 8 October 2026 UTC

**Decision: no-go for pulse collection.** The four local fork states and recovery maps are recoverable exactly. None of the audited control candidates has the full history, support and regime qualification required by the design. Keep the comparison stopped rather than calling summary labels qualified controls.

This is an exploratory saved-data audit of consumed S2-28011 development artifacts. It evaluates no model, trajectory, correction, HVP, pulse, numerical profile or holdout. [Receipt](utility-input-qualification-receipt.json) and [prospective rules](../plans/utility-input-qualification.md) bind the source, every selected input and all raw outputs.

## Local inputs

All four forks are `state[4100, origin]` in the dwell100 common traces: pair ends after **8202 updates**. Recorded alpha at the nominal zero level is slightly negative, so the first nonnegative recorded level is selected without rounding. Origins0/1 are near-orbit/ordinary initialization. The normalization uses loss rows3101..4100, both losses per row; the loss at the fork is row4101, first loss, before its next update. Full state bits, prefix hashes and recovery-reference bits are retained.

| Point / origin | Fork state SHA-256 | Prefix loss mean | Distance to crossing reference |
|---|---|---:|---:|
| hard / near_orbit | `55c3effe673c620fd4d6be8b8b0276dee5d6c8d5564ee8e5ea50c0bf3c35e075` | 0.00553464559 | 0.00031039667 |
| hard / ordinary_initialization | `bb49b69dd75af049fc3bad609027b42c6d29821b0912f46e949ea2caad5adae2` | 0.00708156098 | 1.31505181 |
| soft / near_orbit | `cf9e59e755e2626c35a235c3fd7eac0a3cee05846df3d5ad9e52f40c4d668762` | 0.00462477303 | 0.00148055923 |
| soft / ordinary_initialization | `1cfb60ebc30e10609137b35114a7276b96621055cfb40b6deead65a4980fad8d` | 0.00462477739 | 0.00139375182 |

Recovery rate pairs are **(78.71189928623194, 0.3626495805545865)** at the hard point and **(47.72960569225131, 0.5032628489431)** at the soft point. Their exact binary64 bit strings and log coordinates are in the receipt. Each uses its saved reference[0], with log multiplier near -0.0015. The original two starts remain separate: the hard ordinary fork is far from the local reference; neither it nor an inconvenient support failure may be excluded.

Warmup stores pair-end states and both losses, but omits intermediate states, fork gradients and full reference spectra. Thus the actual one-update group norms, all-phase prefix oscillation, pulse-dose feasibility and complete recovery-map spectrum are not established by this inventory. Pair-end motion is especially misleading here: its RMS is about0.00040 versus0.928 at the two hard forks, which cannot be substituted for one-update scales. The old floating-point audit establishes its stated sampled consistency; it does not fill these missing fields.

## Fixed candidate inventory

The 0.20-decade distance filter leaves **284 periodic and238 chaotic candidates** across the two consequence slices and EXP028 source records. Labels retain their original numerical scope. A deterministic mean-loss rule gives the following four nominees; all pass only the preliminary unequal-window mean-loss screen, not full covariate support.

| Matched point / candidate | log10 hidden, output | Available evidence | Missing state/history |
|---|---|---|---|
| hard / `s2-28011-out-0.0856:127` | 1.634779706275033, -0.08562528015073712 | period2 tail screen | no full state or tail exported |
| hard / `EXP028-chaos:68` | 2.200446788879346, -1.059235430093067 | FTLE4k/10k=0.178184,0.187355 | coarse20k terminal only; no qualifying tail |
| soft / `s2-28011-out-0.0856:0` | 1.55, -0.08562528015073712 | period2 tail screen | no full state or tail exported |
| soft / `s2-28011-out-0.0856:419` | 1.829706275033378, -0.08562528015073712 | two tangent directions at8192 updates; final on-attractor separation | no full state or tail exported |

Two stronger full-state periodic alternatives exist: EXP028 records36 and125, at log-rate pairs (1.2475056124087576,-0.4121766065636553) and (1.7102968465494726,-0.09420659502962986). They retain both phases, primitive closure and attracting spectra (radii0.9973624962 and0.9927565926); exact hashes and all relevant tolerances are in the receipt. Their `other` component label does not establish an independently traced ordinary branch. The coarse final states are not the refined orbit phases or a retained 1000-pair history. Do not silently swap these for the nominees after a failed gate.

The consequence runner computed4096 tail states and8192 losses in memory but exported only summaries. Its two FTLE values are two directions at **one horizon**, not a two-horizon/block robustness test. EXP028 has original/rerun horizons and a20k final state, but omits the rerun terminal trajectory, block exponents, on-attractor growth and precision control. No nominee establishes its regime at the proposed common recovery backoff. A lower-dimensional projection or a visual period band does not repair these gaps.

## Smallest next decision

First, only if separately authorized after review, evaluate **four training gradients at the exact local states**, without advancing any state. If any hidden/output-only dose would exceed25 percent, stop this fixed pulse candidate before buying control acquisition. A proposed five-second internal cap is ample relative to prior small-network workloads but has not been measured for this exact gate. No gradient gate or profile ran here.

Clarification: the old formula equates treatment-minus-sham displacement, `||theta1_arm-theta1_sham||2=R`; it does **not** equate total GD step lengths. Retain those total lengths as a generic explanation. The unchanged horizon is one pulse update followed by8000 common recovery pairs.

If dose feasibility passes, the missing work is specific: recover the four fixed controls with complete per-update states and loss histories; obtain full ordinary-periodic branch qualification or the declared chaos4096/8192/block/perturbation/precision checks; qualify original and recovery regimes; compute the predeclared one-update, all-phase amplitude and transient-response covariates. The same factor2 support bounds must hold for both origins at each point. No post-outcome relaxation or alternative-coordinate search is authorized. Only a reviewed resource profile could then support a pulse launch.

This audit measured **0.194s** internally / **0.304s** externally and **74,520KiB** peak RSS, CPU1nice10. The old warmup measured63.649s for1,232,000 batched cell updates (about51.66microseconds per cell-update); that is historical context, not a throughput guarantee for separate qualification or larger models. All 738,699 bytes of manifested audit outputs, plus manifest/launcher records, are preserved read-only. Source `a921b776953b5fc57d4a51469899022aec56d12f`; manifest `2f962ced4b6f32aca588f1012985591acc670437ff944e8e15375acf4fd2e03f`.

This no-go result avoids an uninformative pulse comparison with weak controls. It does not show that pulse direction has no effect, and it adds no LLM or geometry-specific utility evidence.
