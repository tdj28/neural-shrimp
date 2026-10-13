# Task7: independent hostile 2-cycle fold cross-check

Preparation,13 October2026 UTC. Task1's ongoing certificate and Task6 have priority. This directory reuses the unchanged independent float64 Qwen tensor implementation and independent fold solver; it imports no Claude analysis tool. No target model or passage run has been performed for Task7.

## Scope and existing information

The assigned map uses the pinned Qwen2.5-0.5B-Instruct checkpoint, fixed37-token chat prefix, one appended soft embedding, full151936-token expected-embedding feedback, and layer12 steering at every prefix/current position. The changed vector is exactly `vectors-l12-p1b/vectors.npz["hostility-vs-kindness"]`. All896 state coordinates remain in the calculation. No ordinary discrete-dialogue behavioral claim follows from this map alone.

Producer metadata already read for comparison and initial-guess admission:

- `/data2/neural-shrimp-data/explore/llm-soft-loop/pod64/hnormal/normal-form-p2/normal-form.json`: alpha0.083, period2, log10T−0.04624653570443442, residual7.8047e−11, critical multiplier0.9999999954391827, next modulus0.050015737814941875, a−1.1683279172118877, b44.52577938671375 and ab−52.020711093115274. These are producer values, not independent results.
- The seed is row0 of the producer's `hnormal/curve-p2-fold/states.npy`, SHA `d574d0740da3d026276c508527b6001e642c35daace57da7755ec48fd4036123`. It contains three binary64[896] rows; only the first seeds this check. No producer eigenspace or coefficient is imported into the solver.
- Hostility vector NPZ SHA `d18395239dba63c7deacb8aa9431d768269617c9cc44a98f89cc1e22423b5958`; the selected vector is finite binary64[896]. The loader preserves its stored scale. Checkpoint/config hashes and exact prefix IDs come from unchanged `codex/crosscheck/qwen_map.py`.

This is an exploratory independent reproduction aiming at evidence rung2. Known reference results and the producer's initial guess are disclosed. It is neither a new record claim nor a fresh-data confirmation.

## First bounded resource pilot

`resource_pilot.py` is preparation only. Its externally pinned plan fixes CPU2, nice10,12GiB AS and300seconds. Require at least28GiB MemAvailable before admission (12GiB Task7 ceiling +4GiB Task6 allowance +12GiB certificate headroom); yield on sampled MemAvailable below16GiB. The worker checks between operations. The reviewed outer launcher must also monitor headroom during operations, enforce the absolute deadline, identify/terminate only its worker and preserve its source, logs, failures and complete receipt. Neither the worker nor this note authorizes a launch.

Historical measured independent Task2 profiles were5.53GiB RSS, and5.98GiB for its larger128-direction batch. The prior dense fold took roughly700seconds onCPU4. These motivate a small CPU2 pilot; they do not establish present throughput or nested-AD peak memory during certificate contention.

The pilot loads the pinned model once, records load time/RSS, and exercises exactly one composed F² value, one signed composed JVP, one16-direction composed JVP batch, the log10T parameter JVP, a nested state JVP and the reverse-over-forward critical-gradient graph needed by the existing fold corrector. Deterministic unit probes are cost probes, not asserted critical modes. It saves all full outputs and timing/RSS events. It performs no dense896×896 solve, orbit search or passage trajectories, and always records `fold_qualified:false` and `certificate_passed:false`.

The source bundle consists of the two new modules plus unchanged `codex/confirm-fold/fold_setup.py`, `codex/crosscheck/qwen_map.py` and `codex/crosscheck/numerics.py`. The outer snapshot must include all five at the pinned commit and exact plan; its environment supplies `TASK7_SOURCE_COMMIT` and `TASK7_PLAN_SHA256`. The pilot has no Git operation or model-loader substitution through producer tooling.

## Subsequent numerical work, gated on pilot/capacity review

1. Independently refine the augmented F² fold at fixed alpha0.083 in log10T, retaining every corrector, full Jacobian, iteration count and failure. Use the existing bounded independent solver; choose its resource ceiling after the measured pilot. No default index or producer result substitutes for convergence.
2. Require small full-state closure at both phases, nonzero proper-divisor return, an isolated real+1 crossing with its argument and full spectrum, a complementary modulus below0.95, and small left/right eigenpair residuals. Classify before computing coefficients. Retain full state/modes/matrix.
3. Check the composed float64 F² derivative, not merely a single step: independent dense action versus JVP and a central-difference ladder with measured convergence above its floor. All RMS, rotary, softmax and derivative graph operations remain binary64. No float32 values enter these decisions.
4. Compute a=l·∂pG and b=½l·D²G[r,r] by nested AD, with p=log10T and G=F²; compare each finest FD estimate separately, not only a potentially cancelling average. Record the left/right gauge ||r||₂=1 and l·r=1. Retain raw signs, then flip both modes if needed to use b>0. Report the parameter's exact units and every disagreement from the producer. Compare the sign-invariant ab, |ab| and sqrt(|ab|)/π alongside a,b.

The free-temperature derivative includes ln(10)T automatically. Alpha is fixed, so the prefix is constant with respect to log10T; the reused AD graph nevertheless computes the complete map consistently. The pilot's synthetic tests check temperature, composed state and mixed derivatives on a tiny architecture against independent finite differences.

## Optional passages remain separate

Only after clean fold/precision gates and an explicit new capacity review may six controlled full/midpoint passages at physical offsets p−p_c=−1e−5,−3e−5,−1e−4 be planned. With a<0,b>0, these negative offsets produce motion from negative u toward positive u. Use U=0.1/|b| and record full−U→+U and midpoint0→+U conventions and strict exit/censoring rules. Counts are F² iterates, with underlying forward steps recorded separately. Freeze the discrete-quadratic forecasts before these trajectories; keep the arctangent approximation separate.

Such controlled starts can cross-check a local passage law. They do not independently measure natural reinjection in the producer's P16–20 windows and do not establish global shrimp organization. All missing, failed or censored rows remain reported; the known-target cross-check is not relabeled confirmatory after success.

## Preparation checks

Fourteen synthetic controls passed in2.65seconds, with no pretrained model or target evaluation. They cover exact resources/refusals, pinned vector selection and invalid arrays, reused composed/log-temperature/nested/mixed AD against tiny finite differences, full pilot outputs and early resource-stop behavior. Isolated `-I -B --help` works. The first synthetic run had13 passes and one fixture/import-path failure; moving the resource check before all profile imports corrected the fail-fast behavior, after which all14 passed. No research outcome was generated during that repair.

Inference is the sole Git owner. Root must review the source/card, independently validate the bounded launcher and current resources, and privately push before any real model pilot. New outputs belong only under `/data2/neural-shrimp-data/codex/llm-hostile-fold-crosscheck/`; neither Claude's checkout nor existing certificate/crosscheck source is modified.
