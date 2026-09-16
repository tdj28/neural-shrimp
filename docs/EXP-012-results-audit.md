# EXP-012: the tested CUDA preview pipeline is not qualified

15 September 2026 local / 16 September UTC. This exploratory engineering
experiment completed **all 54 batches and 64,188 case integrations**, retaining
every scheduled result. Keep the scientific CPU backend as the default.
The tested CUDA implementation fails both the fixed numerical-compatibility
requirements and the deployment-speed requirement. No tolerance was relaxed,
step size selected after results, failed batch removed, or paid compute used.

The [protocol](EXP-012-protocol.md), executable source and machine plan were
committed and privately pushed as
`c3a88b9db0a3e82b65d5d55a58d155b9a3275d5f` before known-case integration.
The frozen plan SHA-256 is
`74cef32c3eedf20cba0f3e9b05dc44f1a3f29f6b04c865cf60bee4712750360e`.
This is a compatibility and throughput study of previously observed EXP-010
previews, not fresh evidence for geometry utility. All prior scientific
sources and the original failed EXP-008 strict gate remain unchanged.

## What completed

The comparison uses all 3,456 known 75-unit cases at `dt=0.0025` and
`0.00125`, with 751 saved samples per case. Warm timing uses three paired
repetitions at each of four nested batch sizes; the CPU/GPU order alternates
by repetition. Two cold batches and four full-census GPU layout batches
complete the fixed schedule. CPU and GPU arms execute sequentially, with
four CPU scoring threads and one numerical-library thread per worker.

The local RTX 2000 Ada has 16,380 MiB reported VRAM; the benchmark's largest
requested device-array allocation is 62,581,248 bytes. The float64 CUDA
backend preserves the RK4 expression order and disables fused multiply-add
and fast-math substitutions. Compilation uses CUDA 12.4 with g++-12; the
loaded runtime reports 12.3. These settings are recorded rather than assumed
to make different hardware arithmetic identical.

The producer took 1,542.288 seconds through its qualification analysis,
including 56.634 seconds for that analysis; inventory sealing followed.
The 175 receipt-listed artifacts occupy 1,738,050,920 bytes excluding the
receipt. They remain outside Git in
`/data2/neural-shrimp-data/raw/EXP-012/attempt-001/`.

## Numerical result

| Requirement | Result |
|---|---|
| Complete planned outputs, finite values, identity and progress | Pass; all 54 batches complete |
| Current CPU versus saved same-step EXP-010 reference | Pass in all 25 CPU batches |
| GPU versus same-step CPU state and activation tolerance | Fail on both complete case sets |
| Complete GPU versus CPU metric and phase tolerance | Fail in all 29 GPU batches |
| Complete candidate order, minimum, ordered top four and fixed-control choices | Pass in all 29 tested GPU batches |
| Full-size GPU repeat and restored-layout output bits | Pass in all 10 required comparisons |
| Saved-array scoring and evidence verification | Pass; numerical gate failures reproduce |

The independent full-census comparison gives:

| Diagnostic | `dt=0.0025` | `dt=0.00125` |
|---|---:|---:|
| Maximum absolute state difference | 1.5045654e-7 | 1.2091537e-7 |
| Maximum absolute tanh-activation difference | 2.4309249e-8 | 6.7312504e-8 |
| State coordinates exceeding their gate | 24,596 | 25,648 |
| Activation coordinates exceeding their gate | 16,583 | 17,429 |
| Cases with at least one failed complete metric field, out of 3,456 | 2,177 | 2,204 |
| Failed metric fields | 22,531 | 23,019 |
| Maximum difference across declared RMSE fields | 6.6448419e-6 | 2.8790515e-6 |
| Maximum circular fitted-phase difference, cycles | 5.0760258e-7 | 1.9205198e-7 |

State compatibility requires each difference to be no greater than
`1e-10 + 1e-10*max(abs(cpu),abs(gpu))`; activation and each declared RMSE
difference must be at most `1e-10`, and circular phase at most `1e-8` cycles.
The aggregate aligned RMSE alone exceeds its tolerance in only two cases at
each step size. That narrow statistic does not replace the full waveform,
component, window and phase checks. Complete ordering and policy choices
are unchanged in all 384 independently reconstructed full contexts across
the two steps, but decision equality does not waive the failed requirements.

The discrepancies are reproducible and the saved evidence is intact. Their
specific arithmetic cause has not been established by this experiment.
These engineering tolerances are compatibility requirements, not certified
integration-error bounds or a task-derived acceptable quality loss.

## End-to-end performance

Timing includes input preparation, integration, transfers/synchronization,
scoring, compressed raw and score writes, hashing, readback, and complete
saved-array rescoring. The small final timing record and final experiment
receipt are outside the batch timer; startup/build are reported separately.
The frozen deployment requirement was all three GPU wins and median paired
CPU/GPU speedup at least 1.25 for 3,456 cases at the finer step, conditional
on all numerical and layout gates passing.

| Step | Cases | Median CPU seconds | Median GPU-pipeline seconds | Median paired CPU/GPU ratio |
|---|---:|---:|---:|---:|
| 0.0025 | 18 | 0.438 | 0.595 | 0.753 |
| 0.0025 | 144 | 3.090 | 3.677 | 0.849 |
| 0.0025 | 576 | 12.520 | 13.450 | 0.923 |
| 0.0025 | 3,456 | 77.243 | 83.717 | 0.895 |
| 0.00125 | 18 | 0.476 | 0.823 | 0.560 |
| 0.00125 | 144 | 3.784 | 4.166 | 0.908 |
| 0.00125 | 576 | 12.632 | 14.103 | 0.960 |
| 0.00125 | 3,456 | 69.689 | 75.856 | 0.918 |

The GPU pipeline is slower in **all 24 warm timing pairs**. At the intended
full finer-step batch it takes about 9% longer, with median cost increasing
by 6.167 seconds. The paired ratio and ratio of marginal medians are distinct
statistics; the table reports the frozen paired statistic. There is no
observed amortization break-even: the additional 0.846 seconds of GPU setup
cannot be recovered when the measured warm-batch saving is negative.

For example, the first full coarse GPU batch spends 0.925 seconds in its
CUDA kernel and 0.967 seconds in the complete Python integration call,
40.213 seconds scoring the fresh arrays, and 39.880 seconds rescoring the
saved arrays. Its end-to-end time is 83.717 seconds. This points to scoring
and verification as the next performance work, while avoiding a kernel-only
speedup claim. The CPU arm overlaps per-case integration with scoring whereas
the GPU arm integrates a whole batch before scoring; this benchmark compares
those implemented pipelines, not pure hardware throughput. Measurements on
this shared host do not guarantee timing under other loads or other workloads.

## Verification and provenance

The frozen CLI audit reruns its evidence checks in a separate invocation of
the same frozen implementation, writing to a new external audit directory.
It verifies the receipt inventory and reproduces
the original detailed audit byte for byte, including the failed gates;
`original_audit_and_summary_reproduced` is true. The detailed audit has
SHA-256 `6e87bad3175f5380601836700d0d5bafe26416a5376d04e55da0175d5e32fdf9`.

A separate numerical script, without importing the EXP-012 producer or CUDA
backend, passes **194,275 evidence checks with zero evidence/replay failures**.
It verifies all 175 artifact hashes, 22 source bindings and 3,463 original
inputs; independently reconstructs the old contexts; replays 13,824 canonical
score packets and 34,560 windows exactly; checks 6,912 CPU cases against
EXP-010; and reconstructs all 384 full rankings and control choices. It
independently confirms the failed GPU numerical gates. No new integration,
model fit or fresh outcome is acquired by either post-run audit.

An independently authored standard-library performance/resource audit,
executed by root on the unrestricted host, passes **4,178 checks with zero
failures**. It reconstructs the complete schedule, 24 timing pairs, eight
timing cells and cost arithmetic from saved records; validates inventory,
array identities/progress and resource limits; and checks actual host process
ownership. At 06:06:08 UTC, both recorded worker identities and producer PID
are absent and no task-owned PID holds GPU memory. The producer did not record
its own start ticks, so its final check is PID absence rather than an identity
comparison. The ten unrelated GPU owner PIDs remain listed; no other process
was modified. Minimum recorded available RAM was 68,256,653,312 bytes and
minimum pre-chunk free device memory was 6,762,659,840 bytes, above the frozen
limits. This audit does not recompute numerical parity; the separate numerical
audit supplies that check.

The [compact summary](EXP-012-summary.json) binds the raw receipt, original
summary, complete audit, independent reports and byte-identical audit source
copies. Raw receipt SHA-256:
`8a3119d77d5e07be2039c08411303533b6b67aec800049a5cac500be1b40c020`.
Large reports and trajectories stay in the sibling data directory.

Before the experiment freeze, the suite passed 508 tests with one optional
CUDA test skipped; the separate real-GPU synthetic fixture passed. After
adding the draft EXP-013 final-model module, the suite passes **542 tests,
one skip**, including 34 tests of that new module. Synthetic checks and
engineering tests do not establish scientific GPU qualification.

## Decision and next work

Use local CPU and RAM for the fresh-origin study. The prepared
[EXP-013 design](fresh-origin-tradeoff-design.md) keeps the 24 historical
origins as development data and proposes 24 disjoint origins with complete
budget curves and strong fixed controls. Its final-model module is tested,
but the complete runner, machine plan, independent review and executable
freeze are still required; no final historical recipe or fresh dynamics has
been run for EXP-013. Bind CPU use in that freeze.

Any later acceleration change must preserve this failed attempt and receive
its own prospective compatibility and pipeline-cost check. RunPod remains
a fallback for insufficient local capacity, but this result supplies no
reason to buy a larger GPU for the unchanged scoring-heavy pipeline. It
does not prohibit GPU use for other separately validated work. Geometry
utility, long-target numerical resolution, boundary organization and trained
network cycles remain separate research questions.
