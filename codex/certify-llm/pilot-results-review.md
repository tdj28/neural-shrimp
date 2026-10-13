# Independent retained-pilot results review

Reviewer: Codex inference subagent. Date: 13 October 2026 UTC. This is a
read-only artifact audit: no target model was evaluated, no checkpoint was
loaded, and no producer artifacts were changed. Successful consistency checks
below do not independently regenerate the enclosed transformer computations.

## arb-pilot-002: finite prefix and point; box attempt failed

Raw attempt: `/data2/neural-shrimp-data/codex/llm-p8-certificate/arb-pilot-002`.
Source commit: `0c650f87ad1ba20c3afeeeae443b4d5e7bbcc0a0`.
Receipt SHA-256:
`565ecc98b37b592bb1da16a58ad4109e1dd2ab5f2eac71682f2af860531f33b9`.

All **22 retained artifact hashes and sizes** match, with a unique complete
inventory and no unlisted file other than the receipt itself. All **nine source
snapshot files** match both their manifest records and the corresponding Git
blobs at the declared source commit. Receipt, manifest, input record and recorded
remote main agree on that commit. Standard output equals the event JSONL bytes.
The recorded numerical context is Arb **1024 bits, one thread**; the launch
manifest records one numerical thread, 18 GiB, 3600 seconds, and consistent owned
PID. The process exited **1**, without timeout, after **902.2089455730747 s**.
The final status is `setup_failed`, phase `rigorous_box_jvp`, with
`certificate_passed: false`.

| Stage | Independent saved-data check | Result and evidence scope |
|---|---|---|
| Numerical proposal | Recomputed all four relative returns from the nine binary64 states; checked history, thresholds, recorded token-cycle rotation and proposal hash | Numerical candidate passes its stated gates; this is not an exact-real root or primitive-period proof |
| Fixed prefix | All 24 completion events and all 227,328 saved KV balls checked for shape, exact integer serialization, nonnegative radius and finite reconstruction | Complete saved Arb prefix enclosure, conditional on the reviewed evaluator |
| One point | All 24 decoder completion events; 896 finite primal balls, 896 exactly zero tangent balls; each saved next-proposal defect reconstructed | Complete one-step point enclosure, not an eight-step closure or nonzero derivative test |
| State-box / direction-box | Checked all 896 input coordinates and directions; completed layers 0–2, entered layer 3, then stopped | Failed enclosure setup; no completed box-step artifact and no full Jacobian/contraction gate |
| Root, primitivity, attraction | No such certificate artifacts were produced | Unestablished by this pilot; final certificate flag correctly false |

The numerical relative returns at divisors 1, 2, 4 and period 8 are respectively
`2.003320766759392`, `1.4312115443694848`, `1.3916576566611432`, and
`1.602313717038947e-14`. The last two proposal-history entries are below `1e-12`.
The saved token IDs are a rotation of the declared eight-token pattern. These
checks validate the retained calculations and label relationship; this audit
did not rerun logits or independently tokenize the strings.

The full saved prefix has shape **24 × 2 × 37 × 2 × 64**
(layer, K/V, position, KV head, coordinate). Its largest **stored KV-ball radius**
is **1.0958943346127503e-146**. This must be distinguished from the final prefix
*hidden-state* radius in the progress log, **5.859349654677246e-141**; those are
different objects. Both are finite. Only the KV values are serialized in the
prefix cache.

The one-step point calculation took **97.12563968892209 s**, measured from its
first layer-start event to `point_step_done`. Its maximum stored output radius is
**1.2147891563591135e-142**. The largest enclosed absolute defect relative to the
next binary64 proposal point is approximately **9.622657250048913e-15**. Every
stored coordinate defect agrees with independently subtracting that exact
binary64 dyadic from the decoded output ball. This is neither a period-eight
closure defect nor a contradiction between the real and binary64 maps. All
point tangents were exactly zero because this stage requested no nonzero
input direction.

The box stage used exact state-center dyadics from proposal phase zero and
state radii **2^-80 = 8.271806125530277e-25**. Its last completed layers were:

| Zero-based layer | Logged maximum primal radius | Logged maximum tangent absolute bound |
|---:|---:|---:|
| 0 | 1.5877874331544072e-15 | 1.919484834e9 |
| 1 | 7.556436321110027e-8 | 9.134836059367014e16 |
| 2 | 1.6995551083236933 | 2.227765644956844e24 |

The next event starts layer 3. Both status traceback and standard error identify
the input RMSNorm's refusal: **positive RMS denominator not established**.
No `box-step.json` exists. The retained evidence supports excessive interval
widening in this naive box propagation, not instability of the actual orbit or
nonexistence of the exact-real period-eight cycle. No finite completed one-step
box derivative bound can be inferred from the intermediate layer values.

### Audit correction: outward radius reconstruction

The first audit attempt incorrectly demanded that the saved unit-direction
radius equal the mathematical integer one. The second incorrectly demanded that
reconstructing the saved Arb radius leave it unchanged. Both attempts stopped at
those assertions; neither discovered a producer failure or modified evidence.
The actual stored radius pair is `(536870913, -29)`, namely **1 + 2^-29**.
Reconstructing that pair through the Arb ball constructor expands it to
`(268435457, -28)`, namely **1 + 2^-28**. Both safely contain `[-1,1]`.
The corrected audit checks complete containment and non-narrowing, and passes.
State radii, which are powers of two, remain exactly `2^-80` here. Reported
saved-radius maxima above use the exact serialized radius pairs, not an
unlabelled additional reconstruction widening. This is a serialization precision
detail, not evidence for or against the target dynamics.

## signed-pilot-001: eight signed columns completed

Raw attempt: `/data2/neural-shrimp-data/codex/llm-p8-certificate/signed-pilot-001`.
Source commit: `3fb8844be76ea0d1cd9e0bbe26ac35b794c46b70`.
Receipt SHA-256: `32b73defdf9672e2593ad33370a37fce2535ccb7c9ddc04769d2c16cd70add7d`.

All **27 artifact hashes and sizes** match a unique complete inventory. All
**17 snapshot files** agree with their manifest records and the corresponding
Git blobs. Manifest, receipt, input record and recorded remote main agree on the
source commit. The committed plan hash, its embedded copy, direction inventory,
phase zero, 1024-bit precision and exact assigned parameter values are consistent.
The prefix binding resolves to the audited pilot002 receipt, graph and prefix
hashes. The proposal hash resolves to the same nine-state saved proposal. Receiver
construction is explicitly recorded as checked, with layer 12, while the loader
correctly declines to claim that it authenticates file hashes to array contents
on its own.

The process exited **0**, without timeout. Recorded total launcher time is
**283.6636585770175 s**, signed Arb evaluation **261.8081810520962 s**, and saved
binary64 comparison **15.786404399899766 s**. Recorded maximum RSS is
**6,510,748 KiB** (approximately **6.209133 GiB**). The source/manifest bind
one numerical thread, 18 GiB and a 3600-second watchdog. All 24 layer-start and
24 layer-completion events are present, ordered, and followed by the completed
signed step and comparison. Standard output matches the event log exactly.

All **8,064 stored primal/derivative balls** (896 primal coordinates plus eight
896-coordinate columns) have valid integer midpoint/radius encodings, nonnegative
radii and finite reconstructions. Every one of the eight comparison rows was
independently recomputed from the saved binary64 arrays and the saved Arb
midpoints, including direction, absolute discrepancy, relative L2 discrepancy,
midpoint L2 norm and stored maximum-radius check. No row was omitted.

| Coordinate direction | Maximum absolute midpoint / binary64 difference | Relative L2 difference |
|---:|---:|---:|
| 0 | 7.806255642e-17 | 3.288112115e-14 |
| 1 | 7.350890729e-17 | 2.301270823e-14 |
| 2 | 1.127570259e-16 | 2.293632405e-14 |
| 3 | 3.113828639e-16 | 4.068472703e-14 |
| 4 | 7.372574773e-17 | 3.033796316e-14 |
| 5 | 2.834104479e-16 | 1.239253253e-13 |
| 6 | 1.340073885e-16 | 3.437228254e-14 |
| 7 | 1.134075472e-16 | 3.808168394e-14 |

The maximum stored primal radius is **1.2147891563591135e-142** and the maximum
stored signed-column radius is **2.5258824107502704e-141** (direction 3). The
relative L2 discrepancies range from **2.2936324050859482e-14** to
**1.2392532531427195e-13**. These midpoint comparisons are numerical consistency
measurements between exact-real and binary64 implementations; they are not
containment requirements for the binary64 outputs.

The new primal computation has exactly the same 896 midpoints as pilot002.
894 complete midpoint/radius encodings are identical; two radii differ slightly,
and all 896 enclosures overlap. The first audit pass incorrectly required
identical complete encodings and stopped on that assertion. The corrected test
retains the two radius differences and checks midpoint equality and enclosure
overlap. No producer data or acceptance criteria changed. Identical arithmetic
layout is not required for two rigorous enclosures of the same point.

The final status is **setup_completed**, with **certificate_passed: false**.
The saved result establishes that the reviewed first-order Arb implementation
can complete eight signed coordinate actions at this one point under the bounded
budget. It supplies no bounds for the other 888 directions, no full-box derivative
variation or nonlinear remainder, no exact eight-step root, and no attraction
proof. The earlier box failure remains in the record and is not replaced by this
point calculation.

## What the measured cost supports next

This audit supports using numerical phase Jacobians as *proposals* for
preconditioners and local coordinates, followed by separately admitted rigorous
checks. It does not justify launching a full rigorous matrix from an unmeasured
scaling assumption. Straight linear extrapolation of 261.808 s per eight columns
would be approximately **8.15 CPU-hours for 896 columns at one phase**, or
**65.2 CPU-hours for eight phases** at the same batch strategy, before further
box/refinement work. This is a planning extrapolation, not a measured full-run
cost; larger signed batches can change both overhead and memory substantially.

A 2048-bit trial must recompute the prefix at 2048 bits: increasing the context
cannot recover precision absent from an existing 1024-bit enclosure. Any refined
state centres intended to exploit smaller errors must remain exact high-precision
dyadics (for example mantissa/exponent pairs), rather than being rounded back to
binary64. The present binary64 proposals and eight point columns are suitable
inputs for exploring that next construction, not its completed proof.
