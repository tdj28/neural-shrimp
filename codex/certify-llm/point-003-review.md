# Independent audit of 2048-bit point attempt 003

Date: 2026-10-13. **Accepted as a completed prefix/one-point setup** and as the
hash-bound prefix source for the reviewed first center-refinement pilot. No
checkpoint was loaded and no model was replayed during this audit. This result
is not an orbit, whole-box, primitive-period, token-label or attraction proof.

## Immutable evidence

Attempt: `/data2/neural-shrimp-data/codex/llm-p8-certificate/arb-point-003`.

| Binding | Value |
| --- | --- |
| Source commit | `d674d2101eae58bd09dc919530892f51c56017d3` |
| Receipt SHA-256 | `b286831e06d65280ddc44591ba196e40c2adbb33f125c20a676c615434dd51e5` |
| Manifest SHA-256 | `a5fe49eafe0f5b73667b80c7e7db2f3bcefb6d089ac8fe4d8a3ba6c2c2c17c87` |
| Arb map source SHA-256 | `fcc90849601984506adf26f651067734f9faef408c8f42dffe64bf5c68e98a80` |
| Pilot source SHA-256 | `354f166af4ac0d7afbe68d2b9c61ed17e54aa9f3e4ed2b448bef5e70545ccca1` |
| Prefix cache SHA-256 | `9d4a4193ecbb997a68c067312cae20c1f96de3e1919f7218e841715bad231053` |
| Point output SHA-256 | `6a3823d76cd45f19ef2138a11f2271aea50e8b745b671b8c13225e3c48eb8f68` |
| Comparison pilot002 receipt | `565ecc98b37b592bb1da16a58ad4109e1dd2ab5f2eac71682f2af860531f33b9` |
| Comparison pilot002 point output | `4cdf314f109a3698348b48427ac782de12cc44da9ef28faddc405b908cbfbb78` |

All **32** current-attempt artifact hashes and sizes passed; its actual file
inventory is exactly those entries plus the receipt. All saved artifacts are
read-only. All **20** captured source files match their manifest hashes and
exact Git blobs at the recorded commit. The source commit, input record,
manifest and recorded private-main revision agree. All **22** inventoried
pilot002 comparison artifacts were also rehashed successfully.

Both attempts use the same hash-pinned nine-state proposal
`d8c57bff1d8a986c7c6a0f7d083caa77b4f6373c560d1f60ce0c57461caf668b`, so their
point inputs are identical. Asset hashes, prefix IDs, exact parameter ratios,
seed binding, temperature convention and rotary convention agree. The seed
was rehashed. Attempt 003 records 2048-bit Arb arithmetic and CPU1; pilot002
records 1024 bits. Attempt 003 exited zero without timeout, with empty stderr.
Its `--skip-box` scope is explicit: no box-evaluation event/output is present.

## Every prefix and point phase was checked

The event stream contains ordered starts and completed ends for **all 24**
prefix layers and **all 24** point layers. Every recorded primal bound is
finite and every recorded tangent bound is exactly zero. Completion is
consistent with the reviewed evaluator's denominator/finite-value gates; this
audit did not independently reevaluate those nonlinear operations.

The decoded prefix has exactly 24 layers, two K/V fields, 37 tokens, two KV
heads and 64 head coordinates: **227,328 finite scalar enclosures**. Every
layer's 9,472 entries was checked individually. Every new prefix enclosure is
contained in its corresponding pilot002 enclosure and hence overlaps it.
The layer table below records completed computation time and approximate
maximum saved cache radius; decimals are display summaries, not proof inputs.

| Layer | Seconds | Maximum saved cache radius | Finite and old contains new |
| ---: | ---: | ---: | --- |
| 0 | 76.670 | `4.797439E-614` | 9,472 / 9,472 |
| 1 | 76.316 | `1.249753E-606` | 9,472 / 9,472 |
| 2 | 76.237 | `2.876184E-599` | 9,472 / 9,472 |
| 3 | 77.530 | `9.207520E-592` | 9,472 / 9,472 |
| 4 | 79.825 | `1.535056E-584` | 9,472 / 9,472 |
| 5 | 81.072 | `2.493131E-577` | 9,472 / 9,472 |
| 6 | 80.117 | `1.472398E-570` | 9,472 / 9,472 |
| 7 | 79.966 | `1.042013E-563` | 9,472 / 9,472 |
| 8 | 80.048 | `1.703375E-556` | 9,472 / 9,472 |
| 9 | 81.345 | `6.449315E-549` | 9,472 / 9,472 |
| 10 | 81.485 | `3.954466E-542` | 9,472 / 9,472 |
| 11 | 80.536 | `5.905950E-535` | 9,472 / 9,472 |
| 12 | 77.463 | `1.801045E-528` | 9,472 / 9,472 |
| 13 | 75.704 | `1.844422E-521` | 9,472 / 9,472 |
| 14 | 75.762 | `5.739226E-515` | 9,472 / 9,472 |
| 15 | 75.520 | `5.956807E-508` | 9,472 / 9,472 |
| 16 | 75.428 | `2.807395E-501` | 9,472 / 9,472 |
| 17 | 75.356 | `2.588421E-494` | 9,472 / 9,472 |
| 18 | 75.756 | `4.659763E-488` | 9,472 / 9,472 |
| 19 | 75.978 | `1.684155E-481` | 9,472 / 9,472 |
| 20 | 76.109 | `1.735286E-474` | 9,472 / 9,472 |
| 21 | 75.583 | `6.713061E-468` | 9,472 / 9,472 |
| 22 | 75.883 | `1.818581E-461` | 9,472 / 9,472 |
| 23 | 75.615 | `6.063272E-455` | 9,472 / 9,472 |

## Exact enclosure comparison and precision qualification

The point output has exactly 896 finite values, 896 exact-zero tangents and
896 finite defects from the next numerical proposal. Every saved defect was
also checked to enclose the exact subtraction of the saved point-output
interval and the next binary64 proposal coordinate.

| Field | Enclosures | Overlap pilot002 | Pilot002 contains new | New contains pilot002 |
| --- | ---: | ---: | ---: | ---: |
| Value | 896 | 896 | 896 | 0 |
| Tangent | 896 | 896 | 896 | 896 |
| Defect from next proposal | 896 | 896 | 896 | 0 |

The largest stored point radius is exactly

`98,864,005 * 2^(-1522)`, approximately `6.72024024698383218e-451`.

Pilot002's largest stored point radius was approximately
`1.21478915635911354e-142`. The largest defect from the next numerical proposal
remains approximately `9.62265725004891236e-15`: increased precision narrows
arithmetic uncertainty but does not remove the original binary64 center error.

All comparisons used the **original exact dyadic endpoints** constructed from
saved midpoint/radius integer pairs. Arb decoding was separately checked to
preserve the midpoint and never decrease the radius. This matters because
constructing an Arb ball can conservatively round its stored magnitude radius
outward by one ulp. An initial audit assertion requiring byte-identical decoded
radii failed on this behavior; the audit was corrected to compare exact saved
endpoints, then fully repeated. No raw evidence was changed or radius dropped.
The successful complete audit ran on CPU1 in 6.990 seconds and exited zero.

The prior pilot002 attempt remains a failed whole-box diagnostic; its complete
prefix/point enclosures can be compared without relabeling that attempt a
success. Neither the containment comparison nor the narrower point radius
establishes validity for a neighborhood of the state.

## Measured cost and refinement-plan admission

The complete attempt took **2013.070 seconds** of launcher wall time. Prefix
construction plus cache serialization took **1874.003 seconds**. From the
saved prefix to the saved one-point result took **114.209 seconds**. This last
quantity includes output serialization and is a measured single-phase cost,
not a guarantee for every orbit phase or for a whole refinement pass.

The prepared `center-refine-001-plan.json`, SHA-256
`8cca7c0f2018560a005e524f12f976766ca51f4ca14654f38e5bb400c686bc9c`, was independently
read and is **admitted for the previously reviewed bounded setup scope**:
exactly one correction, two eight-phase point passes, CPU1, nice 10, 18 GiB,
2048 bits and a 3600-second ceiling. Its prefix receipt/map-source hashes match
this audit; its seed receipt/driver/map/preconditioner hashes match the completed
independent seed audit. The current refinement driver, map, cache importer and
launcher hashes still match their reviewed versions.

The plan transparently estimates `16 * 114.208560945 = 1827.337` seconds for
point evaluations, with other overhead extra and no completion guarantee.
The parent's actual `load_bound_inputs` check passed all 32 prefix-attempt and
56 seed-attempt artifacts. The two previous owned PIDs were independently
observed absent at this review. The parent's reported available memory was
55 GiB; this is resource context, not a claim about future peak consumption.

Admission remains a source/plan review, not a launch by this reviewer. The
parent must commit/private-push the final source and plan through the reviewed
launcher. Any timeout remains an incomplete setup with retained phase files.
Even a complete refinement only supplies new centers and residual enclosures;
all whole-box root, primitive-period, token-label and attraction obligations
remain separate.
