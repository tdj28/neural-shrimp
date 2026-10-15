# Task23 final independent saved-evidence assessment

The completed independent CPU float64 replay is internally consistent with its saved finite-twin accounting. It reproduces the producer's finite labels at all four requested cells: no detected return with two positive finite-twin rates at log10T .125 and .129, and detected periods 2 and 14 with two negative rates at .11900000000000001 and .135. It does **not** reproduce the producer's aggregate numbers exactly: all 28 required scalar comparisons have nonzero differences. These differences are retained, not converted into a failed or passed agreement test.

This assessment reads the completed reports, all four own summaries and their 64-lag lists, all 28 comparison records, the auditor's full result structure and discrepancy list, and the existing manifest/receipt/owner links. The complete norm-diagnostic list was parsed and remains in its immutable report; the summary and first/last examples were inspected numerically, without a second elementwise norm audit. No model, raw native-array computation, comparator, auditor, statistical analysis or process inspection was rerun. `coverage.json` is a literal selection of saved fields, not a new verifier.

## Evidence and authority

All paths below are under `/data2/neural-shrimp-data/codex/mechanism-replay-task23`.

| Artifact | SHA256 |
|---|---|
| `run-001/report.json` | `0039ef4c2732bbf4641010543a33b761cd85821167a5ba20ebb7d9f2c391dfd7` |
| `run-001/final-manifest.json` | `9652ee09743854e54bc31b66bcd37c1598c1db3fe866125a059515c3c5ffe82b` |
| `run-001-supervision/closure.json` | `02b2459c55c9fa9bd348e834dc140eb43a211cb1b6cd22614cddc3472c2af498` |
| `run-001-dispatch/outer-closure.json` | `e6072e7fd0aa87a86b9c7f912630ae9cdebffbb2bb7a8773cb9d6f81ccc91b54` |
| `comparison-001/report.json` | `efd89b19259f3d671a78a468158830762d2d697f29d032548f3c57786c50e563` |
| `comparison-001-supervision/receipt.json` | `e77489c8697cf08eccbf6ab5a948244a5e343e135603d54aa69735a1fc59c3e4` |
| `comparison-001-dispatch/outer-closure.json` | `e8d3fcff4d7edf467ac8cd6db70abf86fdff9960d9a854188233e0141ad8c67d` |
| `saved-consistency-001/report.json` | `ac9719898adb66b7f73ea6a3bbb38d1375e002182af6e9323043a8e9acaf06e7` |
| `saved-consistency-001-supervision/receipt.json` | `eb87423f576c06b2947baa99e7bd3de40697c4b70a7c29080789299460ac087a` |
| `saved-consistency-001-dispatch/outer-closure.json` | `b29452252145b8a53791c3f0ba10f0227c2763bd03c0701ac974b3c610d5a8a2` |

Inference observed and authenticated original execution, source capture, output hashes and reaping; these are inherited operational authorities, not observations by this reviewer. The actual source is `3e103261c546d128c9b6212cdaeda78d677148ad`, with five captured Git blobs and seven original output files. Original session 74234 exited zero; the exact original supervisor/worker identities and their groups/sessions are recorded absent. Comparator source `8be809a3b29e820f2c594bec5a7d70e064ddaf2e` and auditor source `c8f98a20b8b042eb294b5b34cb319eb2169de264` each have three captured blobs and completed owner closures. Their receipts record exit zero, no failure, reaped workers and absent owned groups. Neither receipt's `scientific_claims_accepted:false` is changed by computational completion.

The comparator binds the original producer gain JSON `a084574483fed4f7905212412eb2a80259b064ea836f6ee7d424f475bc9184bf`, original source/metadata authority, and the same own report/card/closure used by the saved consistency auditor. The auditor binds the original seven-file manifest before/after and uses reviewed source `00d3e95a0b97b3689fe986dd597257a5a5f087582d5752e1fb45bdbf93bd114b`. Its source review and amendment remain separate artifacts (`e00ecc5e` and `65a343c1`); this final assessment does not repeat them.

## Own finite results

The fixed workload used alpha .0655, beta 0, 1000 transient steps and 2000 measured steps, with epsilons 1e-6 and 1e-9. Both full 201×896 RNG blocks were generated and the original indices 25,29,19,35 selected. Index 19 uses the original stored .11900000000000001, explicitly distinct by one binary64 step from the requested decimal .119. The other requested coordinates match exactly.

The following are exact saved values. Network/readout factors are the first twin's mean natural-log relative gains per model step; entropy is in nats and gap is the top-two logit difference divided by physical temperature.

| Index / log10T | λ, eps1e-6 | λ, eps1e-9 | Mean log g_net | Mean log g_read | Boundary term | Detected period |
|---|---:|---:|---:|---:|---:|---:|
| 25 / .125 | 0.4950137968265117 | 0.4951925533604417 | 0.5917462603039746 | -0.09673621245132731 | 3.7489738624098855e-6 | 0 |
| 29 / .129 | 0.516734394657097 | 0.5179621168353872 | 0.6071165675051129 | -0.0903113529331758 | -7.08199148408509e-5 | 0 |
| 19 / .11900000000000001 | -0.42334643917864945 | -0.4229000409855568 | 0.2391864799012664 | -0.6625329190799171 | 3.3306690738754687e-19 | 2 |
| 35 / .135 | -0.1852612573584068 | -0.18533732202995704 | 0.38499265815998346 | -0.5703094305793245 | 5.551506093574617e-5 | 14 |

| Index | Mean entropy | Mean gap | Mean top probability | Fraction gap<1 | Fraction gap<2 |
|---|---:|---:|---:|---:|---:|
| 25 | 6.328119091338987 | 1.2245618191946803 | 0.2520449131031583 | .5455 | .7665 |
| 29 | 6.59992713280153 | 1.1487947664787075 | 0.22457780100371427 | .561 | .796 |
| 19 | 5.184826964178679 | 1.9480913024177033 | 0.41431130826605317 | .5 | .5 |
| 35 | 6.392994595859047 | 1.3479591832058568 | 0.2664095741249796 | .571 | .714 |

All four observed network factors are positive and all four readout factors negative. For these trajectories, their finite relative-gain accounting yields positive total rates in the first two cells and negative rates in the last two after including the boundary term. This supports the stated finite decomposition; it does not establish that one factor causally determines chaos, nor that entropy alone determines the regime. In particular the period-14 cell has higher saved mean entropy than the .125 cell with a positive finite rate. The second-twin factor splits and every signed residual remain in the full own summaries and coverage artifact.

The recurrence check examines every lag 1–64 against the final state in the 66-state tail, with recorded threshold 1e-8 at all four cells. At index 19 the lag-2 distance is `1.1144712335618982e-15`. At index 35 the lag-14 distance is `3.895063458190642e-14`, while lag 7 is `0.008072090997255305`. All 256 distances were read and retained. The literal chaotic label requires period zero and both rates above 1e-3. It agrees with the saved label and with the producer in all four cases. Finite period detection is not a complete primitive-orbit/stability proof; two finite positive rates and no return through lag 64 are not an asymptotic chaos certificate.

## Every required producer difference

These are the comparator's exact saved **own minus producer** deltas. Both operands and differences are finite in all 28 rows; every exact-equality field is false and every agreement gate is null. Full operands, absolute differences and binary64 representations remain in `coverage.json` and the original comparison report.

| Field | Index25 | Index29 | Index19 | Index35 |
|---|---:|---:|---:|---:|
| λ, eps1e-6 | -0.035152435006497784 | -0.006701266221312174 | -4.28136742725016e-9 | 0.00045621586370286904 |
| λ, eps1e-9 | -0.03465819608056481 | -0.005291129053870303 | -1.6175830368836408e-5 | 0.00091690285406551 |
| Mean log g_net | -0.01299255684798084 | -0.017820636760248254 | -3.887828392556969e-10 | 0.00028366883750102767 |
| Mean log g_read | -0.022068389657685705 | 0.01117799462472005 | -3.892585920262093e-9 | 0.00024211690790287843 |
| Boundary term | -9.148850083323338e-5 | -5.8624085785353566e-5 | 2.0428103653103284e-17 | -6.956988169933146e-5 |
| Mean entropy | 0.012824857970609393 | -0.03175657249898034 | 9.50350909079134e-14 | -0.00401725921915741 |
| Mean gap | 0.004766252599647691 | -0.04216787494750718 | -5.950795411990839e-14 | 0.0006222831343623003 |

The largest rate differences occur at .125, whereas the period-2 first-twin rate and mean entropy/gap are much closer. The period-2 second-twin rate still differs by `-1.6175830368836408e-5`, and the period-14 rates differ by the two positive values above. Neither periodic case is presented as exact agreement. All negative own rates, readout factors, signed boundary terms and signed differences remain visible.

Own execution used CPU float64 with four transient base rows and twelve measured rows. The producer used its original CUDA float64 path with 201 cells/603 rows. Aligned parameter values and original RNG positions do not enforce identical floating trajectories. These implementation/platform/batch differences are reasons to qualify numerical comparison; this evidence does not isolate them as the cause of each delta. No late-state equality, statistical-equivalence threshold, uncertainty interval or favorable tolerance was introduced.

Producer aggregates are source float64 sums/record or finite norms serialized as Python floats without explicit decimal rounding. Their first 1000 saved step arrays are float32 and were not used for the 28 comparisons. The producer's second-twin network/readout split, full 2000-step float64 scalar stream, exact runtime prefix IDs and native twin states are unavailable for an equal-scope independent comparison. Auxiliary power/trace/rank/Rayleigh fields remain preserved raw, without a new comparison or recomputation. This workload does not validate a point-Jacobian spectral estimate as a bound on finite secant gain. Team-level prior exposure and source/runtime authority limits remain; the worker's literal `no_producer_outcomes_read:true` field is not a new claim that the research team was blinded.

## Saved consistency assessment

The sole saved audit reports **16,082 exact comparisons covering 355,224 scalar elements, with zero nonexact comparison records**. It reports the original summed `1e-10` accounting predicate passed. The exact checks include both RNG selections, endpoint/tail state associations, all 2000 saved gain/log/boundary calculations, represented-offset/norm links, scalar accumulation/summary fields, physical parameter identities, all finite recurrence lags and expected progress positions. The full native base matrix was finite-checked and used for norm diagnostics. This does not independently re-evaluate each map transition.

The original per-step residuals are not identically zero. Their saved nonzero counts, in epsilon order, are `[1672,1666]`, `[1697,1690]`, `[1447,1404]`, `[1615,1617]` for indices25,29,19,35. Per-cell maximum absolute step residuals and both signed final residuals are preserved in the coverage summaries; the largest displayed maximum is `6.661338147750939e-16`. The first-twin final residual at index25 is `2.0076408797139317e-15`. These floating residuals meet the original prospective accounting tolerance; they are not new rigorous error bounds.

Separately, the audit retains **3,772 norm-diagnostic records** where independent NumPy reductions differ from the stored Torch reductions. Zero exact-arithmetic mismatches does not mean zero norm differences. For example, `step0/base` contains signed norm differences `[0,5.551115123125783e-17,5.551115123125783e-17,-5.551115123125783e-17]`. All diagnostic objects, including their signed/relative entries and final represented-offset differences, remain in the immutable audit report. No maximum discrepancy, new tolerance or acceptance test was calculated for this final review.

The audit explicitly marks `initial_directional_construction_independently_reconstructed:false`. Exact original RNG selections and represented-offset norms were checked, but initial vector-direction construction still relies on reviewed source. Unsaved intermediate twin/hidden/probability arrays, intervening transient forward calls, and the model itself were not reconstructed. Entropy and top probability are checked as reductions of saved observations; only the saved top-two logit gap is independently reconstructed from its scalar operands. These limits must accompany any claim of successful saved consistency.

## Timing, preserved wording and recommendation

The original worker records `1016.234895714093` seconds through scientific output serialization but before final rehash/report writing, with `5,334,460 KiB` peak RSS observed by then. The separate original supervisor window is `1017.9711758838966` seconds. Comparator supervisor time is `0.05222603399306536` seconds. The audit records `0.39288068283349276` seconds and `122,128 KiB` observed peak before its final report; its separate supervisor window is `0.5561823910102248` seconds. These are distinct scopes, not interchangeable full-worker runtime/peak measurements. The comparator's sampled output maximum of zero and audit's sampled `1,558,548` bytes are sampling fields, not final output sizes; their sealed reports are respectively 52,121 and 2,281,905 bytes.

The original actual report retains stale scope prose: “Profile summaries are cost/setup observations, not1000/2000 results.” That text is an acknowledged unedited artifact. Its `stage:actual`, explicit 1000/2000 horizons, `scientific_full_run_complete:true`, actual card and native output/closure establish that this is the admitted full actual replay. Recommended correction in the separate human-facing report: **“These are the completed 1000-transient/2000-measured finite-twin results; the earlier profile provided cost/setup observations only.”** Do not rewrite the frozen result or treat the stale sentence as a substantive new experiment.

Suggested scientific wording: “An independent CPU float64 implementation reproduced the finite regime labels at the four fixed hostile L12 cells. The measured network/readout factors and boundary term satisfied the prospective floating accounting identity, and a separate saved-evidence audit found no exact same-arithmetic or endpoint/tail-link discrepancies. Aggregate values differed from the original CUDA/batched run, with all 28 required differences retained. The evidence is a finite trajectory/accounting cross-check, not a causal mechanism test, numerical identity of trajectories, or certificate of asymptotic chaos.”

**Final disposition:** accept the evidence as a completed, authenticated, internally consistent finite replay with disclosed producer differences and saved-evidence limitations. No additional computation is required for this assessment. Root retains final scientific acceptance and Inference retains private adoption/delivery.
