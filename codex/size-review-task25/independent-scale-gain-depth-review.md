# Task25 — independent saved-evidence review: scale, gain and depth

15 October 2026. **Finite scientific review complete; parent acceptance and private delivery are separate.** This review reads the sole admitted, completed Task25 outputs. It performs no model, raw-array, reducer, binning, fit, bootstrap, correlation or gate recomputation. PS, CP and plateau are assigned to Theory and are not independently accepted by this note.

The shared scale publication projection reproduces **all 663 compared JSON leaves exactly**. The scientific result is mixed: Qwen satisfies the frozen core P1/P2 comparisons, while OLMo32B contradicts both OLMo predictions. The manuscript-wide Qwen gain comparison satisfies all 12 inequalities, but OLMo violates six of 12. All four Llama3.3-70B depth maps retain finite no-return positive-exponent cells; these maps support a descriptive depth dependence, not disappearance of chaos or a causal parameter-count law.

## Evidence and review coverage

The authoritative output directory is `/data2/neural-shrimp-data/codex/size-review-task25/run-001`. Original report SHA256 is `8a2896947f80e7e9208790da8eabbe6a38fb0837041984236287d3997d3e2a41`; exact own source is `2497d55f529fb0d9b1673c452792a696ee50db19`. The reviewed science files are:

| Output | SHA256 |
|---|---|
| scale.json | `512029022eac16b1e219eae4693c5b02e647382be8140ab31b9ad8dbb51dbc75` |
| gain.json | `def83c177de97819119b6afc1de77e94d476d494a2d4de2387caff86ab474a30` |
| depth.json | `0bb2fbcd3126815e0f1025a6b4bb375f016289ffc0769b2a5f789ed59ac32dab` |
| comparisons.json | `f0c6e61d8aa613f1807a328a048c5f98539f2b20c9176755298186427179e755` |
| input-manifest.json | `705fa1a577d1b7fcec6960613c0ca7f6aab46d4d66f94adcfa1a1152ffbebcd5` |

Inference authenticated all nine output files, eight captured Git blobs, original receipt/source association and exact process closure. Its owner closure is `../run-001-dispatch/outer-closure.json`, SHA `3714d60a82543181f185fb4b321b99e7d9627f7a47c9d840a13687e057ce1645`; seal `d39e8b975d7e30ab8d894716be04278f78c3e72b7104dde5d68934736024c4ff`. The original supervision receipt is `../run-001-supervision/receipt.json`, SHA `3e9a3bef9835941bdb23a81471c2464d24be02be0346fe87d56bfadbc0f666fa`. Session21323 was reaped with exit0; exact original supervisor/worker, process groups and sessions were absent. These are **Inference's direct operational observations**, reused here without another audit.

One authorized bounded stdlib presentation pass copied saved fields and inventoried stored row flags in `/tmp/task25-sgd-presentation-001`; source is `/tmp/task25-sgd-present.py`. It did not derive scientific quantities. Complete cell flags and source provenance remain in the saved originals/presentation. Manual reading covered all 15 scale summaries and their primary/full coverage and duplicate dispositions; all 75 gain bins, the default 24 P3 rows and both 24-row selected views; all 53 unrounded Qwen14B chaotic selected rows; all four depth summaries, histograms, writer comparisons and three baseline comparisons; and every scale/gain/depth published-comparison disposition. Initial verbose scale/gain displays were truncated; subsequent compact selections repaired that coverage. This is full stored-disposition coverage, not a claim to have independently repeated native numerical generation.

Prior producer summaries, Task27 findings and manuscript claims were already known. This is independent **unblinded arithmetic reproduction and saved-evidence interpretation**, not fresh confirmation. Original failed/partial attempts and source/precision limits remain in the admitted authority. A usable final saved file does not make every historical attempt successful.

## Scale: frozen core versus expanded axes

The literal chaos rule is saved LE>0.001 with period<=0. All selected actual period/LE values pass the finite qualification, so period0 is the relevant no-detected-return sentinel here. No source refusal, missing source, invalid retained row, conflicting duplicate chaos label or lower-axis membership discrepancy was reported. The source implements the core by the upper test log10(T)<=0.3+1e-9; the independent lower-axis diagnostic found no points below−1, preserving the intended frozen core on these actual inputs.

The table copies the saved counts and full-precision measures. Core and full-axis denominators are deliberately distinct; full-axis domains differ by model. `C` means finite source-classified chaos, `U` no detected period without passing the chaos threshold. Weighted breadth is the source's sum of `np.gradient(log10T)` weights, including full endpoint spacings, not a rigorous continuum measure.

| Family/model | Core C/N | Core U | Core share | Full C/N | Full max LE | Weighted chaos breadth | Weighted H∈(1,10) breadth |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5 0.5B | 7/512 | 1 | 0.013671875 | 7/512 | 0.28426632849419375 | 0.017808219178082285 | 0.05342465753424663 |
| Qwen2.5 1.5B | 2/130 | 0 | 0.015384615384615385 | 55/224 | 0.854805511814244 | 0.5314465408805031 | 0.19119496855345908 |
| Qwen2.5 3B | 0/130 | 0 | 0.0 | 3/224 | 0.8471919762376028 | 0.030188679245283012 | 0.060377358490566024 |
| Qwen2.5 7B | 65/512 | 6 | 0.126953125 | 89/735 | 0.8774694574141603 | 0.22623786138906316 | 0.2486473679676061 |
| Qwen2.5 14B | 0/130 | 0 | 0.0 | 53/224 | 1.1394301767064778 | 0.5001242887091943 | 0.19119496855345908 |
| Qwen2.5 32B | 0/130 | 0 | 0.0 | 1/224 | 0.3556818793083529 | 0.010062893081761004 | 0.09056603773584904 |
| Qwen2.5 72B | 0/128 | 0 | 0.0 | 0/191 | -0.4473608728407276 | 0.0 | 0.01428571428571429 |
| Llama-3.1 8B | 96/512 | 2 | 0.1875 | 116/735 | 0.6629316521852896 | 0.24080419451316293 | 0.6337733387487848 |
| Llama-3.1 70B | 3/128 | 28 | 0.0234375 | 3/128 | 0.41432951408204777 | 0.03070866141732287 | 0.30708661417322847 |
| Llama-3.3 70B | 0/128 | 19 | 0.0 | 0/128 | -0.26538393286222184 | 0.0 | 0.22519685039370066 |
| OLMo-2 1B | 5/130 | 0 | 0.038461538461538464 | 5/224 | 0.23285797225516813 | 0.05031446540880502 | 0.32201257861635213 |
| OLMo-2 7B | 32/512 | 3 | 0.0625 | 41/735 | 0.3041023056260902 | 0.08333025637238622 | 0.19703011729375153 |
| OLMo-2 13B | 0/130 | 0 | 0.0 | 0/224 | -0.0006737156509113927 | 0.0 | 0.13081761006289305 |
| OLMo-2 32B | 21/130 | 0 | 0.16153846153846155 | 25/224 | 0.6244037461170138 | 0.25157232704402516 | 0.6943396226415096 |
| Qwen3.5 MoE 35B-A3B | 0/128 | 15 | 0.0 | 0/128 | -0.26526369456376475 | 0.0 | 0.14330708661417313 |

All six frozen comparison records are defined. Qwen14B<7B and32B<7B are true, and the nonincreasing Qwen7B→14B→32B→72B sequence is `[0.126953125,0,0,0]`. OLMo13B<7B is true, but **OLMo32B<7B is false**: `0.16153846153846155` versus `0.0625`. The OLMo7B→13B→32B sequence `[0.0625,0,0.16153846153846155]` also fails P2. Thus neither prediction is a universal all-family result. The distinct Llama3.1/3.3-70B and MoE labels must not be pooled into a generic 70B or dense-model claim.

Qwen14B has **0/130** chaotic core points but **53/224** on its expanded axis. The true saved chaotic log10(T) extrema are **0.5194968553459121** and **1.1438095238095238**; an enclosing four-decimal display is `[0.5194,1.1439]`. These are the lowest/highest sampled chaotic coordinates, not a continuous interval filled with chaos. The maximum LE is **1.1394301767064778**, at log10(T)=**0.7973015873015873**. All 53 exact coordinates, source IDs, LE/period/H values and gates are retained in the presentation. The scorer's rounded `chaotic_log10T` list is not used as an exact extremum authority. The expanded weighted chaotic breadth is `0.5001242887091943`, versus Qwen7B's `0.22623786138906316`; the core comparison therefore cannot be restated as absence of chaos at 14B or a monotone full-axis decline.

The four discarded full-view shared endpoints have exact equal coordinates and agreeing chaos labels, but **not equal scalar measurements**. Signed differences below are discarded minus retained; the Qwen72B witness also appears in its primary-join ledger and is not a new independent observation.

| Full-view source | ΔLE | Δperiod | Δentropy |
|---|---:|---:|---:|
| Qwen7B | −3.8494901488661526e−8 | 0 | +2.2497195079074217e−16 |
| Qwen72B | −0.013093993101830126 | +2 | −3.53764546823232e−8 |
| Llama3.1-8B | −3.2741809263825417e−11 | 0 | +1.532107773982716e−14 |
| OLMo-2-7B | +0.27283018098625433 | 0 | +3.4508941179946623e−23 |

The source's stable first-copy, adjacent1e-9 suppression is reproduced literally; no averaging or equality tolerance for these other scalars is inferred. Every source copy and its identity remains available. Primary/full ledgers reuse source observations, so their row counts must not be summed as independent data.

The saved descriptive correlations across the 15 listed models are rho=`0.5500631995505209` for intermediate-entropy breadth versus chaotic breadth, and rho=`−0.44275578849066516` for log parameter count versus chaotic breadth. Exact membership, values and ranks are preserved. Unequal grids, temperature support, architectures/training and inherited numerical precision prevent a causal size interpretation. All unclassified no-period rows remain unresolved, including core counts28 for Llama3.1-70B,19 for Llama3.3-70B and15 for the MoE model; they are not silently called periodic.

## Gain: all bins and three explicit P3 views

All 15 source files are present, each with24 saved cells. All 360 rows are finite-qualified; no row falls outside the five declared half-open bins `[0,1),[1,3),[3,6),[6,10),[10,99)`. The medians use **all cells in each bin**, not just chaotic cells. All medians, chaos counts, factor-specific counts, member IDs and missing dispositions are copied in the companion/presentation. Counts below follow that fixed bin order.

| Source label | Five bin counts |
|---|---|
| qwen1.5b-wide | 8, 3, 0, 2, 11 |
| qwen3b-wide | 10, 2, 2, 1, 9 |
| qwen7b-wide | 5, 3, 3, 1, 12 |
| qwen14b-wide | 11, 2, 2, 2, 7 |
| qwen32b-wide | 9, 2, 4, 2, 7 |
| olmo2-1b-wide | 3, 5, 1, 3, 12 |
| olmo2-7b-wide | 6, 1, 2, 2, 13 |
| olmo2-13b-wide | 8, 2, 1, 1, 12 |
| olmo2-32b-wide | 2, 4, 3, 3, 12 |
| llama3.1-8b-wide | 2, 4, 3, 3, 12 |
| llama3.1-8b | 1, 6, 4, 4, 9 |
| qwen7b | 1, 3, 3, 0, 17 |
| llama3.1-70b | 4, 6, 2, 3, 9 |
| llama3.3-70b | 6, 7, 0, 4, 7 |
| qwen72b | 5, 2, 2, 0, 15 |

Empty bins are Qwen1.5B-wide3–6, historical Qwen7B6–10, Llama3.3-70B3–6 and Qwen72B6–10. Their medians remain null, not zero. The P3 tests use only the three frozen middle bins, separately for each larger model and each net/read factor. Several comparisons have only one reference or target cell; no confidence, robustness or asymptotic claim is supplied by these medians.

| Explicit view | Qwen comparisons | OLMo comparisons | Saved aggregate |
|---|---|---|---|
| All15/default | 12 undefined: two true Qwen7B sources, so ambiguous reference | 6 true,6 false | null |
| Manuscript wide7B references | 12 true,0 false | 6 true,6 false | false |
| Historical narrow Qwen7B reference | 8 true;4 undefined from empty6–10 reference bin | 6 true,6 false | null |

These views do not relabel models or fall back when a selector is missing. All15 identities and the default ambiguity remain; the source-selected views are an explicit, disclosed amendment reflecting the manuscript's wider7B comparison and the historical narrow comparison. Their selection followed source/manuscript exposure, not a fresh blinded test. A null aggregate does not negate the defined OLMo counterexamples.

The six OLMo failures in both selected views are the following exact saved larger-minus7B median differences:

| Model | Entropy bin | Factor | Larger n / 7B n | Signed difference |
|---|---|---|---:|---:|
| 13B | 1–3 | net | 2/1 | +0.4930999663950562 |
| 13B | 6–10 | net | 1/2 | +0.8700685012178457 |
| 32B | 1–3 | read | 4/1 | +0.23250210131757024 |
| 32B | 3–6 | read | 3/2 | +0.566075830382788 |
| 32B | 6–10 | net | 3/2 | +0.11984709583038472 |
| 32B | 6–10 | read | 3/2 | +0.8166918318410678 |

The frozen P3 wording says “the14B and32B models,” while the same roster explicitly names OLMo13B. The admitted family-specific interpretation is **Qwen14B/32B and OLMo13B/32B versus their own7B**, not a literal OLMo14B test. Retain that assumption in any claim. Entropy bins provide coarse observational grouping; they neither match all within-bin distributions nor isolate parameter count. Saved finite-twin log-gain summaries are not a native Jacobian proof or a bound on every trajectory. Original plotting stdout/CLI label order is unavailable, so these results are not claimed equal to an unbound historical full-precision gain JSON.

## Llama3.3-70B depth maps

All four saved maps have finite exponents/entropy, integer period values in the writer domain, and complete final-shaped coverage. Their common nominal domain is log10(T)∈[−0.3,0.3], alpha∈[0,0.12]. Each uses the `open`/`Hello!` prefix, window1,300 transient and300 measured steps,80 recorded states, max detected period32, tolerance1e−8 and twin epsilon1e−6. Args declare float64 computation with bf16 weights and `device=auto`; those declarations and original source/archive association do not independently establish effective historical native precision, split-device equivalence, loaded model bytes or all RMSNorm paths. Per-layer steering vectors were constructed separately. Layer50 has a32×32 grid; the others24×24. Equal alpha therefore does not make this a matched perturbation or identical-coordinate comparison.

The writer's named “positive” count is LE>0.001, without requiring no detected period. The own descriptive no-return qualification adds period0 and finite validity. The one layer20 difference is retained, not repaired. `U` means period0 without LE>0.001; a detected finite return is not a certified primitive orbit.

| Layer | Grid | Writer LE>0.001 | No-return qualified | Qualified share | Detected-period cells | U | Maximum LE |
|---|---|---:|---:|---:|---:|---:|---:|
| 20 | 24×24 | 47/576 | 46/576 | 0.0798611111111111 | 518 | 12 | 0.6149944263114338 |
| 35 | 24×24 | 25/576 | 25/576 | 0.043402777777777776 | 542 | 9 | 0.41826610845363027 |
| 50 | 32×32 | 44/1024 | 44/1024 | 0.04296875 | 512 | 468 | 0.4363767113946329 |
| 65 | 24×24 | 4/576 | 4/576 | 0.006944444444444444 | 571 | 1 | 0.18040902999951214 |

All four per-scan writer cell/positive counts, period histograms and LE extrema reproduce the saved summaries. Complete histograms, raw period-value representations, all cell flags and three layer-versus50 comparison records remain in the presentation. Qualified-share differences relative to50 are **+0.036892361111111105** for20, **+0.00043402777777777624** for35 and **−0.03602430555555555** for65. These descriptive differences do not have a new uncertainty model or common-resolution adjustment.

The actual minimum chaotic sampled log10(T) values are −0.14347826086956522,0.013043478260869545,−0.06774193548387095 and0.19565217391304346 at layers20,35,50,65 respectively; each maximum is0.3. The full exact extrema and outward binary intervals are retained. These are observed sample extremes, not continuous onset boundaries. Layer65 still has4 qualifying cells, so neither disappearance at deep layers nor global absence in70B follows. Neither these maps nor their histograms certify shrimp geometry.

The generic saved depth field `historical_70b_stricter_scorer="unavailable"` is **overbroad specifically for baseline50**. Root had already accepted `/tmp/task25-depth50-metadata-001/inventory-authority.json` (SHA `cf516769371fbaf8e6349cf5a5777c78332c7ea9109afff3a0562293acc9514e`) and its original `tools/analyze_controls.py` (SHA `6d72d4cbd233815f80c7e486f4030c1752906ded8c2a6ba2fa70cb5f18a60ead`). That source uses LE>.001, P<=0 and non-NaN. Its existence is distinct from the writer's exponent-only count and from a dedicated historical scorer for the new L70D layers. On these finite valid period-domain cells, the own period0 qualification agrees with that source definition; this note does not rerun it. Preserve the generic output bytes and attach this source-authority clarification. Initial/tokenizer and post-save plotting failures remain in the original authority. Final NPZ availability is not an erasure of those failures.

## Publication dispositions and suggested wording

1. **Confirmed arithmetic:** the bound scale publication projection has663 exact shared JSON leaves and zero differences. This does not mean every field in the24.85MB combined comparisons file is a shared published value: own witness/representation fields are separate. Gain has no bound historical printed/full-precision comparison; depth uses its per-scan comparisons rather than inventing a dedicated historical L70D score.
2. **Required range qualifier:** “Within the frozen log10(T)∈[−1,0.3] core, the sampled Qwen chaos share decreases from7B to14B/32B/72B. OLMo32B violates the corresponding decline. Wider scans find53 chaotic Qwen14B samples, with saved extrema0.5194968553459121 and1.1438095238095238.” Do not turn sample extrema into a filled interval or a universal contraction bound.
3. **Required P3 qualification:** “Using the declared wider7B references and all-cell medians in the three fixed entropy bins, Qwen satisfies all12 factor inequalities; OLMo violates six of12. Some bins contain only one or two observations, and the historical narrow-Qwen reference leaves four inequalities undefined.” This supports a family- and selection-specific observation, not a generic size mechanism.
4. **Depth wording:** “The saved Llama3.3-70B maps yield46/576,25/576,44/1024 and4/576 finite LE>.001/no-detected-period cells at layers20,35,50 and65. The maps differ in resolution and separately constructed steering vectors; they establish finite sampled depth dependence, not disappearance, identical perturbations or verified geometry.” If quoting the original writer counts, use47/576 at20 and name the exponent-only definition.

## Operational and precision limits

The worker's **7.507569101871923s** is measured through component serialization **before** final rehash/report, with **236656KiB** observed RSS at that point. Inference's separate supervisor window is **7.683673759s**. The supervisor supplies the bounded external elapsed wall; the worker timing and RSS were observed before final rehash/report. No independently measured whole-worker peak is available. The authorized saved-presentation pass completed exit0 under CPU1/nice10/1GiB/30CPU/60wall; its stored observation is0.48474294506013393s and117040KiB before the scope-file write, with direct tool wall about0.5501s. These timings are not new scientific evidence.

The original report's inherited preparation sentence that selected arrays “remain unread until separate admitted scoring” is stale prose after the admitted run. Preserve the original bytes and read it alongside the actual completed output and owner completion clarification; it is not evidence that the saved scoring did not run. Original local archive/member/source/receipt associations are finite provenance, not an invented historical loaded-source/model-weight/native-precision attestation. No selected later patched run silently replaces an earlier full map. No favorable row, duplicate, empty bin, unresolved cell, failed historical attempt or nonzero scalar discrepancy is removed.

Complete stored scalar selections and row-flag inventories: `/tmp/task25-sgd-presentation-001`. Compact coverage and full literal summary tables: `/tmp/task25-scale-gain-depth-coverage.json`. Original authenticated outputs remain authoritative. No further computation is requested by this review.
