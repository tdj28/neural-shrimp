# PM2F independent saved-scalar review — completed recovery run002

The saved arithmetic supports the reported answer-flip counts and the endpoint-based summary: all 42 published numerical model leaves reproduce exactly. It does not establish an exclusive explanation in terms of bfloat16 rounding or near ties, nor an asymptotic chaos diagnosis. The manuscript should distinguish endpoint separation relative to the **nominal** perturbation scale from the fitted slope on a restricted part of the distance history.

This is an independent, unblinded saved-row reconstruction and subsequent reading of its completed outputs, not a new model experiment. The sole successful scalar audit covered all 288 base problems and 576 twins. Its ten manufactured controls passed. I read all saved model summaries, both complete comparison lists, every base/twin diagnostic disposition, and the original receipt/output linkage. No producer code, native distance trace, model, fit, or scalar reducer was rerun in this final review.

## Complete finite results

Model abbreviations below are the original identities: R1-Qwen = DeepSeek-R1-Distill-Qwen-7B, R1-Llama = DeepSeek-R1-Distill-Llama-8B, and QwQ = QwQ-32B. This audit does not independently retest the separate Qwen2.5-7B evidence.

| Saved quantity | R1-Qwen | R1-Llama | QwQ |
|---|---:|---:|---:|
| Base problems / twins | 128 / 256 | 128 / 256 | 32 / 64 |
| Base answers correct | 109 / 128 | 102 / 128 | 25 / 32 |
| Base accuracy | 0.8515625 | 0.796875 | 0.78125 |
| Endpoint above nominal eps0 | 40 | 145 | 34 |
| Endpoint below nominal eps0 | 216 | 111 | 30 |
| Endpoint equal / undefined | 0 / 0 | 0 / 0 | 0 / 0 |
| Twin answer flips | 6 / 256 | 47 / 256 | 7 / 64 |
| Twin flip fraction | 0.0234375 | 0.18359375 | 0.109375 |
| Problems with either twin flipping | 5 / 128 | 28 / 128 | 4 / 32 |
| Different thinking-switch step | 2 | 27 | 13 |
| Twin switches earlier than base | 0 | 14 | 7 |
| Positive / negative / undefined fitted rate | 115 / 37 / 104 | 200 / 29 / 27 | 50 / 3 / 11 |
| Positive fit but endpoint below eps0 | 76 | 57 | 16 |
| Maximum excursion above / below eps0 | 243 / 13 | 254 / 2 | 62 / 2 |

All maximum excursions are defined; none equals eps0 exactly. Every non-null scalar is finite. All 288 independently reconstructed base correctness flags and all 576 twin answer-flip and same-switch flags agree with their saved flags. There are no duplicate source indices or nonempty per-row diagnostic lists. The original grouping is retained: 16, 16, and 8 groups respectively. Missing fitted rates are genuine undefined-fit dispositions, not zero slopes.

The two twin directions remain separate; their aggregate agreement must not be substituted for full row coverage:

| Model, twin | n | flips | same switch | endpoint above / below | fitted + / − / undefined | max above / below |
|---|---:|---:|---:|---:|---:|---:|
| R1-Qwen 0 | 128 | 3 | 127 | 19 / 109 | 55 / 17 / 56 | 123 / 5 |
| R1-Qwen 1 | 128 | 3 | 127 | 21 / 107 | 60 / 20 / 48 | 120 / 8 |
| R1-Llama 0 | 128 | 22 | 114 | 73 / 55 | 98 / 19 / 11 | 126 / 2 |
| R1-Llama 1 | 128 | 25 | 115 | 72 / 56 | 102 / 10 / 16 | 128 / 0 |
| QwQ 0 | 32 | 3 | 24 | 18 / 14 | 25 / 1 / 6 | 31 / 1 |
| QwQ 1 | 32 | 4 | 27 | 16 / 16 | 25 / 2 / 5 | 31 / 1 |

The scalar fitted-rate distributions, copied from the completed independent output, are:

| Model | defined n | minimum | Q1 | median | Q3 | maximum | mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| R1-Qwen | 152 | −0.46323771999334795 | 0.00016101904457060059 | 0.004124054793284102 | 0.0098380135729206 | 0.06287427964354331 | −0.0013389762770220318 |
| R1-Llama | 229 | −0.40684565275786 | 0.004464428135007769 | 0.008201658433743233 | 0.012149097258858998 | 0.11411611968806545 | 0.006305631949178879 |
| QwQ | 53 | −0.020055363243012582 | 0.005780467282503865 | 0.009694848811567047 | 0.014160561866878855 | 0.2352488468547374 | 0.02375140028171656 |

These medians are over **all defined slopes**, not the endpoint-growing subgroup. The fitted rate is the slope of natural-log relative distance versus retained sample index, using only samples satisfying `10*eps0 <= d <= 1e-3` and at least two qualifying samples. It is not an endpoint log-ratio divided by a common horizon. In particular, the positive-fit/end-below counts 76/57/16 make the distinction concrete; R1-Qwen's positive median also coexists with a negative mean and a large endpoint-below majority.

The following are exact saved scalar extrema, not newly estimated native-distance bounds. Values are base-10 logarithms of relative separation:

| Model | max-over-window log: min / max | endpoint log: min / max | endpoint log median |
|---|---|---|---:|
| R1-Qwen | −10.000000008321244 / 0.436275279464522 | −300 / 0.3033511200330593 | −15.889049325857599 |
| R1-Llama | −10.00000000399003 / 0.4980070333346029 | −17.64771031177309 / 0.34350810460975434 | −7.550582514634865 |
| QwQ | −10.000000003740745 / 0.3941046434333723 | −16.887436953312367 / 0.27344093091369315 | −8.992541805784752 |

Thirteen R1-Qwen endpoints hit the writer's −300 floor. This cannot distinguish exact zero from a separation below the floor. Tiny maximum-log differences below −10 remain visible; no tolerance was introduced to relabel them. The base switch ranges are 344–1536, 396–1536, and 381–1536, with medians 793, 1184, and 924 respectively. These are differing finite windows, not a uniform thousand-step observation.

## Published summary and later source clarification

The frozen endpoint projection has **42/42 exact numerical model leaves**. The separately predeclared fitted-sign interpretation has 36 exact leaves and these six intentional differences, all retained:

| Field | own fitted-sign interpretation | published endpoint value | signed difference |
|---|---:|---:|---:|
| R1-Qwen grew | 115 | 40 | +75 |
| R1-Qwen shrank | 37 | 216 | −179 |
| R1-Llama grew | 200 | 145 | +55 |
| R1-Llama shrank | 29 | 111 | −82 |
| QwQ grew | 50 | 34 | +16 |
| QwQ shrank | 3 | 30 | −27 |

They are differences between definitions, not failures of endpoint reproduction. No favorable projection was selected after outcomes. The top-level eps0 is independently checked as 1e−10; the summary's `run` and `source` strings remain inherited descriptive labels, not reconstructed numerical claims.

After the original preparation froze, producer commit `5a507a2a74c89a51265acc0e92bf33275c89c788` supplied `pi_f64_reason_summary.py`, SHA256 `bc8f4b90caf3c1f6b791bef1215e968899f52519995dff40076b7e94ffa80dc0`. Source lines 30–42 explicitly define grew by endpoint `> log10(eps0)` and shrank by `<=`, with growth-rate quartiles separate. This corroborates endpoint semantics, but does **not** establish the historical invocation that produced the older summary. The frozen own projection kept exact equality separate from strict shrinkage; all three actual equality counts are zero. The later implementation also adds intervals and indices outside the frozen summary comparison; no new interval calculation is claimed. Original missing-summary-source prose and original output bytes remain unchanged, with this dated clarification alongside them.

## What the saved measurements do and do not establish

The original writer is `21d300a6c3a6f73a5f098f8a3c8387f7b668a71b`; `pi_reason_f64.py` has SHA256 `4be2bbf44da72084053e3193e68ed32014101ba43b22d531804242f27ba971e3`. Its source-level float64 path uses bf16-stored decoder weights converted for float64 inputs, float64 non-decoder/norm/embedding state, the reviewed RMSNorm/rotary changes, and loader/hook checks. This supports the declared implementation, not independent verification of every historical GPU kernel or model-weight byte. The original recorded hardware is H100 NVL for the two smaller runs and H200 for QwQ. No CPU/model replay or fresh hardware comparison was performed.

The original nominal perturbation is a relative eps0=1e−10 random-direction injection into the first concept token. The *realized* initial separation was not saved. Directions are seeded by group, and base/twins share the per-problem random-number schedule; there is no repeated renormalization. Consequently “ended below nominal eps0” is justified, whereas “closer than they started” is not independently verified by these records.

The writer measures fed-input relative distances before the model call, truncates each twin series at the **base** thinking-switch index, removes NaNs, then fits using the compressed retained index. Fourteen R1-Llama and seven QwQ twins switch earlier than their base. Their base-defined window can therefore include different thinking/answer pipeline phases. Native distances, realized perturbation norms, states/caches, sampled directions, and complete tokenizer/text evidence were not retained. The audit can reconstruct saved scalar classifications and answer-value flags, but cannot independently reconstruct those fits, maxima, final native distances, or answer extraction from tokens.

The answer counts are finite sensitivity observations for this seeded, switched inference pipeline. They are not an asymptotic Lyapunov estimate, a proof of deterministic chaos, or a causal partition of layout effects into rounding and autonomous amplification. Two twins of a problem are dependent; the 576 twin outcomes are not 576 independent problem replicates. No new confidence interval or population-level generalization is introduced. The 102/128 R1-Llama base accuracy here does not retest a separately queued shared-problem precision comparison.

## Concrete manuscript corrections

Frozen manuscript: `/tmp/claude-task26-27-20261017-source-001/tree/paper/llm/main.tex`, SHA256 `910c8e37a201c769937b3e524417d3adcbdb653bfd8bc1bbde808ad6c7c929e7`.

| Lines | Disposition | Correction |
|---|---|---|
| 47–50, especially 48–49 | Must qualify causal attribution | The endpoint-below majority in R1-Qwen does not demonstrate that its layout flips are mainly rounding carried across near ties. PM2F does not save a near-tie measure or isolate competing causal contributions. Avoid “so” or “therefore” assigning an exclusive mechanism. |
| 930–932 | Must specify reference and window | Replace “started 1e−10 apart” with a nominal relative 1e−10 injection, and “closer … than they started” with “endpoint below nominal eps0 in the base-defined pre-switch window.” |
| 933–936 | Must separate estimands | Retain 34/64 and 145/256 endpoint-above counts, but report the 0.00969485 and 0.00820166 fitted medians separately, with n=53 and n=229 and their restricted fit window. Do not label these as the net growth rates of the endpoint-growing subgroup. |
| 934–937 | Confirm counts; qualify interpretation | 7/64 and 47/256 answer flips support approximately 10.9% and 18.4%. “Weak finite-time instability” should be framed as finite pipeline sensitivity, not a new chaos/causal finding, and the mixed-phase and varying-window limitations should be visible. |

Suggested replacement for the PM2F part of the paragraph:

> With a nominal relative 1e−10 perturbation of the first concept token and common random numbers, float64 copies changed 6/256 R1-Qwen-7B answers, 7/64 QwQ-32B answers, and 47/256 R1-Llama-8B answers. At the endpoint of the base-defined pre-switch window, separation was below nominal eps0 for 216/256 R1-Qwen copies and above it for 34/64 QwQ and 145/256 R1-Llama copies. Separately, slopes fitted only where 10 eps0 ≤ d ≤ 1e−3 had medians approximately 0.009695 and 0.008202 per retained step for the 53 and 229 defined QwQ and R1-Llama fits. Positive fitted slopes can coexist with an endpoint below eps0, and some twins switch pipeline phase earlier than the base. These results show finite perturbation sensitivity of the complete pipeline; they do not by themselves identify an exclusive rounding mechanism or establish asymptotic chaos.

The separately supplied current Chaos article snapshot, `/tmp/claude-task27-20261017-chaos-source-002/main.tex` (SHA256 `0fbc6d5215c0737570cfe0c50245a39d3bec128ba30f9cc7efec9aa5b70c8ee8`), lines 51–52 says that “the float64 loop itself grows slowly” in two models and gives an 11–18% answer-change range. This has the same slope/endpoint and pipeline-scope problem. Replace it with: “Float64 copies with a nominal relative 1e−10 initial perturbation changed 7/64 QwQ and 47/256 R1-Llama answers; about half ended above the nominal perturbation scale.” If percentages are given, name the models and use approximately 10.9% and 18.4%, rather than an unqualified 11–18% extremal range. This snapshot is a current local copy, not an assertion of its committed historical source. Only its assigned PM2F abstract sentences were checked; no broader article or blog audit is claimed.

## Failure history, authentication, and coverage

Original PM2F metadata retains all eleven original attempts: successful A/B, eight C startup failures, then successful C. Successful archived selected members match the admitted local records/configs. Historical source/receipt association is finite evidence, not a new observation of those remote processes.

Own `run-001` failed `duplicate_source_rows_retained` before target analysis: an existing list was aliased before appending the current line, so a duplicate witness incorrectly included its own line. Its nine passing groups, one failure, source, failure and receipt remain preserved. The reviewed correction snapshots that prior list before appending; it changes duplicate-witness bookkeeping only, not scientific thresholds or either projection. Separately admitted `run-002` passed all ten groups and performed the first successful actual saved-row analysis.

The final authority is `/data2/neural-shrimp-data/codex/new-results-review-20261017/run-002/pm2f-original-closure.json`, SHA256 `b42a05bd61a0a05fcfbbfc4487e344fb373e5a69dc6fc1d974baaedf3d8bca02`; receipt `pm2f-supervision/receipt.json`, SHA256 `b8c4967d17c8a43351e118981056dfb59cf5796c92578d5807cfe66b83b16854`. Inference directly observed exit0/reaping and original supervisor/worker birth, group and session absence, and authenticated five source blobs plus five outputs. These are Inference's lifecycle observations, not my own process inspection. Actual own source is `8f4e96a1601deae8cb58ad33ba2c82524bc1c5df`.

The saved-only presentation authenticated all five worker files using receipt hashes and sizes, then retained literal fields and complete disposition inventories without rerunning the reducer. Exact principal artifacts:

- Worker `report.json`: `85e11fa9186b7ca27ff1bbfabc348a2d6726a6a23907bc11695f4d33beeb05e0`.
- Worker `models.json`: `1551b34286f4de7a6d32bbf229a14589319c61a62e4376e3d5414343bcef03ba`.
- Worker `comparisons.json`: `95b8aa561bee716c7098ffeafee12207630a55ba9e157588be31fbc48e5fbde6`.
- Worker `manifest.json`: `02aefa41d13f58dd730965663ed1b76375ce74313a2f23214fc600d6a54c7c93`.
- Full literal presentation `saved-coverage.json`: `18201a6ae18ed35bc4368afadce2aac7613df7652d13511e6419570108f59701` (704349 B).
- Full compact per-row table `all-row-table.txt`: `6017ea22a5ebd99ed31cb52eaee5eaf1835b4c10e7679c9fc0bf5857c4e80a15` (92855 B).

Both presentation files are beside this note. All 288 row and 576 twin records, both comparison definitions, every undefined rate, the contrary positive-fit/end-below cases, and all inherited scalar extremes remain available there. No extra verifier was run. Supervisor external elapsed is 0.3406059290282428 s; worker timing is 0.17415966605767608 s through posthash before final report, with 38508 KiB observed RSS at that point. The five outputs total 746395 B. The worker observation is not a separately measured whole-worker lifetime/peak; the supervisor supplies the bounded external wall interval.
