# PM/PM2 independent saved-row audit preparation — 16 October 2026

Status: **prepared, not an admitted numerical run**. Scope is NEED_TO_AUDIT items1–2. All new files are under `/tmp/need-to-audit-20261016-pm`; Inference owns freezing, Git, source capture, execution and closure. No model, producer/analyzer import, target numerical reduction or manufactured control execution has occurred. Current producer source/paper reading already exposes the stated outcomes, so this independent audit is unblinded; it does not re-register the original tests.

## Frozen scientific contract

The designs are `docs/pi-matrix-confirmatory.md` (including its pre-run hardware amendment) and `docs/pi-matrix2-confirmatory.md`. Current source authority is Inference's `/tmp/need-to-audit-20261016-source-001/manifest.json`, SHA256 `3fcbde19749f1cfe53e5309283e0b7a551dff616354e85502ff094abf9c82963`. The original runtime/design ancestry, start chronology and exact analyzer argument order still need the original metadata packet; current source is not automatically historical executed source.

PM tests the primary top15-s64 answer-flip rate >.10 with its cluster95 lower endpoint >.05. Its seven arms use a sequential seed20261016/10,000-draw bootstrap. A cluster is a larger-layout batch, and PM clusters need not contain16 scored problems: same-length test fillers are unscored. PM2 tests the five designated primary soft cells' pooled rate >.05 with lower endpoint >.02, resampling clusters within each cell and weighting by problems. D is a separate A100 replication, not a primary cell. Every missing, partial and unpaired row remains visible; missingness is not automatically failure or success of every other cell.

Independent `core.py` reconstructs paired rows from JSONL, per-layout accuracy/missingness/exactly-one-correct, cluster membership/counts, Wilson intervals, source-scheduled cluster bootstrap, exact two-sided McNemar, all secondary and cross-GPU contrasts and the pooled primary. It imports only stdlib/NumPy, never producer code. The original source projection retains its last-write duplicate semantics, but a separate full duplicate ledger keeps both answers/labels, line numbers and differing fields. Gold mismatches, row-index/layout mismatches, unexpected layouts, incomplete pairs and cluster consistency stay explicit.

Historical argument order matters: PM per-arm draws precede contrast draws; PM2 performs soft/plain/contrast draws per cell, then the pooled draws, then cross-GPU contrasts. The pooled stage uses first-encounter cluster order from the file, whereas per-arm/paired stages sort problem IDs. Exact published-CI reproduction requires these recorded orders. If unavailable, use a disclosed canonical order and report inability to attest the original RNG schedule; never search permutations to fit a receipt. Keep all numeric differences, including tiny floating-point ones, without an added tolerance.

## Answer extraction and limitations

Literal frozen endpoints use stored answer/correct fields. The auditor independently reconstructs extraction from each retained text suffix and verifies correctness against the stored gold. PM retains only the last400 characters; PM2 only the last600, after extracting from the full decoded answer. A differing suffix extraction is therefore not by itself proof that the original extraction was wrong. Full answer text/tokenizer evidence is needed where the lost prefix matters. PM2 does not save full answer token IDs, so this audit must not claim to decode an unavailable token sequence.

R1-Llama's visible byte-level space/newline/tab markers are retained literally. A separately labelled diagnostic maps `Ġ`→space, `Ċ`→newline and `ĉ`→tab before applying the same boxed-number rule. It is not full tokenizer reconstruction and cannot restore a discarded prefix. Raw-source, stored and marker-normalized answers/correctness/flips remain separate; no endpoint is silently overwritten. The eventual report should show whether any interpretation changes accuracy or flip counts, with all affected row IDs. This is a source-definition diagnostic, not a new model experiment.

The writer's stored `top1` series supports a finite first differing saved readout step over the common retained prefix. It does not supply native state divergence. PM2 forced-thinking summaries follow the source's larger-layout records, including any unpaired records, and expose that denominator rather than silently substituting paired-only rows.

## Metadata and admission needed

- Exact19 run/record pairs: seven PM arms and twelve PM2 arms; every original config, model/task/split/layout/budget/seed field, hardware label, model path and source association.
- A lives under `/data2/neural-shrimp-data/explore/pi/pm2-A/explore/llm-soft-loop/pod64/pi/pm2`; D's identically named Qwen-GSM folders live under the ordinary PM2 root. Require distinct pinned paths/content and the H100/A100 metadata association; never identify a cell from its basename alone.
- Original `/data2/neural-shrimp-data/runpod/{PM,PM2-A,PM2-B,PM2-C,PM2-D}-orchestrator.log` scoped commit/start/finish/termination lines, exact executed source and design commit timestamp/ancestry. Chronology is supported by these finite recorded facts, not an invented independent timestamp or historical process attestation.
- Pinned PM/PM2 published summary JSON, analyzer invocation order or an explicit absence, current paper hash/line numbering, source manifests and recorded failure/retry history.
- Cheap metadata byte/hash/schema/line-count timing, not source-preparation elapsed presented as a numerical forecast. Inference is collecting this once; no duplicate raw inventory is needed here.

The actual driver will bind that concrete packet, hash before/after, require a fresh output directory, retain failures separately and distinguish computational completion from favorable scientific verdicts. Proposed envelope, still unadmitted: CPU1 plus supervisor, nice10,1GiB AS,55CPU/60wall,32MiB output. The source-only manufactured control set has12 focused groups and one complete seven-arm/six-cell synthetic schedule through serialization; it measures the full scheduled bootstrap workload and output, not native model work. Exact card follows metadata and Root review. No tests have run yet.

## Current paper claims to check after the sole admitted run

Line numbers refer to the source first read from `/data2/neural-shrimp/paper/llm/main.tex`; final findings must use Inference's frozen identical snapshot or disclose any delta.

| Lines | Claim | Required finite check |
|---|---|---|
|44–47|PM21%; PM2 well-answering models86–90%, flips15–20%, correctness changes12–15%|Exact count denominators and true min/max, named model/task/GPU/arm/layout subset; distinguish approximate rounding from enclosing bounds.|
|141–143|PM21%, replay6%; PM2 accuracy83–92%|Same raw units; determine whether83–92 covers both soft/plain while86–90 describes soft only. Do not silently mix A and D.|
|860–866|Pre-data freeze, H100,256 fresh training problems,54 flips,77-cluster-style interval|Original chronology/config/index coverage; actual variable scored cluster sizes and full sequential bootstrap. Sample freshness relies on retained prior-use authority, not model recomputation.|
|867–874|Replay,16-step,greedy,top1 contrasts and McNemar values|Every paired ID, discordance, unrounded CI and p; retain uncorrected multiplicity status.|
|875–877|“clear null” and four-test correction|A CI containing zero does not establish equivalence; suggest “no statistically resolved difference under this test.” Any multiplicity claim must name the four prespecified secondary predictions and method.|
|877–880|Full-vocabulary rate,22–30% accuracy,6.2% correctness changes|Named arm/layout scope and actual extrema.|
|901–910|PM2 fresh selection, five primary cells,265/896 and pooled interval|Every primary role, partial completion and cluster order; replication D excluded.|
|911–914|Qwen/QwQ per-cell flip rates, accuracy and correctness changes|Both layout accuracies and exact extrema, soft-only subset.|
|915–917|No soft/plain significant difference; effect attributed to feedback|All six cells and both endpoints, including adverse/missing rows. Nonsignificance is not equivalence or proof of a unique causal mechanism.|
|918–923|A100/H100 difference, MATH caps, R1-Llama accuracy|Correct separate A/D paths; all forced-step denominators; stored versus suffix/byte-marker extraction for Llama.|
|924–925|Cold Stop omitted|Source/config scope; “published settings” must acknowledge implemented stopping/budget and pipeline differences.|

The current source references later float64/horizon evidence in neighboring lines; those are separate queue items, not silently re-audited or used to prove PM/PM2 native chaos. This audit establishes finite batch-layout answer dependence and scoring/provenance fidelity, not a new geometry, chaos or causal theorem.

## Concrete source/input amendment (16 October, still unexecuted)

The metadata packet is now bound by `inputs.json`: all19 original JSONLs contain8192 saved records and36,729,067 bytes; original configurations, five complete orchestrator logs, historical source and published-summary order are explicitly selected. The earlier source/design/inventory release is retained in `preparation-001`. This is a metadata-only selection, not raw answer/statistic decoding. The source-only selection took about0.007 seconds reading530,721 metadata bytes; this is not a scoring forecast. Inference's0.5995s/79.12MB measurement had a broader, multi-lane metadata scope.

Historical PM source0f24b344 differs from the frozen current analyzer only in its `cfg['mode']` versus fallback to`cfg.get('arm')`; PM configs have`mode`. PM2 analyzer, harness and reasoning writer at3639819a match the frozen current relevant source exactly. The exact historical analyzer CLI order was not recovered. The saved published mapping order is therefore a disclosed reconstruction proxy: PM top15-s64,replay,greedy,top1-s64,full-s64,top15-s16,full-s16; PM2 A-gsm8k,A-math,B-gsm8k,B-math,C-gsm8k,D-gsm8k, followed by A/D cross-GPU comparisons. No order permutation is searched and every resulting leaf delta is retained. Source metadata/pod timestamps establish recorded chronology, not an independent historical clock or loaded-weight attestation.

The single proposed command is `python -I run.py --card CARD --out /data2/neural-shrimp-data/codex/need-to-audit-20261016/run-001/pm`. It runs the12 manufactured groups and complete source-scheduled synthetic workload once under `controls/`, then the admitted actual saved audit. A manufactured failure stops before actual scoring. Prepared admission refuses execution. The driver authenticates each selected pin before/after, binds every run config literally, separates A's retained root from D's ordinary root and preserves every log line. Output is `report.json`, `row-diagnostics.json`, `comparisons.json`, `chronology.json`, `manifest.json`, plus `controls/manufactured.json` and `controls/report.json`. The manifest inventories the other six files. Computational completion is distinct from scientific agreement or favorable primary outcomes. Input hash ceiling48MiB and output32MiB are finite hard failure ceilings; the coordinator supplies CPU/AS/wall/process limits and serial ownership. No numerical run or manufactured check has yet occurred.
