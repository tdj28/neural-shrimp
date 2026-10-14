# Task18 independent loop-gain crosscheck: recovered own results

**Status:** the completed scientific events support the finite numerical loop-gain result. Original `run-002` remains a failed execution with exit1; its separate saved-summary recovery completed. **Producer numerical comparison is pending.** This report does not substitute the recovered result for the original failed receipt, and does not claim a new interval certificate.

## Evidence read and scope

This note reads the exact recovered JSON at `/data2/neural-shrimp-data/codex/llm-loop-gain-crosscheck/run-002-summary-recovery-001/report.json`, SHA256 **f8f42ddb1dcd25ce13cd602f2a8aa670335d79b72cc0fbd21f5aed3a46825fef** (892,971 bytes). I read all eight phase summaries and gates, all64 directional rows, all eight cyclic summaries with their leading signed/complex multipliers and singular values, all three composed controls, original authority/input binding, limitations and56 reconciliation rows. No model, array reduction, dense product, JVP, eigensolve, SVD or bootstrap was run for this report. The previously released theory/source review is `/tmp/task18-task19-source-review-93dfb761.md`, SHA256 **5eafb53ad4c1638e00bc3c09bdd40ca13893de1c3410cbfbb0ed38a378911211**.

Original source **7918af949fa60cb8bb123504e78c2dc959943d9c** uses the independent ManualQwen implementation: Qwen2.5-0.5B-Instruct pinned checkpoint, dimension896,24 layers,151,936-token vocabulary, tied embedding/readout, full-vocabulary softmax, philosophy vector at layer12 and the retained37-token prefix. The fixed parameters are alpha0.071674 and log10(T)=−0.115. After1000 full forward steps from the fixed start it retained eight consecutive phases and the closing image. State derivatives hold the steering parameter, temperature and regenerated prefix fixed. These are floating-point evaluations of the declared map at its literal binary64 parameters. Model/prefix/parameter pins do not by themselves identify a previous exact-real interval witness or prove equality to a separately rounded/dyadic parameterization. This report makes no such identification.

The factorization under review is

\[
 F(x)=E^\mathsf{T}p(x),\qquad
 p(x)=\operatorname{softmax}(Eh(x)/T),\qquad
 DF(x)=\frac1T\,\Sigma_p(x)\,J_h(x),
\]

where \(\Sigma_p=E^\mathsf{T}(\operatorname{diag}(p)-pp^\mathsf{T})E\), and \(h\) includes final normalization. The independent code acquired all896 coordinate directions per phase for the hidden map and full map, plus eight separately generated original-map directional checks at each phase. Saved numerical agreement supports the implementation of this chain rule; the algebraic identity is distinct from the finite error measurements.

## Own numerical results

The eight-step relative closure is **3.250425216758018e−15**, and the minimum pairwise relative phase separation is **0.1503826910030647**. Both frozen numerical closure/distinct-phase gates passed. This supports a numerically primitive eight-cycle at the evaluated point; exact existence/primitivity is not newly certified by this floating computation.

Every phase passed all three saved factorization, dense-direction and covariance gates. The reconstructed readout matched the original ManualQwen forward image with recorded relative difference0 at all eight phases. The largest directional factorization error among all64 original-map directions was **8.265928151997498e−14** (phase6, direction0). This is different from the largest dense-matrix-versus-direct-direction error, **3.011501335615533e−13** (phase6, direction6). The largest full dense factorization relative error was **3.442218647050999e−14** at phase6. These are the original relative Euclidean/Frobenius error metrics, not maxima of elementwise relative errors.

| Phase | Step σ1 | Step σ2 | Step σ2/σ1 | Covariance λ2/λ1 |
|---:|---:|---:|---:|---:|
|0|11.0912516613|7.91814721659|0.713909255542|0.406434692934|
|1|16.6834401047|3.81088265846|0.228423073093|0.667441075105|
|2|20.9077769571|17.1937369351|0.822360835889|0.310246803732|
|3|14.7123596446|4.81556419176|0.327314197593|0.435183695623|
|4|5.36504003307|1.68767818872|0.314569542505|0.351450900467|
|5|2.61888828528|2.33984387249|0.893449287485|0.738006291312|
|6|1.24008195358|0.189163134453|0.152540833214|0.236303659446|
|7|10.7688562738|2.12853668252|0.197656708234|0.697988458603|

Thus **each step expands at least one direction**: its largest singular value exceeds1. The measured range is1.2400819535801617–20.90777695714715. Saying “approximately1.24–20.9” is legitimate rounding; claiming a literal upper bound20.9 is not. The minimum step singular ratio is **0.15254083321426204**, approximately0.153, not a literal lower bound≥0.153. The covariance is not numerically rank-one on these points: its second/first eigenvalue ratios remain substantial, and the minimum saved smallest covariance eigenvalue is positive, **4.863336302525681e−10**. This numerical observation is not a formal eigenvalue enclosure or an exact support/rank claim.

The ordered eight-step derivatives have a leading signed multiplier approximately **+0.0110401558701276** and a second signed multiplier approximately **−2.9611745420e−7** at every cyclic start. The negative sign must not be dropped unless the quantity is explicitly called a modulus. The first two saved singular values depend on the starting phase:

| Cyclic start | Return σ1 | Return σ2 |
|---:|---:|---:|
|0|0.192899120094848|7.77541905358518e−6|
|1|0.260373982638411|1.23133563240175e−5|
|2|0.0594734892406083|9.44510982802735e−6|
|3|0.0897852303183271|1.12648101604166e−5|
|4|0.122954021251983|2.40137522864786e−5|
|5|0.622064367132936|2.44286984088048e−6|
|6|0.643702293505275|1.10471014952189e−6|
|7|0.328753490253188|2.97681624852712e−6|

In particular, **0.19289912009484797 is the phase0 Euclidean return norm**, not a phase-independent norm. All eight observed cyclic return norms are below1, with the overall range0.05947348924060834–0.6437022935052745. The product of the individual step norms is approximately **1.0679877352872368e7** (the last few bits vary with multiplication order). The much smaller norm of the ordered product demonstrates the numerical importance of directional alignment and composition. It does not mean the product-of-norms inequality fails, nor that individual step norms can never give useful sufficient conditions.

The three direct composed-map directional controls at phase0 all have recorded primal relative difference0 and derivative relative errors **1.1169603757423053e−13**, **4.731808350561387e−13** and **1.8022422904377204e−13**, respectively. They pass the frozen1e−8 derivative/1e−12 primal gates. These support the ordering and state clock of the saved composition. The small second return singular value describes local linear compression of this evaluated return derivative; it does not prove a one-dimensional invariant nonlinear attractor.

## Failure, recovery and provenance limits

Original `run-002` finished all scientific event groups but its final status JSON failed on a NumPy scalar boolean; the subsequent exclusive failure-status write also failed because the partial file already existed. Its original exit remains **1**, with recorded launcher elapsed **2031.384801283013s**. The truncated status, stderr, original source and original receipt remain immutable. Original peak RSS and final backend identity were not recorded in complete terminal metadata and are **unavailable**; registration RSS or the recovery process's current versions are not substitutes. Exact source/asset admission and recorded dtype/interface checks remain finite premises, not a fabricated full backend record.

The separate recovery source **49006cea46e3a4c5891fdb5d55a5146988b05a0ad29adb6fd67e5b70ac3a7648**, admitted card **2ad4e639b87f35afcc8c653be76ff15efe1caaba0ef39ccb9907fa28f8785ef0**, reports `saved_summary_recovered`,7925 passed checks, complete scientific events and passed saved numerical gates. All56 saved-scalar reconciliation entries have zero nonzero differences and zero maximum absolute discrepancy. The recovery inherited the accepted original Git/process-closure authority and independently authenticated retained inventory/source bytes; it did not issue new model/JVP/product/eigen/SVD calculations. Historical asset postchecks are inherited from the original control flow reaching final serialization, not new model-asset loading in the recovery. Its own0.6999467078s/39,872KiB measurements are recovery-only.

Exact original authorities:

- Plan SHA **98a01706d756e3271f3f566dedf792719bf088055ba084c753b199b649b198c4**.
- Receipt `/data2/neural-shrimp-data/codex/llm-loop-gain-crosscheck/run-002/receipt.json`, SHA **4a3153f174bec42f034e245b63d5fe48978962908cbf82478ace3498b632b4fa**.
- Original closure `/data2/neural-shrimp-data/codex/llm-loop-gain-crosscheck/run-002-closure-001/report.json`, SHA **2d3a679a6dabde4379f95c56c66b303d9a82a5e1ddad6b76f4a3d7296c66b047**. Inference observed the original execution/closure and relayed that authority to Root; this note does not claim a fresh process observation.
- Partial-status SHA **9d05a513dd69dad790bfeffee46e8f160f49cc2565942726750d11cbd8fd0df0**; stderr SHA **6cf6250d0e25a829223bed7c4e2759fda699338eff08ee4029b3a7d4a1a93401**.

Native phase and monodromy timings retain their original event definitions. A monodromy event's `event_seconds` was recorded before serialization; the unfinished final summary's later timing cannot be reconstructed exactly from it. No cumulative timing is silently substituted.

## Scientific disposition and pending comparison

The own computation supports the narrow statement: **at this numerical eight-cycle, every one-step derivative expands some direction, whereas their correctly ordered eight-step derivative is contracting at every cyclic start.** Independent source formulation and separate directional/composed checks support the chain-rule factorization and composition result. The experiment supplies finite binary64 evidence, not a new exact-real orbit proof or interval stability certificate.

The theory review's broader qualifications remain: low-temperature suppression away from a fixed positive logit gap is not a uniform exclusion of small-positive-temperature chaos near ties; a high-temperature contraction argument needs its fixed-parameter compact-domain/bounded-derivative hypotheses. This single evaluated cycle does not establish either global temperature regime. Likewise, local linear compression does not establish a smooth one-dimensional nonlinear reduction.

**Producer comparison remains pending in this release.** No producer numerical artifact or comparator result was read to write this note. The original producer claims are known exposure; agreement or disagreement must be reported from the separate exact phase/input-matched comparator once available. The own result above does not depend on that comparison or on promoting the original failed attempt to success.
