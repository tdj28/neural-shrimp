# Broader-data tiny-LM development outcome — 8 October 2026 UTC

**TLDR:** On this one fixed text pool, training on 256 distinct windows instead of 8 greatly reduced memorization and improved the disjoint development-probe loss in both initialization seeds. This made continuous learning progress measurable without requiring rare bursts. The declared forecasting test was negative: update-plane features B3 were only slightly better than matched-history B2 and substantially worse than simple comparators. These are exploratory saved-output findings pending independent outcome review; no shrimp, NS, inference or training-control benefit has been established.

The four cases and one analysis completed exactly once. The earlier failed burst-calibration unit stays stopped. No assessment, protected final 10%, reserved seeds 29000–29999, inference run or adaptive policy was used.

The execution/card commit was `44ed5ef60bdb300087c04d5a7b7d3fd7faf75dee`; bound executable and plan bytes match reviewed source `ca9e96b362189d3220370d798f649eb24fbfdc00`. Root source review `e0c09a21be4ef2cd370e19511d71e664754f0dd7` independently passed 32 focused tests. The reviewed profile passed resource checks; none of its learning/probe outcomes selected the collection.

The 15,776-parameter causal transformer used plain full-batch GD, tied embeddings, ridge 0.001, float32 and TF32 off. Each seed 28161/28162 supplied one initialization shared by 8/256-window arms and one common eta from the 256-window initial Ritz rate scale. All four arms ran 800 updates. One fixed nested training pool and one disjoint development-only probe were used; the seeds do not replicate data-pool choice. Exact token/byte intervals and hashes remain in the unchanged [plan](plans/explore-tiny-lm-broader-data.md) and raw data receipt.

Learning endpoints (cross-entropy in nats per target token):

| Initialization seed | 8-window train CE at 800| 256-window train CE at 800| 8-window probe CE at 800| 256-window probe CE at 800| 256-window probe CE at 25|
|---|---:|---:|---:|---:|---:|
|28161|0.009542|2.222921|7.350480|2.587573|3.111169|
|28162|0.009720|2.166343|7.363545|2.608421|3.102936|

At equal 800 updates, the expanded-minus-small probe difference averaged −4.759016 nats (seed range −4.762907 to −4.755124). This compares 6,553,600 with 204,800 processed training targets. At equal 204,800 training targets (expanded 25 updates versus small 800), the difference averaged −4.249960 nats (range −4.260609 to −4.239311). These are descriptive paired contrasts on this pool. The rate was shared and scaled on the expanded objective; neither arm was separately optimized. A lower development-probe loss does not establish broad language-model generalization.

All 3,204 parameter states and 3,200 feature rows were retained; each case has 784 complete 16-update progress targets and 16 right-censored rows. Fit/tune/test rows were 352/184/184 per case, with the specified 16-update horizon embargo. All candidate models and eligible comparator rows were valid. The development probe did not enter fitting, tuning, features or selection.

Late 256-window forecast normalized MSE (fit-only target-RMS normalization):

| Comparator |Seed 28161|Seed 28162|
|---|---:|---:|
|persistence|0.122318|0.120837|
|AR1|0.098139|0.094886|
|AR2|0.105983|0.088533|
|B0|0.775846|0.182114|
|B1|0.102668|0.209416|
|B2|0.305050|0.515333|
|B3|0.299926|0.502608|

The tune-selected ordinary comparator was B1 for 28161 and AR1 for 28162. B3 late MAE was 0.029163/0.039032 nats, versus 0.017066/0.017105 for the selected comparators and 0.029642/0.039679 for B2. Its normalized-MSE improvement over B2 was about 1.68%/2.47%, below the required 5%; it also lost to the selected ordinary comparator. The declared positive-forecast interpretation is false for each seed and overall. AR1/AR2 and persistence remain essential controls: the B3 feature regression did not add useful predictive accuracy here. B3 is a nonlinear representation of the same retained history available to B2, with no claim of new information.

For completeness, late 8-window control normalized MSE:

| Comparator |Seed 28161|Seed 28162|
|---|---:|---:|
|persistence|1.48022e-09|1.88724e-09|
|AR1|5.24028e-15|2.66042e-14|
|AR2|4.93386e-15|2.59084e-14|
|B0|0.000199813|0.00533717|
|B1|0.00120715|0.00218628|
|B2|0.000253226|0.000252188|
|B3|0.000198317|0.000250586|

The small-data controls are nearly stationary late in training, so their tiny simple-baseline errors should not be interpreted as a broad forecasting success. Full per-row predictions, candidate fits, train/tune/test metrics, MAE, signed bias and all scheduled probe values remain in the saved analysis; no overlapping windows are treated as independent replications.

Collection PID 35872 exited 0 after 31.034187 s; CPU analysis PID 38169 exited 0 after 2.406882 s. Including the prior 4.409418 s engineering profile, measured process time was 37.850487 s, within the 600 s unit ceiling. Peak CUDA allocation was 125,251,584 bytes (excluding driver/context); the retained unit occupies 220,500,453 bytes, below 512 MiB. CPU/BLAS 1 and nice 10 were used. Foreign PID 27512 occupied 7,414 MiB and GPU utilization was 100% at launch; 8,493 MiB was free. That process was untouched and remained after our exit. Runtime under this contention is not an isolated accelerator benchmark. The GPU reservation was released.

Evidence: [compact hash-bound receipt](explore-tiny-lm-broader-outcome-receipt.json), raw `/data2/neural-shrimp-data/explore/tiny-lm-broader-data/attempt-002-collection`, analysis at adjacent `attempt-002-analysis.json`; exact launcher/log paths and hashes are in the receipt. Manifest SHA256 `2ec1948521fe50a05a81b0ddec2e3e741cb25745c58f887868e3c5db43e2413d`; analysis SHA256 `a84558767147a0d9521e4d9f597982075145d96e36c4dffabe1039b7536179c3`. Source, plan and receipts are backed up by the private Git push; the external raw data require their own separately verified backup.

**Stop status:** the fixed unit is complete, with the negative forecast result preserved. Independent saved-outcome audit is pending. No retry, rate/seed search, policy, assessment or inference stage is authorized or running.
