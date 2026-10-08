# Tiny-LM drift/prediction machinery: engineering profile

Exploratory, 2026-10-08 UTC. Source `5c477cca9445d507f1e1c58f8c93154ef35097e3`; compact evidence is in [the receipt](explore-tiny-lm-prediction-receipt.json). The committed [plan](plans/explore-tiny-lm-drift-controls.md) separates this completed engineering profile from the unrun calibration, assessment and policy stages. Prior tiny-LM evidence is unchanged.

The collector now records causal CE/AR forecasts, gradient norm/cosine, secant curvature, norm drift, the complete eight-step history of four-direction Gram matrices, and adjacent-plane coherence. B0–B3 share missingness masks. B2 has all raw information needed to reconstruct B3; success would mean useful nonlinear featurization of this information. This is neither new information nor an NS diagnostic. The offline analyzer uses training-only imputation/scaling, equal four-value penalty searches, 16-step label embargoes, tied-score average precision, temporal and leave-one-seed-out checks. It validates the exact input file inventory and binds committed analyzer/imported module bytes plus NumPy/SciPy versions. It has no assessment or policy execution path.

## Measured profile

The existing 15,776-parameter causal transformer used full-batch fixed-rate GD, float32 with TF32 disabled, seed 28151 and factor 0.8. Its eight context 32 windows have 256 next-character targets per update. Data/source/license attribution and target convention are hash-bound in `data.json`; no validation losses were evaluated. The initial maximum Ritz estimate 8.11137 had relative residual 0.008863 and supplied the learning-rate scale only, giving eta 0.197254.

| Measurement | Result |
|---|---:|
| Completed updates | 100 |
| Collection including features/state writes | 0.2910 s |
| Feature computation | 0.04488 s (15.4% of collection) |
| In-function total including Ritz/precision checks | 1.9498 s |
| Entire launched process | 3.6100 s |
| Peak CUDA allocation | 24,808,960 bytes; excludes driver/context |
| Preupdate CE, step 0 → step 99 | 4.17524 → 1.65966 |
| Saved states | 101 × 15,776, all finite |
| Feature trace | 100 × 81; all finite from step 32 |
| Uncensored 16-step burst labels | 53; zero positives |
| Manifest | 13 files, 6,467,760 bytes |

Hardware was a locally owned idle RTX 2000 Ada, driver 580.178.04, CUDA 13.0, Torch 2.14.1+cu130, one CPU/BLAS thread and nice 10. Internal 60s/external 75s limits were not approached. At the same rounded initial state, CPU float64 versus GPU float32 gradient/HVP/update relative discrepancies were `9.26e-8`, `1.23e-7`, and `1.93e-8`. This validates a matched-state implementation check, not trajectory precision equivalence.

Twelve focused tests passed in 1.97s, covering causal-prefix invariance, B2/B3 information parity, degenerate geometry, label embargo/censoring/guards, training-only scaling, selection invariance to late test changes, tied-score AP, request restrictions and exact input manifests; existing reduced dense/JVP controls remain included. All source and raw manifest identities verified. Raw profile files and launcher evidence are retained read-only under `/data2/neural-shrimp-data/explore/tiny-lm-prediction/attempt-001-profile` and the corresponding data-store logs.

## Interpretation and next gate

This measures a cheap, functioning feature pipeline and a low-rate training trace. No prediction model was fitted, and zero observed burst events offer no predictive-utility evidence. Six 800-update calibration cases would retain about 303 MB of states and, by this low-rate throughput, roughly 14s collection plus setup; that is a planning estimate, not a guaranteed runtime. At that checkpoint they were unrun pending independent source review. The later reviewed card reduced the collection cap to internal 240s/external 260s, and the [completed calibration result](explore-tiny-lm-calibration.md) now records the failed continuation gate. All assessment text/seeds, validation outcomes, protected final 10% outcomes, reserved seeds 29000–29999 and policies remain untouched. No period, Floquet, NS, chaos or generalization claim is supported by this profile.
