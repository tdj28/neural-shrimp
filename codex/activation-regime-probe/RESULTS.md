# ARP1 results: local activation-based regime diagnosis

Short recordings contain a measurable signal about the loop’s regime, but this pilot did not show an advantage over direct recurrence. At 128 observations, activation dimensionality classified 7 of 8 held-out settings correctly and spatial roughness classified 6 of 8. Both recurrence comparators separated all eight by 64 observations in the closed-set comparison.

This supports a local diagnostic workflow without a new parameter-plane scan. It does not establish that activation appearance alone can certify chaos: a non-repeating trajectory remains unresolved until a separate growth diagnostic supports the finite-time chaotic label.

## Setup and reference labels

Ten parameter settings of the same Qwen2.5-0.5B soft-token loop were run: two for calibration and eight held out by setting. Every setting used the same fixed prompt and ordinary initial state. After 400 settling steps, the first 128 states supplied activation features. A separate 512-step continuation supplied the reference diagnostics. Two renormalized twins used absolute offsets 1e-6 and 1e-8. All model arithmetic and full-state diagnostics were float64; saved MLP activations were float32.

The eight test trajectories comprised four numerically periodic cases (periods 1, 2, 6 and 20) and four finite-time chaotic cases. Every row passed its specified recurrence/contraction or two-offset positive-growth gate; none was unresolved. The latter label is finite-horizon numerical evidence, not a proof of infinite-time chaos.

## Every observation budget

Entries are correct classifications out of eight held-out settings. Thresholds and their direction were learned only from the two calibration settings; no layer or window was selected using test outcomes.

| Observations | Raw dimensionality | Standardized dimensionality | Spatial roughness | Recurrence: three cycles | Recurrence: two cycles |
|---:|---:|---:|---:|---:|---:|
| 16 | 5/8 | 6/8 | 6/8 | 6/8 | 7/8 |
| 32 | 7/8 | 7/8 | 7/8 | 7/8 | 7/8 |
| 64 | 7/8 | 7/8 | 6/8 | 8/8 | 8/8 |
| 128 | 7/8 | 7/8 | 6/8 | 8/8 | 8/8 |

**Interpretation of the recurrence columns:** these are closed-set scores, treating absence of a short repeat as a chaotic candidate. Operationally, the baseline returns “unresolved” when it sees no repeat. At 64 observations it detected all four periodic cases; the separate long-window growth tests supported the other four chaotic labels. The activation classifiers’ chaotic outputs are also predictions, not certificates.

The two-cycle comparator was added in [amendment 01](amendment-01.md), committed before viewing target arrays or scores. It avoids giving activation features an artificial advantage over an unnecessarily delayed recurrence detector. The original three-cycle comparison is retained.

## Counterexamples to a simple order-versus-chaos rule

- The chaotic `test-table` trajectory had standardized effective dimensionality about **2.54**, lower than the periodic P6 case (**4.12**) and the P20 case (**3.52**). Its two positive finite-time exponents were approximately **0.08632** and **0.08635** per step. Both dimensionality classifiers incorrectly called it periodic at 128 observations.
- The fixed-point control had a rough-looking static activation pattern and was incorrectly called chaotic by the spatial-roughness classifier. The roughness classifier also incorrectly called `test-stripe-b` periodic. Spatial texture and temporal dynamics are different properties.
- The hypothesis did hold in the calibration pair: the P8 trajectory’s standardized dimensionality was about **4.18**, compared with **7.58** for the chaotic calibration trajectory. That contrast did not generalize perfectly to the held-out cases.

Effective dimensionality describes the spread of variance across collective activation patterns. It is not a direct count of neurons, a Lyapunov exponent or an attractor dimension. In particular, an exact period-p trajectory has centered covariance rank at most p-1; some reduction is expected from periodicity alone. The standardized measure removes neuron-wise scale differences, while the raw measure retains them. Both are invariant to rearranging neurons on the screen.

## Checks, artifacts and limits

- Six synthetic metric tests passed. All 30 batch rows passed central-difference convergence against an exact float64 JVP. There were no offset collapses and no source changes during acquisition.
- The local audit verified all input/output manifest hashes, independently checked all ten reference labels and all twenty evaluation counts, checked both short-window recurrence comparators, and compared Gram-matrix dimensionality against SVD plus grid differences against explicit edge calculations on forty layer/window cases.
- Collection took about **6.4 minutes** on the local CPU with four threads, including loading and preflight. No rented compute or new parameter-plane scan was used.
- [Overview figure](/data2/neural-shrimp-data/audits/activation-regime-probe/figures-002/overview.png) and [PDF](/data2/neural-shrimp-data/audits/activation-regime-probe/figures-002/overview.pdf). Both panels and their labels were visually checked. Original figures and numerical outputs remain preserved.
- [Compact receipt](receipt.json), [protocol](README.md), [run log](research-log.md). Raw data and all per-layer scores are under `/data2/neural-shrimp-data/audits/activation-regime-probe/`.

This is a small exploratory challenge set with one start per setting, one model and one prompt. It was selected using prior dynamical knowledge, not sampled randomly from the plane. It neither demonstrates a general-purpose classifier nor a shrimp-specific effect. Longer periods, quasiperiodicity, slow transients and other models remain untested. Combining the activation features might help, but that would be a new hypothesis requiring fresh held-out data.

Acquisition source: `b9b96dc910de14b9e21a4be1cc5aeb66856330d9`. Scoring source: `78baf332e51654552557cfa22c686626b20d42de`. Numerical results SHA-256: `2532c5a21d0e7b24d4440a75b373f7626ba88ac774d2c67635cfc31e0f919a0f`.
