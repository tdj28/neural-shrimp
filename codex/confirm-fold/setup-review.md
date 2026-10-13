# Task 5 saved-setup audit

Date: 13 October 2026 UTC. Scope: manually inspect the independent setup artifacts, recompute saved-matrix spectra and coefficient contractions, and audit the provenance chain before the passage design is frozen. No producer numerical artifact or model checkpoint was opened by this audit, and no new model evaluation or passage trajectory was generated. All NumPy/SciPy checks used one BLAS/OpenMP thread. This is a saved-result numerical cross-check, not an interval certificate or an independent model replay. The reviewer also authored the setup implementation; the separate design/replay reviewer supplies additional implementation-independent scrutiny.

## Current disposition

- **Target A: PASS for freezing a passage prediction**, using `qualification-A-01/run/setup.json`.
- **Target B: pending**, with no complete setup artifact inspected yet. No approval for B is implied by A's result.
- The prospective passage forecasts themselves remain untested. Setup qualification establishes that the proposed fold and local coefficients meet the numerical prerequisites; it does not establish forecast accuracy.

All paths below are relative to `/data2/neural-shrimp-data/codex/task5-fresh-prediction/`.

## Target A: independent fold at fixed log10(T) = −0.140

The original solve `setup-A-01` used source commit `5d0df12be5255d1ca8ea88682e85ef4b8fd35c36`. Its three accepted Newton corrections reduced the augmented residual from 0.007728832684090239 through 0.0003391835964115407 and 8.724232109505e−8 to **4.2178252438347017e−14**. All three linear residuals were below 3e−15 and all damping factors were one. The original saved orbit relative residual is **8.560986300584375e−15**; the augmented residual must not be mislabeled as the orbit closure.

The resumed qualification `qualification-A-01` used source commit `ee32575a839f6f79b1d39727addbb9e7cabd63de`, ended with status `complete`, stage `qualified`, and receipt exit code zero. Its corrected-state/matrix checkpoint is byte-identical to the original. The resume JSON, NPZ, original input manifest and original worker hashes were checked against the saved provenance. Original and resumed `qwen_map.py` and `numerics.py` hashes agree. All **44** original receipt entries and **30** qualification receipt entries matched their recorded sizes and SHA256 hashes.

### Crossing, orbit and modes

| Quantity | Audited value |
|---|---:|
| Active parameter and return map | physical alpha; G = F² |
| alpha_c | 0.06705485936004332 |
| Critical multiplier from independent dense eigensolve | 1.0000000000000413 + 0i |
| Critical argument / turns | 0 |
| Next-largest modulus | 0.08798289266724593 |
| Right-mode residual against +1 | 4.0639054511327286e−14 |
| Relative left-mode residual against +1 | 3.831892373184137e−14 |
| Norm of right mode | 1 |
| Left/right pairing | 0.9999999999999996 |
| Norm of left mode | 28.845209588039285 |
| Relative one-step return (proper divisor) | 1.0217208732854681 |
| Relative P2 closure from saved phase points | 8.551961099371066e−15 |
| Independently evaluated closure at other saved phase (reported worker value) | 3.705780791052064e−15 |

The matrix is 896×896; the saved orbit contains all three full 896-coordinate points, including the return. The saved points reproduce both the proper-divisor separation and closure above. The critical eigenvalue is real +1, with a complementary radius strictly below 0.95. Its tiny floating-point excursion above one is not interpreted as repulsion. The mode pairing shows substantial nonnormality; transverse kicks must lie in ker(l), rather than merely being Euclidean-orthogonal to r.

### Coefficients and arithmetic checks

Direct contractions of the saved raw left mode with the saved AD vectors reproduce

- raw a = −7.054924948575865 per unit physical alpha;
- raw b = −24.92734030789243;
- raw ab = 175.86051503979107.

The saved orientation is −1. Both modes are flipped together, preserving `||r|| = 1` and `l·r = 1`. The positive passage coefficients are **7.054924948575865** and **24.92734030789243**, the physical ghost side is **alpha = alpha_c + epsilon**, and **U = 0.004011659437583009**. The oriented arrays were checked to equal minus their raw arrays exactly.

Every FD row was inspected; each of the finest two estimates separately agrees with the AD coefficient within the declared relative tolerance 1e−3 and has the same sign. No averaging cancellation is used as a gate.

| Parameter h | FD a | Full-vector relative error against parameter AD |
|---:|---:|---:|
| 7e−6 | −7.054932995924692 | 6.250908066862941e−7 |
| 2e−6 | −7.054925607138868 | 5.1108360238450276e−8 |
| 7e−7 | −7.054925041586483 | 6.602795954154427e−9 |

| State displacement h | FD b | Composed-map first-derivative relative error |
|---:|---:|---:|
| 1e−3 | −24.94793255063069 | 3.6070974541233886e−4 |
| 3e−4 | −24.92919283317255 | 3.245456562631709e−5 |
| 1e−4 | −24.92754613686375 | 3.6060465221904624e−6 |

The composed derivative errors decrease in proportion to h² across this ladder. The final normal-form gates all pass: closure, primitive period, eigenpairs, closure of every phase, nonzero a and b, per-row parameter/state FD agreement, and composed-map precision.

### Preserved pre-outcome interruption

`setup-A-01/preoutcome-interruption.json` records the deliberate stop of the owned worker at **2026-10-13T01:30:07.252374+00:00**, after the converged corrector was saved. Its receipt records exit code **−15**, not a successful completion. The raw failed/interrupted attempt remains intact.

The original artifact inventory contains no `second-ad-*` stage. In the original captured worker source, the parameter FD loop follows the second-AD call and its persisted stage. Together with the guard's `second_ad_absent: true` record and the receipt, this supports that the old overlapping FD ladder was not executed. The resumed input manifest fixes the replacement offsets **7e−6, 2e−6, 7e−7**, disjoint from the future passage distances **3e−6, 1e−5, 3e−5, 1e−4**. The qualification contains no passage trajectories. This pre-outcome method amendment does not turn setup arithmetic into a confirmatory result.

### Artifact bindings

| Artifact | SHA256 |
|---|---|
| `setup-A-01/run/corrector-001.json` | `f0c3a9d90910aefda4cb828ff236c3ddad711f731359abe3cff2749447373057` |
| `setup-A-01/run/corrector-001.npz` | `ba86bcd81f71407a5c635c3f711753e2cb9f53de59da14cd82fbcfafb574801d` |
| `setup-A-01/preoutcome-interruption.json` | `9d6ad7b89f0ea000ad48789fe0b4a58467dbbcf863105cef1b367d6adbc3d936` |
| `qualification-A-01/run/setup.json` | `fc1bf51990b039bc789db7b49bcefaf56d4e5750e75f4921536684f1b611156b` |
| `qualification-A-01/run/setup.npz` | `41cdbc346f56a6f0ed2e60c470ff9d4331f8bd7f14fe267e34ecd8dff365861c` |
| `qualification-A-01/run/status.json` | `2c7e3a23b0557422371c86e05c7ace45cf28949186ccc87f614fa5bd02572253` |
| `qualification-A-01/run/inputs.json` | `ac42d8808c82212cb576c400ba44309aaa7b28ca95833c5c99a3cc3079d852ef` |

## Target B

Pending completion and row-by-row review of its independent fold/corrector, disjoint qualification derivatives and preserved interruption provenance. No B setup or passage claim is accepted yet.
