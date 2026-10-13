# Task 5 saved-setup audit

Date: 13 October 2026 UTC. Scope: manually inspect the independent setup artifacts, recompute saved-matrix spectra and coefficient contractions, and audit the provenance chain before the passage design is frozen. No producer numerical artifact or model checkpoint was opened by this audit, and no new model evaluation or passage trajectory was generated. All NumPy/SciPy checks used one BLAS/OpenMP thread. This is a saved-result numerical cross-check, not an interval certificate or an independent model replay. The reviewer also authored the setup implementation; the separate design/replay reviewer supplies additional implementation-independent scrutiny.

## Current disposition

- **Target A: PASS for freezing a passage prediction**, using `qualification-A-01/run/setup.json`.
- **Target B: NOT QUALIFIED on `qualification-B-01`**. Its fold/orbit gates pass, but its state-derivative FD and composed-map precision gates fail. A pre-outcome B-only refinement of the displacement ladder is justified below; `qualification-B-02` must pass independently before any B approval.
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

## Target B: fold accepted, first derivative qualification failed

The original `setup-B-01` solve used source commit `7fda7ac18d4b0afcd912543432b4a2479d41d64d`. At fixed physical alpha = **0.0717**, the active parameter is **log10(T)** and G = F⁸. The corrected parameter is **−0.1033462421844031**. A single undamped correction reduced the augmented residual from 1.71222076581266e−7 to **5.302028987296867e−12**, with linear relative residual 2.0969525346247531e−16. The final orbit relative residual was **4.396019119164493e−13**.

The 896×896 saved-matrix eigensolve gives **1.0000000000052838 + 0i**, angle zero, and next modulus **1.1075749560650417e−6**. Right and relative left residuals against exact +1 are 5.283814405574861e−12 and 5.284198518836221e−12. The saved modes have unit right norm, pairing one and left norm 8.207774615213541. Thus the numerical real +1, isolation and attracting-complement gates pass.

The saved nine full phase points reproduce proper-divisor returns **2.1159121613872816**, **1.4850797386890862**, **1.4482546019973817** for periods 1, 2, 4 and full P8 closure **4.3960191339005917e−13**. The independently evaluated per-phase closures reported by the worker are all below 8.14e−13. All are below the unchanged closure threshold 1e−10; the primitive threshold is 1e−6.

### Preserved interruption and resumed provenance

The guard stopped the original owned B worker at **2026-10-13T01:53:32.230440+00:00**, after saving its converged corrector. The receipt retains **exit code −15** and all **30** artifact hashes/sizes check. There is no original `second-ad-*` artifact; the captured source orders the parameter FD loop after the persisted second-AD stage. The guard and retained inventory support that the old overlapping parameter-FD offsets were not executed.

`qualification-B-01`, from source commit **213fb7463bc54a9c071a237e2561bce829aeaf5c**, resumed the identical corrector binary. All original worker, independent graph, input-manifest and checkpoint hashes in the resume chain were checked. Its final receipt has **exit code 2**, and all **32** listed artifact hashes/sizes check. Failure remains a recorded outcome of setup, never a silently discarded run.

### Failed qualification, inspected row by row

Contractions of the saved derivative vectors reproduce raw **a = −11.89977885013483 per unit log10(T)** and **b = −152.05451694889547**, with **ab = 1809.4151248559342**. These are numerical coefficient estimates, not yet qualified for freezing under the declared derivative gates. The candidate positive orientation would have ghost side log10(T) = p_c + epsilon and U = 0.0006576588581949812.

| Parameter h | FD a | Relative scalar coefficient error |
|---:|---:|---:|
| 7e−6 | −11.900011873004317 | 1.958211765294297e−5 |
| 2e−6 | −11.899797873701713 | 1.598648775168747e−6 |
| 7e−7 | −11.899781083080834 | 1.8764600857048613e−7 |

Both finest parameter rows pass the existing 1e−3 relative agreement gate, with the AD sign preserved.

| Actual state displacement h | FD b | Relative coefficient error | Composed-map derivative relative error |
|---:|---:|---:|---:|
| 1e−3 | −156.30683543323804 | 0.02796574919094145 | 0.020343552468709315 |
| 3e−4 | −152.43024990235338 | 0.0024710410515735894 | 0.0018050521181906167 |
| 1e−4 | −152.09619802020646 | 0.0002741192576672903 | 0.000200312289435309 |

The state-AD/FD gate requires **each** of the two finest rows to agree within 1e−3. The 3e−4 row fails. The finest composed-map derivative error is also greater than the unchanged 1e−5 threshold. Consequently, `state_ad_fd = false`, `composed_precision = false`, `passed = false`, and stage is `derivative-gates-failed`. All other gates pass. The failure reflects finite-difference truncation on this ladder; it does not by itself refute the fold or the passage mechanism.

### Reviewed pre-outcome amendment for B only

Before freezing or observing any passage, refine B's state-displacement ladder to **(1e−4, 3e−5, 1e−5)**, keeping the same corrected state, parameter-FD ladder, AD coefficients' definitions and all acceptance thresholds. A's accepted qualification is unchanged. The old B qualification remains immutable and cited.

The observed quadratic decay of both errors motivates this numerical refinement: extrapolation suggests a composed-map error near 2e−6 at h = 1e−5 and relative b errors near 2.47e−5 and 2.74e−6 at the finest two new displacements. These are planning estimates, not observations or gate approvals. The state probes remain at p = p_c, while prospective passage trajectories use positive epsilon on the ghost side; the amendment introduces no passage outcomes. **`qualification-B-02` requires its own row-by-row audit before B is accepted.**

### B artifact bindings

| Artifact | SHA256 |
|---|---|
| `setup-B-01/run/corrector-001.json` | `67843de0babc02e7b2b887b83a52a375cef777f464d559409604ef744f053d00` |
| `setup-B-01/run/corrector-001.npz` | `9fa46f2e284f59945970a581322de2176aa5362cb9d8f9c4e46fd661071af4a1` |
| `setup-B-01/preoutcome-interruption.json` | `33dc3b868f0d9433d3c41cc48268c8a335a3793f8399d2401f78aff5ab083b99` |
| `qualification-B-01/run/setup.json` | `4fbfdd66b4d4eb4fc31bb4fd0298ec146c0b5d53ccb54e7f67e6a0fb36b92692` |
| `qualification-B-01/run/setup.npz` | `d3bccdc48595c2d41206db8147b42c3cf668a9c977ac399063b036b1794da48e` |
| `qualification-B-01/run/status.json` | `f73b163d40ddef9380be5a6566d292f3c54d0550a366f7d7cd422abb89b1a492` |
| `qualification-B-01/run/inputs.json` | `016e9d08f1c2e4a29b2bc1a0cabcba8ac4f24d90e231860ee83a208f7ef59438` |
