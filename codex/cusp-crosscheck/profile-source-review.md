# Task14 resource-profile source review

13 October 2026. Independent reviewer: Codex literature_cascades.

**Verdict: no source or mathematical blocker found for the bounded resource-only profile.** This accepts the reviewed implementation and prepared card; it does not authorize an uncheckpointed launch, qualify either endpoint, or verify a cusp. The current plan has `admission: prepared`, which the worker refuses before model construction. Root and the sole Git/process owner must supply the reviewed, privately pushed plan and fresh capacity admission.

This pass read the implementation, unchanged independent map/asset loader, plan, card, and test source. It did not execute tests, producer code, a model, or the profile; it did not decode candidate states or coefficient outcomes. The authors reported 12 numerical controls and 27 launcher controls passing. Their test sources were inspected, not rerun here. The already completed metadata-only ancestry review and mathematical derivation are separately retained in `map-ancestry-001` and `/tmp/task14-independent-math-review-20261013.md`.

## Map and dense derivative

`inputs.py` authenticates the exact candidate/ancestry manifests and immutable bytes before decoding. The selected row is 13, with its original physical alpha and log10-temperature; the composed map is period seven. The loader uses the unchanged independent ManualQwen implementation, the pinned model/config, the exact hostile vector without renormalization, and the accepted fixed prefix. Runtime checks bind the prefix IDs, zero-based layer 12, dimension 896, 24 layers and full vocabulary 151936. Historical producer checkpoint/tokenizer-byte usage remains declared provenance, as the card explicitly states; it is not reconstructed by the ancestry check. The new evaluator's actual assets are independently pinned.

`dense_profile.profile` sends every identity-basis row exactly once in 56 batches of 16. Independent batch elements have the same original state and one signed basis tangent each. Each returned row is `D(F^7)(x)e_j`; concatenating then transposing therefore gives the conventional column-oriented 896-square Jacobian. No missing direction can be replaced by a reduced eigensolver or an estimated tail. Batch shape and finite-return checks apply before completion. Original inputs and returned arrays are saved before post-call resource/finite refusals. Later matrix reuse is valid only for the identical state, parameter values and map/source identity; a refined state needs a new matrix.

The single composed JVP is compared with the dense matrix on a deterministic normalized direction at relative tolerance 1e-10. This is a consistency check of the same float64 map and differentiation implementation, not an independent high-precision enclosure. The second, third and mixed directional derivatives use fixed vectors through nested `torch.func.jvp`; they time the required derivative orders without selecting a critical mode. The physical-alpha derivative deliberately omits the cached prefix, so prefix steering participates in parameter differentiation. The probe does not mutate weights, the seed or cached tensors. All 896 columns and all declared probes must finish to report computational completion.

## Algebra helpers and scope

The independently derived formula uses the coordinate convention
`x(u)=x*+r u+h2 u^2/2+...`, unit right mode, `l^T r=1`, and `l^T h2=0`. At a simple real +1 multiplier, the implementation's quadratic equation and `c=l^T C(r,r,r)/6+l^T B(r,h2)/2` are consistent with that convention, including nonzero b. Simultaneous reversal of r and l reverses a and b but leaves c unchanged; comparing magnitudes still requires the same normalization.

`fold_basis` refuses a missing real +1, unresolved isolation, nonhyperbolic complementary spectrum, nonfinite eigensystems or excessive eigenpair residuals. It records complementary attraction separately rather than rejecting a legitimate fold with other unstable directions. `homological` supports the near-multiplier operator `mu^2 I-M`, retains border, orthogonality and residual diagnostics, and rejects nonfinite outputs. `cubic` retains the direct and center-manifold terms plus cancellation ratio. These helpers are not called by this profile and do not constitute an endpoint qualification pipeline. The later worker must enforce the stated closure, primitive-orbit, composed finite-difference and individual coefficient agreement gates before reporting a normal form.

A b-sign change between two endpoints and nonzero endpoint cubic coefficients can support a cusp candidate. They do not locate b=0 or establish the rank-two parameter unfolding needed for a numerical nondegenerate cusp. The prepared design preserves that distinction and defers P28.

## Resource and evidence controls

The exact envelope is a CPU2 worker plus lightweight supervisor, nice 10, hard 12 GiB address space per process, 2400 s outer deadline, at most ten seconds owned cleanup, 28 GiB start/16 GiB runtime available-memory floors, and a sampled 64 MiB output limit. The worker reserves 4 MiB for closing evidence. These available-memory/output observations are sampled, not hard aggregate-memory or byte guarantees. The inherited hard-AS refusal and exact nice admission occur before Git/model activity. The launcher captures the complete five-file worker inventory, itself, the pinned ownership helper, plan and card from the exact Git source, checks private-main ancestry, launches isolated Python, retains Popen ownership before further admission, and requires birth/session/command/limits equality. Failures and uncertainty are retained rather than promoted to completion. Task1 and the total auxiliary reservation still require the owner's fresh prelaunch capacity check.

The estimate from the measured Task7 period-two batch16 cost is disclosed as a forecast: 56*3.5*5.329830808 seconds, doubled plus 300 seconds, is about 2389.3 seconds. Third-order AD is unmeasured and can exhaust the 2400-second cap. In that case this attempt remains incomplete; there is no silent retry or permission to enlarge it. Native operation timings, dense-loop timing and total runtime have distinct meanings; the dense-loop timing includes recorded I/O. No coefficient, fold, cusp or certificate flag can become true from profile completion.

## Reviewed byte identities

| File | SHA256 |
|---|---|
| inputs.py | 29bd207180b26c94124cfbb8a8d50af4842cd658c151f6c4d02c3f08f187eff7 |
| numerics.py | d861466fe80fdb1e4c631f4bf6a59f126486e54290dfc3bdffdd6c58ec47d111 |
| dense_profile.py | 218eae0027475eb8cffe6457fb2b8e9cd979e51e042e9d3ff161da0429e2b98e |
| design.md | a326ce2ce7c58db7e725c024d7eb6f051f2e8e72177a94e57dfe969f472898c9 |
| profile-001-plan.json (prepared) | d4a5c493614accac0b5d00594655b1bc6724702d61f065d3a31f0b9c822b52aa |
| profile-001-run-card.md | 597a1bd4b11a1bea85587cdb0ba54455db072bedc2efd3dee4d335d08073bef3 |
| launch.py | b614bf3be1afd2a04c739e6e713106fa9bbe8fb218e3ce688ad3e9a1de6f39a6 |
| tests/test_codex_cusp_numerics.py | b963ff9c405571fbab1c3032c212a939d0d1a42ad5b692e5f686e75a35b1b9b1 |
| tests/test_codex_cusp_launch.py | 96b9862bc236b3e7647363b49ed6261c005dbc52a4d81bdcb5b373167a1b2226 |
| unchanged hostile-fold assets.py | 06477ee36a78f15ee8f1e8ae196dc53ae230ff19368fbda638eb6a6e0ef758e0 |
| unchanged crosscheck qwen_map.py | 493a3d1d57fec227f5ba2e3f8f9cfcfdd02d22495d1d44e6449ede901310ae6e |

Changing only the prepared admission field to reviewed necessarily changes the final plan hash. The owner must pin that exact prospective plan, not reuse the prepared hash above. No scientific source change is requested by this review.
