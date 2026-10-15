# Task23 saved consistency auditor: independent source review

15 October 2026. Read-only source/control-fixture review; no imports, controls, tests, target outputs, model calls, or Git operations. The original auditor and fixtures are preserved as reviewed. This note is not an actual audit result or launch approval.

Full sources read:

- Auditor `/tmp/task23-saved-consistency-prep/audit.py`, SHA256 `91c15f332d86e15052422b4d1b2853eab3c7e951222457cb255d5080f2a7a86e` (287 lines).
- Controls `/tmp/task23-saved-consistency-prep/controls.py`, SHA256 `a4181853263d598a9ab341c62c78e336a759e645eeb855f112318871f4ea6977` (98 lines).
- Design SHA256 `2df683f31bb404068c10638bd7a7ef9c2da88a7dcfae1c6934976ce820773b52`, release `739d8dc00a5d86f51b0dc6d90469164da07f6107dbb66a66f1bdf746e73735f5`.
- Accepted original `run.py`, SHA256 `a6e83da06332cb5a27e1a167537226aaa38cfccdd1f87e92e28f17a5cc1093ed`, and `gain.py`, SHA256 `229d79a788589b5f860a0774f7273e9e93b54cecf8e6f474a3c5c3734eca751f`, read fully from `/tmp/task23-preparation`.

Disposition: the per-step arithmetic, accumulation order, saved-array associations, RNG selection, and finite recurrence conventions match the accepted worker. Three narrow coverage/wording changes are requested below; Root has agreed to preserve this version and release a correction. No change of scientific threshold, model replay, or additional numerical experiment is required.

## Concrete changes

1. **Bind the remaining physical parameter metadata.** Auditor lines 228–233 bind source hashes, indices, log temperatures, dtype/device and asset/input pins. Line 249 then trusts the saved physical temperatures. Lines 247–248 do not compare per-summary alpha, beta or physical temperature. Check metadata alpha/eps and the original card's fixed identities, independently reconstruct the physical temperatures using the source's `10. ** np.asarray(LOG_T, dtype=np.float64)` convention, and compare each summary's alpha, beta=0 and physical temperature. This closes an identity association; it does not introduce a scientific agreement tolerance. Exact report/source/input bindings already remain mandatory.
2. **Explicitly retain the original final summed accounting predicate.** Lines 136–138 enforce the original per-step `1e-10` predicate. Lines 161–184 reconstruct and compare the final residual, but do not explicitly state its finite/absolute-threshold disposition, although original `Accumulator.finish` does. Retain the original final predicate on the reconstructed residual and the saved final residual, keeping all discrepancies. Do not introduce a different threshold or claim the accounting identity establishes a mechanism.
3. **Narrow the initialization claim.** Lines 111–113 exactly regenerate both full 201-row RNG blocks and select the four original row positions. Lines 108 and 145 check initial base-state equality and represented offset norms. They do not independently reconstruct initial twin vector directions from the RNG vectors and Torch-normalized initialization formula. State this explicitly as reviewed-source authority. The norm diagnostics and exact RNG checks remain useful, but neither proves that directional construction. No Torch/model invocation or new norm tolerance is needed.

Root agreed to these three narrow corrections before final source acceptance. This original review does not pre-accept a later diff.

## Verified source coverage

The auditor requires all seven original files, authenticated hashes before/after, the original actual-card hash, exact accepted scientific source hashes, completed 1000/2000 horizons, and the supervisor/owner closure associations. Inference confirmed that the planned owner authority has the exact full-pin objects and fields expected by lines 196–215, including five captured source blobs, reaped exit zero, and original PID/group/session absence. That is a coordinator schema confirmation, not an independent process observation by this reviewer. Actual exact pin/card admission remains separate.

Native arrays are loaded with `allow_pickle=False`. The full `(2001,4,896)` base-state matrix is finite-checked in chunks; initial/final/tail native states and directions must be float64 with exact expected shapes. Initial and final base blocks and all 66 saved tail states are compared componentwise. Every measured base/feedback norm is independently recomputed from saved states, and both initial/final represented twin offsets are norm-checked. The absent intermediate twin/hidden/probability arrays prevent an independent full map or renormalized twin-vector chain replay; the saved norm links are what can be checked.

All 2000 stream positions must be present, ordered and numerically defined. Positive finite operands are required for both twins. The log-growth, network/readout relative-gain, step boundary and residual formulas reproduce the original NumPy operation order. Every nonexact value remains in the discrepancy ledger. Sums, gap counts, residual maxima/nonzero counts and means use the worker's sequential order. Gap arithmetic uses the saved top-two logits and physical temperature; vocabulary IDs are range-checked. Entropy and top probability are reduced from saved observations, not independently recovered from unsaved distributions.

The final recurrence uses precisely the last-state, norm-scaled `1e-8` rule and every lag 1 through 64 on the actual 66-state tail. Its period, threshold, all distances and the literal two-exponent chaotic label are compared. This is a finite recurrence check, not an all-phase primitive-cycle, asymptotic-chaos or stability proof. Expected transient and measurement checkpoint positions and saved valid-prefix row counts are checked; no intervening transient map is recomputed.

NumPy state norms are explicitly separate diagnostics from the original Torch norm reductions. Every nonzero difference is retained without a new tolerance. Exact same-operation scalar/state checks and these cross-library reductions are not conflated. Completion does not erase mismatches or imply scientific acceptance. Runtime and RSS fields stop before final report serialization and must keep that scope.

## Controls and finite resources

All five manufactured groups were read without executing them. They cover a tiny analytically specified factor-two stream, structural/zero refusals, changed summary/tail/RNG/log fields, cross-library-style norm differences without an exact gate, and float32 native substitution. The fixture has no claim to be a model trajectory. Its tampering cases currently preserve exact mismatches rather than demanding favorable science. Corrections should keep that distinction.

The proposed CPU1 plus supervisor, 1GiB AS, 55CPU/60wall and 16MiB output are bounds, not measured guarantees from this source review. A concrete output-size caveat is `Ledger.exact` at lines 67–77: one mismatch causes entire compared blocks to be copied into the report. The 66×4×896 tail could therefore exceed 16MiB after three serialized copies even for one altered element. A compact shape/index/differing-value representation would preserve every discrepancy without matched-value duplication; otherwise the card must retain output-cap exhaustion as an audit failure rather than promise complete worst-case serialization. No new scientific gate or benchmark is requested by this observation. Exact original input bytes and the final supervisor/card are still to be bound by Inference.
