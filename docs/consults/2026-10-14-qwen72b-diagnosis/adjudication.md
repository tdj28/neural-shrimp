# Consult 8 adjudication (14 October 2026, about 15:50 UTC)

Review: gpt-6-astra Pro, xhigh, research-strategy, brief at 1f649d11; 63,119 tokens. The raw bundle is in
../neural-shrimp-data/consults/2026-10-14-qwen72b-diagnosis/execute-001; review.md is copied here unchanged. The review
is advisory only, and every point was checked against our evidence. It is not an independent audit: the reviewer saw
only the brief.

## Accepted

1. **Framing.** "No detected chaos" is plausible for the 72B, but "Qwen2.5-72B has no chaos" is not established, and
   size gives no prior either way. The paper will state the restricted result: no numerically resolved chaos over the
   stated prompts, temperatures, initial states and horizons.
2. **Steering.** Until validated, the steered map gives "eight single-offset candidate cells", not "0.8 % chaotic".
   The paper is corrected now.
3. **Definition.** Chaos is defined by infinitesimal asymptotic growth, measured by validated finite-time estimates.
   Finite-size (offset) exponents are reported separately. "No period up to 64" is a diagnostic, not a definition of
   aperiodicity.
4. **First diagnostics.**
   - Check that identical inputs give identical outputs.
   - Localize the first divergence before any new dynamics.
   - Compare with a small model on one device and on a split.
   - Use affine controls with known exponents (+-log 2).
   - Check JVPs against central differences over a range of h.
   The local part is done; findings 1-3 below.
5. **Tangent exponent.** Use the JVP tangent exponent as the offset-free estimate, validated against central
   differences on resolved scales. Settle for 256 steps, measure 512 with two tangent directions, and compare the two
   halves. Report the jump as numerically unresolved if no derivative can be trusted.
6. **Steered candidates.** Test the eight at 1e-4, 3e-5 and 1e-5, plus validated JVPs, with matched alpha = 0 cells
   and an extension of the candidate slices toward log10 T 0.4. Report confirmed/8. Unresolved and invalid cells do not
   count as negative.
7. **Llama.** Recheck representative Llama-70B positives at resolved offsets or by JVP, not at 1e-7.

## Findings made while adjudicating (local, no paid compute)

1. **Comparison at matched entropy.** The single-device runs settle at stable period-1 cells, all with entropy above
   one nat, to 1e-15 to 1e-14. Every two-GPU run of Qwen2.5-72B and Llama-3.3-70B wanders at 1e-8 to 6e-8.
2. **Local reproduction.** Qwen2.5-1.5B with half its layers split off through accelerate (CPU offload) wanders at
   5e-8 to 1.5e-7. On one device the same model settles to 2e-15 to 6e-15.
   - One step is repeatable: identical inputs give bitwise-identical outputs, across calls and across batch
     positions.
   - The split and one-device loops agree bitwise when the one-device model is loaded first in the same process.
3. **Cause.** accelerate wraps the forward of every module it hooks when the model is dispatched. Our float64 patches
   for RMSNorm and the rotary embedding are applied to the classes after loading, so a hooked module keeps calling the
   original forward: float32 for RMSNorm. In two-GPU runs the hooked modules include the final norm (`model.norm`) and
   `model.rotary_emb`. With CPU offload, the norms inside the offloaded layers are hooked as well. Loading the
   one-device model first had patched the classes before the second model's hooks were attached, which explains the
   bitwise agreement. The decoder linear layers were never affected, because their upcast patch is applied before
   loading by design.

## Not adopted, or deferred

- **Fixed-cache reimplementation.** A second, explicit implementation of the step (immutable prefix, explicit
  transfers) is not needed now that the defect is localized and reproduced. The fix is checked against the
  one-device path instead.
- **Feedback-margin intervention.** Direction 4, the logit scaling z_beta at fixed points, is deferred. It is a
  good local experiment, but not on the critical path for the 72B question.
- **Cost estimate.** The reviewer's estimate of 1.5 to 3 two-GPU pod-hours assumes per-trajectory work. One batched
  pod session with the fixed code is planned instead, with a hard cap in the private ledger.
