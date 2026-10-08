# Independent broader-data LM outcome audit

8 October 2026 UTC. Prospective audit, prepared before this reviewer opens
numerical target outcomes. Target source reviewed at
`ca9e96b362189d3220370d798f649eb24fbfdc00`, design `2a3f074a`, root source review
`e0c09a21be4ef2cd370e19511d71e664754f0dd7`. Collection/analysis completion and
hashes were communicated during preparation; forecast or learning results
were not inspected. This record does not authorize the audit launch.

Plain-language scope: check whether the saved experiment supports its stated
learning and prediction comparisons. It cannot establish a shrimp-specific
benefit, orbit, generalization to a new text pool, seed transfer or a controller.
The four cases use only two initializations and one nested development text pool.

## Complete saved-data checks

`tools/audit_tiny_lm_broader_data.py` independently verifies exact inventory,
file sizes and hashes; recorded execution commits and their full required
source inventories against Git and the reviewed source bytes; collection plan;
original corpus receipt, full hash and vocabulary; all selected byte windows,
token IDs and next-character input/target arrays directly against original
`input.txt`. Rehashed internal token metadata cannot substitute different text.
The audit reads corpus bytes for identity and vocabulary only; its selected
training windows are fixed and no protected outcome is evaluated.

It verifies all four state sequences, contiguous chunk indices, float32 shape,
accepted-state counts, exact independently reconstructed initializations,
common initial-state bytes and learning rate within each seed. The rate must
exactly follow the recorded positive initial expanded-objective Ritz estimate,
whose stored residual meets the declared limit. This binds the estimate; it
does not recompute an HVP or prove a largest eigenvalue. State transitions and
all stored loss/gradient primitives are not exhaustively reevaluated.

For every case, independently reconstruct CE(t+16)-CE(t), the censored tail,
fit/tune/test indices and fit-only RMS scale with the .01 floor. Refit all
32 ridge candidates through augmented least squares, separately from the
producer normal-equation solver. Check every mean, scale, coefficient, candidate
tuning score, selected penalty, selected ordinary comparator and every eligible
prediction, failure status, metric and gate. Candidate selection and gate
booleans must agree exactly; numerical comparisons use rtol=2e-8, atol=2e-10,
fixed before outcomes. This comparison tolerance never changes producer
qualification thresholds. A mismatch fails the audit and is retained; it
cannot be repaired by relaxing thresholds after seeing outcomes.

Independent AR1/2 reconstruction explicitly assembles causal 32-point designs,
checks full rank and recursively predicts 16 steps. All eligible comparator
failures remain in scoring and block positive interpretation. Probe values
supply descriptive checkpoint summaries only. Source inspection and a synthetic
probe-mutation control verify that they do not select fitted models, predictions
or gates. No real probe forward evaluation occurs in this audit.

At all saved rows from t=32 onward, verify loss/AR/state features, Gram history,
Gram moments, B3 algebra and availability masks. These checks reconstruct B3
from retained Gram entries and do not independently establish every Gram entry's
gradient origin. Gradient norms in trace/features and state norms are cross-bound.
The summary's cases must equal the individually saved cases.

## Fixed sampled feature reconstruction

After the complete arithmetic checks, reconstruct feature rows t=32 and t=600
for each of four cases. Each uses exactly the 33 saved states ending at that
row: 264 objective-gradient evaluations in total. The independent synthetic
control establishes that 33 states suffice for the helper's finite causal
history. Use the reviewed `TinyTransformer`, `LMProblem` gradient and
`CausalFeatures` as explicitly shared primitives, CPU float32, one thread,
no GPU, HVP, parameter updates or trajectory generation.

Feed saved CE to the causal helper, isolating gradient/device parity from
ill-conditioned AR fits. Separately compare recomputed CPU CE at those same
states against stored CE with absolute tolerance 2e-5. Sampled feature finite
masks must match exactly; finite values use rtol=.003, atol=.00003, fixed before
outcomes. These are cross-device implementation diagnostics, not rigorous
error bounds or trajectory equivalence. Save the 264 CPU gradients, their CE
values and eight reconstructed feature vectors in the new audit directory.
Do not alter inputs, target claims or forecast gates to resolve a discrepancy.

Arithmetic and sampled-feature conclusions are separate. A sampled failure
makes the overall audit fail even if arithmetic passes. Deadline expiration
makes the audit incomplete; preserve completed stages and samples, and neither
skip a predeclared sample nor extend the budget. No new computation is implicit
in a negative or incomplete outcome.

## Resource and retention card

Use the existing clean isolated reviewer worktree on
`codex/review-tiny-lm-broader`. Source, these instructions, synthetic tests and
research-log card must be committed and privately pushed before inspecting
numerical target artifacts. Root reviews and authorizes the exact command after
collection plus analysis exist. Recheck CPU load, available RAM and GPU occupancy
(read-only; no GPU allocation). CPU/BLAS/OpenMP1, nice10, internal60 seconds,
external90 seconds plus5 seconds termination grace; raw output cap32 MiB.
No paid compute or protected data. An external timeout backs native calls;
SIGKILL or filesystem failure may interrupt finalization, but prior fsynced
stages/arrays remain. Never overwrite or remove an incomplete attempt.

Planned exclusive output:
`/data2/neural-shrimp-data/audits/tiny-lm-broader-data/attempt-001`.
Launcher log/receipt prefix:
`/data2/neural-shrimp-data/logs/tiny-lm-broader-data-audit-attempt-001`.
Root pins the actual producer commit, collection manifest hash and analysis
hash as explicit command arguments. The audit guards all its executable/shared
source against its committed HEAD and hashes that exact source in its receipt.
Manifest input hashing is complete; there is no target computation until root
signoff. Raw arrays are under32 MiB (264x15776 float32 gradients are16,659,456
bytes); fixed JSON/prediction overhead has more than14 MiB headroom, and writes
check the cap. Parent owns separate-disk backup including failed attempts.

Verification status: all20 synthetic controls passed in4.86s on CPU1/nice10,
including full32-candidate reconstruction, semantic prediction/model/gate tamper
rejection, rehashed token substitution against the original synthetic corpus,
33-state causal-history sufficiency and retained deadline failure. Root source
review and saved-data execution remain pending. No numerical target result is
asserted here.
