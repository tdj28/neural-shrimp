# EXP-024 implementation checkpoint

22 September 2026. **Synthetic component tested; neural execution not ready.**

**Human-friendly TLDR:** No new neural behavior was measured here. The next
question is whether successive inputs create a linearly readable interaction
inside the recurrent trajectory. The analysis now distinguishes that from an
additive memory combined with a nonlinear readout or observation.

The [protocol](EXP-024-protocol.md) adopts the completed Pro consultation with
the corrections in its [adjudication](consults/2026-09-22-pro-adjudication.md).
`tools/two_write_interaction.py` contains pure analysis functions and imports
no neural model or evidence. Its 25 synthetic tests passed in 0.11 seconds
with BLAS/OpenMP/MKL limited to one thread. The [local component review](EXP-024-component-review.json)
binds the tested source and records the scope of verification.

Controls cover additive parity covariance, a misleading 75% XOR classification
example, analytically known injected parity, SVD against a dense normal-equation
solution, static observation mixing, centering errors, negative R-squared,
coordinate changes and explicit invalid-input failures. These are mathematical
controls, not new outcomes for either neural cycle.

Next implement and validate first/second-order sensitivity integration with
fixed pulse jumps; bind the exact reference artifacts and write directions;
complete acquisition, observer sealing, numerical comparisons and immutable
task census. Push a complete development execution freeze before any target
trajectory. Use the measured development throughput to finalize the held-out
run. No EXP-024 target attempt has been created, and EXP-023 remains deferred.
