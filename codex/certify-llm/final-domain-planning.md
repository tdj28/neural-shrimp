# Final-domain planning, 13 October 2026

This is an engineering proposal, not a frozen derivative design or a successful
inclusion. The final exact centres, radii, prefix and source bundles must be
written and hash-bound after the complete centre pass has been audited.

Continue the unchanged bounded refinement until the exact residual upper bound
is at most 1e-210, with at most twelve further one-candidate attempts starting
at center-parallel-004. Every attempt retains its complete prior capsule,
history decision and all eight outputs. Stop the sequence for a failure,
regression or uncertain audit; a stopping threshold is not an orbit certificate.
Each candidate still gets its own committed run card and independent audit.

An initial uniform nominal radius worth testing is 2^-640 (about 2.2e-193).
The completed original-prototype phase-zero 64-column run at radius 2^-800
had maximum derivative-entry radius about 6.12e-75. Linear rescaling by 2^160
would suggest about 8.94e-27 at the larger radius. This is a heuristic only:
interval wrapping is nonlinear, these are different centres, only one phase
and 64 directions were sampled, and no transverse coverage or rigorous error
bound for the proposed final boxes follows from this scaling.

The existing numerical factored-budget calculation tolerated a common induced
infinity-norm Jacobian discrepancy of roughly 2.59e-9, with a limiting
coefficient near 3.86e8. These are numerical planning quantities, not bounds
admissible to the proof. They motivate spending a few more short centre passes
to provide substantial margin before the much longer derivative campaign.
Actual point images, whole-box derivatives and the exact shooting inequalities
must decide the final radius and inclusion; none of these estimates can do so.

The final centres may retain their exact 2048-bit dyadic records. Arb midpoint
storage can hold exact constants with more bits than the current arithmetic
precision; the collector explicitly checks exact nominal-record preservation.
The lower-precision derivative receivers must contain the full nominal boxes
with rational endpoint witnesses, including any rounding in their actual
construction. Point images remain bound to precisely those centres.

First measure a final-domain derivative shard before scaling concurrency.
The existing 64-column prototype took about 30 minutes on one CPU thread and
5.14 GiB peak RSS. The roughly 56 CPU-hour full-coverage estimate is only a
single-phase extrapolation. A full campaign needs a separate current resource
inventory, exact ownership, immutable outputs, bounded jobs and all-row audits.
No campaign or final-domain derivative run is launched by this note.
