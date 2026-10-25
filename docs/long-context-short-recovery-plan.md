# LCM1 short-control recovery proposal (2026-10-25)

Status: prepared, not authorized for rental. The original maximum-context arm
is complete and will not be rerun. Keep the first attempt immutable.

Compute short-prefix indices 49–255 (207 missing settings), plus index 48 as
one disclosed overlap check. `long_context_map.py --short-recovery` selects
exactly 48–255 in the original grid order. Same model revision, short prefix,
steering vector, float32 settings, seed per grid index, 256 settling and 512
measured updates, and twin radii. Use a new external output directory
`llm-soft-loop/pod64/lcm1/recovery-01` and new exact ownership record.

The changed diagnostic records each twin's measured/zero/nonfinite/disabled
status. After a twin fails, its remaining growth is unavailable and its row is
reset to the base; the unperturbed base continues. Never average only surviving
steps to manufacture a full-window exponent. Nonfinite base states yield explicit
partial-cell status, not fabricated history. A completed file can therefore be
a scientifically unresolved setting, and must be plotted as such. Recurrence
screening can use a complete finite base even when growth is unavailable.
Compare overlap index 48's base-history recurrence and growth with the first
attempt, disclosing device/runtime differences; do not silently replace its
original evidence or claim bitwise CUDA reproducibility.

Prior A100-SXM4-80GB measurements: short-cell computation approximately 11.5–11.7
seconds; checkpoints and verified acknowledgments raised end-to-end throughput
to roughly 18 seconds per cell. 208 cells imply approximately 62 minutes plus
setup and final retrieval. Expected resource duration 1–1.5 hours. Proposed
absolute lifetime 120 minutes, 20-minute retrieval reserve, bounded tool 90
minutes and runner 95 minutes. Price, rate cap and proposed monetary ceiling
are in the private ledger; new explicit approval is required before rental.

Start the mandatory collector before rental, with the new exact source/name/
approval binding, `--relative-output llm-soft-loop/pod64/lcm1/recovery-01`
and `--hard-minutes 120`. Do not reuse the original collector's path or approval.
Each cell waits for local hash verification; collector loss fails closed.
No automatic rerental. Retain explicit per-cell numerical statuses in final
manifest. Verify exact termination, separate-disk backup and final hashes.
Audit masks before plotting; show any unresolved growth distinctly from missing
execution and from contracting dynamics. Bind combined figures to both attempts.

Preparation validation: CPU synthetic tests inject zero/nonfinite twin distance,
nonfinite base, and known linear contraction. Retrieval tests cover interrupted,
missing and corrupt files and acknowledgment/deadline failure. A bounded CPU
model run at original failed index 49 tests the real model path; it does not
reproduce or settle the historical CUDA failure cause.

The worker now writes its own progress.jsonl alongside stdout, avoiding the
first run's operational progress symlink amendment. The fixed short recovery
has an 85-minute internal numerical guard within its 90-minute tool limit.
