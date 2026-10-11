# EXP-025 numerical refinement amendment

23 September 2026, after original outcomes. The original attempt is complete
but unqualified, and remains immutable. Its receipt SHA-256 is
c326f3f9d80c0aa5cfa440bcef44d02810bac58658291e719f67f7edb786c044.

Both original step sizes give an inconclusive/heterogeneous primary decision:
mean periodic-minus-irregular memory R2 approximately -.0231, conditional
interval approximately [-.0679,.0211]. Every R2 step difference is below .002,
but only two settings pass the stricter full-trajectory gate. The maximum
activation disagreement is .01656. All sampled independent DOP853 checks pass.
No scientific threshold is changed and no failed setting is dropped.

The complete numerical failure census identifies 73 contexts of 4,992, with
at least one amplitude/sign exceeding 1e-4 between paired RK4 traces. Refine
all eight amplitude/sign cases in each such context at steps .000625 and
.0003125, totaling 1,168 new two-pulse trajectories. This is a fourfold decrease
from each original step. Selection is solely by the pre-existing numerical
gate, not by task score. The exact 73 contexts are frozen in the refinement
manifest and independently regenerated from the original arrays at runtime.

Inputs, exact starting states, train/test split, geometry classes, ridge
penalty, all task endpoints, qualification gates and primary decision rules
remain unchanged. Physically copy original inputs to a new exclusive attempt
directory, replace only the listed context rows there, and recompute readouts
and scores from the full composite. Original rows and receipts remain intact.
Record the per-context source and step mapping; the composite is explicitly a
mixture of original and refined numerical executions, not a uniform fine run.
Original autonomous geometry, tapes, unforced repeatability and independent
sampled solver checks are reused byte-for-byte with parent provenance.

The refinement is not a fresh replication and cannot strengthen the original
statistical design. Its only purpose is resolving numerical qualification.
Any remaining failure stays failed; another calculation would require a new
dated amendment and pre-outcome push. The auxiliary independent readout audit
already passed on the original: 416 fitted readouts checked, maximum prediction
difference 3.15e-11 using an augmented-least-squares alternative.

The reviewed implementation preserves original files through physical copying,
retains partial writes and exceptions in the new directory, verifies parent
hashes, enforces an exclusive output directory, and requires a clean private
source freeze. Six CPU workers, single-thread BLAS, 600-second execution ceiling
and 700-second owned-process external backstop. The original full acquisition
took 246.58 seconds and peaked at 188,660 KiB RSS; this targeted refinement fits
comfortably on the same shared local CPU resources. No paid resources.
