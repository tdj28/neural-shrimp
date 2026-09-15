# EXP-011 feature adapter: local pre-fit audit

14 September 2026, project local time. This is a completed feature-construction
work unit for the planned retrospective development pilot. It is not the
complete EXP-011 execution freeze. No real model was fitted and no new dynamics
were integrated during this work.

The new `preview_shortlist_features.py` implements the previously published
[37-field shared schema and six geometry fields](shortlist-feature-adapter-appendix.md).
Its pure builder has no preview/target argument and reads only the specified
origin, certificate and unperturbed sensitivity fields. A separate loader
reads exactly 281 receipt-bound inputs: the original receipt and manifest,
two baseline files, 181 original atlas rows, and 48 sensitivity JSON/NPZ pairs.
It never opens old mixed feature packets, perturbed previews, targets,
decisions or fitted model files.

An independent reviewer checked the data flow, schema, grouping, numerical
arithmetic and bound input census. The review strengthened the file-access
test: its exact allowed paths are constructed from the frozen dataset
specification, rather than trusting the loader's own binding list. Thus a
future accidental reader cannot authorize itself by adding a binding.
No feature-adapter blocker remains.

All fifteen tests pass. They cover forbidden key/file access, exact schema,
complete real-data census and original directions, endpoint exclusion, tail
response projection, zero velocity, cancellation of large columns, nonfinite
inputs, crop handling and physical-origin aliases. The crop-edge test initially
used a decimal expression that lies one binary64 step inside the edge; its
fixture was corrected to produce the exact represented edge. No numerical
feature tolerance or scientific endpoint was changed to pass that test.

Separate historical parity checking confirms all 3,456 reconstructed candidate
directions are bitwise identical to the original realized directions. Every
shared feature agrees with its historical mathematical counterpart within
9.033e-15 absolute difference; geometry within 1.735e-18. This parity checker
may read the old mixed feature files, but is separate from the builder and
never supplies information to a shortlist.

Direct projected and unprojected response norms replace the old rounded Gram
quadratic forms. The saved activation sensitivity and velocity arrays remain
unchanged; they are neither reintegrated nor regenerated in this adapter.
The original tail mask remains `75 < t <= 300`, and the reference endpoint
is excluded. These are pre-preview unperturbed calculations, not perturbed
target labels. Physical groups use the common dynamical-system identifier and
exact hexadecimal alpha/gamma coordinates, keeping the twelve EXP-007/008
aliases together.

A representative saved-input construction took 2.70 seconds on this server.
That combines receipt checks, historical atlas reads, shared and geometry
feature construction; it is neither separate per-policy cost nor prospective
atlas/sensitivity acquisition time. The loader retains original sensitivity
call times with an explicit old-host qualification. All eighteen-entry
learned policies inherit sensitivity through candidates 16/17. Fixed axes
and diagonals do not, and geometry additionally requires the old atlas.
No net speedup follows from this timing.

The complete EXP-011 model, grouped validation, label access, prediction seals,
numerical checks, outcome handling, cost accounting and run verification still
require independent review, integrated tests and a private pre-fit push.
