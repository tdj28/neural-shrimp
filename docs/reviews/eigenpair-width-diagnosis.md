# Saved width-scaling diagnosis — 10 October 2026

TLDR: The current period-18 boundary chart's error is dominated by Jacobian
variation rather than the quadratic predictor's remaining Taylor error. A
simple saved-data estimate suggests a doubled-width chart is worth trying,
with a small margin. This estimate itself proves no additional branch coverage.

The fixed last chart of completed tangent unit002 was reduced at256bits with
conservative saved-ball decoding. Center defect≈3.32e−73; degree0/1/2 forcing
contributions are negligible here, while the cubic contribution divided by
coordinate radius is bounded by≈.00268813. The observed whole-box defect is
≈.46471410. The same-width triangle budget is≈.46740223. Doubling the observed
variation and scaling Taylor degrees by1/2/4/8 gives a **heuristic** budget
≈.95093323, below1 with only≈.049 margin. Actual widened derivatives, coefficients,
predictor and C can change; none was acquired or bounded by this diagnostic.

Execution `b71ad685d11202b558091be7c99d628c46534767`; all exact pins, returned
arrays and resource records are in [the receipt](eigenpair-width-diagnosis.json).
The one saved-only run completed8.25s/784240KiB with zero native calls. All raw,
source, inventories and process exits are verified and immutable. The initial
synthetic upper-bound/enclosure assertion repair is disclosed in the plan.

Next: one fresh two-chart pilot at halfwidth1/64, starting from the whole proved
endpoint19/128 under the same normal/origin/native order and radii. It must
reacquire all widened C1/path returns and strict root/primitive/full-anchor tests.
No automatic radius change, no transferred wider-domain proof, and no changed
genericity or cusp-association claim.
