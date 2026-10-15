# Raw NaN-period bin correction — source preparation only

15 October2026. Root's full source review found that constructing the raw period histogram with `p==q` assigned zero to a NaN bin because NaN is unequal to itself. The correction uses the multiplicities returned by `np.unique(p, return_counts=True)`, preserving every raw period entry and its existing invalid-period qualification.

The existing depth-semantics manufactured control now inserts one NaN period, requires raw NaN count1 and invalid-period count1, and verifies that the raw histogram still sums to all576 grid cells. No new test group or scientific threshold was added.

Original depth205da36e, controlseb72a17a and release2171659a are preserved byte-identically under preparation-001. No controls, target arithmetic, model or producer code were executed. This is a correction before execution, not a numerical retry. Other reducers, source associations and scientific interpretations are unchanged.
