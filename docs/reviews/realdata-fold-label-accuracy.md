# Real-data fold labels: manuscript accuracy amendment

2026-10-08 UTC. New source-reading finding, separate from the S1R gap result.
Reviewed repository base: `c9fa055b84189c4c37136d5caa84a57bd74e7310`.

For the 529-parameter digit-image example, `continue_ray` classifies the dominant
multiplier at its last attracting point. `classify_multiplier` checks distance
from the unit circle and relative imaginary magnitude; a positive real result
receives the historical string `fold` (continuation.py:31–37,95–105). The two
real-data runner paths call that routine and copy its type and multiplier
(explore_realdata.py:196–238,282–300). No fold simplicity, parameter
transversality or quadratic nondegeneracy test is made by this classifier.
A real multiplier near +1 is compatible with a fold but does not establish one.

The four files already counted by build_numbers.py:565–572 contain one complex
label and three `fold` labels. The three retained real multipliers are
0.9999998987530999, 0.9999998466673945 and 0.9999998719090355. Their raw labels,
counts and files are unchanged. The manuscript now calls them real near-+1
boundary candidates consistent with folds. The historical real-data note gains
a dated interpretation amendment; its original result descriptions remain.
No other directly matching 529-parameter fold count occurs in the manuscript
or README. This bounded correction is not an audit of every legacy boundary
label, a new bifurcation test, or a correction to the independently audited
S1R fixed-state diagnosis. Exact source and saved-file identities accompany this
note. No model, derivative, figure, proof, builder or scientific operation ran.
