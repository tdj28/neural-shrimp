# EXP-008 presentation corrections, 13 September 2026

These corrections were made after the only complete attempt, before the
results release. They change no experimental source, plan, trajectory, feature,
model, action, example selection, measured endpoint or numerical gate.

Independent review found that the first waveform renderer passed the complete
three-column reference array to each neuron panel. Each panel therefore drew
all three reference signals under a single-neuron label. The renderer now
selects the matching reference column. The saved reference and candidate arrays
are unchanged; each panel displays its corresponding neuron. All four selected
candidate traces still coincide in the prospectively fixed example. A focused
synthetic test requires exactly one matching reference signal per neuron panel.
All figure source/output receipts were regenerated and checked.

The post-run manuscript generator now renders small quantities with explicit
base-ten exponents. Timing totals use a sorted sensitivity census and `fsum`;
every sensitivity file read is checked against its immutable receipt hash.
No timing-order mismatch was observed. Its presentation-JSON comparison uses
rtol=1e-12 and atol=1e-14 for reconstructed finite floating-point values;
types, keys, counts, IDs and other discrete content remain exact, as does the
generated TeX text. This is a post-run presentation tool, not a change to the
frozen producer's scientific or model-replay tolerances. The independent SVD
coefficient failures remain failures.

The final manuscript contains 34 pages, 15 empirical figures with 73 panels,
and 10 numerical tables. All pages were rendered at 125 dpi and visually
inspected. After these corrections, pages 27 and 31 were reinspected; exact
rendered PNG hashes confirmed the other 32 pages unchanged. Page 15's sentence
ends with "dose." on its next line; a tentative truncation concern was checked
against both the source and rendered page and was not reproduced. No clipping,
overlap, broken glyphs or unintended blank pages remain. The final LaTeX log
contains no warnings or box overflows. Full-page landscape plates intentionally
leave whitespace at preceding section transitions.

The full local suite passes 250 tests, including the new scientific-display
regression test. Frozen-source validation, source-only isolated CLI checks,
all 7,008 target/3,456 preview replays, figure checks and manuscript-derived
summary checks pass. GitHub CI at the final release commit is a separate merge
gate. No paid review or compute was used.
