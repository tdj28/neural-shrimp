# Task 38 adjudication (Claude, 19 Oct 2026)

Review: `review.md` (Codex, delivered 09:28Z, evidence 90f84c8c). Every finding was checked against its cited evidence.

## Must fix: all five accepted and applied (article and report)
- **M1 (DSS6 delta).** Accepted. The sentence on "delta negative in 151 cells" is replaced by Codex's text:
  - at 289 cells the top-two pair is real or conjugate-closed, with 143 negative products;
  - at 240 cells a complex subleading pair means the first two define no real 2-D determinant, so no determinant-zero
    curve is inferred.
  The conditional model is kept as a model.
- **M2 (C labels).** Accepted.
  - C now means that the usable phases disagree on the last symbol (153 cells).
  - The two Newton-checked C cells have small nonzero complex pairs. C does not locate a superstable curve.
  - The sign-change curve is a candidate superstable curve.
  - Pale and full colors are the multiplier sign predicted by the label's parity.
  - The text, the word-figure legend and caption, and the Table p6words caption were all changed.
- **M3 (pooled bound).** Accepted. The text and caption state the pooled test: 48 of 58 below 5 %, largest passing
  ratio 4.7 %, 23 with none, a single phase of a passing cell up to 21.7 %. The figure legend now reads "passes / fails
  the coding test".
- **M4 (Singer).** Accepted: "at most one attracting periodic orbit can have an immediate basin that avoids the
  interval's endpoints". The observation is stated as consistent with coexistence, establishing neither a global
  reduction nor its Schwarzian sign.
- **M5 (masks).** Accepted.
  - PV: the text and caption now say chaotic means lambda > 1e-3 alone, which overlaps a detected period at 3-18 cells
    per map, and that gray includes unresolved nonperiodic cells.
  - HP5: the caption says colors mark lambda > 1e-3, which overrides a detected period at 33 cells of the followed
    domain, and the report's 12 % share says the same.
  - A period-aware redraw is left for later.

## Should fix
- **S1 (endpoint rule, 2/43 not 7/38).** Accepted; Codex's sentence is in the report.
- **S2 (four points above 1e-7).** Accepted: "within 1e-7 on the body and kite curves; four points of the extended leg
  curves reach 2.9e-7".
- **S3 (p6-only branch).** The branch is older than main; it is the human's alternative version. Before it can be
  used it must be synchronized with main (new geometry, Fig. 1 distances, six cusps, kite, symbolic dynamics). It is
  flagged to the human; it is not edited tonight.
- **S4 (7.7e-9).** Accepted in both papers.
- **S5 (offset cause).** Accepted: the offsets are consistent with the alignment transient, whose contribution has not
  been isolated.
- **S6 (MSS filter and denominator).** Accepted: the text names the MSS filter (240 cells rejected) and gives the
  eligible periodic denominator (1,469 of 46,046) and the Euclidean distance (922, 63 %).
- **S7 (approximation).** Accepted: the paragraph now calls this an approximate scalar coding motivated by Jones's
  interpretation, uses W|X orbit labels rather than measured critical-point itineraries, and says red cells fail the
  coding test, reserving "more than one resolved turning point" for the 7 cases.
- **S8 (rescoring).** Accepted: "scored by a subsequent clean-phase rule ... reproduce the scan's label at 16 cells;
  the two C cells have complex leading pairs; 5 unconfirmed; these checks support selected labels, not the whole map".
- **S9 (random starts).** Accepted: chaos labels agree at 98.9-99.0 %; detected periods agree at 86.5-93.1 %.
- **S10 (repro caption).** Accepted: "the median periodic distance stays at the rounding level, with exceptions".

## Evidence gaps
1. **P6 boundary counts (294/419, 384/742).**
   - These came from an earlier, unrecorded body restriction. The text now uses the recorded recipe
     (`tools/plot_p6_shrimp.py` boundary_gate, 8-neighbor pairs, midpoint distance to the nearest traced curve of the
     right kind, in each axis's grid steps).
   - Within the close-up window of Fig. p6shrimp a (`boundary_gate_window` in fig-p6-shrimp.json): flip 232 of 232
     within two steps (median 0.19); fold 459 of 459 (median 0.19).
   - Over the whole followed domain: 388 of 916 (median 4.3) and 1,720 of 1,888 (median 0.24); the far legs leave the
     traced curves.
2. **basin-check-stipple.** The record is
   `/data2/neural-shrimp-data/explore/llm-soft-loop/kite/basin-check-stipple/basin-check.json` (local data folder, not
   in git). Command: `tools/kite_basin_check.py --alpha 0.07292258215962441 --log10t -0.15382910798122065
   -0.15382801251956182 --threads 16 --out llm-soft-loop/kite/basin-check-stipple` at commit b4e31cfb.
3. **SD Newton arrays.** Not saved; the verify JSON holds summaries only. The papers report them as stored results of
   the float64 Newton runs. Independent admission would need a rerun that saves orbit states and monodromy
   eigenvectors, which is not done tonight.
