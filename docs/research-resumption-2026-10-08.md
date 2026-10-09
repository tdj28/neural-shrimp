# Research resumption — evening of 8 October 2026

The morning stop was premature: completed pilots and document preservation were
treated as completion of the research goal. The user challenged that decision.
We resumed with an ordinary local exploratory experiment, keeping code and run
cards committed before each run and retaining every outcome. The exploratory
protocol does not require a new prospective auditor for every small pilot.

**What we learned about network behavior:** the dominant two-step dynamics of
these small networks are not reliably described by just the sharpest curvature
direction at each phase. In the two real-mode controls, the retained plane
contains almost none of the actual leading directions. This explains a concrete
failure of our proposed scaling shortcut; it is not a proof about large models.

The fixed six-reference pilot requalified every original orbit and spectrum,
but the rank-one/bulk approximation missed its accuracy target at every point.
It falsely predicted complex leading modes in both real controls, and real
leading modes in one complex control. Mean improvement over the matched-strength
commuting controls was insufficient because absolute errors remained large.
An independent post-outcome saved-arithmetic review passes.

A separately disclosed adapted analysis reused those saved matrices, with no
new derivatives or decompositions. The retained plane captures less than 0.021%
of each leading right eigenvector's squared norm in both real controls, versus
29.7–95.4% in the complex cases. These are whitened-coordinate measurements.
Projection of the actual full map into that plane still has material error;
leakage and residual-action terms show that omitted directions/couplings matter.
[Independent saved-arithmetic checks](reviews/curvature-plane-saved-arithmetic-review.json) pass. This is target-informed explanation,
not a replacement predictor or a rescue of the first failed test.

The next concrete scaling question is whether directions selected by the
**two-step update dynamics** can be found cheaply through matrix-vector actions,
with accuracy and cost compared against direct tangent-map diagnostics. The
failed single-direction assumption should not lead directly to another LM
forecast sweep. The earlier 72D certificate remains the strongest rigorous
advance; practical training/inference utility is still unestablished.

## Reading and evidence

- [Current 32-page paper](../paper/gd-route/gd-route-working-draft.pdf): page 25
  records the fixed pilot; all 303 previous source-derived entries are unchanged,
  with seven new entries. Compilation, source verification and visual checks pass.
- [Fixed pilot outcome](reviews/curvature-six-reference-outcome.md) and
  [independent arithmetic review](reviews/curvature-panel-saved-arithmetic-review.md).
- [Adapted saved-plane explanation](reviews/curvature-saved-plane-outcome.md).
  This later diagnostic is recorded separately and is not yet in the manuscript.

All work used local CPU, with no paid compute, GPU job or holdout access.
Profile and pilot took under two seconds combined scientific time; the adapted
analysis took about 0.10 seconds. Development and review took longer. Every
recorded experiment process exited. New raw evidence has a verified separate-disk
copy: 343 files / 14,543,690 bytes; see the
[backup receipt](backups/2026-10-09-curvature-resumption.json). Private Git backs
up code, compact evidence receipts and the current PDF. Main and the original
Claude-owned dirty log remain untouched. This remains a private reading draft.
