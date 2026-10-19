# Task39 integration consistency pass

Reviewed `/tmp/task39-final-review.md`, SHA256 `6f70a08b2496520da9b59528d95ca418ffeca0466d1d2b0966df84b8aa584df5`, against the completed numerical subreview (`ec4c95b608481dc953ddae62872d676006c05bb232ab997f27b78edda732270f`) and claim ledger (`9a8a9fb6a793a464ba6135ddb27d7fb06b5705fc8c581f77392b1bffb54dbbde`). This was a read-only integration pass, with no repeated calculation, model call, test, or Git mutation. Only this note is new.

## One material refinement

**M1:** Remove the assertion that the displayed square-root formula itself assumes the studied fold's sign/orientation. This overstatement originated in my earlier R6 and should not be propagated. In `mu = 1 - 2 sqrt[ab(alpha_c-alpha)]`, the radicand can be positive with either `ab > 0, alpha < alpha_c` or `ab < 0, alpha > alpha_c`. The formula selects the center-direction stable branch on whichever side it exists.

The genuine defect remains: excluding other unit-circle multipliers does not exclude unstable transverse multipliers, so the generic conditions do not establish a full-state attracting branch. The proposed specialization to the actual Sec. V fold, with `ab > 0` and all noncritical multipliers inside the circle, is correct. It is defensible to make that narrowly framed correction a pre-submission must-fix while explicitly stating that the studied fold and its measured coefficients are intact. Suggested M1 opening:

> Nonzero fold coefficients and absence of other unit-circle multipliers do not imply an attracting full-state branch: other multipliers may be outside the circle. The displayed formula gives the stable center-direction multiplier on the side where its radicand is positive. For the actual fold studied in Sec. V, ab>0 and all noncritical multipliers are inside the circle. State those hypotheses when calling the branch attracting.

No change to the proposed replacement sentence is needed. The original completed subreview is preserved; this note records the qualification.

## Consistency checks passed

- S1 preserves the exact forecast `154.74482095152212`, the correction to `154.74`, and unchanged error `5.74`/`42/48` results.
- S2 correctly treats the flip disclaimer as scope ambiguity, not absence or invalidity of the separate doubling-bubble calculation.
- S3 preserves the prior crossing `1.0775400117776854` and amplitude fit `1.0778010471978443`, with the missing coefficient and partial-spectrum diagnostics retained.
- S4 and S5 correctly separate the distinct saved R orbits and local basin evidence from a simultaneous bisected crossing or an entire-edge claim.
- The close-up counts `232/232` and `459/459`, medians, and upward-rounded maxima match the retained result. The narrower claim that all fold pairs lie within one step is supported. Whole-domain `388/916` and `1720/1888` are explicitly reused, not rerun.
- The random-start agreement ranges, one body-qualified P6 example, P5-removal consistency, and unresolved symbolic-state/provenance limits agree with the ledger.
- The integrated review clearly distinguishes historical report comparison, prior original-record receipts, the one new saved-data reduction, and root/companion-owned build/citation checks. It does not claim every historical number was freshly raw-verified or use the active R-pair run as evidence.
- Build counts `28 pages / 24 numbered figures / 4 tables` are root-owned results, consistently carried into the report. This integration pass does not independently redo that build or the bibliography/license/public-access checks.

After the M1 refinement above, no additional substantive numerical discrepancy or scope overclaim was identified in this integration pass. The scoped numerical verdict remains **GO WITH FIXES**.
