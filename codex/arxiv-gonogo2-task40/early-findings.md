# Task40 early article fixes

This is an early actionable list for article
`bb42a82189da91c37f50804c077832df9e2237ae`, not the completed go/no-go report.
The human requested the arXiv review first and the Zenodo content audit last.
The article remains unedited by this audit. Line numbers refer to the frozen
article's `main.tex`.

## Confirmed corrections to make before submission

**New highest-priority correction, fold section lines 926–928:** “within
4–8%” is a statement about the **median** relative error at each offset, not
every natural passage. Under the original definition (`|u|<3U` and transverse
distance `<0.05`, complete phases only), median signed errors range from
-4.08% to -7.69%; the largest individual absolute error reaches70.59%.
Suggested replacement: “For complete phases defined by `|u|<3U` and transverse
distance `<0.05`, median relative errors at the four offsets were4.1–7.7%;
individual errors reached71%.” State the denominator as the forecast.
Figure15d uses `|u|<U` alone and must be distinguished from these statistics.
TableIV's mean fractions0.60–0.67 remain valid under the original definition.
This does not affect the separately verified42/48 prospective passage result.

1. **Abstract, lines 43–44: distinguish the operator from published Soft
   Thinking.** Delete “as in Soft Thinking” after “over the full vocabulary”.
   The published method filters and renormalizes its distribution. The
   introduction already distinguishes these maps. The abstract must do so too.
   Primary-paper evidence: `citations-review.md`, F1.

2. **Introduction, lines 167–168: qualify the synchronization summary.**
   Replace the assertion that copies synchronize near the theoretical coupling
   with: “Coupled copies have transverse growth rates that approximately follow
   the prediction from the single-loop Lyapunov exponent; observed
   synchronization thresholds additionally depend on initialization, duration,
   tolerance and attractor selection.” The independently reduced distance
   arrays reproduce all 18 table brackets, including the exceptions to a
   general threshold claim. Evidence: `raw-evidence.json`, `citations-review.md`.

3. **Cusp paragraph, lines 1379–1380: correct the bracketing-coefficient
   range.** The twelve initial bracketing-point magnitudes span
   `0.17047709685584778` to approximately `39.10824`, not `2.4` to `39.2`.
   Say “about 0.17–39.11 at the twelve bracketing curve points.”
   The final bound `|b| <= 9.5e-5` does pass. Evidence:
   `raw-evidence.json`, section `cusps`, and `raw-review.md`.

4. **Data Availability, lines 1770–1778: state the actual access status.**
   “Openly available” and “is archived at Zenodo” must be true when submitted.
   The Task40 brief describes the DOI as reserved and unpublished; repository
   policy keeps the repository private. If that remains the state, use
   “available from the author on request” and describe the DOI as reserved for
   a planned deposit. If the author separately authorizes and completes public
   release first, verify the public landing pages before retaining the present
   wording. This audit neither publishes anything nor changes visibility.
   Package-content checking is deferred as requested.

## Additional wording corrections

- **Kite basin edge, lines 1463–1465:** a bifurcation is a property of the map,
  but it need not appear in an attractor diagram from every start. Suggested
  replacement: “Bifurcation curves are properties of the map, whereas a
  fixed-start scan shows only attractors reached from that start. Together with
  the coexisting-orbit and saddle checks, the disappearance of this edge
  supports a basin-boundary interpretation.”
- **Two-token threshold, lines 291–293:** state Backour's assumptions:
  unbiased two-token reduction, linear logit transport, positive constant
  coupling. The threshold 2 is not an unrestricted mixture-collapse criterion.
- **Trace interpretation, Table VIII:** describe the trace/saddle account as
  supported on the sampled patch. It is a retrospective finite-grid numerical
  explanation, not a certified global characterization.

## Strong result already reproduced

An independent SVD/projection-overlap tracking implementation, using the saved
three Ritz vectors without importing the producer, reproduces the decisive
DELTA2D quantities: 852 tracked cells, 38 overlaps below 0.9, 849 sign
agreements, 42/42 and 60/64 zero-set matches; trace saddle
`(0.07318804087584509, -0.15208331868800173)`, value
`0.044884483163678275`; leading-real-multiplier saddle
`(0.07319482971306128, -0.1521384898381376)`, value
`0.06736198957428159`; no fitted determinant stationary point under the
declared mesh/window/filter test. This is saved-data numerical verification,
not a new model or derivative evaluation. Evidence:
`delta-evidence.json`, `delta-001.receipt.json`.

## Further checks completed at 03:31 UTC

- **Cascade, lines 1102–1103:** the phrase “after the first, shrink by factors
  of 3.5 to 6.5 per flip” includes a factor of 9.0653. Say instead:
  “After the first ratio, the deviations have one sign. The reductions from
  n=4 onward are approximately 3.52, 6.54, 3.93 and 5.26; the last is not
  resolved by the stated location uncertainty.” The nine flips, seven
  ratios, 0.008% comparison and disclosed sensitivity ranges otherwise
  reproduce. `cascade-evidence.json` retains the failed intermediate attempt
  and the original action-based Ritz residuals.
- **Ditch paragraph, lines 1408–1412:** replace the blanket claim that the
  continued orbit is also the fixed-start attractor throughout the patch.
  The nearest HR10 cells are period six at 593/595 solved patch points and
  at all 26+21 line points; the remaining two patch neighbors are unresolved.
  These are nearby parameter cells, not independent same-parameter
  full-state orbit-identity checks. A safe statement is that the continued
  orbit retains its six-token text and is attracting across the sampled
  patch/lines, while nearby fixed-start scans predominantly find period six.
- **Synchronization, lines 1615–1620:** the selected saved full-state return
  checks verify some far-start period-two synchronization, but do not
  authenticate the denominator for “most”. Change “most” to “some” unless a
  full-state classification of the complete claimed subset is supplied.

The CPU and GPU fine patches both solve all 225 cells and agree at every
quoted precision. Maximum leading-multiplier difference is
`6.5533828363939506e-12`; maximum next-modulus difference is
`3.7638295258268784e-13`. Alpha coordinates are identical; two temperature
rows differ by one ULP (`2.7755575615628914e-17`). This is agreement on the
same intended lattice, not a bit-identical-input device benchmark. The
initial exact-axis check failed and is preserved; the separately disclosed
follow-up records both exact axes and the comparison.

The complete verdict still awaits the final fold-passage and
bundle/build/visual review. Zenodo content remains last.
