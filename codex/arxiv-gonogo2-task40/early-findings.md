# Task40 early article fixes

This is an early actionable list for article
`bb42a82189da91c37f50804c077832df9e2237ae`, not the completed go/no-go report.
The human requested the arXiv review first and the Zenodo content audit last.
The article remains unedited by this audit. Line numbers refer to the frozen
article's `main.tex`.

## Confirmed corrections to make before submission

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

The complete verdict still awaits the remaining fold/cascade cross-checks,
CPU/GPU fine-grid comparison if complete, and bundle/build/visual review.
