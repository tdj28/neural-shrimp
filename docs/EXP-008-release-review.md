# EXP-008 presentation and CI release review

**Platform status update, 13 September 2026:** the results below were audited
locally on macOS. Linux CI at release `1d12ebeda040` does not pass the frozen
verifier. [PR-run log](EXP-008-linux-pr-failure.txt) records a baseline
coefficient difference of 3.68269315e-8 at the unchanged 1e-7/1e-9 rule; the
[push-run log](EXP-008-linux-push-failure.txt) records an earlier shared-feature
difference of 1.39784018e-14 at the unchanged 1e-12/1e-14 rule. Each run stops
at its first failure. Local replay and local SVD action agreement do not
establish full Linux portability. The strict gate remains failed and private
PR7 remains unmerged; a separate platform diagnostic does not replace it.

A subsequent two-worker Linux diagnostic preserves all three penalties and all
288 learned actions after refitting on saved features; maximum prediction
differences are 1.39e-9 and 1.96e-9. Feature/coefficient failures persist. This
is saved-input decision agreement, not a full regenerated-pipeline pass; see
[the platform results and complete reports](EXP-008-platform-status.md).

Independent post-run review, 2026-09-13. Scope: the new paper-summary
generator, the three new figure families and their provenance checks, and
the EXP-008 additions to CI. This review performs no neural integration,
model fitting, paid call or frozen-source edit. Scientific evidence remains
source `3c5604497ec57b19d62fbb595da7b190102f4b33`, manifest
`fb0e546947e0f36b77ff4978f60769db3541fc64330022bd664da22c20c321cd`.

## Findings and disposition

1. **Reference-neuron leakage in the waveform display: resolved.**
   `tools/plot_hopfield_information.py`, `waveform_figure`, originally passed
   the entire three-column reference to each neuron panel. Matplotlib drew
   three reference-neuron lines in every panel, incorrectly implying that
   each was that panel's reference. The maintainer changed this display to
   `sync[:, neuron]`. The reported error, saved three-column reference and
   candidate selection were unaffected. All three figure families and
   receipts were regenerated after the source correction; their saved-data
   `--check` passes. A focused new
   synthetic test in `tests/test_information_presentation.py` requires
   exactly one reference line, with the correct neuron column, in each of
   the three panels. **1 test passed in 1.07 seconds** after the correction;
   it does not integrate or fit a model.

2. **Paper-summary numerical replay portability: resolved.**
   The initial `summarize_information_paper.py` compared generated JSON text
   exactly, including full-precision NumPy means and quantiles. Such a check
   treats harmless arithmetic representation differences as artifact drift.
   It also initially summed sensitivity timings in filesystem enumeration
   order. The maintainer has sorted those paths, added exact hashes for
   every timing input, and used `math.fsum` for sensitivity and overlapping
   preview totals. Native, reversed and sorted sensitivity sums happened to
   agree for all 48 actual timings; no observed timing mismatch is claimed.
   The final generator compares derived JSON float values at the established
   presentation tolerance `rtol=1e-12, atol=1e-14`, while keeping types, keys,
   discrete fields, raw hashes and generated LaTeX text exact. Its saved-data
   check passes for 7,008 targets and 10,650 artifacts. An additional small
   synthetic check accepts a 1e-15 float difference and rejects a material
   float change, integer-to-float conversion, changed ID and boolean-to-int
   conversion. This is a post-run presentation check; the frozen model
   replay tolerance and experimental sources are unchanged.

## Scientific content accepted

The generator's numerical results agree with the independently audited saved
summary. The pooled primary benefit is 0.0008576276406 with descriptive
interval [-0.007701467321, 0.006545433785], leaving an advantage over the
learned baseline unresolved. The secondary 17.17% prediction-MSE reduction is
expressly distinguished from action-selection utility. Minimum-preview
selection has lower mean error at every dose and a lower pooled error than
geometry. Negative model scores, the stronger control, the unfavorable last
test block and the exhaustive candidate set remain visible.

The original figure contract is fulfilled by **4 + 6 + 4 = 14 empirical
panels**: all attempted origins and geometry coverage; all four policies,
four doses, both numerical steps, all twelve blocks, finite-action oracle
headroom and development CV; then three reference/selected-neuron panels and
all eighteen candidate preview-versus-tail errors. The displayed example is
the prospectively fixed first test origin, phase 675, radius 0.016, fine
step. All four policies choose candidate 4 there. It has not been replaced
with a case that separates policies or favors geometry.

Labels correctly identify a parameter map, pixel/crop limitations,
descriptive block intervals, finite-action hindsight oracle and original
waveform epoch. They do not claim an exact state-space basin, a deployable
oracle, independent preview and target dynamics, a topology-specific causal
mechanism, a population-generalization result or a net speedup. The generated
text retains the substantial exhaustive coarse/fine disagreements rather
than claiming uniform pointwise convergence. The surrounding EXP-008 text
explicitly retains the independent SVD coefficient-comparison failures and
distinguishes those failures from the passing frozen same-algorithm replay.

## Provenance and CI

The plot loader first requires a complete, failure-free receipt, verifies
the frozen manifest and source allowlist, and checks every receipt artifact's
hash and size. Complete target and baseline censuses are required. It
reconstructs the frozen summary from saved outcomes and decisions. Each
figure receipt binds source files, generators, exact plotted-array bytes and
all three output formats. `--check` rederives scientific arrays and
conditions with the existing arithmetic tolerance; it verifies stored output
bytes rather than demanding cross-platform re-rendered image-byte identity.
The new plotter's imported scientific helpers are covered by the explicit
generator list or frozen source allowlist.

The new CI commands validate the sealed plan, import the exact source bundle,
verify saved evidence and fits, and replay the presentation. None launches
new target trajectories. The pinned Linux/Python 3.12 workflow is structurally
appropriate; this local review does not substitute for executing it on Linux.
In particular, the known conditioned-ridge coefficient sensitivity makes an
actual same-algorithm CI result necessary. If it fails, preserve and diagnose
the failure without loosening frozen tolerances. The presentation JSON issue
above can be corrected independently of that numerical gate.

**Accepted in this scope.** The focused waveform test, full three-family
saved-data plot check, paper-summary check and synthetic presentation
comparison checks pass. The maintainer separately runs the full test suite
and final Linux CI. No remaining scientific-content or presentation-check
blocker was found in this scope.
