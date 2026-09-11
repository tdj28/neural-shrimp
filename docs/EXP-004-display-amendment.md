# EXP-004 display amendment — 11 September 2026

This post-run amendment changes figure layout and interpretation labels only.
The completed EXP-004 raw arrays, plan, model, selection rules, numerical
classifications, and experiment attempt remain unchanged. No target simulation
is authorized or performed by this amendment.

The first rendered maps contained more peak-period categories than the
synthetic layout fixture. Their multirow legends overlapped the plot axes and
colorbars. The revised display reserves vertical space from the number of
observed categories, retains every category separately, and supplies enough
distinct categorical colors without recycling the original 20-color palette.
The asymmetric finite-time-exponent colorbar explicitly labels both endpoints
and zero so its smaller negative range remains numerically visible.
No observed period is dropped, merged, or relabeled. The exact plotted-array
archives are checked against the preserved first render after regeneration.

The role named `exterior` by the frozen coarse-grid selection rule returned
period nine in both saved step-refinement runs. The waveform comparison must
therefore be described as a **grid-selected comparison**, with its refined
peak count reported directly from the saved refinement records. It is not an
established exterior or chaotic control. All role identifiers and raw
selection records are preserved unchanged.

The original source, twelve images, four plotted-array archives and their
receipts/index are preserved locally in
`/private/tmp/EXP-004-original-display-20260911/`. Git history preserves the
original frozen generator. The amended generator writes new output hashes and
per-figure receipts, and must pass exact-array comparison, byte-identical
regeneration and visual inspection before use in the manuscript.

Completed checks: all four plotted-array archives retain exactly the original
arrays, including NaN positions. All four updated figures passed visual review.
The final `--verify` run re-derived every plotted array and source/output hash,
and regenerated all twelve image files byte-identically.
