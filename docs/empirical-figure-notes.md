# Empirical manuscript figures, 2026-09-11

These six figures are post-run descriptive analyses of the original exploratory
pilots. They use saved trial records, archived full states, and frozen data.
No target experiment, training update, map iteration, paid compute, or paid
review was executed to produce them. Re-evaluating predictions and objective
loss at saved states is explicitly labeled post-run analysis.

The canonical research figure contract in
`/Users/d7082791602/PROJECTS/agent-skill-documents/EXPERIMENT_INTEGRITY_SKILLS.md`
requires deterministic charts, a separate receipt for each figure, exact derived
plot values, alternative encodings, and source/output/code hash verification.
Every empirical figure therefore ships as vector PDF, selectable-text SVG, and
300-DPI PNG, accompanied by a `.receipt.json`. The receipt index is
`paper/figures/empirical-provenance.json`. Exact trial IDs, frozen input hashes,
source root receipts, plotted arrays, transformations, exclusions, alt text,
and claim boundaries live in the per-figure receipts.

## Build and verify

From the repository root, using the repository virtual environment:

```sh
MPLCONFIGDIR=/private/tmp/neural-shrimp-matplotlib PYTHONPATH=src .venv/bin/python tools/build_evidence_figures.py
MPLCONFIGDIR=/private/tmp/neural-shrimp-matplotlib PYTHONPATH=src .venv/bin/python tools/build_evidence_figures.py --check
```

`--check` verifies every original evidence file through `neural_shrimp.cli.verify`,
checks the index, figure receipt, code, and output hashes, and independently
re-derives every plotted array and selection through the generator's analysis
functions without overwriting outputs. Re-derived floating values use absolute
tolerance `1e-14` and relative tolerance `1e-12`; integer counts, selections,
transformations, and strings must match exactly. All finite neural and utility
terminal test-MSE readouts are recomputed from archived full states and frozen
test data before downstream analyses. Recorded maximum readout error is
host-dependent metadata; each readout must satisfy the stated tolerance.

A rebuild in the same pinned Python/NumPy/Matplotlib environment emits identical
output bytes. Rendering can differ across library or font versions; committed
output hashes still protect the released artifacts, while the independent
numerical check permits only the declared floating roundoff tolerance.

## Figure inventory and caption facts

| Stem under `paper/figures/` | Size (inches) | Question and evidence |
|---|---|---|
| `empirical-henon-map` | 6.4 × 5.3 | Which sampled controls support verified periodic roots, and how strongly do those roots attract locally? All 65 controls: 39 verified attracting roots and 26 unresolved tails. Period counts: 2→24, 4→10, 7→2, 8→2, 16→1. Actual parameter coordinates, no interpolation. Spectral radius is reported only for verified attracting roots. |
| `empirical-henon-orbits` | 6.4 × 6.5 | What do the labels mean in full state space? Three post-run representatives: period 2 at `(a,b)=(0.8,0.1)`, period 7 at `(1.25,0.3)`, unresolved at `(1.4,0.3)`. Left panels use all 512 saved states, right panels the final 48. Lines connect discrete iterates and do not imply continuous trajectories. |
| `empirical-neural-map` | 6.4 × 5.5 | Does finite-state classification determine held-out quality? All 162 controls, two seeds. Seed 101: 53 unresolved and 28 divergent; seed 102: 54 unresolved and 27 divergent. Test MSE uses a shared logarithmic color scale for finite trajectories; failures remain crosses. Coordinates are the exact sampled learning rates on log axes, without interpolated surfaces. |
| `empirical-neural-tail-diagnostics` | 6.4 × 5.4 | Can full weights, predictions, and loss change on different scales? All four specified combinations of seeds 101/102 with equal rates 0.3/1.8. Three fractional one-step changes use separate denominators: full-weight L2 norm, prediction RMS over frozen test inputs, and regularized training loss. The receipt also supplies unnormalized changes. These are descriptive post-run readouts, not recurrence certificates. |
| `empirical-utility-census` | 6.4 × 7.25 | Does a controller help each paired task/rate case, and did it actually intervene? All 24 cases × 4 policies, with every finite MSE printed and all 36 failures visible. Each policy has 9 failures. Triggered runs: fixed 0, schedule 15, stagnation 12, recurrence 0. Recorded intervention counts are 0, 30, 21, 0. No success-only mean. |
| `empirical-utility-trajectories` | 6.4 × 6.4 | How do interventions change training motion and objective loss? Seed 202, first three manifest-order rate pairs where all four policies stayed finite: `(0.3,0.3)`, `(1.2,1.2)`, `(1.8,1.8)`. All 1,200 saved update indices shown. Rows show regularized training loss, full-weight L2 update size, and realized common hidden/output learning rate. Scales are shared within rows; lines mark actual recorded interventions. This finite-case illustration is not the complete policy census. |

Hénon representatives follow a deterministic disclosed post-run rule:
lexicographically smallest `(a,b,trial_id)` among verified attracting period-2
and period-7 trials, plus the lexicographically largest unresolved trial.
This is illustrative selection, with all other controls retained in the map.
Utility trajectory selection is explicitly conditional on all-policy finiteness;
it cannot justify aggregate improvement. The complete census accompanies it.

## What the visuals establish, and what remains open

The verified Hénon regimes correspond to actual repeated full states, with
measured local stability. The initial neural scan has no certified cycles or
shrimp geometry; unresolved points must not be relabeled chaotic or treated as
proof that cycles are absent. The neural maps and tails demonstrate why state
motion and task readouts require separate measurements.

The complete utility census shows consequential policy differences in some
finite cases and deterioration in others. For example, seed 202 at rates
`(1.8,1.8)` has terminal test MSE `0.041878571420910005` under fixed rates,
`0.00012173176934500859` under scheduled decay, and
`0.0001689460781299543` under stagnation decay. Seed 204 at rates `(0.3,0.3)`
is visibly worse under scheduled decay. These contrasts are existing pilot
outcomes, not demonstrations of a shrimp-based benefit.

Recurrence never triggers in this pilot. The figure generator checks every
fixed/recurrence pair and establishes exact equality of their complete saved
state arrays, including the divergent cases. Therefore an apparent failure of
this detector to add benefit does not settle the utility of known, verified
periodic geometry. The next experiment must first establish and characterize
such a regime, then compare geometry-informed control against these ordinary
loss and schedule controls under an explicitly matched compute budget.

The utility pilot matches a 1,200-update cap, not total compute. Some runs
terminate at the operational divergence threshold, and diagnostic work differs
between policies. These figures make no equal-total-compute or benchmark claim.

## Completed artifact audit

A same-environment rebuild reproduced all 25 empirical files byte-for-byte:
18 PDF/SVG/PNG outputs, six per-figure receipts, and their index. A temporary
PNG mutation was rejected by the output hash check. A temporary plotted-data
mutation was rejected by numerical re-derivation even after the receipt and
index were updated to agree internally. Both negative-test mutations were
immediately restored; original source evidence was never edited. All six PNGs
were visually inspected, with final layout corrections re-inspected.

The Hénon spectral radius is the radius of the Jacobian product over the full
primitive cycle. Because the periods differ, its magnitude is not a common
per-update contraction rate across all plotted controls.
