# Visual manuscript revision — 11 September 2026

This revision asks how a dynamical structure becomes a measurable task
consequence and a useful decision. It adds ten figures without running another
target experiment or changing the original pilot classifications. The original
three attempts, their source revision, and their receipts remain unchanged.

## What each figure contributes

| Figure | Evidence type | Question answered |
|---|---|---|
| [Controls, state, and readout](../paper/figures/conceptual-spaces.pdf) | Schematic | Where does a parameter window live, and where should its consequences be measured? |
| [Visible and invisible motion](../paper/figures/conceptual-readout.pdf) | Exact algebraic counterexample | Can a strong internal oscillation disappear under a readout? |
| [Hénon parameter map](../paper/figures/empirical-henon-map.pdf) | Original pilot | Does the classical positive control detect attracting periodic orbits? |
| [Hénon trajectories](../paper/figures/empirical-henon-orbits.pdf) | Original pilot; disclosed post-run selection | What motion corresponds to the parameter-map labels? |
| [Neural parameter maps](../paper/figures/empirical-neural-map.pdf) | Original pilot | How do the complete status and predictive-error maps compare? |
| [Neural tail diagnostics](../paper/figures/empirical-neural-tail-diagnostics.pdf) | Post-run analysis of saved states | How much of the continuing weight motion appears in predictions and loss? |
| [Policy census](../paper/figures/empirical-utility-census.pdf) | Original pilot | Which policies act, and what happens across all task–rate pairs, including failures? |
| [Policy trajectories](../paper/figures/empirical-utility-trajectories.pdf) | Original pilot; disclosed finite-case selection | How do actual interventions change the optimization path? |
| [Exact neural training cycle](../paper/figures/analytic-training-cycle.pdf) | Closed-form toy model | Can periodic training cause prediction oscillation, bias, and distinct averaging outcomes? |
| [Research roadmap](../paper/figures/conceptual-roadmap.pdf) | Proposed work | What evidence is needed before choosing to stay, escape, aggregate, or stop? |

The analytic example is a tied-weight scalar neural network trained on one
sample. It proves a particular cycle has observable consequences. It neither
establishes shrimp geometry nor beats the available zero-error fixed-point
solution. Its derivation is in [analytic-cycle.md](analytic-cycle.md); seven
analytic-control tests independently substitute its phases into the update map,
differentiate the map, and reconstruct prediction errors.

## Provenance and verification

Every figure has PDF and SVG vector exports, a 300-dpi PNG, and a
`*.receipt.json` file with construction details, exact plotted values where
applicable, claim scope, alternative text, and source/output SHA-256 hashes.
The two figure indexes bind all ten receipts. The manuscript provenance binds
the final PDF, its source, the indexes, and the supporting analysis/plan files.

`tools/build_evidence_figures.py` verifies the complete census and original
receipt hashes before reading saved state arrays. It also reconstructs every
finite neural and policy terminal test MSE from the frozen test input and final
saved weights, using explicit absolute/relative tolerances. Selection of
illustrative trajectories is post-run and disclosed; the complete maps and
policy census retain failures. Plotting and further analysis do not overwrite
the original status labels.

`tools/build_concept_figures.py` constructs diagrams and evaluates stated
algebra. Its receipts explicitly identify these figures as nonempirical.
`--verify` rebuilds the conceptual assets in a temporary directory and checks
byte equality in the same environment. Rendering differences across platforms
are not empirical changes; CI uses `--check` to verify committed artifacts.

```sh
uv run --frozen pytest -q
uv run --frozen python tools/summarize_pilots.py
uv run --frozen python tools/build_evidence_figures.py --check
uv run --frozen python tools/build_concept_figures.py --check
```

## The next decision

The next experiment must first establish an accessible neural periodic regime.
The current coarse tanh scan found none that passed the verification gate.
Once a regime is verified, measure phasewise task outputs and perturbation
recovery; then compare boundary-directed interventions with equal-dose
controls, a zero-dose sham, and ordinary performance-informed directions.
Charge the cost of finding and monitoring the structure.

The [next-step sequence](next-steps.md) and
[observable-consequences matrix](observable-consequences.md) specify the
measurements, alternative explanations, stop conditions, and uses that would
follow. They are design documents, not a claim that these experiments ran or a
substitute for freezing executable protocols before new outcomes.
