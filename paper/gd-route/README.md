# Gradient-descent route manuscript (working draft)

This directory holds the manuscript on the dynamics inside learning-rate planes
of full-batch gradient descent: numerically validated periodic windows inside chaotic
training, and an exploratory quasi-periodic route (Neimark-Sacker bifurcation of
the edge-of-stability 2-cycle, invariant circles, phase-locked windows). The
draft is private; framing, authorship, license and release are the human
author's decisions.

| What | Where |
|---|---|
| Manuscript source | `main.tex`, `references.bib` |
| Numbers quoted in the text | `generated-numbers.tex` (generated), manifest `data/numbers.json` |
| Figures and their receipts | `figures/f*.py` write `f*.pdf`, `f*.png` and `f*.provenance.json` |
| Index of all receipts | `provenance.json` |
| Review copy of the compiled draft | `gd-route-working-draft.pdf` |

Section 4 reports the prospectively designed, independently audited map study.
Section 5 gives narrowly scoped exact-real orbit proofs and a local scalar NS proof from exploratory selection.
Section 6 is exploratory: its analyses were chosen while exploring and have not
been tested on fresh networks or seeds. The evidence-and-scope table distinguishes mathematical rigor from design timing. Three orbit certificates (two 13-dimensional period-10 and one 72-dimensional period-2), one local scalar NS certificate, controlled scalar/ramp pilots and actual-text transformer pilots are integrated. The transformer runs show drift without an accepted periodic orbit; a drift-aware calibration and a broader-data continuous-progress pilot both failed their predictive-utility gates. The proposed frozen-curvature LM follow-up is deferred because its outcomes would not change the current strategy. A new S1R full-state pilot supports one locally consistent attracting segment, and a follow-up associates it numerically with a new reproduction of the upper bootstrap procedure at identical controls. Missing historical states, the lower-family association, distant windows and coexistence remain unresolved. The 72D proof and bootstrap outcomes have separate independent audits with their different scopes made explicit. A CIFAR diagnostic with the original Krylov budget still fails correction. The reviewed S1R saved-map audit retains unresolved cells and does not establish resonance ancestry. Orange boxes identify
remaining research or human-authorship decisions. Every citation must be checked at its
primary source before release; entries marked `[title to verify]` carry only
identifiers.

## Rebuild and verify

Figures read saved evidence from the data store (`NEURAL_SHRIMP_DATA`, default
`/data2/neural-shrimp-data`); figure 6 recomputes four short trajectories.

```sh
cd paper/gd-route/figures && for f in f[0-9]_*.py; do python "$f"; done
cd .. && python scripts/build_numbers.py
latexmk -pdf main.tex
python scripts/verify.py
```

`verify.py` re-derives every number, regenerates every figure byte for byte
against its receipt, replays the scalar, warmup and initial transformer feasibility receipts, all three rigorous orbit certificates and the scalar NS certificate,
checks that each macro used in the text is generated, and
that every repository path the text cites exists in the committed tree.

## Preparation for eventual submission

The draft is maintained for human review as experiments finish. Its current
claims concern optimization and nonlinear dynamics. A language-model contribution
would support the intended cs.CL audience; final primary/cross-list choices
(including nlin.CD, and alternatives cs.LG/cs.AI where appropriate) follow the
actual contribution and remain the human author's decision. No upload, submission,
public repository release or license selection is authorized by this preparation.

Before release, complete the outstanding science, author review, reference-content
checks, a self-contained source archive and a clean compilation of that archive.
Rigor labels follow the certified-computation playbook; historical filenames
containing `certificate` do not by themselves denote interval proofs.
