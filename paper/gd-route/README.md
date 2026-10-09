# Gradient-descent route manuscript (working draft)

This directory holds the manuscript on the dynamics inside learning-rate planes
of full-batch gradient descent. Its selected exact-real certificates prove primitive
periods 10 and 18 in 13 parameters and period 2 in 72 parameters, with full-state local
attraction at fixed layer-specific rates. A separate interval certificate proves a local nondegenerate period-18 fold at fixed output rate, neutral at the fold with an attracting complement. A further certificate proves a local subcritical period-18 flip on the exact historical two-rate exponential ray, with unstable nearby doubled cycles and an attracting complement. Neither bifurcation is another attracting-orbit certificate. A separately reviewed uniform tube extends the same center period-18 orbit over a small nonzero interval, without reaching a bifurcation or increasing the count of four individually certified attracting orbits. These are internal project results. The
numerical study also reports periodic windows inside chaotic training and an exploratory quasi-periodic route (Neimark-Sacker bifurcation of
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
Section 5 gives narrowly scoped exact-real orbit proofs, one local period-18 fold, one local subcritical period-18 flip and a local scalar NS proof from exploratory selection.
Section 6 is exploratory: its analyses were chosen while exploring and have not
been tested on fresh networks or seeds. The evidence-and-scope table distinguishes mathematical rigor from design timing. Four orbit certificates (two 13-dimensional period-10, one 13-dimensional period-18 and one 72-dimensional period-2), one local period-18 fold certificate, one local subcritical period-18 flip certificate, one local scalar NS certificate, controlled scalar/ramp pilots and actual-text transformer pilots are integrated. The transformer runs show drift without an accepted periodic orbit; a drift-aware calibration and a broader-data continuous-progress pilot both failed their predictive-utility gates. The proposed frozen-curvature LM follow-up is deferred because its outcomes would not change the current strategy. A new S1R full-state pilot supports one locally consistent attracting segment, and a follow-up associates it numerically with a new reproduction of the upper bootstrap procedure at identical controls. Missing historical states, the lower-family association, distant windows and coexistence remain unresolved. A fixed-state lower-gap diagnostic finds that the stored direction solves its local Newton equation accurately but gives no improving retained finite step; the nonclosed terminal state and coordinate-dependent orientation do not establish a fold or branch turn. A subsequent two-edge pseudo-arclength pilot passes local replay and full-state qualification, with a consistently oriented tangent sign change on an unstable period-two segment. This is a numerical turning candidate, not a fold proof, attracting-window boundary or lower-family connection. The 72D proof and bootstrap outcomes have separate independent audits with their different scopes made explicit. A matched CIFAR restart comparison improves the linear forcing but yields no accepted correction; all four new damping trials increase residual merit. The reviewed S1R saved-map audit retains unresolved cells and does not establish resonance ancestry. Orange boxes identify
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
against its receipt, replays the scalar, warmup and initial transformer feasibility receipts, all four rigorous orbit certificates, the scalar NS certificate and the local period-18 fold and flip,
checks that each macro used in the text is generated, and
that every repository path the text cites exists in the committed tree. The fold and flip replays require a clean checkout with the bound kernel/card bytes and retain separate exclusive scratch directories (printed by the verifier) on success or failure; each driver runs with one CPU thread at nice10 under a 135s timeout plus 5s cleanup.

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
