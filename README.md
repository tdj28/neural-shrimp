# A neural shrimp and what the network does inside it

Private exploratory research on periodic-window geometry, neural activity, and
whether geometric information can improve model control.

**EXP-004 numerically reproduces an actual shrimp-shaped window in a
three-neuron recurrent neural network.** This is the fixed-weight tanh system
reported by da Silva and Rech (2022), not a newly trained modern network. The
network's three potentials evolve while two connection weights define the map.

![Measured overview and focused neural shrimp maps](paper/figures/neuralshrimp-map.png)

The frozen experiment measured **39,322 grid settings and 201 independently
initialized transect points**. Smaller-step, longer-settling checks and
independent DOP853 shooting verified two nine-peak cycles and a nearby
eighteen-peak cycle. At the core, the full state repeats after
**37.63637538 time units**, with relative closure **7.46e-10**. The flow has its
required neutral multiplier; the dominant transverse multiplier is **0.06506**.

The shape therefore corresponds to something the parent network actually does:
a repeating three-neuron activation waveform with a definite period and local
transverse stability. Aligned maps and an independent transect show its
relationship to finite-time sensitivity and actual tanh-activation variability.
The activation readout preserves the complete waveform's period; saturation
compresses some amplitude differences.

A useful failure remains visible. The point selected as an exterior by the
coarse grid becomes a nine-peak candidate at both refinement steps. EXP-004
retains that failed interpretation. **EXP-005 independently certifies a
nine-peak orbit at the same coordinate**; it does not overwrite the failure.

The separately frozen [EXP-005 follow-up](docs/EXP-005-protocol.md) starts all
four controls independently, settles for 10,000 time units, and measures for
2,000 at both `dt=0.01` and `0.005`. The lower and upper transect endpoints
remain irregular under its finite-horizon rule: no detected short period and
positive finite-time exponents at both steps. After a **1e-6 state perturbation
and 300 time units**, the actual activation-vector separation is **1.53e-6 at
the core versus 1.41 and 2.13** at those two controls. This is a large measured
difference in neural waveform repeatability, without an asymptotic-chaos or
task-accuracy claim.

Read the [numerical working paper](paper/neural-shrimp-working-draft.pdf)
([LaTeX](paper/main.tex)), [EXP-004 protocol](docs/EXP-004-protocol.md),
[exact manifest](experiments/hopfield/EXP-004-plan.json), and
[display amendment](docs/EXP-004-display-amendment.md). Figures contain measured
ODE outcomes, with lossless plotted-array archives and source/output receipts.
This reproduces known small-network dynamics; it does not establish novelty,
exact bifurcation boundaries, trained-task accuracy, or an optimization benefit.

## Does the geometry help preserve a useful output?

**EXP-006 gives a mixed, dose-dependent answer for a defined neural waveform task.**
All 12 attempted anchors qualified and all 4,320 planned trial-step records are
complete. The task measures error in all three activations during the first
300 time units after a held connection-weight change, allowing one constant
phase offset for the whole record. Competing directions have equal native
weight norms. The declared local comparator uses actual local weight
sensitivity; its two direction signs are averaged without selecting the better
outcome. The atlas policy uses only the earlier parameter map.

At the largest tested radius, **0.016**, atlas RMSE is **0.301672 versus 0.466653**
for symmetric local sensitivity, about **35.4% lower**. The paired benefit is
0.164981, with a descriptive anchor-bootstrap 95% interval [0.013853, 0.326023];
8 of 12 anchors favor the atlas. **At both smallest radii, local sensitivity wins
at every anchor.** The prespecified contrast averaged across all four radii is
0.025738 [−0.027862, 0.083081], leaving the aggregate advantage over local
sensitivity unresolved. These intervals describe the fixed selected anchors;
they are not population or confirmatory claims.

The atlas is **not the best observed arm**: the `tangent_plus` diagnostic has lower
mean error at every dose, including 0.260795 versus 0.301672 at radius 0.016.
That exploratory arm comparison must inform the next frozen baseline set; it
does not replace the prespecified primary comparison after outcomes.

Eventual nine-peak retention is 188/192 for the atlas and 308/384 across both
local signs. This separate assay follows 10,000 settling and 2,000 measurement
time units. **Eventual nine-peak retention does not guarantee immediate waveform
preservation.** The frozen example, `anchor-08`, favors both local signs over the
atlas, although every displayed arm eventually has nine peaks. It remains the
illustration. Transients, phase, frequency, and shape can all affect the short
waveform error; their contributions have not been isolated.

The new figures retain favorable and unfavorable cases:

- [All anchor attempts and atlas margins](paper/figures/neuralshrimp-drift-anchors.png).
- [Waveform error, all arms, and paired benefits](paper/figures/neuralshrimp-drift-utility.png).
- [Frozen waveform example that favors the local comparator](paper/figures/neuralshrimp-drift-waveforms.png).
- [All four continued branches and transverse stability](paper/figures/neuralshrimp-drift-continuation.png).

Independent pseudo-arclength continuation retains 196 node records, including
122 unstable records and five failed corrections. These include overlapping
branches and repeated seeds, not 196 distinct cycles. Geometric shooting checks
agree at 27/28 checkpoints. One checkpoint fails the unchanged closure threshold
(1.0591e-8 against 1e-8) and remains failed. Candidate multiplier crossings and
parameter reversals are not exact bifurcation boundaries.

Read the [EXP-006 protocol](docs/EXP-006-protocol.md),
[utility audit](docs/EXP-006-utility-results-audit.md),
[continuation audit](docs/EXP-006-continuation-results-audit.md), and
[complete evidence](evidence/EXP-006/attempt-001/receipt.json).
The next experiment should test a frozen hybrid of local sensitivity for small
changes and geometric constraints for larger changes against stronger local and
equally funded task-quality methods. A coupled controller must then succeed
under held-out disturbance streams. Neither an optimization speedup nor
trained-model transfer follows from the present fixed-network task. See the
[concrete next steps](docs/next-steps.md).

## Implemented instruments

- Native continuous-time three-neuron tanh integration, complete-state peak
  recurrence, tangent sensitivity, and all-neuron state/activation statistics.
- Independent adaptive shooting with primitive-period, complete-state closure,
  neutral-flow, and transverse-Floquet checks.
- Native-weight sensitivity, one-global-phase waveform comparison, a matched
  eleven-direction utility design with shams, and failure-retaining analysis.
- Pseudo-arclength continuation through stable and unstable orbit segments,
  transverse Poincaré spectra, and independent shooting checkpoints.
- Frozen row-wise evidence, independent-initialization transects, step and
  settling refinements, and paired perturbation traces.
- Deterministic figure generation with exact plotted values and source hashes.
- Earlier full-batch tanh training maps, gradients, Hessians, and Jacobians;
  Hénon/logistic controls; fixed/scheduled/stagnation/recurrence policies.
- Offline RunPod packaging and watchdog preparation. Current experiments use
  local CPU resources; GPU and provider-lifecycle automation are not validated.

## Reproduce and verify

Dependencies are pinned by `uv.lock` and the hashed `requirements.lock` export.
From a clean clone:

```sh
uv sync --frozen --extra dev
uv run --frozen pytest -q
uv run --frozen python tools/run_hopfield.py validate
```

Existing EXP-004--006 evidence and figures can be checked without advancing the ODE:

```sh
uv run --frozen python tools/run_hopfield.py verify \
  --output evidence/EXP-004/attempt-001
uv run --frozen python tools/plot_hopfield.py evidence/EXP-004/attempt-001 \
  --destination paper/figures --check
uv run --frozen python tools/run_hopfield_controls.py verify \
  --output evidence/EXP-005/attempt-001
uv run --frozen python tools/plot_hopfield_controls.py evidence/EXP-005/attempt-001 \
  --destination paper/figures --check
uv run --frozen python tools/run_hopfield_drift.py verify \
  --output evidence/EXP-006/attempt-001
uv run --frozen python tools/plot_hopfield_drift.py evidence/EXP-006/attempt-001 \
  --destination paper/figures --check
```

The plotting commands re-derive plots from saved measurements. `--verify` also
requires byte-identical image regeneration in the same library/font environment.
No source outcome is changed by rebuilding the figures.

Fresh target execution requires clean source, a pushed source commit, a frozen
plan, and a new attempt directory. Never overwrite a consumed plan or reset an
attempt. The original EXP-004 used four CPU workers and 140.54 seconds of
measured wall time; EXP-005 used one worker and 4.17 seconds. EXP-006 was frozen
at `ddb568e6cd67ef2cfbdb45e05710cd86025ef548` and used 423.866 seconds for collection
and analysis, excluding its final verification pass. Its 48 local sensitivity
solves took 26.627 seconds **within** that time. The old atlas's 140.54 seconds
is a separate cost, not free information. Paid compute and paid review spend
were zero. The source-freeze CI passed with 136 tests. Integrated evidence and manuscript
revisions are checked by the repository workflow; current run status is retained
in GitHub.

To build the paper, run `pdflatex main`, `bibtex main`, then `pdflatex main`
twice from `paper/`. The two main parameter maps and the four-control plate use
landscape pages for readability; references resolve to primary sources.

## Earlier pilot history

EXP-001–003 are preserved separately. The Hénon control verified periodic
orbits, the coarse neural **training** scan found no verified cycles, and the
recurrence-triggered policy never acted. Those results neither established a
neural shrimp nor an optimization advantage. See the [retained evidence](evidence/README.md)
and [initial results](docs/initial-results.md).

The earlier [figure guide](docs/visual-revision.md),
[exact scalar-cycle derivation](docs/analytic-cycle.md), and
[observable-consequences plan](docs/observable-consequences.md) remain as research
history and proposed methods. Their conceptual diagrams are not the numerical
shrimp reproduced in EXP-004.

Paid research reviews require explicit approval for each request. See the
[compute plan](docs/compute-plan.md) before considering paid resources. This
repository remains private until its owner authorizes publication.
