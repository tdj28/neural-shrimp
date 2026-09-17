# From the measured cycles to a neural-memory hypothesis

17 September 2026. Local literature assessment after EXP-014 acquisition;
not advice from Pro, a preregistration, or evidence that our network was trained.
This note motivates later designs; no new trajectory is generated here.

## The connection is plausible, but has substantial prior art

Pals, Macke and Barak (2024) trained 512-unit continuous-time tanh recurrent
networks on a phase-coding working-memory task with an external reference
oscillation. Their analysis identifies stable phase-locked cycles, evaluates
Floquet stability, and connects stimulus-driven switching to bifurcations.
They also map regions of bistable phase locking in forcing frequency and
amplitude. This is direct prior evidence that periodic dynamics can support
a trained neural task. Their parameter-space islands are not evidence that
our autonomous three-neuron system has the same boundary organization or
computational mechanism. Full primary article and methods inspected:
[PLOS Computational Biology / PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10868787/).

Ichikawa and Kaneko (2021) report trained recurrent networks whose short-term
memory uses transient oscillation amplitudes and attraction toward a slow
manifold. That supplies a relevant alternative to persistent phase memory:
transverse activity can carry useful information while other disturbances
contract. We inspected the abstract, not their full experiments:
[primary manuscript, revised 2021](https://arxiv.org/abs/2010.15308).

Wedgwood, Lin, Thul and Coombes (2013) develop phase-amplitude descriptions
when phase-only reductions are inadequate, including weak attraction and
stronger perturbations. Our selected near-edge cycle's slower transverse
decay therefore makes a phase-only discrepancy unsurprising in principle.
Their analysis of periodically forced neural oscillators does not establish
chaos in our single-pulse experiment. Abstract inspected:
[primary manuscript](https://arxiv.org/abs/1302.0476).

Ganguli, Huh and Sompolinsky (2008) analyze memory of past inputs under
dynamical noise. Their treatment makes input direction and transient
amplification central in nonnormal networks: eigenvalue moduli alone do not
determine accessible memory. Our present deterministic pulse trajectories
with stipulated measurement noise are a different experiment, so their
capacity bounds cannot be applied directly. Full primary article inspected:
[PNAS / PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC2596211/).

## What our experiment can contribute

The qualified reference calculations give dominant transverse multipliers
of about 0.0651 and 0.8560 at two selected nine-peak cycles. This is a
specific, reproducible contrast in local relaxation, not a trained-network
or shrimp-geometry advantage. EXP-014 also contains a numerically consistent
selected phase-null example whose delayed readout matches the full linear
prediction and differs from the phase-only prediction. Both setting-level
experimental results remain unqualified because other pulse cases fail
the frozen numerical gates; see [the audit](EXP-014-results-audit.md).

The immediate follow-up is numerical resolution of all affected large-pulse
contexts and complete paired controls. Agreement of saturated classification
accuracy is insufficient; unsaturated readout scores and full states must
remain visible. Solver disagreement alone cannot establish a new attractor,
chaos, or a basin boundary.

If the finite-time responses qualify, the next mechanistic analysis should
separate the neutral phase component from transverse response using the
saved full propagators, and quantify observable gain at the frozen clock
windows. This can test whether the delayed response is explained by ordinary
phase-amplitude dynamics. It requires a separately frozen analysis before
opening new target summaries; changing decoders after seeing performance
would answer a different question.

A subsequent geometry test must compare several certified parameter regions
and ordinary periodic controls, accounting for period, waveform, input gain,
local contraction and transient amplification. Geometry earns incremental
value only if it predicts a prespecified task outcome beyond these dynamical
quantities on held-out regions. Matching two hand-picked points is inadequate.
Full boundary organization remains a separate dynamical requirement for a
verified shrimp claim.

Finally, generalization to learned networks needs an actual training task,
held-out inputs and seeds, and dynamical certification of the trained
solutions. A low-rank recurrent-network pilot modeled on the published
phase-memory task is a credible bridge and a more plausible GPU workload
than our currently CPU-favored three-state integration. It is a later
experiment, not an outcome already obtained here.
