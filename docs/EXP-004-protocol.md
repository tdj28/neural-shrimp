# EXP-004: an actual neural shrimp and its parent-network behavior

Prospective exploratory protocol, 11 September 2026. The previous manuscript
did not contain a measured neural shrimp. This experiment addresses that
missing object before proposing applications. It does not change EXP-001–003.

The target is the asymmetric three-neuron tanh network in da Silva and Rech
(2022), DOI 10.33434/cams.1064713. Its fixed connectivity weights alpha and
gamma are the two controls; all three neuron potentials evolve. We reproduce
the exact Figure 2.3b rectangle and a finer rectangle around a period-nine
shrimp identified visually in that published figure before our outcomes.

The strongest allowed conclusion is that a numerically reproduced periodic
window corresponds to a particular repeating network waveform, activation
pattern and local perturbation response in this model. Failure to reproduce
the recognizable window or to verify representative periodic orbits counts
against reproduction at our stated discretization and horizon. Neither result
establishes anything about modern gradient-trained or latent-reasoning models.

The complete machine contract is
`experiments/hopfield/EXP-004-plan.json`. It binds both grids, the
independent-initialization transect, steps, transients, recorded measurements,
selection rules, refinement doses, verification, and failure handling. Every
cell retains its terminal full state and recent complete-state peak events;
period maps are derived from those events, not visual interpolation.

Attracting periodic motion in an autonomous flow has a neutral time direction.
All Floquet multipliers will be retained; transverse contraction is tested
separately. Full-state and tanh-activation traces are both measured because
saturation can hide potential fluctuations. Nearby-state separation, local
Floquet contraction and output variability are different observables and will
not be treated as equivalent evidence of useful task performance.

The paper uses a finer grid and much longer transient. Our lower-cost map is
an explicitly labeled reproduction attempt, checked with smaller steps and
longer settling at deterministically chosen representatives. The exact source
initial state and row resets were not published; our choices are explicit.
The source's oblique-transect equation is not used because its printed range
and plotted extent do not agree.

All code, tests, this protocol, and the manifest must be committed and pushed
before any target outcomes. Numerical implementation tests and an instrument
benchmark outside the target rectangle precede execution. Four CPU workers
and a two-hour hard ceiling bound the run; no paid resources or reviews.

The pre-outcome local audit passes 79 tests across the existing instrument,
native engine, independent orbit diagnostics and driver fixtures. Independent
reviews corrected partial-row preservation, required agreement at both
refinement steps, and enforced the wall ceiling with a separate termination
timer. A synthetic interrupted-row test retains exact completed-cell data and
the next attempted coordinates. No target dynamics were generated in these
tests. The pre-outcome plotting fixture is explicitly synthetic and stays
outside the repository and evidence tree.
