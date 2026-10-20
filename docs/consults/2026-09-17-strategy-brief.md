# Consultation brief: what would neural shrimp dynamics let a network do?

Prepared 17 September 2026 for one explicitly authorized GPT-6 Astra Pro
consultation, with a strict total incremental charge ceiling of [amount redacted].
**Prepared only: no consultation has been sent.** This is a compact strategic
brief, not a claim that a reviewer inspected the linked code or raw evidence.

The researcher's question is whether this research is missing an important
scientific direction, and what meaningful consequences shrimp dynamics have
for neural activity or computation. Prefer a bounded experiment that separates
causal mechanisms over another demonstration that a parameter map looks like
a shrimp. Attack the strongest confounds and specify a decisive falsifier.
Do not flatter the project, invent literature, or promise modern-network
transfer. Treat all result summaries below as disclosed evidence, not results
independently verified by the reviewer. A same-family model consultation is
advisory and cannot substitute for independent scientific validation.

## System and established scope

We reproduce the fixed-weight continuous three-neuron tanh system studied by
da Silva and Rech (2022). With activation `a_i=tanh(x_i)`, its dynamics are:

```
dx1/dt = -x1 + 1.5*a1 + 2.9*a2 + alpha*a3
dx2/dt = -x2 - 1.5*a1 + 1.18*a2
dx3/dt = -x3 + gamma*a1 - 22*a2 + 0.47*a3
```

Alpha and gamma are two connection weights. There is no training loss,
trained task, input stream, language model or biological preparation. RK4
approximates this continuous flow; numerical step size is not a neural
parameter. We distinguish autonomous waveform stability, useful downstream
computation, and incremental benefit from knowing parameter-space geometry.

## Compact evidence and failed interpretations

- EXP-004 mapped 39,322 settings and 201 independent transect initializations.
  Full-state shooting and Floquet checks support two nine-peak cycles and a
  neighboring eighteen-peak cycle. The core period is 37.63637538; relative
  closure is 7.46e-10 and dominant transverse multiplier is 0.06506, alongside
  the autonomous flow's neutral multiplier. These are numerical certificates,
  not interval proofs or a complete bifurcation-boundary characterization.
- EXP-005: after a 1e-6 state perturbation and 300 time units, activation-vector
  separation is 1.53e-6 at the core versus 1.41 and 2.13 at two finite-time
  irregular controls. A supposed coarse-map exterior later proved periodic;
  that failed interpretation is retained. Finite-time irregularity does not
  prove asymptotic chaos, and repeatability does not establish a task benefit.
- EXP-006: equal-native-weight-norm parameter interventions at twelve starts
  compare atlas and local sensitivity. At radius .016, atlas mean waveform
  error is .301672 versus .466653 for the symmetric local comparator. Local
  sensitivity wins at every start at both smallest radii; the pooled contrast
  remains unresolved. A tangent diagnostic beats atlas at every radius.
  Eventual return to a nine-peak cycle does not guarantee immediate waveform
  preservation. No phase/frequency/transient/shape decomposition is established.
- EXP-007: twelve new starts, four phases and stronger controls fail to show
  a resolved pooled hybrid advantage. Tangent and coordinate policies select
  the identical gamma-only actions in all 192 contexts. Their apparent success
  supplies no independent evidence for detailed shrimp geometry.
- EXP-008's original strict Linux replay remains failed. Saved-trajectory
  regeneration preserves aggregate scores, all 288 learned actions, 96
  minimum-preview actions and three penalties, but nineteen eigenvalue or
  coefficient checks fail. Stable direct residual features and QR/SVD fits are
  separately specified future methods, not a rewrite of old evidence.
- EXP-009 qualifies all 220 original selected long-horizon actions, while 952
  of 1,012 selected sensitive candidates still differ by more than 1e-4 between
  finer native steps; only 12/19 independent-panel cases meet all waveform
  checks. Do not treat the complete 300-unit label table as uniformly resolved.
- EXP-010 qualifies all 3,456 shorter 75-unit labels, sixteen independent
  sentinels and all 192 candidate rankings. EXP-011 then uses all 24 historical
  origins as development evidence: four-preview mean errors are .0162661
  shared features, .0107899 with geometry, .00856899 signed axes, and .00617504
  full-eighteen reference. Geometry minus shared is promising descriptively,
  but the reverse-signed benefit interval includes zero and simple axes win
  pooled. Local coverage wins smaller radii; geometry wins the largest. These
  observed radius winners are not a validated switching policy.
- EXP-012 tests a float64 CUDA implementation. All 54 batches/64,188 repeated
  known-case integrations finish. GPU state/metric parity fails the fixed
  gates; all warm pairs are slower end-to-end, despite unchanged rankings and
  repeat/layout bits. Fine full-batch median is 69.69 seconds CPU versus 75.86
  GPU. The native CPU remains the scientific default.

## Immediate draft and open scientific decision

EXP-013 is under implementation and has no fresh outcomes. It proposes 24
new physical starts, excluding all 36 previous distinct starts, conditional
on the observed atlas. It measures the full 75-unit loss-versus-preview-budget
curve at 2/4/8/18, fixed coverage controls, shared and geometry learners, four
radii and two phases. Four models tune only on old origins, using all five
penalties and both stable solvers. Models, fresh features and shortlists seal
before fresh labels; selected winners seal before full-eighteen evaluation.
It keeps invalid origins without replacement, uses all-candidate multistep
qualification, and reports actual acquisition costs. It establishes neither
fresh-population inference nor the longer 300-unit control task's utility.

Beyond finishing that bounded comparison, should the next scientific step
test a genuinely useful consequence of the periodic dynamics? Candidate
directions include phase storage versus transverse forgetting, transient
response or disturbance rejection, controlled input-driven computation, or
an eventual test in trained recurrent networks. These are hypotheses, not
findings. We need one tractable experiment with a manipulation/readout that
can distinguish the mechanism from saturation, ordinary contraction,
frequency drift, phase-alignment artifacts, selection bias and scalar
parameter anisotropy. A necessary tradeoff (for example robust oscillation
versus loss of input history) could be more informative than a claimed benefit.

## Requested advice and practical constraints

Rank at most five directions, then recommend exactly one next experiment
after EXP-013. Give the hypothesis, cheapest strong baseline, matched negative
and positive controls, independent unit, manipulation, primary endpoint with
units, prespecified falsifier and interpretations of positive/null/invalid
outcomes. Separate what is already a textbook consequence of an attracting
limit cycle from what this particular neural system could newly establish.
Explain why that result would matter for a network's computation, and what it
would not imply for trained modern networks. Propose a small pilot and the
decision it enables; avoid simply buying a larger atlas or weaker tolerances.

Local server: i9-14900K, 24 cores/32 threads, about 92 GiB usable RAM, shared
RTX 2000 Ada 16 GiB. Use explicit modest worker caps and leave other agents
alone. Native CPU integration is fast; CPU waveform optimization/replay can
dominate costs. No paid rentals are authorized. Freeze new plans and source
before outcomes, retain failed attempts and raw evidence, and keep work private.
There is no human-review availability during this run. A public release or
claim of novelty requires later literature and independent scientific review.

Local provenance pointers (not transmitted contents): README.md;
docs/EXP-010-results-audit.md; docs/EXP-011-results-audit.md;
docs/EXP-012-results-audit.md; docs/fresh-origin-tradeoff-design.md;
native/hopfield.cpp. No raw arrays, source tree or full manifests are included.
