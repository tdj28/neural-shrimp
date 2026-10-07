# Research roadmap and paid-compute plan for director-level review

Date: 7 October 2026. Private, unpublished work. Target: arXiv cs.LG primary,
cross-listed to nlin.CD. The human researcher is the final reviewer; nothing is
released before submission.

## Claim under development

Beyond the edge of stability, full-batch gradient descent with per-layer
learning rates follows the quasi-periodic route to chaos: the period-2
oscillation loses stability by a Neimark-Sacker bifurcation of the two-step
map; training becomes quasi-periodic on an invariant circle whose rotation
number is predicted by the crossing multiplier; rational rotation numbers give
phase-locked windows (Arnold tongues, period 2q at rotation m/q) arranged in a
devil's staircase; tongues contain period doubling and further Neimark-Sacker
bifurcations; binary trainability maps hide all of this.

## Evidence status (compact)

- **Confirmed** (frozen design, independent audit): 61 Newton-certified attracting
  periodic windows embedded in chaos in per-layer learning-rate planes of two
  small tanh networks (13 and 72 parameters, two seeds each); the published
  binary convergence score labels most non-convergent training "convergent".
- **Exploratory, quantitative:** Neimark-Sacker onset in both networks with
  birth angle matching the measured rotation (0.218 vs 0.2175; 0.132 vs
  0.12-0.135); along the traced curve the measured rotation just beyond it
  matches arg(mu)/2pi to a median of 0.0018; 91% of even-period periodic cells
  in a 256 x 256 map are m/q-locked (74% with q = period/2, 17% period-doubled
  inside a tongue); 35 of 56 testable confirmed windows are Birkhoff m/q
  orbits (null expectation 3.6); boundary census over 61 windows: fold 1,316,
  Neimark-Sacker 1,395, flip 1,012 rays; staircases with Farey-neighbour
  plateaus (weak evidence: 3 of 3 pairs in one network, uninformative null in
  the other).
- **Open:** odd-period windows (5 confirmed) are not circle orbits; a minimal
  two-parameter model shows only period doubling; no real-data result yet; no
  training-relevant consequence yet; float32 sensitivity untested.
- A 14-page manuscript draft with six reproducible figures exists; every
  number is generated from committed data and re-verified by a script.

## Roadmap

**R1. Mechanism completion (local CPU and RTX 2000 Ada; days 1-4).**
Orbit-following maps (initialized on continued orbits) for boundary figures;
1024 x 1024 GPU rotation maps for both seeds of the 13-parameter network and
512 x 512 for the 72-parameter network; Neimark-Sacker curves for all four
network-seed pairs; a quantitative tongue-root test (locked regions converge
to the predicted roots); odd-window analysis as advised by a separate
mechanism consult; minimal model candidate: two-mode depth-2 linear network.

**R2. Generality on real data (days 2-9).** (a) Running now: a small tanh MLP on
scikit-learn digits with full-batch gradient descent and a single learning
rate, testing for Neimark-Sacker of the 2-cycle, rotation numbers, locking and
staircases. (b) Canonical benchmark: full-batch gradient descent on a
5,000-image CIFAR-10 subset (the Cohen et al. 2021 edge-of-stability setup),
MSE loss with weight decay so that attractors are isolated, a two-hidden-layer
MLP and possibly a small CNN; single learning rate and the per-layer plane.
Diagnostics that scale: top two-step Floquet multipliers by Arnoldi on
Jacobian-vector products; rotation numbers from principal-plane projections;
loss power spectra. Explore in float32 locally to choose slices; final figures
in float64 on rented H100s.

**R3. Consequence (running now as an exploratory pilot, days 1-8).** Does the
regime or rotation number predict reproducibility (parameter and function-space
disagreement across perturbation sizes, uncertainty exponents), loss behaviour
(time-averaged, averaged-iterate and held-out loss) or a practitioner-visible
signature (loss spectra), beyond what the largest Lyapunov exponent and
sharpness already predict? Distant ordinary-periodic and qualified chaotic
controls are mandatory; nested models test incremental value.

**R4. Confirmatory test (days 8-12).** Frozen and pushed before running, on
reserved fresh seeds (29000-29999, never used in exploration) and at least one
network not used in exploration, with an independent audit. Draft predictions:
(a) a quasi-periodic band with near-zero Lyapunov exponent between the period-2
region and chaos whose rotation number just beyond the Neimark-Sacker curve
matches arg(mu)/2pi within a frozen tolerance; (b) at least 80% of even-period
periodic cells in the band are m/q-locked with q = period/2 or period-doubled
inside an m/q tongue; (c) rotation increases monotonically across plateaus
along frozen slices; (d) the period-2 orbit's stability boundary at the band's
onset is Neimark-Sacker. A consequence prediction is added only if R3
supports one.

**R5. Writing and release (days 10-14).** Manuscript built around whatever R1-R4
support; citations verified at primary sources; reproducibility package and a
clean arXiv source archive; human review; release only at submission.

## Paid-compute plan (RunPod, on-demand community H100, $2.69/h listed today)

Rationale: our engine and the scientific diagnostics run in float64; the local
GPU's float64 throughput is about 0.19 TFLOPS (measured 0.057 and 0.476
microseconds per cell-step for the two toy networks), so a 1024 x 1024 map of
the 72-parameter network takes about 4.5 hours locally and float64 CIFAR runs
are infeasible locally. An H100 offers about 26-34 TFLOPS float64 (51-67 with
tensor cores) and 80 GB of memory.

| Phase | Work | Estimate | Gate |
|---|---|---|---|
| 0 | Rebuild the CUDA engine for sm_90; parity check against the CPU reference on sampled cells; time one 72-parameter map unit and one float64 CIFAR batch | at most $5 | parity passes and timing measured |
| 1 | High-resolution rotation maps; the frozen confirmatory maps on about 16 fresh networks | about $20 | design frozen and pushed |
| 2 | Float64 CIFAR-10 benchmark: batched learning-rate sweeps (about 256 cells per batch) and one per-layer map, 2-3 seeds | about $70-95 | float32 exploration shows a candidate effect |

Expected total about $100-120; proposed hard ceiling $200 with a check-in at
$100. Controls: each phase priced from measured throughput and approved
before launch; maximum wall time and no-progress timeouts frozen; only
task-created pod IDs touched (the account has unrelated running pods);
credentials stay on the local controller; artifacts hash-verified before
termination; termination verified by direct lookup and full inventory.

## Known risks

1. CIFAR networks may drift at the edge of stability instead of settling onto
   isolated attractors; weight decay may be needed and may change the regime.
2. Float32 rounding could blur invariant circles or erase narrow tongues.
3. Generality beyond tiny networks is unproven.
4. Partial anticipation: Hofmann (2026) reports complex 2-cycle multipliers in
   a two-mode linear network; trainability-fractal and edge-of-stability work
   is extensive.
5. Reviewer monoculture: the executing agent and this consult are both
   frontier language models; the human researcher (a dynamical-systems
   expert) is the independent channel.

## Decision requested

Is this roadmap the most efficient path to a strong cs.LG and nlin.CD paper?
What should be cut, added or reordered? Is the paid-compute plan justified and
correctly sized, and are its gates right? Which single experiment would most
increase the paper's value, and what would a skeptical referee attack first?
