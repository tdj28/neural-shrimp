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
- **Exploratory consequence pilot (three slices, two networks):** regimes differ
  in reproducibility as their Lyapunov exponents imply (period 2 decays,
  quasi-periodic grows linearly with the perturbation, locked windows show phase
  slips after chaotic transients, chaos saturates), but rotation number and
  locking add no robust prediction of loss or reproducibility beyond Lyapunov
  exponent and sharpness (Delta R^2 about 0). The rotation-specific consequence
  is a diagnostic: the even-step peak of the training-loss spectrum equals the
  rotation number within 1e-3 (125/125 quasi-periodic and 97/97 locked cells on
  one slice; similar on the others), and with small update noise an AR(6) fit
  of the loss recovers the period-2 orbit's complex multiplier before the
  transition (median angle error 1.8e-4 and 3.6e-4 in the two networks).
- **Open:** odd-period windows (5 confirmed) are not circle orbits; a minimal
  two-parameter model shows only period doubling; no real-data result yet;
  float32 sensitivity untested. The Neimark-Sacker curve is W-shaped; the
  birth rotation peaks just above 2/9 next to the widest (2/9) window.
- A separate mechanism consult (same model family) recommended, and we
  accepted: resonance-anchored ancestry by pseudo-arclength continuation of a
  window's stable and saddle orbits and both fold edges toward its predicted
  root, with a resonant normal-form positive control and an embedded unimodal
  negative control; an effective one-dimensional return-map test for odd
  windows; a coupled four-weight two-mode model as the minimal candidate;
  invariant-circle solves; independently generated losses as replication
  units; a bounded CIFAR pilot profiled locally before renting.
- A 14-page manuscript draft with six reproducible figures exists; every
  number is generated from committed data and re-verified by a script.

## Roadmap

**R1. Mechanism (local CPU and RTX 2000 Ada; days 1-6).** Resonance ancestry of
the 1/5 window (selected by a fixed rule; fallback 2/9) with positive and
negative controls; then one invariant-circle solve outside the locking
interval; the coupled four-weight model scan (c in {0, 0.3, 0.6, 0.9}, two
ridges, eight rate ratios); a one-window unimodal-quotient pilot for odd
windows; 1024 x 1024 GPU rotation maps and Neimark-Sacker curves for all four
network-seed pairs for figures.

**R2. Generality on real data (days 2-9).** (a) Running now: a small tanh MLP on
scikit-learn digits with full-batch gradient descent and a single learning
rate, testing for Neimark-Sacker of the 2-cycle, rotation numbers, locking and
staircases. (b) Bounded real-data pilot: full-batch gradient descent
in float64 on two independently drawn CIFAR-10 subsets of 256-512 images (the
second reserved), one 32-unit hidden layer, MSE with weight decay; track the
fixed-point flip and the 2-cycle with matrix-free Arnoldi multipliers, replay a
validated toy Neimark-Sacker through the same machinery, and compare the
loss-spectrum readout with state-based rotation. Profile 1,000 updates locally
first; scale up (larger subsets, a CNN) only if the bounded pilot succeeds.

**R3. Consequence (days 3-9).** Develop the loss-only diagnostic: rotation
readout from the loss spectrum and the noise-driven AR precursor of the
Neimark-Sacker transition; test it with minibatch noise instead of isotropic
noise and on the real-data pilot. Reproducibility classes are reported as
generic (explained by Lyapunov exponents), not rotation-specific.

**R4. Confirmatory test (days 8-12).** Frozen and pushed before running, on
reserved fresh seeds (29000-29999, never used in exploration) and at least one
network not used in exploration, with an independent audit. Replication units are independently generated losses
(data, targets and initial weights from one reserved seed). Draft predictions:
(a) the period-2 orbit's stability boundary at the band's onset is
Neimark-Sacker and the first quasi-periodic rotation number equals
arg(mu)/2pi within 2e-3; (b) at least 80% of even-period periodic cells in
the band are m/q-locked (q = period/2) or period-doubled inside an m/q tongue;
(c) the loss-spectrum readout equals the state-based rotation within 1e-3;
(d) with small update noise, AR recovery of the 2-cycle multiplier within
2.5e-3 in angle and 0.02 in modulus; (e) ancestry passes for the rule-selected
window in at least half of the losses where one qualifies.

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
| 0 | Rebuild the CUDA engine for sm_90; parity check against the CPU reference on sampled cells; time one 72-parameter map unit and one float64 bounded-CIFAR batch | at most $5 | parity passes and timing measured |
| 1 | Confirmatory maps for the 72-parameter architecture on fresh losses (each 1024 x 1024 map is about 4.5 local GPU-hours); the 13-parameter maps stay local | about $10-20 | design frozen and pushed |
| 2 | Real-data scale-up beyond local capacity (larger subsets, more learning-rate cells, a CNN), only if the bounded local pilot succeeds | about $20-60 | bounded pilot positive and profiled |

Expected total about $35-85 (Phase 2 is conditional); proposed hard ceiling
$150 with a check-in at $75. Controls: each phase priced from measured throughput and approved
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
