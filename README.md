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

## What this can inform

The first concrete utility hypothesis is **tolerance to connection-weight drift**.
A broad body or narrow extension may predict how far the weights can move before
the desired rhythm is lost. This differs from a Floquet multiplier, which measures
small state disturbances while the weights remain fixed. The next experiment
will test whether window width or boundary distance predicts drift-induced
failure beyond the current period, amplitude, waveform quality, and multipliers.

A subsequent controller must preserve a declared activation waveform and phase
tolerance under held-out disturbance streams, and beat ordinary parameter search
and equally funded local-diagnostic controllers. Mapping cost, failed probes,
and all control actions count. See the [concrete next experiments](docs/next-steps.md).

Phase and transverse recovery remain distinct: an attracting autonomous cycle
can retain a same-time displacement because its phase direction is neutral.
The current perturbation probes show finite-horizon repeatability, not a completed
phase-aware control task.

## Implemented instruments

- Native continuous-time three-neuron tanh integration, complete-state peak
  recurrence, tangent sensitivity, and all-neuron state/activation statistics.
- Independent adaptive shooting with primitive-period, complete-state closure,
  neutral-flow, and transverse-Floquet checks.
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

Existing EXP-004--005 evidence and figures can be checked without advancing the ODE:

```sh
uv run --frozen python tools/run_hopfield.py verify \
  --output evidence/EXP-004/attempt-001
uv run --frozen python tools/plot_hopfield.py evidence/EXP-004/attempt-001 \
  --destination paper/figures --check
uv run --frozen python tools/run_hopfield_controls.py verify \
  --output evidence/EXP-005/attempt-001
uv run --frozen python tools/plot_hopfield_controls.py evidence/EXP-005/attempt-001 \
  --destination paper/figures --check
```

The plotting commands re-derive plots from saved measurements. `--verify` also
requires byte-identical image regeneration in the same library/font environment.
No source outcome is changed by rebuilding the figures.

Fresh target execution requires clean source, a pushed source commit, a frozen
plan, and a new attempt directory. Never overwrite a consumed plan or reset an
attempt. The original EXP-004 used four CPU workers and 140.54 seconds of
measured wall time; EXP-005 used one worker and 4.17 seconds. Paid compute and
paid review spend were zero.

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
