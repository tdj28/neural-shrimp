# Next steps after the actual neural shrimp

Updated 2026-09-11 after EXP-004. The earlier plan is preserved in Git history.
This is an experiment agenda, not an executed utility result or authorization
for paid resources. Each target packet must be complete, locally audited, and
pushed before its outcomes.

## What is now established

EXP-004 evaluates 39,523 neural settings, including an independently initialized
201-point transect. It reproduces the published nine-peak shrimp-shaped window
in a fixed-weight three-neuron recurrent network. Independent shooting verifies
two nine-peak orbits and one eighteen-peak orbit. At the core, the full neuron
state and activation vector repeat every 37.63637538 time units; the dominant
transverse Floquet multiplier is 0.06506010. The nearby eighteen-peak orbit
repeats after 74.48169921 time units. Those are different network behaviors,
not graphical decorations.

The window is in the plane of two connection weights. A trajectory evolves in
three-dimensional state space; changing its state does not move its connection
weights into or out of a shrimp. Control changes and basin changes are different
interventions. The original grid-selected exterior becomes a nine-peak candidate
under both refinements. EXP-005 separately checked fixed controls using the
source paper's 10,000-unit settling horizon. Both distant comparison points
retained absent short periods and positive finite-time exponents at both steps.
After a 1e-6 state perturbation and 300 time units, the actual activation-vector
separation was 1.53e-6 at the core, versus 1.41 and 2.13 at the lower and upper
controls. This is a strong finite-horizon repeatability difference, not a task
quality or asymptotic-chaos theorem. The failed near-edge interpretation remains
in EXP-004; EXP-005 independently certified a nine-peak cycle there.

## 1. Measure whether the shape predicts tolerance to weight drift

**Question:** do broad bodies and narrow extensions differ in the connection
perturbations they tolerate, beyond knowing that their current orbit is stable?
This is the most direct potential use of the two-parameter geometry. A Floquet
multiplier concerns small changes of state at fixed weights; it does not measure
how far the weights can change before the attractor disappears or changes.

Continue the certified nine-peak orbit and locate its saddle-node and flip
boundaries. Retain failed continuation steps, unstable branches, and ambiguous
locations. Sample interior, extension, and boundary settings under a prospectively
specified rule, with multiple initial states and phases. Measure period, physical
cycle duration, neutral mode, transverse multipliers, basin capture, and control
margin. Distinguish branch stability from which branch a given initial state
reaches. A visual edge in a finite map is not an exact bifurcation boundary.

At those settings inject matched perturbations of the two weights, with the
control metric fixed before outcomes. Use the focus-rectangle widths for one
explicit dimensionless metric, and report perturbations in raw weight units too.
Measure retention of the nine-peak regime, waveform error after phase alignment,
phase drift, recovery time, and failure probability. Separately perturb states
at fixed weights to test orbital recovery. Do not infer weight robustness from
state contraction or a broad state basin from a wide parameter window.

**Decision criterion:** boundary distance or local window width must predict
weight-perturbation failure beyond period, amplitude, clean waveform quality,
and local Floquet multipliers on held-out settings. A null result is useful:
it says the expensive map adds little to simpler local diagnostics.

## 2. Test one concrete use: preserve a neural rhythm under drift

Treat the recurrent network as a rhythm generator with a fixed reference
activation waveform, specified allowable phase error, and a prescribed stream
of connection drift and state disturbances. Freeze the reference and tolerances
using development cases. Final disturbance streams and parameter settings are
held out. This defines a real measured control task, but remains a small-model
benchmark, not an established application in a trained modern network.

At shared saved checkpoints clone all policy arms. Compare a controller using
window geometry with an ordinary parameter search, a controller using complete
local recurrence/period/amplitude/stability diagnostics, and a sham with identical
diagnostic overhead. Match allowed weight changes, action opportunities, sensor
access, tuning budget, and total computation. Include random directions and a
shuffled-boundary control to test whether the direction supplied by the geometry
matters. A task-quality map with the same cost is an essential competing baseline.

**Primary endpoint:** fraction of the evaluation horizon satisfying the frozen
waveform and phase tolerances, counting every failure. Secondary endpoints are
recovery time, control effort, and total compute including atlas construction,
tuning, failed probes, and online diagnostics. Report the number of uses needed
to amortize mapping cost. Do not call improved recurrence alone useful control.

## 3. Determine whether the mechanism transfers to training or reasoning

The Sohl-Dickstein training map and the latent-reasoning basin map have different
evolving states and controls from this fixed-weight ODE. Pin one small published
model and reproduce its reported dynamics before trying to transfer a policy.
Include optimizer buffers, schedule phase, caches, or secondary latents in the
state when they evolve. A repeated scalar loss or decoded answer is insufficient.

Once a full-state periodic regime is verified, measure phase-resolved predictions,
held-out quality, correctness, settling time, and resource use. For a solver meant
to reach a fixed answer, a persistent cycle can be a failure mode; for a rhythm
generator it can be the desired behavior. Which interpretation applies is a
property of the task and readout, not the shrimp outline. Compare entry, exit,
averaging, and stopping only against equally funded ordinary methods. Preserve
all unresolved and divergent runs. The original training/policy pilots supply
no demonstrated benefit and remain unchanged.

## Resource decision

The actual EXP-004 grid completed locally with four CPU workers in 140.54 seconds.
The immediate continuation and controlled-perturbation work should therefore
start locally. A GPU is justified only by profiling the exact later workload,
checking numerical parity, and preparing a measured bounded cost plan under
[compute-plan.md](compute-plan.md). Paid execution and paid Pro review each need
their own explicit approval; none was used for the present numerical result.
