# EXP-016: two-write memory under interference

17 September 2026. Exploratory pilot following the
[completed Pro consultation](consults/2026-09-17-pro-adjudication.md) and
[revised design](two-write-memory-design.md). No new neural solve, reference
gain calculation or target timing pilot precedes the complete reviewed,
privately pushed executable freeze. Earlier unqualified studies remain so.

**Human-friendly TLDR:** We have evidence that the selected near-edge neural
cycle retains a disturbance longer than the core cycle. This experiment asks
whether that lingering activity helps a fixed observer recover an earlier
input after a second input arrives. State-component interventions test what
contributes to that observer. Linear decoding failure will not be described
as erasure of all information or absence of memory in every possible sense.

## Inputs, phases and complete census

Use only the two qualified reference packets from EXP-014 recovery attempt
002, source `406b89553b956bac1b97ef05413c799e383a9103`, external receipt SHA256
`2494966914cb68d8c5286a5dca6710212ade171ead42c9cfaed82cebd44c12a8`.
The new plan binds the receipt plus each reference JSON/NPZ byte inventory.
Original pulse trajectories, fitted task scores and previous phase propagators
are not input to this study. The inherited cycles are selected settings in a
fixed-weight three-neuron tanh network, not a sample of trained networks.

Draw 24 phase fractions with `Generator(PCG64(2026091701)).random(24)` and
store their exact float values in the plan. Both settings use the same ordered
fractions. Indices 0–7 are development, 8–23 held out. No fraction equals an
original eighth-cycle phase. The scientific specification is fixed for both
splits; development supplies measured engineering cost, not response-based
tuning. Complete all reference work before any finite-pulse trajectory.

For every setting and phase, first and second signs are each {-1,0,+1};
retention arms are identity, timing and transverse. This gives 1,296 histories,
432 development and 864 held out. All histories have coarse/fine native RK4
traces. Every one of the 27 histories at phase indices 0 and 8 in each setting
also has an independent DOP853 trace: 108 independent traces. Total traces
are 2,700, divided into 918 development and 1,782 held out. Two full-state/
tangent reference methods at every one of the 48 contexts add 96 jobs, for
**2,796 total planned jobs**. Comparisons are all 1,296 coarse/fine pairs and
108 fine/independent pairs, 1,404 in total. The complete ledger exists before
dispatch; blocked, failed and interrupted identities are never removed.

## Reference construction and event chronology

At the sampled initial phase, obtain the state and adjoint from the unchanged
periodic cubic-Hermite templates. Evaluate the physical RHS at that state.
Construct the unit phase-null direction by projecting coordinate axes off
the normalized adjoint and selecting the column with maximum norm, first-index
tie rule. This direction is reference-only. Do not renormalize the adjoint
or silently repair its physical normalization error.

For each context directly integrate `[x, Phi]`, with Phi initially I, over
[0,T], then restart a fresh identity Phi at T and integrate [T,8T]. Do not
invert a nearly collapsed fundamental matrix or assume exact period closure
by repeating a one-period matrix. The two reference algorithms are DOP853
(rtol 1e-12, atol 1e-14, maximum step T/2048) and RK45 (same tolerances,
maximum step T/4096). Save all three state components and all nine tangent
entries at T/4096 intervals, with both segment endpoints and derivatives.
Each algorithm propagates its own unforced endpoint into the second segment.

The primary DOP853 state at T supplies the common intervention reference.
Compute its physical flow `fT`; obtain `zT` from the saved periodic adjoint
at the declared nominal epoch plus T. This is an approximate periodic adjoint
at the actual unforced endpoint, with its error checked explicitly. Define
`Q=fT zT^T`, `P=I-Q` without repairs and record normalization, idempotence and
leakage diagnostics. All methods use the same sealed Q/P and reference state.

Write `s1 * 1e-4 * v` at time zero. Integrate exactly to T, save T-minus,
and let d be the deviation from the sealed reference. In the timing or
transverse arm replace the state by `xrefT+Qd` or `xrefT+Pd`. Identity retains
the original state bit-for-bit while recording the same event diagnostics;
it avoids introducing an artificial subtract/add round-trip. Only after
retention add `s2 * 1e-4 * v`, save T-plus, and restart integration to 8T.
No event-detection or rounding to a nearby timestep is used. Preserve every
requested and realized addition, both event sides, replacement magnitude,
resulting state deviation, phase leakage and projection amplification.
Oblique operators can amplify a deviation; they are not orthogonal energy
partitions or exact finite-amplitude isochrons.

Native RK4 uses T/32768 and T/65536 and saves T/4096. The independent DOP853
panel uses rtol 1e-13, atol 1e-15, maximum step T/4096 and exactly the same
sample/event times. Separate segment arrays prohibit interpolation through
the discontinuity. These period-locked configurations require the new
qualification checks; old fixed-step results do not automatically certify them.

## Sealed observers and predictions

Use 32 midpoint samples of all three tanh activations across each cycle
ending at 2T, 4T and 8T, plus an immediate window over [0,T/8]. These are exact
saved grid nodes. Independent observation noise has sigma 1e-4 per activation
sample; the reference is calibrated and noiseless. The observer knows nominal
starting phase and elapsed time. No noisy ODE, fitted threshold, noisy-trial
training, inferred hidden-bit baseline or phase optimization is introduced.

For each delayed window, use the direct variational matrices and activation
derivative at the primary unforced state to form first-write response A and
second-write response B. The first response includes the T intervention
before later propagation; the second is added afterward. For identity, seal
unit h proportional to `(sigma^2 I+B B^T)^(-1) A`, oriented so h dot A is
positive. The immediate observer has B=0. Zero/undefined primary A invalidates
that context; no favorable fallback or held-out fitting is permitted.

For all arms and second-present/absent comparisons, apply the same identity-arm
h and subtract the same presealed zero-write reference under the event schedule.
For exact zero deviation, every retention operator leaves that reference
unchanged. Never subtract an observed sham correction or a reference depending
on either unknown bit. Finite solver drift is tested using the acquired
zero/zero histories. Signed margins are h dot the activation difference divided
by `sigma*||h||`. For nonzero first signs, the probability of correct decoding
is the standard normal CDF of first-sign times that margin. Average balanced
bits analytically; retain the unsaturated signed values as well as probabilities.

Preseal three comparator families: full-linear responses under the actual
event schedule; an arm-specific covariance observer and the derivative-based
clock observer; and the periodic one-mode approximation `A=rho*B`, matching
the identity arm's second-write response B at that window. Rho is the signed
dominant real transverse multiplier of the saved cycle certificate. A missing
or complex dominant multiplier makes that comparator unavailable, not a
post-hoc substitute. The one-mode model is an observability-matched local
approximation, not a simulation of the whole nonlinear neural system. The
immediate window is a calibration, not a one-mode forgetting prediction.

Undefined secondary observers remain explicitly unavailable. Secondary
observer agreement is flagged separately and cannot replace the primary
observer or rescue a failed primary conclusion. The covariance construction
is not a Bayes-optimal binary-mixture decoder. Synthetic tests include the
four-level scalar counterexample: a zero-threshold linear observer can fail
while a nonlinear decoder can distinguish the noiseless bit pairs. Removing
a component may reformat information for a readout without erasing it.

## Numerical qualification, failure handling and estimands

Reference prerequisites compare both complete unforced state trajectories and
all direct tangent columns, including the pulse-scaled activation response.
State/activation relevance thresholds are 1e-6 and 1e-7. Operator differences
are scaled by epsilon 1e-4, with all window A/B predictions and primary signed
margins also compared under the same Q/P. Preserve raw operator maxima and
their locations; an algebraic identity or neutral-flow check alone does not
qualify the transverse operator. Initial and event physical adjoint
normalization must satisfy absolute residual <=1e-7. A failed prerequisite
blocks its dependent trajectories while preserving every planned terminal row.

For all 1,404 planned trajectory pairs, compare complete saved state and
activation arrays, both event sides, and every primary signed readout. Maximum
absolute state difference must be <=1e-6, activation difference <=1e-7, and
standardized primary margin difference <=.01. Analytic correctness-probability
agreement must be <=.001. All zero/zero histories additionally require absolute
primary standardized drift <=.01 for every arm/window. A first-zero,
second-nonzero input legitimately creates a nuisance signal: its q is not
required to be zero. Balanced pseudo-first-label accuracy is analytically .5.
Immediate physical detectability and injected-observation synthetic controls
are reported, without choosing a dose/noise pair from their outcomes.

Setting-level qualified means require every own reference and history to meet
the frozen primary gates, including development and independent-panel cases.
Keep completion separate from numerical qualification. Otherwise withhold the
qualified mean and retain all per-history diagnostics and conservative bounds.
The primary contrast is near-edge minus core first-bit balanced accuracy at
4T, identity arm, unknown balanced nonzero second bit, equally weighted across
the sixteen held-out phase pairs. Secondary outcomes include no-second accuracy,
interference loss, fixed-observer retention contrasts, other windows and
reference-only arm-specific/one-mode comparators. No threshold is labeled a
minimum practically useful effect.

Use 20,000 paired phase-block bootstrap draws with seed 20260917 and the central
95% interval when both settings qualify. This is descriptive uncertainty over
phase sampling conditional on these two selected cycles. Do not resample bits,
arms, windows or solvers as independent networks. Failure bounds assign each
unqualified held-out phase block accuracy [0,1]; include all sixteen blocks
and form the contrast from the two setting bounds. Bounds are not a qualified
mean or a confidence interval. All old failed studies remain unchanged.

## Execution, evidence and verification

Use the shared local CPU with four owned workers and all six numerical thread
environment variables fixed to one. Require at least 16 GiB available RAM
before dispatch and preserve an 8 GiB reserve. Limit combined owned-process
RSS to 12 GiB (shared pages counted in each process) and external raw output
to 8 GiB. Record CPU load, RAM, disk space and exact process identities. Do
not terminate unrelated processes. EXP-012's tested GPU path was slower and
numerically incompatible; no new GPU or paid-cloud run is included.

The producer ceiling is 3,600 seconds, with 90 seconds per reference task,
60 seconds per trajectory and 180 seconds final sealing grace. These are
ceilings, not measured promises. Prior EXP-014 refinement acquired 425 traces
in 178.036 seconds using four workers. New direct tangent work, denser grids
and event handling add cost. Record complete fixed development-stage timings;
do not inspect development effects to change the held-out specification.

The controller requires exact committed and privately pushed source/plan/local
review, a clean checkout, unchanged input hashes and a new exclusive attempt
under `../neural-shrimp-data/raw/EXP-016/`. Persist input/context identity before
references and all reference predictions/observers before any pulse dispatch.
Read back the saved seals and bind their hashes into the event chronology.
Code and compact receipts go to private Git; raw arrays remain external.

Every failed/incomplete attempt and returned partial solver array is retained.
The lifecycle records owned PID/start identities, deadlines and verified worker
termination. A forced kill or storage failure can prevent an in-process buffer
or final receipt from being saved; existing partial files and the initial full
ledger remain. Do not claim completeness in that case. No automatic scientific
retry, changed tolerance, replacement phase or selected favorable method is
part of this run.

The saved-array verifier checks byte inventory, frozen source/plan/input
bindings, lifecycle and chronological seals, then reconstructs references,
readouts, all scores, comparisons, failure rules and summaries without new ODE
integration. This replay is separate from a local independent review of the
evidence. Strict arithmetic discrepancies must be reported distinctly from
physical/noise-scale gates rather than silently relaxed. No extra Pro call,
public release or merge to main is part of execution.
