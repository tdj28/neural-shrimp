# EXP-019: complete two-write memory regression with the refined reference

17 September 2026. Exploratory regression repeat of
[EXP-016](EXP-016-protocol.md), following the globally qualified [EXP-018
reference](EXP-018-results.md). No new neural solve, reference gain or timing
pilot precedes a complete, reviewed and privately pushed executable freeze.
Prior failures remain intact.

**Human-friendly TLDR:** Earlier checked cases suggest that this small network
forgets an input quickly at the selected core, while near the selected edge it
can retain the input for several cycles. A second input can interfere with that
memory. This repeat tests that behavior across the complete planned phase set,
using the newly qualified repeating-orbit reference. It does not yet test an
advantage unique to shrimp geometry or transfer to trained networks.

## Inputs, phases and complete census

Use only the two globally qualified EXP-018 reference pairs at source
`7c9b1977f5d12e4354ff41c537778a3a284a4027`, plan SHA256
`b1bd00f851bff8ef600745e09941682c6c58829203180b9e1cf238f9d3eef143`,
external receipt SHA256
`b4b56a8fbcc592285cebdbf15c2182a3173c62d3e770ebc1681cd9a142066ce1`.
Bind exactly eight historical inputs: that receipt, both reference pairs'
JSON/NPZ packets (four files), the raw summary and manifest, and the final
`docs/EXP-018-summary.json`. Verify all declared bytes and qualified canonical
packet identities before constructing contexts. The compact summary binds
the frozen saved-array verifier and independent evidence review. Both audits'
strict alternate-arithmetic failures remain recorded; categorical evidence
agreement is not relabeled as overall strict-arithmetic agreement.
Original finite-pulse trajectories and fitted memory outcomes are not inputs.
The network is the same fixed-weight, three-neuron tanh system at two selected
parameter settings.

Reuse all 24 original ordered phase fractions exactly, as stored in EXP-018's
bound packets and the new plan. Both settings use the same fractions. Indices
0–7 retain the schema label `development`; 8–23 retain `heldout`. These are
historical labels only: every phase is reused, so this is a regression cohort,
not fresh held-out evidence. No response-based tuning, replacement phase or new
split is allowed. Complete and qualify all reference/context work before any
finite-input trajectory.

For every setting and phase, first and second signs are each {-1,0,+1};
retention arms are identity, timing and transverse. This gives 1,296 histories,
432 historically development and 864 historically held out. All histories have
coarse/fine native RK4 traces. Every one of the 27 histories at phase indices 0
and 8 in each setting also has an independent DOP853 trace: 108 independent
traces. Total traces are 2,700, divided into 918 historically development and
1,782 historically held out. Two full-state/ tangent reference methods at every
one of the 48 contexts add 96 jobs, for **2,796 total planned jobs**.
Comparisons are all 1,296 coarse/fine pairs and 108 fine/independent pairs,
1,404 in total. The complete ledger exists before dispatch; blocked, failed and
interrupted identities are never removed.

## Reference construction and event chronology

At each sampled phase take the exact dense state and adjoint stored by
EXP-018's canonical DOP853 reconstruction. Both new reference algorithms start
from this same state and same unchanged gradient. Do not interpolate the older
Hermite template, perform a new solve, or use method-specific starting states
in context preparation. Evaluate the physical RHS at that state. Construct the
unit phase-null input direction with the unchanged EXP-016 rule: project
coordinate axes off the normalized adjoint, choose the column with maximum norm
using the first-index tie rule, then normalize that direction. Normalizing the
input direction does not renormalize the physical adjoint.

Use the new `refined_two_write_context` adapter, with exact source, plan,
receipt, packet-byte and packet-semantic bindings. Omit the obsolete unused
Hermite-template velocity field instead of fabricating its provenance. Reuse
the frozen EXP-016 acquisition, memory, runtime and evaluation kernels
unchanged through an explicit new controller. Do not monkeypatch old module
globals or invoke the old controller's hardcoded context loader/run/verifier.

For each context directly integrate `[x, Phi]`, with Phi initially I, over
[0,T], then restart a fresh identity Phi at T and integrate [T,8T]. Do not
invert a nearly collapsed fundamental matrix or assume exact period closure by
repeating a one-period matrix. The two reference algorithms are DOP853 (rtol
1e-12, atol 1e-14, maximum step T/2048) and RK45 (same tolerances, maximum step
T/4096). Save all three state components and all nine tangent entries at T/4096
intervals, with both segment endpoints and derivatives. Each algorithm
propagates its own unforced endpoint into the second segment.

The primary DOP853 state at T supplies the common intervention reference.
Compute its physical flow `fT`; use a bitwise copy of the same canonical phase adjoint as nominal `zT`
at the declared epoch plus T. This is an approximate periodic adjoint
at the actual unforced endpoint, with its error checked explicitly. Define
`Q=fT zT^T`, `P=I-Q` without repairs and record normalization, idempotence and
leakage diagnostics. All methods use the same sealed Q/P and reference state.

Write `s1 * 1e-4 * v` at time zero. Integrate exactly to T, save T-minus, and
let d be the deviation from the sealed reference. In the timing or transverse
arm replace the state by `xrefT+Qd` or `xrefT+Pd`. Identity retains the
original state bit-for-bit while recording the same event diagnostics; it
avoids introducing an artificial subtract/add round-trip. Only after retention
add `s2 * 1e-4 * v`, save T-plus, and restart integration to 8T. No
event-detection or rounding to a nearby timestep is used. Preserve every
requested and realized addition, both event sides, replacement magnitude,
resulting state deviation, phase leakage and projection amplification. Oblique
operators can amplify a deviation; they are not orthogonal energy partitions or
exact finite-amplitude isochrons.

Native RK4 uses T/32768 and T/65536 and saves T/4096. The independent DOP853
panel uses rtol 1e-13, atol 1e-15, maximum step T/4096 and exactly the same
sample/event times. Separate segment arrays prohibit interpolation through the
discontinuity. These period-locked configurations require the new qualification
checks; old fixed-step results do not automatically certify them.

## Sealed observers and predictions

Use 32 midpoint samples of all three tanh activations across each cycle ending
at 2T, 4T and 8T, plus an immediate window over [0,T/8], before the second
write at T. These are exact saved grid nodes. Independent observation noise has
sigma 1e-4 per activation sample; the reference is calibrated and noiseless.
The observer knows nominal starting phase and elapsed time. No noisy ODE,
fitted threshold, noisy-trial training, inferred hidden-bit baseline or phase
optimization is introduced.

The observation budget is fixed per cycle. Report each period and elapsed
window times in addition to 2T/4T/8T: periods and sample spacing differ between
settings, so the contrast is not an equal-elapsed-model-time or equal-bandwidth claim.
Elapsed times are model time units, not experimentally calibrated seconds.

For each delayed window, use the direct variational matrices and activation
derivative at the primary unforced state to form first-write response A and
second-write response B for positive epsilon*v basis writes; the first and
second signs combine these responses afterward. The realized floating-point
additions are retained separately. The first response includes the T
intervention before later propagation; the second is added afterward. For
identity, seal unit h proportional to `(sigma^2 I+B B^T)^(-1) A`, oriented so h
dot A is positive. The immediate observer has B=0. Zero/undefined primary A
invalidates that context; no favorable fallback or held-out fitting is
permitted.

For all arms and second-present/absent comparisons, apply the same identity-arm
h and subtract the same presealed zero-write reference under the event
schedule. For exact zero deviation, every retention operator leaves that
reference unchanged. Never subtract an observed sham correction or a reference
depending on either unknown bit. Finite solver drift is tested using the
acquired zero/zero histories. Signed margins are h dot the activation
difference divided by `sigma*||h||`. For nonzero first signs, the probability
of correct decoding is the standard normal CDF of first-sign times that margin.
Average balanced bits analytically; retain the unsaturated signed values as
well as probabilities.

Preseal three comparator families: full-linear responses under the actual event
schedule; an arm-specific covariance observer and a derivative-matched clock
observer using the saved direct physical reference velocity; and the periodic
one-mode approximation `A=rho*B`, matching the identity arm's second-write
response B at that window. Rho is the signed dominant real transverse
multiplier of EXP-018's canonical DOP853-candidate/ DOP853-verification
certificate. A missing or complex dominant multiplier makes that comparator
unavailable, not a post-hoc substitute. The clock observer has fixed
positive-flow orientation and uses direct physical velocity, unchanged from
EXP-016. Earlier studies used a periodic-Hermite template derivative; the
current clock is not an exact replay of that older clock. The one-mode model is
an observability-matched local approximation, not a simulation of the whole
nonlinear neural system. The immediate window is a calibration, not a one-mode
forgetting prediction.

Undefined secondary observers remain explicitly unavailable. Secondary observer
agreement is flagged separately and cannot replace the primary observer or
rescue a failed primary conclusion. The covariance construction is not a
Bayes-optimal binary-mixture decoder. Synthetic tests include the four-level
scalar counterexample: a zero-threshold linear observer can fail while a
nonlinear decoder can distinguish the noiseless bit pairs. Removing a component
may reformat information for a readout without erasing it.

## Numerical qualification, failure handling and estimands

Reference prerequisites compare both complete unforced state trajectories and
all direct tangent columns, including the pulse-scaled activation response.
State/activation relevance thresholds are 1e-6 and 1e-7. Operator differences
are scaled by epsilon 1e-4, with all window A/B predictions and primary signed
margins also compared under the same Q/P. Preserve raw operator maxima and
their locations; an algebraic identity or neutral-flow check alone does not
qualify the transverse operator. Initial and event physical adjoint
normalization must satisfy absolute residual <=1e-7. A failed prerequisite
blocks every finite-input trajectory in this experiment while preserving every
planned terminal row. This global execution barrier is stricter than EXP-016's
per-context dispatch, an explicit prospective policy change with no relaxation
of a scientific threshold.

Before any finite worker starts, all 96 reference jobs must be complete, all 48
contexts must pass the full reference/primary-observer prerequisites, and all
48 seals plus their index and the global gate must be persisted and read back.
If any prerequisite fails, all 2,700 finite rows are blocked; no favorable
subset proceeds. Secondary comparator availability remains separate from the
primary gate. Every finite worker start must postdate the qualified persisted
global gate and all prerequisite seals. Retain both per-context details and the
cohort decision in the receipt and replay.

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
The primary contrast is near-edge minus core first-bit balanced accuracy at 4T,
identity arm, unknown balanced nonzero second bit, equally weighted across the
sixteen historically held-out, reused phase pairs. Secondary outcomes include
no-second accuracy, interference loss, fixed-observer retention contrasts,
other windows and reference-only arm-specific/one-mode comparators. No
threshold is labeled a minimum practically useful effect.

Use 20,000 paired phase-block bootstrap draws with seed 20260917 and the
central 95% interval when both settings qualify. This is descriptive variation
over the reused phase cohort conditional on these two selected cycles, not
fresh validation uncertainty or a confirmatory population interval. Do not
resample bits, arms, windows or solvers as independent networks. Failure bounds
assign each unqualified historically held-out phase block accuracy [0,1];
include all sixteen reused blocks and form the contrast from the two setting
bounds. Bounds are not a qualified mean or a confidence interval. All old
failed studies remain unchanged.

## Execution, evidence and verification

Use the shared local CPU with four owned workers and all six numerical thread
environment variables fixed to one. Require at least 16 GiB available RAM
before dispatch and preserve an 8 GiB reserve. Limit combined owned-process RSS
to 12 GiB (shared pages counted in each process) and external raw output to 8
GiB. Record CPU load, RAM, disk space and exact process identities. Do not
terminate unrelated processes. EXP-012's earlier preview workload showed no
end-to-end GPU speedup and failed its device-parity thresholds. That was not a
benchmark of this memory workload. Here the existing qualified CPU kernels and
measured EXP-016 cost support CPU execution; no new GPU benchmark or paid-cloud
run is included.

The producer ceiling is 3,600 seconds, with 90 seconds per reference task, 60
seconds per trajectory and 180 seconds final sealing grace. These are ceilings,
not measured promises. Historical EXP-016 completed 96 references in about
122.27 seconds and 2,079 finite traces in about 133.16 seconds with four
workers. Scaling only the finite count gives about 173 seconds for 2,700
traces, an engineering proxy rather than a new pilot or promise. The full saved
grids require about 3,460,512,000 uncompressed bytes: 2,831,328,000
finite-state and time bytes plus 629,184,000 reference state/tangent/derivative
bytes, before metadata, seals and compression. Record fixed stage timings; do
not change the specification in response to reused development-phase effects.

The controller requires exact committed and privately pushed source/plan/local
review, a clean checkout, unchanged input hashes and a new exclusive attempt
under `../neural-shrimp-data/raw/EXP-019/`. Serialize and reload all 48 input
contexts before reference dispatch. Persist and reload all reference
predictions/observers, the seal index and the qualified cohort gate before any
finite dispatch. Bind their hashes into each event chronology. The initial full
ledger has 2,796 identities. With two segments per job there are at most 5,592
IVP/segment calls: 192 reference and 5,400 finite calls. Each actual segment
attempt and returned partial packet remains visible. Code and compact receipts
go to private Git; raw arrays remain external.

Every failed/incomplete attempt and returned partial solver array is retained.
If a worker returns a partial packet before a parent timeout, preserve and hash
it without promoting its timed-out ledger row to complete. The lifecycle
records owned PID/start identities, deadlines and verified worker termination.
A forced kill or storage failure can prevent an in-process buffer or final
receipt from being saved; existing partial files and the initial full ledger
remain. Do not claim completeness in that case. No automatic scientific retry,
changed tolerance, replacement phase or selected favorable method is part of
this run.

The saved-array verifier checks byte inventory, frozen source/plan/input
bindings, lifecycle and chronological seals, then reconstructs references,
readouts, all scores, comparisons, failure rules and summaries without new ODE
integration. This replay is separate from a local independent review of the
evidence. Strict arithmetic discrepancies must be reported distinctly from
physical/noise-scale gates rather than silently relaxed. No extra Pro call,
public release or merge to main is part of execution.
