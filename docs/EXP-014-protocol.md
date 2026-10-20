# EXP-014: selective timing memory in two neural cycles

17 September 2026. Exploratory mechanistic study, not a formal preregistration.
This executable supplement incorporates the scientific definitions in
[phase-memory-design.md](phase-memory-design.md), with the prospective
implementation clarifications below. EXP-013 must finish with its outcome
retained before this study's first target solve. The complete code, this
protocol, machine plan, tests and exact-plan local review must then be
committed and privately pushed before any new reference or pulse trajectory.
Synthetic tests and metadata-only preparation are allowed before that freeze.

The question is whether equal-norm state pulses leave a readable timing
component while transverse disturbances decay, and whether an ordinary
neuron-one pulse makes that timing accessible. Phase-writing advantage is
predicted by construction. A phase-only emulator is mandatory and may explain
the result completely. Neither a positive contrast nor an emulator mismatch
establishes special shrimp computation, trained-network generalization,
learning, memory capacity, or an advantage over other oscillator systems.

## Fixed system, references and interventions

Use the unchanged three-neuron tanh RHS and its Jacobian. The two settings
are the original EXP-005 `core` (alpha 1.064, gamma 10.260722222222222) and
`failed-near-edge` (alpha 1.0661333333333334, gamma 10.297166666666667).
The latter name preserves its failed original irregular-control role; both
later had attracting primitive nine-peak numerical certificates. They are
selected settings, not independent draws or a causally matched geometry pair.
Bind the exact EXP-005 plan, manifest, receipt and two point files. Historical
section states/periods seed new unchanged certificates; no alternate seed,
replacement setting or duration extension is allowed after outcomes.

Require every original certificate gate, primitive nine-peak identity,
correct physical parameters and attraction. Then integrate state plus the
3-by-3 identity-initialized variational matrix over one period with DOP853,
rtol 1e-12, atol 1e-14 and max_step 0.025. Save 4,097 uniform endpoint-inclusive
times, full potentials, flow, tangent matrices and their derivatives. Keep
raw closure and the actual endpoint. The periodic interpolation uses the
first 4,096 samples with a declared wrap to the initial sample; it never
replaces the saved raw endpoint. Require 8*T <= 320.

The time-valued phase gradient is the right singular vector of M^T-I for
its smallest singular value, normalized once by its product with the flow.
No monodromy inverse, inverse tiny multiplier or per-time normalization is
used. Integrate the adjoint backward from T to zero along the fine solver's
raw dense orbit, and retain all residuals. Prospective implementation gates:

| Check | Requirement |
|---|---|
| Fine-orbit relative closure | ||x(T)-x(0)||/max(1,||x(0)||) <= 1e-8 |
| Neutral-left and neutral-flow relative residuals | each <= 1e-7 |
| Unique singular direction | second-smallest singular value > 1e-12*max(1,largest) |
| Phase normalization conditioning | absolute unit-neutral-vector/flow cosine >= 1e-12 |
| Once-normalized initial phase gradient | |Z^T f - 1| <= 1e-12 |
| Backward-adjoint relative endpoint residual | ||Z(0)-Z(T)||/max(||Z(0)||,||Z(T)||) <= 1e-7 |
| Whole-orbit normalization | max |Z(t)^T f(t)-1| <= 1e-7 |
| Reference arrays and derivatives | complete, finite float64, exact prescribed grids |

Use eight phases j=0..7. The realized phase state, flow, gradient and epoch
come from saved sample j*512, including its exact binary64 time. At each
phase, integrate a separate full-period identity-initialized propagator with
the same fine solver. Require relative closure <= 1e-8 and retain neutral
flow/transport residuals. The propagator is a nonperiodic matrix segment;
linear response is Phi_j(s) M_j^k delta_x in that order, with no inverse.

For epsilon 1e-4, 1e-3 and 1e-2, apply both signed state pulses along:

- phase_write: Z/||Z||;
- phase_null: normalized projection of the coordinate axis with largest
  projected norm onto the plane orthogonal to Z, lowest-index exact ties;
- neuron_one: (1,0,0).

Require unit-norm error <= 1e-12 and normalized null projection <= 1e-12.
Keep requested and realized floating-point displacements, addition residuals,
state norms and initial activation displacements. Do not rescale using an
observed response. Include one zero-dose sham per setting/phase/method.
Weights are constant after each instantaneous state pulse.

Every trajectory covers [0,320] with exactly 12,801 saved potentials at
spacing 0.025. The native methods are RK4 dt=0.0025 and 0.00125 (128,000 and
256,000 steps); the independent method is DOP853 with the fine settings above.
Raw potential arrays are retained; activations are derived as tanh(x).

## Census, sealing and readout

There are 288 pulse cases plus 16 shams, each with three methods: **912
trace tasks**. Two reference jobs and sixteen phase-propagator jobs give
**930 terminal task rows**. Preserve every planned identity on failure.
Flat case metadata uses setting, phase_index, kind, dose_index, epsilon,
direction, sign and method; the machine plan fixes every exact ID.

Before any pulse or sham starts, persist, reload, exactly replay and hash-seal qualified reference
and propagator packets, phase states/directions, all clock coefficients,
requested-pulse phase-only outputs and realized-pulse full-linear predictions.
Unavailable prerequisites remain explicit. This seal is derived under the
prospective source freeze; do not imply its numerical values existed in the
metadata-only machine plan. If a setting's prerequisite fails, block its
associated tasks while preserving the other planned setting's opportunity
to run. Never substitute a setting or silently shrink a denominator.

For k in {1,2,4,8}, read 32 midpoint samples across the cycle ending at k*T:
t_m = T*(k-1+(m+0.5)/32). Reconstruct potentials with the prescribed cubic
Hermite rule, then apply tanh. The experiment supplies an external clock and
known starting phase. Reference U0 and its derivative G define w=G/(G^T G),
and the time-shift score is q=w^T(U-U0). Here G is the derivative of the
periodic Hermite potential template followed by tanh; its nodal derivatives
originate from the physical RHS. Preserve derivative energy and interpolation
closure diagnostics. It is a numerical approximation, not an error theorem.
No phase optimizer, learned threshold or pulse-dependent readout is allowed.

Under independent measurement noise sigma on each of the 96 activation
samples, score standard deviation is s=sigma*||w||. The primary sigma is
1e-4; fixed secondary values are 1e-5 and 1e-3. For opposite signed pulses,
BA=[Phi(c*q_plus/s)+Phi(-c*q_minus/s)]/2. The orientation c is the sign of
Z^T d (positive for exact zero), except phase_null always uses positive
orientation. Noise affects measurement only; the reference is calibrated
and noiseless. There is no noisy ODE or Monte Carlo class training.

The phase-only null uses the **requested** displacement sign*epsilon*d,
exactly as the design's delta_t=sign*epsilon*Z^T d. Retain the realized-delta
phase shift only as a rounding diagnostic. Full-linear response uses the
realized displacement and is labeled accordingly. Its dense diagnostics
are limited to saved times <=8*T; full numerical trace comparisons still
cover all 320 units. The zero-response null has BA exactly 0.5.

The exact analysis censuses are 3,648 signed trace readouts (3,456 pulse and
192 sham), 5,184 paired BA cells across all methods, and 1,728 distinct
method-independent phase-only null BA cells. There are 608 raw cross-method
comparison rows (304 per setting), 2,432 signed-readout comparisons and
3,456 paired-BA comparisons. Methods are numerical checks, not replicates. Retain 432 opposite-sign
state/activation response diagnostics and 1,728 fixed-delay odd/even score
rows, including explicit unavailable records.

## Qualification, estimands and interpretation

Compare coarse versus fine RK4 and fine RK4 versus DOP853 for every case.
Require full-trace maximum absolute potential difference <=1e-6 and
activation difference <=1e-7. Every signed-readout q/s difference at primary
noise must be <=0.01. Every paired BA difference at every declared noise,
delay, phase, dose and direction must be <=0.001. Preserve RMS values,
maxima, their locations and components regardless of passing status.
All methods must be complete, finite, float64 and bound to the exact initial
condition, grid, solver and pre-acquisition seal.

Every sham readout must satisfy |q/s|<=0.01 at primary noise. A sham is a
reference-drift/interpolation check; never subtract its observed score from
primary pulse scores. Failure makes the associated setting unresolved under
this plan. No automatic refinement, tolerance relaxation or favorable-method
selection is included. An amended refinement would require a separate freeze.

Fine RK4 is the primary numerical readout. At each setting, report the
phase_write-minus-phase_null BA contrast at the cycle ending at 4*T and
sigma=1e-4, plus both absolute accuracies and ordinary neuron-one accuracy,
with equal weight across eight phases and three doses. Keep all dose/phase
cells and all secondary delays/noise levels. Only two selected settings exist:
no bootstrap over phases, signs or solvers as independent network replicates.
A setting-level qualified average requires all of its own prerequisites and
456 traces to qualify; a complete-study claim requires both settings.

Retain phase and transverse linear projections, state/activation residuals,
odd/even signed responses, saturation diagnostics and finite-amplitude escapes.
They are local linear coordinates, not exact finite-amplitude isochrons.
Second-order phase shifts of the phase-null input may be expected. A good
phase-only match supports generic oscillator timing memory. A discrepancy
requires diagnosis; it is not automatically evidence of richer computation.
No task-derived acceptable accuracy margin or novelty claim is declared.

## Execution and evidence

Use local CPU: four owned workers, one numerical thread each, no GPU or paid
compute. The prospective producer ceiling is 1,800 seconds, with 120 seconds
per reference/propagator and 60 per pulse/sham. Limit combined owned-process
RSS and retained raw data to 2 GiB each; require 8 GiB available RAM before
dispatch and retain a 4 GiB reserve during work. Record measured synthetic
infrastructure timing and a conservative historical CPU estimate in the
final plan. These ceilings are not measured runtime promises. Finalization
must retain failures under a separate 120-second grace period.

The integrated four-suite synthetic check passes 142 tests in 24.80 seconds,
including analytic Hermite/time-unit behavior, Gaussian score arithmetic,
finite positive clock-energy validation, actual owned worker scheduling,
cooperative timeout evidence and exact persisted-seal replay. Interpolation
and raw endpoint checks plus the all-sham floor have declared prospective
gates; there is no undeclared interpolation cutoff or new-target timing pilot.

EXP-013's saved 75-unit median integration costs are 0.006895007 seconds
(native .0025; 3,504 cases), 0.012064974 seconds (native .00125; 3,504 cases)
and 0.240359109 seconds (fine DOP853; 48 cases). Duration-only scaling to 320
units and 304 cases per method gives 336.35 task-seconds, or an idealized
84.09 seconds at four workers. This compute-only extrapolation omits the
17.05-fold larger saved sample count, reference/adjoint/propagator work,
roughly 264 MB reference seal, serial readouts, replay and storage. Budget
10–15 minutes as a conservative planning estimate, retaining the 30-minute
producer ceiling. The plan records exact historical ledger bindings. These
are engineering estimates from prior completed work, not new target outcomes.

Preparation and validation are metadata-only. The live runner requires a
clean checkout, exact committed source/plan/review, exact private remote HEAD,
unchanged historical hashes and a new exclusive external attempt. Use
`../neural-shrimp-data/raw/EXP-014/`; never resume or overwrite evidence.
Registered PID/start identities and parent-death handling limit termination
to owned processes. Retain setup failures, completed partial native/adaptive
arrays and interrupted unaccepted RHS snapshots when cooperative cancellation
reaches Python. Files already persisted survive forced termination; an
in-process native buffer or final snapshot cannot be guaranteed after a forced
kill or parent death. Keep every terminal record, resource log, source/runtime/
library hash, temporal seal and cleanup receipt. The distinction limits
evidence recovery and does not relax any scientific qualification gate.

The read-only verifier reconstructs saved identities, reference-derived
seals, template/linear/null outputs, all scores, numerical gates, complete
censuses and contrasts without reintegrating. Metadata replay uses the same
safe JSON conversion as writing; raw array dtype/grid/hash checks remain
strict. Same-implementation replay and an independent evidence/scientific
review are separate requirements. Source or artifact drift invalidates
qualification without deleting the failed attempt.

The Pro consultation was not run because its [amount redacted] hard ceiling could not
be guaranteed; incremental spend is USD0. This study follows local review,
not purported Pro advice. Keep the repository private, push coherent code
checkpoints, and preserve all earlier failed evidence and merge gates.
