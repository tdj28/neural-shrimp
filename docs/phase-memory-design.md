# Draft: selective timing memory in two verified neural cycles

17 September 2026. **Unnumbered design draft; not an execution freeze.**
This develops the [local strategy review](research-strategy-2026-09-17.md).
It is local analysis, not advice received from a Pro consultation; the
consultation was [not run, with USD 0 incremental spend](consults/2026-09-17-status.md).
EXP-013 remains the current research priority. Do not generate the trajectories
described here before EXP-013 finishes and this successor's complete code,
tests, local review, source inventory and machine plan are privately pushed.
No new fit, reference solve, pulse trajectory or timing pilot was run to write
this document. The numerical constants below are proposed prospective choices,
subject to result-free implementation review rather than outcome-driven tuning.

## Question and estimand

At fixed network weights and a specified initial phase, does the same-norm
input retain a readable timing component while its first-order phase-null
component decays? Can an ordinary input to neuron one support delayed sign
recall under an explicit observation-noise model?

The first question is a causal intervention on pulse **direction within the
same setting**, not a causal intervention on shrimp geometry. The second
tests whether the measured mechanism is accessible without engineering the
input direction from the full orbit. Neither is a memory-capacity theorem,
learning experiment, advantage over other recurrent networks or evidence
about trained-model generalization.

The phase-writing direction is chosen using phase sensitivity. Its first-order
advantage over a phase-null direction is therefore predicted by construction.
This is not an independent discovery endpoint. The experiment checks its
finite-amplitude persistence, observability and limitations against a cheap
model that contains exactly that generic explanation. A null-equivalent
result is useful closure, not grounds to search for a more favorable task.
This implements the canonical integrity playbook's section 4A warning about
predictors and outcomes that share the same mathematical factor.

## Fixed settings and source bindings

Use only these two existing EXP-005 controls:

| Role retained from EXP-005 | Alpha | Gamma |
|---|---:|---:|
| Core | 1.064 | 10.260722222222222 |
| Failed near-edge comparison | 1.0661333333333334 | 10.297166666666667 |

The second name retains its failed original irregular-control interpretation.
Both later received numerical certificates of attracting primitive nine-peak
cycles. Their similar periods and differing transverse contraction motivate
the pair; they do not make the between-setting comparison causally controlled.

The result-free plan must bind byte counts and SHA-256 for:

- `experiments/hopfield/EXP-005-plan.json`;
- the original EXP-005 manifest and receipt;
- `evidence/EXP-005/attempt-001/points/core.json` and
  `points/failed-near-edge.json` under the same attempt;
- the exact unchanged neural RHS/native integration implementation, independent
  diagnostic implementation and all new source/test files.

Read the old certificate's section state and period only as the declared
starting seed for a new reference qualification. Preserve old files unchanged.
No atlas lookup, nearest-neighbor replacement or new origin selection is
allowed. All phases start directly on the qualified cycle; this deliberately
studies local response rather than basin size or unknown initial conditions.

## Reference stage before pulse access

For each fixed setting, rerun the existing independent primitive-orbit
certificate without relaxing its closure, neutral-mode, primitive-period or
transverse-stability gates. Record all original seed data and root diagnostics.
A failed certificate leaves the corresponding planned cases explicit as
dependency failures; it does not authorize selecting another orbit.

From the qualified state and period `T`, propose a DOP853 reference and
variational calculation with `rtol=1e-12`, `atol=1e-14`, `max_step=0.025`.
Keep raw endpoint mismatch. Store 4,096 equally spaced phase samples on
`[0,T)` plus the separate integrated endpoint, state derivative and tangent
data. The derived periodic template uses cubic Hermite interpolation with a
declared wrap to the starting state; the size of that wrap is audited, not
silently removed from the raw orbit. Its construction and interpolation must
be tested independently before execution.

Use a time-valued phase coordinate. Its adjoint gradient obeys

```
dot Z(t) = -J(x0(t))^T Z(t)
Z(t)^T f(x0(t)) = 1
Z(t + T) = Z(t).
```

Obtain the neutral left direction of the full monodromy by a declared
SVD-based null-vector calculation for `M^T-I`, normalize once by its product
with the flow, and integrate the adjoint backward over a period. Do not form
the inverse of an ill-conditioned monodromy, invert a tiny multiplier, or
normalize independently at every saved time to hide adjoint error.
Proposed additional reference gates are relative adjoint periodic residual
at most `1e-7` and `max |Z^T f - 1| <= 1e-7`, with all residual arrays saved.
The existing certificate remains a numerical certificate, not an interval
proof; these extra gates do not upgrade it to one.

At each of eight phases `j*T/8`, `j=0,...,7`, compute the unperturbed
one-period state/tangent propagator directly from the phase state, with
initial tangent identity and the same fine DOP853 settings. This requires
**16 named phase-propagator jobs**, not a potentially unstable inversion
of a common fundamental matrix. Their relative state closure must be at
most `1e-8`; retain the neutral flow residual and all monodromy entries.
The linear response at `k*T+s` is reconstructed as
`Phi_j(s) M_j^k delta_x`, in this order, without discarding transient growth.

Save a reference-stage seal binding all phase states, directions, tangent
predictions, readout coefficients and null outputs **before launching any
pulse or sham case**. References and propagators are new computations under
the future freeze, not unchanged historical evidence. Their CPU time and
storage are included in the attempt's cost.

## Intervention census

For each setting, phase and dose `epsilon in [1e-4,1e-3,1e-2]`, integrate
both signed initial states `x0 + sign*epsilon*d` for these unit directions:

| Direction identifier | Exact rule |
|---|---|
| `phase_write` | `d = Z/||Z||`, fixing the sign through `Z^T f=1`. |
| `phase_null` | Let `Pz=I-Z Z^T/(Z^T Z)`. Choose the coordinate index maximizing `||Pz e_i||`, breaking exact ties by the lowest index; set `d=Pz e_i/||Pz e_i||`. |
| `neuron_one` | `d=(1,0,0)`. |

This phase-null plane is defined by the adjoint, not Euclidean orthogonality
to the velocity. Preserve normalization, `Z^T d`, exact realized addition
`initial-x0`, its norm and rounding residual. Require direction norm error
at most `1e-12`, and normalized phase-null residual
`|Z^T d|/||Z|| <= 1e-12`. Never rescale a case using observed response size.
Equal state norm does not guarantee equal initial activation displacement:
save that displacement explicitly as a saturation/input-gain diagnostic.

There are `2*8*3*3*2 = 288` pulse cases plus `2*8 = 16` shams per method.
Each sham uses the same phase state with no pulse. No independent noise
trajectory is needed for the analytic observation-noise readout below.

| Numerical method | Pulse traces | Shams | Total |
|---|---:|---:|---:|
| Native CPU RK4, `dt=0.0025` | 288 | 16 | 304 |
| Native CPU RK4, `dt=0.00125` | 288 | 16 | 304 |
| Independent DOP853, `rtol=1e-12`, `atol=1e-14`, `max_step=0.025` | 288 | 16 | 304 |
| **All methods** | **864** | **48** | **912** |

All traces run over elapsed `[0,320]` and save exactly 12,801 state samples
at spacing `0.025`, including initial and final samples. RK4 needs exactly
128,000 or 256,000 steps. Require `8*T<=320` at both references; failure
does not trigger an outcome-selected duration extension. The weights remain
constant after the instantaneous pulse. The impulse represents a state
initialization change, not a permanent connection change, learned update or
fully characterized finite-duration electrical stimulus.

Candidate IDs are deterministic tuples of setting, phase index, dose index,
direction, sign and solver, with separate sham IDs. Create the entire
912-row task ledger at attempt creation. All signs, doses and methods remain
in every denominator; numerical or operational failures are never dropped.

## Clock-referenced delayed-sign readout

Use `u(t)=tanh(x(t))`. The experiment supplies an external clock and the known
starting phase. It does not solve synchronization or infer absolute phase
from an unobserved initial state.

For each delay `k in [1,2,4,8]`, take 32 fixed sample times

```
t_m = T * (k - 1 + (m + 0.5)/32), m=0,...,31.
```

This is a readout **over the cycle ending at `k*T`**, not an instantaneous
observation at that endpoint. Reconstruct potentials at these times by the
frozen cubic Hermite rule using saved states and RHS derivatives, then apply
tanh. Do not alter sample count, choose favorable sample phases or average
over a longer window after outcomes.

Concatenate all three reference activations at these times into a
96-dimensional vector `U0`, and their physical-time derivative into `G`.
The frozen linear matched-filter vector is `w=G/(G^T G)`. For a pulse vector
`U`, its time-shift score is `q=w^T(U-U0)`. `G^T G` must be positive and
finite; ill-conditioned or unresolved readout coefficients are a reported
qualification failure. No pulse outcomes enter `U0`, `G`, `w` or the
classification threshold. In particular, **no phase optimization is allowed
in this task score**.

The ordinary and phase-writing classifier's orientation is the sign of
the reference prediction `Z^T d` (use `+1` for an exactly zero prediction).
The intentionally phase-null control always uses fixed orientation `+1`,
so numerical roundoff in a theoretically zero projection cannot choose its
label orientation. Denote the orientation by `c`; the fixed rule predicts
positive sign when `c*q>0`.

Add independent Gaussian **measurement** noise with standard deviation
`sigma=1e-4` to each of the 96 activation samples. This is a declared
observation model, not a stochastic perturbation of the ODE or an estimate
of biological noise. The reference is treated as calibrated and noiseless;
the cost and idealization of that calibration are explicit. Let the two
deterministic signed responses give `q_plus` and `q_minus`. With
`s=sigma*||w||`, exact balanced sign accuracy is

```
BA = (NormalCDF(c*q_plus/s) + NormalCDF(-c*q_minus/s))/2.
```

Use a stable library normal CDF and independently test its calculation; no
Monte Carlo noise, class balancing fit or threshold training is needed.
Save both signed score means and `s`, including accuracies below chance.
The fixed secondary noise levels are `1e-5` and `1e-3`; never select one
as the successful noise level after inspection.

Primary descriptive estimands at each setting are the phase-writing minus
phase-null BA at the cycle ending at `4*T`, and both absolute accuracies,
averaged equally over eight phases and three doses. Report the ordinary
neuron-one BA beside them as the required practical-access check. Retain
all dose/phase effects, with delays `T`, `2*T`, `8*T` secondary. There are
only two selected physical settings: do not bootstrap phases, signs or
solvers as independent biological/network replicates. No arbitrary
utility-acceptance threshold is implied by an accuracy difference alone.

## Nulls and mechanism diagnostics

**Mandatory cheap null:** for each pulse, set `delta_t=sign*epsilon*Z^T d`
and predict `u_null(t)=u0(t+delta_t)`. Use the same periodic template,
external clock, 96 samples, frozen classifier and noise calculation. This
phase-only emulator has no transverse state, geometry feature or fitted
parameter. It is the first-order phase-reset prediction; differences at
large doses may simply expose nonlinear phase response, not new computation.
Its 288 signed responses are interpolation outputs, not extra ODE traces.

The zero-response null has BA exactly 0.5. Readout validation also uses
synthetic exactly shifted baseline signals at the frozen predicted phase
shifts. These validate arithmetic and the measurement floor, not the
network's capacity. Neither null can be omitted if it matches the network.

Compare the full saved state difference with `Phi_j(t) delta_x` for the
declared small doses and delays. Diagnose its linear phase component with
`Z(t)^T delta_x(t)` and its complementary component with
`[I-f(t)Z(t)^T]delta_x(t)`. These are local linear coordinates; at finite
amplitude they are not exact nonlinear phase/isostable coordinates.
Report both state and activation residuals and the odd/even response under
opposite pulse signs. Second-order phase shifts in the phase-null control
are expected to be possible. Do not call any nonzero finite-dose residual
a failure of phase reduction without its dose scaling and numerical floor.

An optional-looking but scientifically tempting nearest-orbit phase fitter
is deliberately **not required in this first design**. It adds phase
ambiguity and optimizer work while the fixed readout and linear prediction
already test the declared mechanism. In particular, the nine local neuron
three peaks cannot serve as nine interchangeable full-cycle origins.
Adding a nonlinear phase estimator later requires a labeled amendment;
it cannot retroactively replace this task score.

## Numerical qualification and possible falsification

Preserve every completed raw trace before scoring. Proposed all-case
comparison gates, evaluated separately for coarse-versus-fine RK4 and
fine-RK4-versus-independent DOP853, are:

- maximum absolute potential difference no greater than `1e-6`, and maximum
  absolute activation difference no greater than `1e-7`, over every saved
  component/time;
- for each signed readout and delay, difference in `q/s` no greater than
  `0.01` at the primary noise level;
- absolute BA difference no greater than `0.001` at every declared noise,
  delay, dose, direction and phase;
- every solver complete, finite, correctly bound and at the exact grid,
  with every reference/direction prerequisite qualified.

These are prospective consistency requirements, not certified error bounds.
Save RMS differences, maxima, their locations and component values even
when a gate passes. Report all numerical comparisons, without selecting a
favorable method or deleting the worst case. No automatic smaller-step
rescue is included. A failure leaves the experiment numerically unresolved
under this plan; an amended, separately frozen refinement may follow.

Shams must be retained in the same method/grid census. Compare their scores
to the calibrated template and the synthetic zero-response null; require
`|q/s|<=0.01` for every primary-noise sham readout. This checks that reference
drift or interpolation bias is not masquerading as stored input. Do not
center primary pulse scores with their observed sham after the fact.

Suppress qualified setting-level averages if any required case or prerequisite
for that setting fails. Preserve the full planned denominator and individually
available diagnostics. A complete-study claim needs both settings qualified.
Retain finite-amplitude escapes, transient amplification and unreadable
inputs as outcomes, rather than identifying and excluding them post hoc.

The selective-memory interpretation is falsified at these task conditions if
the numerically resolved phase-writing contrast does not persist, if retained
timing decays with the transverse component, or if the apparent contrast lies
below the qualified measurement floor. A phase-writing advantage with an
unreadable ordinary-neuron input establishes only engineered accessibility.
If the phase-only emulator explains the task responses, report generic
oscillator timing memory rather than special shrimp computation. An emulator
residual calls for diagnosis, not an automatic claim of superior capacity.

## Resource envelope and operational contract

The task uses local CPU only. EXP-012 supplies a measured reason to keep this
small job off the tested GPU path; no GPU reservation, driver operation,
RunPod launch or further paid consultation is part of this design.

Proposed caps are four owned worker processes, one BLAS/OpenMP thread each,
2 GiB combined owned-process RSS, and 2 GiB retained raw-output budget.
Require at least 8 GiB available host RAM before dispatch and retain a
4 GiB reserve during work. Record CPU/load availability; postpone dispatch
when shared resources cannot meet the declared envelope. Stop only owned
processes, identified by PID plus start identity, never another task's work.

The raw state payload alone is `912*12801*3*8 = 280188288` bytes, about
267.2 MiB. This excludes timings, time grids, reference/tangent arrays,
failures, logs and receipts. Derive activations from saved potentials rather
than silently duplicating bulky evidence. Save each case independently,
read back its exact schema/grid/dtype/initial condition, and hash it before
acknowledging completion.

Propose a 1,800-second whole-attempt watchdog, 120 seconds per reference
or phase-propagator job and 60 seconds per pulse/sham job. Failure sealing
has a separate bounded grace period. These are safety ceilings, **not
measured runtime estimates**. The eventual plan must include a measured
synthetic/infrastructure timing and a conservative estimate from qualified
historical CPU throughput; no new pulse may be timed before its freeze.
If the caps look inadequate before execution, revise the draft openly.
If a cap is reached during execution, preserve an incomplete attempt;
do not quietly extend it.

The future machine plan should include at least:

```
kind: neural_cycle_selective_timing_memory
status: exploratory; selected existing settings
source_commit_expected: <complete privately pushed future freeze>
source_files: <exact paths, bytes, SHA-256>
historical_bindings: <EXP-005 seed/certificate/manifest/receipt hashes>
settings: [core, failed-near-edge]
phase_indices: [0,1,2,3,4,5,6,7]
doses: [0.0001,0.001,0.01]
directions: [phase_write,phase_null,neuron_one]
signs: [-1,1]
methods: [rk4-0.0025,rk4-0.00125,dop853]
required_pulse_sham_tasks: 912
required_phase_propagator_jobs: 16
reference_certificate_jobs: 2
task_inventory: <all IDs and result-free dependency rules>
readout_and_noise: <the exact equations, samples and fixed constants above>
qualification: <all fixed numerical and operational gates>
resources: <caps, owned process schema, measured estimate>
output_root: <exclusive external attempt directory>
paid_compute: false
paid_review: false
```

`prepare` and `validate` must inspect only source, metadata and historical
bindings; neither may integrate. The runner must check clean source,
expected HEAD, remote containment, immutable input hashes and exclusive
external output before the first target solve. Reference-derived directions
are realized inputs generated under that freeze; store their later seal
without pretending those arrays existed in the result-free plan.

Create terminal records for all planned jobs even on setup failure. Preserve
partial native prefixes, adaptive-solver status and elapsed work. Distinguish
not started, dependency failed, integration failed, numerical disagreement,
timeout and completed. Source postflight and receipt inventory failures
invalidate qualification without deleting evidence. A new read-only verifier
must reconstruct nulls, scores, all 912 identities and the primary contrasts
from saved arrays, with complete hash/byte checks and no reintegration.

## Before implementation can become an execution freeze

Finish EXP-013 and retain its outcome regardless of sign. Review this design's
reference/adjoint algorithm, readout arithmetic and caps independently;
specifically challenge whether the proposed task can say anything beyond
its phase-only null. Implement synthetic analytic-cycle, signed-pulse,
readout-noise, interpolation, identity, failure-preservation and watchdog
tests, with no target outcomes. Measure the infrastructure budget, bind all
sources and source versions, and push the complete plan/code before running.
No experiment number, execution date or runtime success is asserted here.
