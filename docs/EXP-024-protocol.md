# EXP-024: Does the recurrent trajectory mix successive inputs?

22 September 2026. **Exploratory design under implementation; not an execution
freeze. No target integration or held-out outcome has been generated.**

**Human-friendly TLDR:** We already know that changing the readout can recover
some memories in the selected near-edge rhythm. Now we will test whether the
network itself combines two inputs into a new, linearly readable feature. This
distinguishes computation in the trajectory from interpretation by a nonlinear
decoder. A positive result would concern two selected neural cycles, not a
special advantage of shrimp geometry or trained networks.

## Motivation and scope

The [Astra Pro consultation](consults/2026-09-22-pro-adjudication.md) recommends
deferring EXP-023's matched-cycle acquisition. This experiment instead targets
the origin of a cross-input response. Fixed weights, an external clock, known
initial phase and a phase-specific reference observer define the task. It does
not test autonomous clock recovery, learning, generalization across weights,
joint two-bit information capacity or robustness under process noise.

Use the same qualified core and near-edge anchors as EXP-019/021. Exact input
artifacts, hashes and cycle certificates must be bound by the machine plan;
labels alone are insufficient. All prior failure records remain unchanged.

## Proposed fixed census

| Factor | Levels |
|---|---|
| Weights | Core and qualified near-edge anchors |
| Fresh initial phases | 8 development, 32 held out; independent uniform draws shared across settings |
| Both write amplitudes | 0.0001, 0.001, 0.01 in state-coordinate units |
| Signs in array order | (++), (+-), (-+), (--) |
| Nominal writes | 0 and one reference period T |
| Windows | [T, 1.25T] and [1.75T, 2T], 32 equally spaced inclusive samples per window |
| Observations | State x primary; activation tanh(x) secondary |
| Models | Full nonlinear ODE, first order, second order, cross-term-ablated second order |

This is 192 development and 768 held-out full trajectories, 960 total before
numerical comparisons, no-write references or sensitivity solves. Both windows
and both observation types use the same full trajectory. Samples are flattened
in time-major order to 96 coordinates. The sample at T is **after** the second
instantaneous write. T is setting-specific; cross-setting scores compare
normalized periods, not identical physical bandwidth.

Write directions must be unit norm, reference-defined and frozen before pulse
outcomes, with no sign-dependent orientation or perturbed-state recomputation.
Reuse a precisely specified reference phase-null construction where possible;
its source, projection, normalization and degeneracy policy remain to be
bound. No direction search on finite outcomes is allowed. Sign-independent
constant state translations have identity derivative, so the second write
introduces no cross sensitivity by itself.

Set measurement noise SD to 0.0001 per sampled state coordinate, iid Gaussian.
This is an analytic measurement channel, not injected dynamical process noise.
Use the same numerical SD for the secondary activation channel but explicitly
different units. Scores are not directly comparable to EXP-021 as the same
task. Primary cell: near-edge, amplitude 0.01, early state window. Its endpoint
is the mean across 32 held-out phases of the sealed reference observer's
noise-averaged parity R-squared.

## Mechanistic equations

Let two signed pulse magnitudes be a1 and a2. Around the zero-write trajectory
define u_i = partial x / partial a_i and v_ij = partial² x / partial a_i partial a_j.
Between pulses:

    dx/dt = f(x)
    du_i/dt = J(x) u_i
    dv_ij/dt = J(x) v_ij + H(x)[u_i, u_j]

For f(x) = -x + W tanh(x),

    J = -I + W diag(sech²(x))
    H[u,v] = W (-2 tanh(x) sech²(x) * u * v).

Initialize u1 to the first fixed unit write at time 0, u2 to zero, and all v
to zero. At T add the second fixed unit write to u2; leave x, u1 and all v
continuous in the sensitivity system. The finite-pulse system separately
receives its actual signed state jump. Before T, v12 is zero.

For common amplitude a, the second-order state coefficients are

    m_x = x0 + a² (v11 + v22) / 2
    A_x = a u1; B_x = a u2; C_x = a² v12.

The cross term has no extra factor of one half: the symmetric Hessian has two
equal off-diagonal terms. Self terms are sign-independent because s_i² = 1.

For componentwise g(x) = tanh(x), include observation derivatives:

    m_g = g(x0) + a² [g'(v11+v22) + g''(u1²+u2²)] / 2
    A_g = a g' u1; B_g = a g' u2
    C_g = a² [g' v12 + g'' u1 u2].

Products with g' and g'' are elementwise at x0. Report the state-induced and
observation-induced portions of C_g separately. Removing v12 alone must NOT
silently remove g'' u1 u2. Also transform first-order state predictions through
the actual tanh as a separately labeled static-observation control; this is
different from linearizing the observation.

## Linear observer and analytic risk

For reference coefficients m,A,B,C, observations have the model

    Y = m + s1 A + s2 B + s1 s2 C + epsilon.
    covariance = AA^T + BB^T + CC^T + sigma² I
    w = covariance^-1 C
    predicted parity = w^T (Y-m).

Do not normalize w. Freeze m and w from reference sensitivities before finite
test means exist. Evaluate all four full-ODE sign means mu with

    R² = 1 - mean_signs[(s1*s2 - w^T(mu-m))² + sigma² ||w||²].

An additive response has C=0, zero optimal weights and R²=0. An arbitrary
nonzero observer on an additive system cannot gain positive population R².
This avoids the misleading possibility of above-chance threshold accuracy on
an XOR arrangement. Preserve negative scores and reference-centering error.

Direct contrast C_direct = (mu++ - mu+- - mu-+ + mu--)/4 includes all finite
cross effects. Its amplitude scaling and agreement with predicted a² v12 test
the second-order account. A descriptive affine oracle may use the exact direct
factorial coefficients to compute the best possible affine risk under this
specified noise model; label it as using test means. It does not replace the
reference observer or show all information recoverable by nonlinear decoders.

## Controls, numerical qualification and decision rules

Synthetic controls precede target execution: additive negative, orthogonal
injected parity with known C²/(C²+sigma²), dense covariance cross-check, static
nonlinear observation of additive states, and analytic second-order flow/jump
solutions. Development finite differences must separately test u1, u2, v11,
v22 and v12, including the post-write convention and observation derivatives.

The exact solver tolerances, independent-method census, comparison thresholds,
retry configuration and qualification rules are still implementation work.
They must be concrete in the pre-outcome machine plan. Every comparison must
retain absolute state and contrast errors; cancellation in C_direct makes a
trajectory-level relative error check alone inadequate. Risk uncertainty from
numerics must be resolved separately from phase sampling. No silent clipping,
selective retries or favorable imputation.

An unresolved primary phase blocks a complete-panel primary inference. R² is
unbounded below, so imputing zero is not a conservative failure bound. Report
all intended and observed counts, failed attempts and descriptive available
cases. Leaving a local cycle after a finite pulse is an observed behavior, not
an automatic exclusion; it limits the local expansion interpretation.

Use a fixed-sample phase bootstrap for descriptive uncertainty conditional on
these two weights, with its exact seed and resample count sealed in the machine
plan. R² > 0.05 with an interval excluding zero and resolved direct state
interaction is a proposed engineering trigger for one targeted extension, not
a universal computational-utility threshold. An affine oracle below 0.01 over
all cells motivates the process-noise branch. A reference/oracle gap motivates
diagnosing approximation or observability. Static-activation-only mixing does
not establish nonlinear state computation. Solver-sensitive results are invalid.

No automatic extra 32-phase panel after inspecting uncertainty. Any extension
requires its own dated frozen plan; the original fixed panel remains reported.

## Implementation status and resource limits

`tools/two_write_interaction.py` implements the pure four-sign decomposition,
reference LMMSE observer and analytic risk. Synthetic checks do not validate
neural sensitivity integration. No target files are loaded by this module.

Remaining: exact input hashes/write-direction contract, sensitivity integration
and tests, finite acquisition and sealed observer storage, complete numerical
comparison census, immutable task ledger and attempt receipts, development
machine plan, local adversarial review, private source push, then development
execution. Development results and timing may inform a dated held-out freeze,
but the held-out phase census and target outputs must remain sealed until then.

Before running, recheck shared CPU/RAM/GPU availability. Start with at most four
single-thread CPU workers, record owned PIDs and peak memory, and benchmark a
fixed 32-trajectory development batch. Initial wall-time cap: eight hours per
pilot attempt; checkpoint and preserve partial output on timeout. Record the
exact timeout enforcement and cancellation mechanism in the executable plan.
No GPU, RunPod or further paid review is required. The CUDA path's prior failed
qualification does not transfer to this experiment. Keep raw artifacts under
`../neural-shrimp-data/raw/EXP-024/` and only compact receipts in Git.
