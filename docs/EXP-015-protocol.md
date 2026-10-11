# EXP-015: conditional phase and transverse observable gain

17 September 2026. Exploratory analysis of already acquired reference data, proposed after EXP-014 and its numerical refinement. This is not independent confirmation of a task effect. No new ODE, fitting, new target gain calculation or timing pilot precedes the reviewed private freeze. The original EXP-014 settings remain unqualified; its numerical and strict arithmetic-audit failures remain unchanged.

The aim is to identify how much of the frozen **linear** readout prediction is carried by its propagated neutral-direction part and by the complementary part, conditional on the saved reference and variational calculation. This tests the generic phase/amplitude explanation already available from those objects. Correlation with those same objects is not new evidence for geometry utility. It does not establish finite-amplitude memory performance, trained-network transfer, capacity or boundary organization.

## Immutable input and complete census

Use only EXP-014 recovery attempt 002 at `/data2/neural-shrimp-data/raw/EXP-014/attempt-002`, source `406b89553b956bac1b97ef05413c799e383a9103`, receipt SHA256 `2494966914cb68d8c5286a5dca6710212ade171ead42c9cfaed82cebd44c12a8`. The controller binds the exact receipt inventory, two qualified reference packets, sixteen phase-propagator packets and original reference-output seal. Original pulse trajectories and measured target readouts are not inputs to this analysis.

Each cell is exactly `(setting, phase index, direction, delay)` in the original setting order `core`, `failed-near-edge`, phase indices 0 through 7, directions `phase_write`, `phase_null`, `neuron_one`, and clock delays 1, 2, 4, 8. Thus there are **192 cells**. IDs are `{setting}--p{phase}--{direction}--k{delay}`. All cells remain in the ledger, including failed dependencies, arithmetic failures and failed approximation criteria. No setting or pooled task means are produced.

Every cell also retains all six original dose/sign inputs: doses .0001, .001, .01, each with signs -1 and +1. These create **1,152 compatibility rows**, not new integrations or independent replicates. The requested vector, exact original initial-state addition and realized displacement must match the seal. The unit-direction readout has three fixed noise diagnostics, giving 576 unit/noise rows; the six realized responses at all three noise scales give 3,456 pulse/noise rows. The noise levels remain 1e-5, 1e-4 and 1e-3, with 1e-4 primary.

The adapter may remove unrelated dense diagnostic arrays when extracting a compact input bundle, but every retained field and array must be an exact copy of the original sealed value. Mathematical reconstruction starts only after the full executable plan and source are privately pushed. No implicit file or target access occurs inside the mathematical helper.

## Exact algebraic partition

At the original starting phase use the saved flow `f0`, saved adjoint `z0`, and matrices

```
Q0 = f0 z0^T
P0 = I - Q0.
```

Do not renormalize `z0`, orthogonalize either operator, recompute a direction or remove a small neutral residual. Record `z0^T f0 - 1`, `P0^2-P0` and `z0^T P0`. These are oblique operators; state or activation energies are not an additive orthogonal decomposition.

Reconstruct the elapsed propagator independently in matrix form from the original nonperiodic tangent segment:

```
Psi(k*T+s) = Phi(s) M^k.
```

The multiplication order matters. Periodic wrapping applies to the reference state, not to the full tangent propagator. The original 32 clock samples lie within the declared eight-cycle horizon.

For either a fixed unit direction `d` or an exact realized displacement `delta`, let the input be `v` and define `a=z0^T v`. Compute

```
total = Psi v
phase_algebraic = Psi f0 * a
transverse = Psi P0 v.
```

`total = phase_algebraic + transverse` is an algebraic identity even if `z0^T f0` is only approximately one. Check it directly in state, activation and scalar-readout space. Also compare total propagation against the unchanged original `full_linear_response` primitive. This is internal arithmetic consistency, not an independent solver-accuracy certificate.

## Physical neutral motion and the frozen clock

Keep two reference velocities distinct:

- `f_ref(t)` is the unchanged Hopfield RHS evaluated at the saved periodic Hermite state.
- `v_template(t)` is the Hermite template derivative used by the original clock.

The ideal physical neutral response is `f_ref*a`; the ideal clock response is `v_template*a`. Preserve the separate defects

```
transport_defect = (Psi f0 - f_ref) * a
clock_flow_defect = (f_ref - v_template) * a.
```

Neither is silently absorbed into the transverse part. Report each raw state/activation array, entrywise maximum and RMS, maximum location and signed value; report the relative physical-flow transport residual descriptively. If a physical-flow sample is exactly zero, retain its count and mark that relative ratio unavailable. Relative flow residuals are not a derived eight-cycle accumulated-error bound.

For activation response use only the original first derivative `D(t)=diag(1-tanh(x_ref(t))^2)`. This is not the finite nonlinear difference `tanh(x+delta)-tanh(x)`.

The original clock weights `w_m` and sample times remain byte-exact. Compute the observable row

```
L_k = sum_m w_m^T D_m Psi(t_m)
```

and the signed scalar score for each total/component/defect array. The decomposition is

```
q_total = a + q_transverse + q_transport_defect + q_clock_flow_defect
          + (q_phase_clock - a) + arithmetic residual.
```

The parenthesized term records clock normalization/interpolation residual; do not force it to zero. Report `sum_m w_m^T D_m v_template(t_m)` separately. Report the phase-algebraic, transverse, physical-neutral and clock-neutral gains, state/activation response statistics, row operator and signed contributions at all three measurement-noise scales. Standardization is by the original `sigma*||w||`; orientation is the original frozen integer sign. These are signed linear readouts under an assumed independent Gaussian measurement-noise model. No Gaussian trial sampling, classifier fitting, BA endpoint or orientation change is introduced.

## Compatibility and qualification gates fixed before gain outcomes

Input identities, finite float64 arrays, epochs, time grids, monodromy endpoints, directions, orientations, clocks and all original prerequisite qualification flags must match. Every current clock is reconstructed from the unchanged template and compared exactly with the original sealed clock.

The **algebraic qualification** requires exact input identity, complete calculation, direct propagation agreement, state/activation/readout partition agreement, observable-row agreement and all six original full-linear compatibility checks. Componentwise arithmetic comparisons use `atol=1e-14, rtol=1e-12`; this preserves the original saved-input replay tolerance. Each residual and its maximum tolerance ratio is saved. The six compatibility checks cover the original full-linear state array, activation array, scalar q and noise gain at that clock. All six use **realized** displacements. The original nonlinear finite-shift phase-only q, which used **requested** displacement, is retained separately and never equated with the linear neutral component.

The **ideal-neutral qualification** additionally requires `|z0^T f0-1|<=1e-7`, inherited from original reference qualification, and all three approximation checks below for all six realized pulses:

1. Physical-flow transport defect alone.
2. Physical-RHS versus template-velocity defect alone.
3. Combined difference between algebraic phase and ideal clock phase.

Each check uses the existing numerical relevance scales: maximum absolute state discrepancy <=1e-6, maximum activation discrepancy <=1e-7, and absolute scalar-readout discrepancy divided by `1e-4*||w||` <=.01. These are the same state/activation/readout scales used in the original pulse comparison; they are prospective approximation criteria, not a certified error bound. Testing each defect separately prevents fortuitous cancellation, while the combined check bounds the complete ideal-clock approximation. No favorable noise scale is selected after outcomes.

A cell can be complete and algebraically qualified while its ideal-neutral interpretation fails. Persist both flags. The convenience `qualified` field means ideal-neutral qualification and never erases the separate algebraic flag. A malformed input or computational exception leaves a failed cell plus all six retained compatibility placeholders or completed prefixes. Numeric gate failures preserve complete arrays and measured residuals. No fallback, reconditioning, alternative solver selection or threshold change is permitted.

All these checks are conditional on the inherited variational solution. Testing neutral-flow transport does not independently validate every transverse column; no second reference or tangent solve is added. Later comparison with finite-pulse responses needs a separate explicit numerical-eligibility contract, after sealing these reference-only predictions. This run performs no such target comparison and no original-study requalification.

## Execution and evidence

Root's controller/plan must bind this helper, original dependencies, metadata adapter, tests and protocol; require a committed privately pushed clean source; create an exclusive external attempt; and persist the input seal before any new gain calculation. Run on local CPU with one numerical thread, no GPU or child workers. Frozen limits: 300 seconds for producer work, 10 seconds per cell, 60 seconds sealing grace, 4 GiB owned RSS, at least 8 GiB launch availability and 4 GiB remaining reserve, and 256 MiB raw output. Saved-array verification is separately invoked. The earlier 13-second original-input replay is a planning reference, not a measured new timing pilot. No paid compute or paid review is needed.

Create all 192 cell and 1,152 compatibility identities before calculations. Every interrupted or failed row remains; partial arrays/prefixes are retained where available. Store numerical arrays and verbose outcomes outside Git. A receipt binds exact code/plan/source/input hashes, invocation, timings, resource observations and all output bytes. A frozen saved-array verifier replays complete rows, validates partial status/census and preserves numerical failures. Compact count/flag summaries may be committed; no pooled task means or gain-driven subset selection is permitted.

Defer handled signals only during the short terminal/journal record commit,
then deliver pending cancellation before further work. An interrupted,
unreadable input seal can be verified only as preserved invalid partial bytes
with a recorded input-stage failure and zero dispatched cells. It is never
accepted as a valid input seal or a successful saved-array replay. Uncatchable
kills and storage failures may prevent final metadata sealing; the initial
complete ledger and any existing files remain evidence of the attempt.
