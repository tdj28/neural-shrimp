# EXP-005: independent, longer control checks after EXP-004

Prospective exploratory follow-up, selected after EXP-004 outcomes on
11 September 2026. EXP-004 supplied actual neural parameter maps and three
certified cycles, but its nearest initially irregular exterior became
period-nine under longer integration. That failure remains in EXP-004.
This experiment asks whether fixed controls offer a more persistent
irregular comparison under independent initialization and longer settling.
It does not replace the earlier experiment or retroactively preregister
the new choices.

The complete contract is `experiments/hopfield/EXP-005-plan.json`. Four
fixed points are retained: the earlier core, both endpoints of its declared
independent transect, and the failed near-edge comparison. Every point is
run with `dt=0.01` and `dt=0.005`, restarting from `(0.1,0.1,0.1)` each time,
with 10,000 physical time units discarded and 2,000 measured. These are
eight independent runs. No point, step or changed outcome is dropped.

The existing native flow engine, event detector, state recurrence classifier
and independent DOP853 shooting/Floquet code are reused. The 128-event
buffer supports four repeated blocks up to 32 maxima per candidate cycle.
The finite-time exponent is accumulated over the final 1,900 measurement
time units after its fixed 100-unit tangent alignment.

An independent orbit certificate is attempted only when both steps agree
on a positive peak period; agreement with the old EXP-004 period is not
required. Positive finite-time exponents above 0.005 and absent short
recurrence at both steps supply a **finite-horizon irregular comparison**.
They do not prove chaos, rule out longer transients or periods, or establish
any task benefit. All other outcomes remain explicit unresolved or
step-sensitive controls.

From each finest-step terminal state, the existing behavior probe records
300 time units at spacing 0.05, with a paired perturbation of norm `1e-6`
in direction `(1,1,1)/sqrt(3)`. Full neuron-potential, activation and
separation arrays are saved. Their comparison concerns this parent recurrent
network's behavior, not accuracy of a trained model.

The entire code and plan must be clean, committed and pushed before any
EXP-005 target outcome. The output directory is exclusive. Each completed
cell is saved and read back before the next starts; raw native failures,
partial diagnostics and progress remain preserved. A 600-second hard
watchdog bounds the attempt. All result artifacts are hashed; verification
recomputes classification, period times and residuals, checks coordinates,
initialization and timing, and checks behavior-array identities. No paid
resources, review or new dependencies are used.

## Implementation audit

The new orchestration reuses the frozen EXP-004 native ABI and diagnostics.
Its tests use synthetic arrays and injected stubs: they check eight fresh
initializations, saved payloads and failure preservation, positive-period
agreement, finest-step certificate seeding, absent-period/positive-exponent
label gates, declared plan validation and raw-cell verification. These tests
do not integrate or classify any new target-network trajectory.

Pre-run check: all six `tests/test_hopfield_controls.py` tests passed in
0.31 seconds, and the plan validator reported exactly eight cells. No
EXP-005 target trajectory was generated during implementation or testing.

Independent root review before target execution checked the complete driver,
manifest, and synthetic tests against the reused native binding and certificate
implementation. The explicit per-call step overrides the plan's two-step list;
burn and measurement horizons remain 10,000 and 2,000. Every call constructs a
fresh initial state. All eight payloads are saved before diagnostics. Both
positive periods must agree, and only the finest-step state seeds shooting.
The irregular-support label requires both positive finite-time exponents and
both absent short periods. No blocking discrepancy was found. The six new
synthetic tests passed before this review; no target trajectory was run to
obtain that test result.
