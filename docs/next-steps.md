# Next experiments after the initial exploratory pilots

These are proposals, not consumed or frozen experiments. Do not alter EXP-001,
EXP-002, or EXP-003 or relabel their outcomes as confirmatory.

## 1. Establish a neural training cycle before studying its utility

The broad tanh pilot has not provided verified periodic candidates. First
implement an analytically tractable neural training positive control, alongside
an exact reproduction of a published training period-doubling setup. A useful
instrument fixture is a depth-two scalar linear network with tied weights,
prediction `y_hat = theta² x`, one training example `(x,y)=(1,1)`, and half
squared loss. Its gradient map is

`theta_next = theta - 2 eta theta (theta² - 1)`.

At `theta=±1`, the derivative is `1-4 eta`, reaching the flip threshold at
`eta=1/2`. This identifies a prospective neighborhood to explore; it does not
by itself prove a stable period-two branch or shrimp geometry. An implementation
must check branch existence, stability, and full-state recurrence. This toy
model is a neural positive control, not evidence of utility in practical models.

Use the same intervention opportunities and rate-reduction doses for recurrence,
loss stagnation, fixed schedule, and no intervention. Include predictive cycle
averaging as a separate component where meaningful. Charge the detector's
cost. A tie with scalar diagnostics is a useful result, not a failed study.

## 2. Separate long transients from stable periodic windows

Build a new tanh search manifest informed by the disclosed pilot. Test longer
horizons and a grid focused on bounded high-rate dynamics; independently vary
task and initialization. Add the Sohl-Dickstein architecture and scaling as an
explicit reproduction arm, keeping its no-bias/no-ridge choices distinct from
our initial biased, regularized teacher-regression model. Inspect long-lived
recurrence and true asymptotic cycles separately. Empty short-horizon cells do
not establish nonexistence, and visually oscillating loss is not full-state
recurrence.

## 3. Qualify geometry, then ask whether it adds information

Once actual neural periodic branches are available, implement predictor-corrector
continuation of the orbit equation and saddle-node/flip boundaries. Retain
multistability and failed correctors. A shrinking window or attractive silhouette
does not substitute for the bifurcation checks.

For utility, compare geometry-informed exit directions or robust interior
settings with ordinary recurrence, local stability, and an equally expensive
local performance map. Match actions and tuning budgets; hold out whole tasks
and whole windows. Report cold-start and amortized costs separately. A periodic
window is worth entering, averaging over, or avoiding only if independently
measured task quality and cost justify that action.

## 4. Extend to latent reasoning only after the cheap gates

Choose a pinned small open recurrent checkpoint and task population. Probe the
full evolving state, not decoded-answer recurrence alone. Relaxation and
momentum can preserve the fixed-point equation while changing stability; treat
feedback-gain changes as a different solver. Test rescue, early stopping,
incorrect termination, and productive-transient disruption with all timeouts
retained.

No large GPU is currently justified by the pilot. Price and request approval
for a specific paid run only after the numerical path and measured throughput
show that it is necessary. No RunPod resources have been created.
