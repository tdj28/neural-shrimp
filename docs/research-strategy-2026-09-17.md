# Local strategy review: test selective timing memory next

**Historical strategy.** The current behavioral priority and completed
boundary comparison are recorded in [EXP-025 results](EXP-025-results.md).
The dated consultation and planning account below is preserved as written;
it is not a statement of current consultation status or the next executable run.

17 September 2026. **Local adversarial analysis, not a Pro review.** The
requested GPT-6 Astra Pro consultation was prepared but not sent: no enforceable
[amount redacted] cap or available authenticated, included subscription route could be
established. Incremental consultation spend is **USD 0**; see the
[not-run record](consults/2026-09-17-status.md) and
[hash-bound receipt](consults/2026-09-17-receipt.json).

After the already designed fresh-origin EXP-013, the next scientific question
should be whether these recurrent neurons **retain an input's timing effect
while forgetting its waveform-shape effect**. One bounded state-pulse
experiment can test that mechanism and a simple delayed readout. Further
controller ablations alone would not answer it. This recommendation does not
depend on EXP-013 turning out favorably.

This is a proposed next design, not an executable protocol, preregistration or
completed experiment. No new trajectory or model fit was generated for this
review. Exact numerical qualification, executable code, manifest and resource
limits must be reviewed, committed and privately pushed before target outcomes.

## What the evidence supports, and what remains missing

The inspected network is the asymmetric three-neuron tanh ODE in
[native/hopfield.cpp](../native/hopfield.cpp) and
[the manuscript](../paper/main.tex). Changing alpha and gamma changes two
connections between runs. There is no training objective or demonstrated
learning process; the name Hopfield does not supply the energy-descent or
associative-memory guarantees of a symmetric classical Hopfield network.

- **Dynamical structure is real.** EXP-004 independently verifies selected
  primitive periodic orbits, and EXP-005 verifies two nine-peak cycles after
  longer settling. Its small versus large activation-separation contrast is a
  real finite-time behavioral difference. However, a same-time difference mixes
  phase displacement, transverse recovery and transient amplification. Neither
  a small output difference nor the sign of a finite-time exponent identifies
  the information retained. See the [EXP-004](EXP-004-results-audit.md) and
  [EXP-005](EXP-005-results-audit.md) audits.
- **Geometry has not earned a task advantage.** EXP-006's favorable controller
  examples coexist with stronger baselines and failed numerical checks;
  EXP-007's tangent and coordinate policies choose exactly the same 192
  actions. EXP-008's primary geometry benefit remains unresolved, and its
  original strict Linux replay failure remains a failure. Saved-input
  regeneration is not a new integration. See the
  [EXP-006](EXP-006-continuation-results-audit.md),
  [EXP-007](EXP-007-results-audit.md) and
  [EXP-008](EXP-008-results-audit.md) records.
- **Qualified short labels do not qualify long behavior.** EXP-009 leaves
  952/1,012 designated sensitive cases unresolved at finer steps. EXP-010
  qualifies all 3,456 short preview labels and their tested rankings. EXP-011
  then measures a retrospective four-preview selection problem: geometry's
  mean RMSE is 0.0107899, shared features 0.0162661, fixed axes 0.00856899,
  and all 18 candidates 0.00617504. Its geometry-versus-shared contrast is
  heterogeneous and descriptively unresolved; local coverage has the lowest
  policy mean at the three smaller radii. See the
  [EXP-009](EXP-009-results-audit.md), [EXP-010](EXP-010-results-audit.md)
  and [EXP-011](EXP-011-results-audit.md) audits.

EXP-013's fresh origins, frozen shortlists and explicit cost/error curves are
the appropriate next test of that narrow controller proposition. They do not
test memory, information processing capacity, useful learning dynamics or
generalization to trained neural networks. A negative EXP-013 would close an
important controller hypothesis without invalidating the measured dynamics.

## What prior work makes unsurprising

Da Silva and Rech study periodic-window and bifurcation organization in this
neural system; that establishes the dynamical subject, not computational
utility. [Published article](https://dergipark.org.tr/en/pub/cams/article/1064713).

For an attracting autonomous cycle, phase has a neutral direction while
transverse disturbances contract locally. Phase-amplitude reduction makes
these separate responses explicit. Thus finding persistent phase displacement
is not a new shrimp mechanism by itself. The open question here is its
finite-amplitude, finite-time manifestation in observable neural outputs and
a defined readout. [Shirasaka, Kurebayashi and Nakao, 2017](https://arxiv.org/abs/1701.05428).
Asymptotic phase also need not accurately predict transient event timing;
the measured observable and delay matter.
[Wilson and Ermentrout, 2018](https://epubs.siam.org/doi/10.1137/17M1153261).

An input-driven Hopf oscillator has already been tested on computation tasks,
with strong dependence on forcing and resonance. That work does not establish
that our unforced neural cycle is a useful reservoir. Its explicit input,
clock, sampled readout and task are precisely the missing ingredients here.
[Shougat, Li and Perkins, 2022, accepted manuscript](https://link.aps.org/accepted/10.1103/PhysRevE.105.044212).

Information-processing capacity concerns specified input-driven dynamics and
accessible readouts under explicit assumptions. Counting peaks, periods or
parameters is not measuring memory capacity; a persistent initial phase does
not automatically provide the fading-memory property used in reservoir
analysis. [Dambre et al., 2012, full author-deposited paper](https://backoffice.biblio.ugent.be/download/2972574/2972575).

The full da Silva/Rech, Shirasaka et al., Shougat et al. and Dambre et al.
papers were opened; only the publisher abstract was inspected for Wilson and
Ermentrout. These sources motivate the design below, not a claim that they
have tested this network's proposed task.

## One proposed experiment: pulse direction changes what survives

Use the **two existing EXP-005 certified cycles**, binding their exact
coordinates and orbit artifacts from the original manifest rather than rounded
manuscript coordinates. The core has period 37.6363753816 and dominant
transverse multiplier modulus 0.0650600985; the retained formerly failed
near-edge comparison has period 37.3314995885 and modulus 0.8560332413.
Both have nine peaks and periods differing by less than one percent, but very
different slow transverse return. They are two deliberately selected examples,
not independent population samples or a controlled geometry intervention.

Keep weights fixed throughout every pulse response. Requalify the unperturbed
reference and variational solution before the pulse stage, using frozen
methods; seal their hashes, derived directions and predicted responses before
opening any pulse output. This avoids mistaking the frequency change caused
by a permanent weight perturbation for stored input information.

At each of eight equally spaced phases, write a signed state pulse
`delta x = sign * epsilon * d`. Use three declared norms
`epsilon = 1e-4, 1e-3, 1e-2`, both signs, and these three unit directions:

1. **Phase-writing direction:** the normalized adjoint phase gradient `Z`.
2. **First-order phase-null direction:** a deterministic unit vector in the
   null space of `Z^T`. Choose the coordinate basis vector with the largest
   projection into that plane, with lowest-index tie breaking and a fixed
   sign. Euclidean orthogonality to the flow is not sufficient.
3. **Ordinary accessible input:** neuron-one coordinate direction `(1,0,0)`.
   This checks whether the mechanism survives without an engineered input
   direction that uses the full orbit's phase information.

Normalize phase in **physical time units**, so `Z(t)^T f(x0(t)) = 1` and
`dot Z = -J(t)^T Z`. The local predictions are a time displacement
`Z(0)^T delta x` and the full linear response `Phi(t,0) delta x`.
Use the full propagator for finite-time predictions: a Floquet modulus alone
does not describe nonnormal transient amplification. Do not invert or report
precise exponents for the numerically tiny transverse multiplier.

There are **288 pulse traces plus 16 sham traces = 304 per numerical
method**, with every declared phase, direction, sign and dose retained. Run
to 320 time units, covering eight full periods at both settings; save raw
states at spacing 0.025. Proposed numerical arms are unchanged native CPU
RK4 at 0.0025 and 0.00125, plus an independently implemented DOP853 arm
at every case: **912 traces** in total, excluding separately inventoried
reference/variational solves. The DOP853 tolerances, endpoint interpolation
and all pass limits must be fixed during implementation review, not selected
from these outcomes. Previous 75-unit qualification is insufficient here.

## Read a delayed input, without aligning its information away

The task is binary delayed recall of pulse sign from **actual tanh
activations relative to an external clock with known starting phase**.
Neither initial phase uncertainty nor clock recovery is supplied for free:
those are outside this first test. The neural state alone is not being
claimed to contain an absolute external timestamp.

Freeze a linear matched-filter readout from the unperturbed activation
waveform and its time derivative. At 32 fixed equally spaced samples in the
cycle ending at delay `k*T`, concatenate all three activations, subtract the
unperturbed clock-referenced template, and project onto its derivative.
The template and scale use no pulse outcomes. Reverse the classifier's
orientation when the baseline prediction `Z^T d` is negative; use a fixed
positive orientation when it is zero. An unobservable or unresolved prediction
is reported as such, not removed. Use the same sampling/readout for all
directions and nulls. **Do not optimize phase in the task score.**

Specify independent additive Gaussian readout noise with standard deviation
`1e-4` per activation sample. The fixed linear score then has an exactly
calculable Gaussian noise distribution, so balanced sign-classification
accuracy can be computed from both deterministic signed responses without
Monte Carlo trials or fitting a favorable decoder. This is a stipulated
measurement-noise model, not biological noise or noisy dynamics. Show the
predeclared `1e-5` and `1e-3` levels only as sensitivity analyses.

The **primary mechanistic contrast** is phase-writing minus phase-null
balanced accuracy at delay `4*T`, averaging phases and doses equally within
each setting and reporting the two settings separately. Report absolute
accuracies, not only their difference. The coordinate-input readout is the
practical-access check; its absolute accuracy must accompany any memory
claim. Delays `T`, `2*T`, `8*T` and dose-specific values are secondary.
Phases, signs and repeated numerical methods are not independent subjects;
do not manufacture a population confidence interval from them.

Separately reconstruct time displacement and transverse residuals from all
three state coordinates and compare with the frozen variational predictions.
A finite-amplitude nearest-orbit projection is a diagnostic, not an exact
isochron or isostable. Qualify its phase uniqueness and unwrap continuously:
the nine local maxima of neuron three are not nine interchangeable full-cycle
phase origins. Include potential and activation diagnostics so tanh saturation
cannot masquerade as dynamical forgetting. These diagnostics must never
replace the clock-referenced task score with an aligned RMSE.

## Cheap nulls, falsifiers and decision

The primary cheap null is a **phase-only emulator**, using the same saved
baseline waveform, period, initial phase, input displacement `Z^T delta x`,
readout and noise. After the input it advances uniformly around that waveform;
it has no transverse dynamics, geometry features or trained parameters. It
requires only reference interpolation. A zero-response/sham null gives chance
sign accuracy. Synthetic exactly phase-shifted reference signals test the
readout and numerical detection floor without generating favorable ODE cases.

The within-setting direction manipulation isolates which input component
persists under the stated norm constraint. It does **not** match every input
direction's initial activation displacement: saturation and coordinate gains
must be shown explicitly. Between-setting recovery differences are descriptive;
alpha and gamma also change waveform shape and input/readout sensitivity.

The claimed selective-memory mechanism fails at the declared task conditions
if numerically resolved phase-writing responses have no delayed sign advantage
over phase-null responses, if the supposed retained timing signal decays with
the transverse component, or if the signal is below the qualified readout
floor. A smaller-dose first-order phase-null response should suppress the
linear phase term; finite-amplitude phase shifts of second or higher order
are not automatically a failure. Check the frozen dose scaling and the full
variational prediction rather than demanding identically zero residuals.

Even a clear engineered-direction effect is insufficient for a practical
input channel if the ordinary neuron-one pulse cannot be read at the declared
noise and delay. Conversely, if the phase-only emulator explains the observed
delayed recall, the warranted result is **generic oscillator timing memory
with measured neural readout**, not special computational power of shrimp
geometry. An unexplained residual is a model discrepancy to diagnose, not an
automatic discovery. Geometry-specific utility would require a later matched
intervention beyond this experiment.

The useful outcome is a measured separation between retained timing and lost
shape, including how quickly it appears and whether a simple input/readout can
use it. A negative or null-equivalent result is also decisive: it prevents
confusing repeatability with computation and prevents an open-ended search
for controller improvements from standing in for a neural mechanism.

## Execution boundary

Prefer four local CPU workers and one numerical-library thread per worker.
EXP-012's tested GPU pipeline was slower and failed fixed compatibility gates;
do not reopen that engineering effort for this small experiment. The proposed
raw state census is about 0.28 GB before metadata/compression, plus reference
artifacts. That is a planning calculation, not measured runtime or permission
to skip a cost/resource preflight. Freeze a measured runtime estimate and hard
watchdog in the executable plan, with bulky evidence outside Git, complete
failed-attempt records, owned-process cleanup and independent saved-array
readout replay. No paid compute or review is needed for this proposal.

Do not extend this first experiment to an input stream, multiple stored
memories, learning-rate stability, training a larger network or an unbounded
parameter search. Those claims require additional mechanisms and controls.
The proposed test first establishes exactly what this verified neural cycle
can preserve, forget and expose to a fixed observer.
