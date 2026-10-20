# Research-director consultation: neural behavior inside shrimp regimes

17 September 2026. Updated decision-level brief for the user's one authorized
GPT-6 Astra Pro consultation, after EXP-013 through EXP-015. This supersedes
the unsent earlier brief; it does not replace old evidence. Same-family model
advice is not independent scientific validation. The reviewer receives only
this synthesized brief, not the source tree, figures or raw trajectories.

**Human-friendly TLDR:** In the two tested shrimp regimes, the network repeats a
stable activity pattern. A disturbance can shift its timing or leave a separate
transverse signal. Core signals attenuate quickly; near-edge signals persist
and can temporarily reverse the expected readout. This suggests consequences
for memory and robustness, without proving a benefit unique to shrimp geometry
or transfer to trained networks.

## Question and emphasis

What important neural consequence or scientific alternative are we missing?
Recommend the most informative next bounded experiment, and challenge whether
our present numerical-audit priorities are serving the scientific question.
The researcher suspects shrimp regimes have consequences beyond visual map
geometry. Do not assume this intuition is correct or equate generic oscillator
memory with shrimp-specific utility. Explain both the promising mechanism and
the cheapest experiment capable of falsifying its importance.

Each recommendation and the final synthesis must include a human-friendly
TLDR answering: **What did we learn, or would we learn, about the behavior of
neural networks in regimes of the shrimp?** Distinguish this fixed three-neuron
model from trained networks, biological brains or modern language models.

## System, intervention and readout

The autonomous fixed-weight tanh network has potentials x and activations a=tanh(x):

    dx1/dt = -x1 + 1.5*a1 + 2.9*a2 + alpha*a3
    dx2/dt = -x2 - 1.5*a1 + 1.18*a2
    dx3/dt = -x3 + gamma*a1 - 22*a2 + .47*a3

Alpha and gamma are two weights. No training, loss function or ongoing input
exists. Instantaneous state pulses are followed by unchanged autonomous flow.
An external clock only OBSERVES the system; it does not drive or entrain it.
It takes 32 midpoint activation samples in windows ending at 1,2,4,8 periods,
with a frozen matched filter built from the reference waveform's time derivative.
The signed score q has analytic Gaussian measurement-noise SD sigma*||w||,
primary sigma=1e-4 and diagnostics 1e-5,1e-3. Reported balanced accuracies are
analytic probabilities, not sampled noisy trials or fitted classifier scores.

Two selected primitive nine-peak full-state cycles are used: core alpha=1.064,
gamma=10.260722222222222, period37.63637538, dominant transverse Floquet
multiplier .0650601; near edge alpha=1.0661333333333334,gamma=10.297166666666667,
period37.33149959,multiplier .8560332. Each has the autonomous neutral direction.
These are numerical certificates, not rigorous enclosures, a matched causal
pair, complete boundary organization or proof that every map island is a shrimp.

## Evidence, including failures

Earlier work mapped39,322 settings plus201 transect starts and certified selected
primitive cycles. A supposed coarse-map irregular exterior later proved periodic;
that failed interpretation remains. Detailed shrimp boundary organization is open.

Geometry-informed waveform control has no established pooled advantage over
simple controls. In24 historical development origins, four-preview mean errors
were .0162661 shared features, .0107899 geometry, .00856899 signed axes;
full18preview .00617504. Geometry's contrast interval includes zero. Radius-specific
winners are not a validated switching rule. Original strict Linux replay remains failed; later regenerated-input diagnostics
retain19 coefficient/eigenvalue discrepancies despite unchanged tested actions.

EXP013 acquired24 fresh origins/3,456 candidates/10,608 previews. One candidate
fails waveform agreement, so the frozen whole-cohort utility means remain withheld.
No policy selected it as a winning action. Its separate finer-step diagnosis
passes10 new traces and30 comparisons; original qualification is not rewritten.

EXP014 acquired930 tasks: two new reference certificates,16 one-period phase
propagators, and912 finite-pulse/sham trajectories (three methods; duration320).
At8 phases per setting, directions are phase_write=Z/||Z||, a deterministic
unit vector in Z's null plane, and neuron_one. Pulse norms1e-4,1e-3,.01,
both signs. Z is the saved time-valued adjoint gradient, Z^T f=1 approximately.
Phase writing generally is NOT flow-tangent and can have a transverse component.
Predictions, requested-versus-realized floating-point pulses, clocks and orientations
were sealed before pulse outcomes. Full-linear prediction uses realized displacement;
the finite-shift phase-only emulator uses the requested displacement.

All references, phase propagators, shams, smaller doses and phase-null comparisons
pass the original scientific gates.30 trajectory comparisons fail at .01; both
setting averages are withheld. A separately frozen425-trace tighter-solver panel
covers85 selected cases including both signs/all doses/matched shams. All52
smaller-dose cases and7 shams agree across all8 methods. Eight large-pulse cases
still disagree, including7 between the two tightest DOP853 solves. Their
primary-noise binary accuracy has zero solver spread even while state and signed
q disagree greatly: bounded probability saturation can hide numerical error.
No chaos, basin crossing or other attractor has been established by disagreement.

EXP015 now completes192 saved-reference cells (2settings*8phases*3directions*4windows)
and1,152 realized-pulse compatibility rows. No new ODE, decoder fit or target-pulse
comparison. With Q=f0 Z0^T, P=I-Q and elapsed Psi(kT+s)=Phi(s) M^k,
the linear response is split into Psi f0 (Z0^T delta) and Psi P delta.
The physical flow and Hermite-template velocity are distinct; their defects and
neutral-transport error are retained separately. No projector repair, phase
renormalization or orthogonal-energy interpretation is allowed.

Frozen algebraic checks pass191/192; one near-cancelling unit-state partition
fails the unchanged atol1e-14/rtol1e-12 rule. The ideal-neutral approximation
passes179/192:12 additional cells exceed state1e-6 or activation1e-7 defect
scales only for .01 pulses; scalar q/noise defects still pass .01. All1,152
original full-linear compatibility rows pass. Independent evidence/recorded-decision
audit passes240,153 checks. Alternate SciPy Hermite/explicit products/compensated
sums retain2,917 strict numeric discrepancies and direct-propagation failures in9
additional originally algebra-qualified cells. All3,456 independent physical/noise
approximation classifications match. Maximum realized q/noise discrepancy is
2.675e-9. Do not call this an overall strict-arithmetic pass or new solver certificate.

All64 phase-null cells pass frozen producer checks: q's timing component is nearly
zero by construction but a transverse readout remains. Across all8 phases, core
signed unit transverse q ranges from [-.04744,.04744] at window1 to roughly
[-2.93e-9,2.14e-9] at8. Near-edge ranges are[-.413,.782] at1 and[-.167,.338] at8;
some small near-edge responses grow rather than decline over these windows.
At near-edge phase1/window4, exact realized +/-1e-4 pulses have linear q about
+/-6.29666e-5 almost entirely transverse (primary standardized mean +/-2.45676).
The prior individually consistent finite-pulse case had analytic accuracy .9929902,
full-linear .992990214 and phase-only .5. This is one explained case, not a
setting mean or independent proof of a mechanism unique to shrimp geometry.

A further prediction is early sign reversal: all near-edge phase-writing algebraic
phase scores are positive, yet total linear q is negative at4/8 phases in window1,
3/8 in window2, and none at windows4 and8. Transverse opposition outweighs the timing part.
All16 early-window cells pass frozen checks. This is not a newly validated finite-pulse
performance result. Transverse modes, observability and nonnormal transients matter;
one Floquet modulus alone is not a bound on every finite-horizon readout.

## Proposed successor and alternatives to challenge

Our current next design independently integrates the unperturbed state plus all9
tangent entries directly through8 old periods at every16context, using fixed
DOP853 and Radau settings (candidate rtol1e-13,atol1e-15,maxstep.00625):32jobs.
Keep old states, periods, projectors and clocks; do not reshoot or retune. Separate
cross-solver agreement, old stitched Psi versus direct propagation, and each
new solver's direct solution versus its own first-period composition. Compare
operator row-norm errors scaled to .01 input against the inherited state/activation/
noise scales; retain strict arithmetic separately. This is a proposal, not an
executed or frozen study. Is this the best next scientific use of effort, or
an overemphasis on negligible arithmetic instead of a meaningful neural test?

Other candidates: controlled continuous input with fading-memory and nonlinear
processing tasks, robustness-memory tradeoffs, matched cycle controls outside
the shrimp shape, bifurcation-boundary continuation, or eventual trained-RNN
transfer. Generic oscillator memory is prior art. Published driven phase-coded
RNN examples motivate the direction but their forcing differs from our observation
clock. Do not invent citations; identify literature concepts needing verification.

## Requested decision and constraints

Rank no more than5 directions; choose exactly one next experiment with hypothesis,
independent unit, strongest cheap baseline, matched controls, manipulation, endpoint
and units, falsifier, minimum complete pilot and positive/null/invalid interpretations.
Name which effects are generic limit-cycle facts and what could establish added
value of shrimp geometry. Give a practical48-hour plan and2-week roadmap. Recommend
what to stop doing, and identify any claim or missing control that could invalidate
our interpretation. Do not merely prescribe stricter bookkeeping or a larger atlas.

Local resources: i9-14900K24cores/32threads,~92GiB usable RAM,shared RTX2000Ada16GiB.
CPU is the qualified path; the tested CUDA pipeline was slower and failed numeric
parity. No paid rental, external message or public release is authorized. We may
use suitable local resources with owned-worker limits and shared headroom. New
target outcomes follow executable private freezes; failed evidence is retained.
The user authorizes this one consult up to [amount redacted]; no retry or second review is authorized.
