# EXP-006: does the measured shrimp geometry help preserve a neural waveform?

Prospective exploratory utility experiment, 2026-09-11. The question and design
were chosen after EXP-004/005. Anchor attempts and atlas-policy directions are
selected from those retained data before any EXP-006 target trajectory. The
local-policy algorithm is frozen then, and its directions are subsequently
calculated from each newly certified baseline orbit. The executable contract
is `experiments/hopfield/EXP-006-plan.json`. Its complete implementation, analysis,
tests, and this protocol must be committed and pushed before the exclusive first
attempt starts. No paid compute or paid review is authorized or required.

## Question and claim boundary

The fixed-weight three-neuron Hopfield network already has numerically certified
periodic orbits in the observed nine-peak shrimp. A finite change to its two
variable connection weights can change the actual activation waveform. We ask
whether a policy using the old two-parameter atlas preserves that waveform better
than a policy using only local weight sensitivity of the current orbit. This is
an explicitly defined recurrent-network signal-preservation task. It does not
establish a benefit for gradient training, reasoning, generalization, or larger
models. A negative comparison is an informative result and will be reported.

The shrimp lies in a connection-parameter plane. Its periodic attractors lie in
the network's three-dimensional state space. We do not describe a state
trajectory as entering a shrimp in the parameter plane. Neither preservation of
nine peaks nor proximity to an old periodic pixel is the primary utility outcome;
those would partly restate how the atlas was constructed.

## Fixed network, anchors, and coverage

Use the EXP-004/005 equation `dx/dt = -x + W tanh(x)`, with
`W = [[1.5, 2.9, alpha], [-1.5, 1.18, 0], [gamma, -22, .47]]`.
The full activation vector `tanh(x)` is the task output. Other weights remain
fixed. The old focus map and control summary are bound by exact SHA-256 hashes.

Within the eight-connected nine-peak component containing the original core,
select 12 anchor attempts: four from each rank third of the raw Euclidean distance
to the nearest observed non-nine-peak pixel. Exclude prospective candidates for
which that distance is not strictly smaller than the distance to the crop edge.
The seeded first choice and subsequent maximin choices are deterministic
(20260911); width normalization is used only to spread anchor coverage. Margins,
directions, and interventions use native alpha/gamma units. The map is pixelated,
and unresolved pixels are not thereby proved chaotic.

Each anchor undergoes two new independent common-initialization integrations
from `(0.1, 0.1, 0.1)`, at steps 0.01 and 0.005, with burn 10,000 and sample 2,000
physical time units. Both must have nine recurring maxima and the finer result
must yield an independently verified attracting shooting solution. Failed
anchors are retained without replacement; every associated planned record is
marked `baseline_not_evaluable`. Coverage always uses all 12 attempted anchors.

Four initial states are exact quarter-cycle samples of the certified orbit's
2,700 endpoint-excluded points (indices 0, 675, 1350, 2025). All arms at a given
anchor and phase start from the same state. Certification, native recurrence
tolerances, four-repeat requirement, 128-event capacity, and maximum 32-peak
search remain unchanged from the existing methods.

## Matched interventions and competing policies

Use radii 0.00025, 0.001, 0.004, and 0.016. Since only two distinct entries of W
change, `sqrt(delta_alpha^2 + delta_gamma^2) = ||delta_W||_F` exactly. At every
anchor, phase, and radius, evaluate eleven unit-direction arms:

1. Outward and inward old-pixel normals.
2. The two tangent signs.
3. The two signs of a locally calculated minimum-error direction.
4. The two signs of a seeded random direction.
5. Outward and inward normals from a fixed cyclic derangement of the anchors.
6. `atlas_margin`, the prospective primary geometry policy.

`atlas_margin` evaluates 512 fixed equally spaced angles using only the old atlas
and maximizes the destination's observed pixel margin in the same component.
Out-of-crop or noncomponent destinations score zero. Ties use the first angle;
all-zero scores use the declared inward fallback. Save all candidate scores,
pixel indices, crop flags, and selected directions. The frozen old-data selection
has zero fallbacks and 12 crop-censored selected scores among 48 anchor-radius
choices. Retain and report that limitation; the scores are not exact distances to
a fully observed bifurcation boundary. No post-outcome refinement is permitted.

The local baseline integrates the actual weight sensitivity
`S' = J(x) S + B(x), S(0) = 0`, where `B[0,0] = tanh(x3)` and
`B[2,1] = tanh(x1)`. Transform S to activation sensitivity, project off the single
global activation time-shift vector, and use the minimum eigenvector of the
resulting two-by-two Gram matrix in native weight units. Average errors from its
two signs: do not select the better sign from outcomes. Degeneracy is reported.
This is a stronger comparator than an attractor's Floquet multiplier alone.

One zero-weight-change sham is evaluated per anchor, phase, and integration step.
There are `12 * 4 * (4 * 11 + 1) = 2160` conditions at each of two steps, hence
4,320 planned trial-step records including ineligible-anchor records. Every
eligible record contains both a short probe and an independently initialized
long assay. A weight change is applied immediately and held fixed throughout.

## Waveform endpoint, mechanism, and numerical checks

The primary endpoint is aggregate root-mean-square error over all three
activations during 300 physical time units, sampled every 0.1 time unit including
both endpoints. Compare with the original certified baseline waveform, allowing
one constant phase offset for the entire window. Use 128 phase-grid candidates,
then bounded refinement around the best one. Do not refit frequency, warp time,
or realign individual cycles. A changed physical period therefore accumulates
task error. The reference epoch includes the starting quarter-cycle phase.

Keep synchronous error, component errors, constant phase offset, window errors,
and the Fourier cadence diagnostic. Aliasing flags do not trigger exclusions or
resampling. Zero shams expose interpolation and integration floors. The primary
step is 0.005; 0.01 is a separately reported numerical sensitivity, never a
choice made according to which looks better.

The independent long assay starts from the same initial state, not the short
probe's endpoint: burn 10,000, sample 2,000. Retain full peak-section evidence,
nine-peak retention, recurrence residual, physical period, and finite-time
Lyapunov estimate. They are secondary mechanism measurements. A finite-time
positive estimate alone is not a chaos proof, and a change of attraction basin
is not proof that a periodic orbit has ceased to exist.

Raw verification checks exact receipt hashes, complete record census, declared
arm directions, matched norms, exact certified phase states, both long horizons,
and independently recomputed waveform metrics. Only derived floating-point
identity checks use portability tolerance (relative 1e-12, absolute 1e-14);
raw artifact identity and scientific classification gates remain exact.

Every saved aggregate, neuron and window RMS is recomputed at its recorded
single global phase under that tight tolerance. Fresh phase minimization is a
separate portability audit: equivalent final optimizer phases need not have
identical last digits or evaluation counts. The fixed phase allowance is
`4*(sqrt(2.2e-16)*(1+1/128)+1e-10/3)`, reflecting the bounded optimizer's stopping
term over its frozen brackets. Multiply this allowance by the maximum RMS slope
of the periodic linear reference interpolant, then add
`1e-14 + 1e-12*abs(fresh_RMS)`, when comparing the recorded-phase RMS with a fresh
numerical optimum. This bound applies only to reoptimization replay; it changes
no saved scientific score, certificate, eligibility, missingness or analysis
rule. Evaluation counts remain provenance, not equality gates. No window gets
an independent phase fit. See `EXP-006-driver-audit.md` for the bound's derivation.

## Fixed analysis and missingness

The three declared contrasts are comparator error minus `atlas_margin` error:
mean of the two local signs, mean of the two random signs, and shuffled inward.
Positive values favor the atlas policy. First average equally over four phases
within an anchor, then equally over four radii, then equally over eligible
anchors. Also report each radius, all eleven arms, and both numerical steps.
The plain inward normal is a secondary diagnostic, not the sole comparator.

Use 10,000 anchor-cluster bootstrap draws (seed 20260911) and percentile 95%
intervals. These describe variation across the fixed selected anchors, not a
random population sample or a confirmatory hypothesis test. Any missing or failed
required eligible condition makes its affected contrast undefined. Do not remove
unsuccessful trials, choose a successful subset, or replace failed anchors.
Always show eligibility out of 12 and all planned versus completed conditions.

## Independent orbit continuation

Separately follow the EXP-005 original core and formerly unresolved near-edge
certified nine-peak orbits, in each initial gamma direction, with alpha fixed.
Use pseudo-arclength coordinates `gamma = seed_gamma + 0.41*s`, state scales
`(1,1,20)`, and unit scales for log period and s. This can traverse a fold and
retain unstable orbit segments. It never chooses drift anchors or directions.

The frozen limits are initial step 0.005, minimum 1e-5, maximum 0.01, 48 accepted
successor nodes in addition to the initial seed (at most 49 stored nodes), and
96 correction attempts per branch, ten Newton iterations and six
line-search halvings. Keep gamma in [9.9,10.7] and period between one quarter and
four times its seed value. Keep every accepted node and every correction failure.
Validate primitive nine-peak recurrence at accepted nodes, and independently
check shooting every eighth node and the last node. The detailed equations,
tolerances, and analytic fold tests are in `EXP-006-continuation-method.md`.

Compute the two-dimensional Poincare transverse spectrum, avoiding ambiguous
deletion of a +1 multiplier at a fold. Sign changes near +1 or -1 are numerical
candidate brackets; they are not exact bifurcation theorems. A failed corrector
does not demonstrate branch disappearance or basin loss. Report the finite
branch extent and stopping reasons, including branches with no crossing.

## Figures fixed before outcomes

Produce saved-evidence figures, with PDF/SVG/PNG and exact plot-data receipts:

1. The old atlas with all 12 anchor attempts, their margin strata, and new
   baseline eligibility. No unsuccessful anchor disappears from the plate.
2. Waveform error by dose for the atlas, symmetric local and random comparators,
   shuffled inward, and inward diagnostic; an all-arm heatmap; paired benefit
   intervals. Keep steps distinct and undefined results visibly undefined.
3. All four continuation branches, gamma versus physical period and dominant
   transverse multiplier, with both transverse multipliers retained in the plot
   data. Display stability, candidate crossings/reversals, node counts,
   failed-correction counts, and stopping reasons.
4. If an upper-margin anchor is eligible, use the first such frozen anchor,
   phase zero, radius 0.016 and primary step 0.005 for an illustrative
   three-activation trace plate:
   baseline, atlas, both local signs, and random plus. This selection is made
   now and does not choose the largest observed policy difference. If no
   upper-margin anchor qualifies, show the panel as unavailable rather than
   substitute another stratum or an outcome-selected example. Failed selected
   arm traces also remain visibly unavailable.

These complement rather than replace EXP-004/005 shrimp and control plates.
Captions must distinguish old-map selection evidence from new target outcomes.

## Resources, failures, and release

Local CPU only, four trial workers, exclusive attempt directory, hard whole-run
ceiling 1,800 seconds. The ceiling includes builds, baselines, continuation,
sensitivity, probes, analysis, and verification. Preserve partial files,
timeouts, failure metadata, and journals; no resume or outcome-dependent rerun.
The native workload is approximately eight billion steps. Non-target timing
fixtures gave a 300-time probe about 0.0121 s, a local sensitivity solve 0.770 s,
and one waveform comparison 0.013 s; these are estimates, not promised runtime.

Report the receipt's actual collection-and-analysis wall time and
local-sensitivity cost. The current receipt captures its wall clock before the
final verification pass, so it does not measure total time including that pass;
the hard whole-run ceiling still covers verification. The prior atlas took
140.54 s and is not free. Do not claim net optimization speedup from a probe
comparison. A prospective correction after failure requires preserving the
failed attempt and a new labeled source freeze; no consumed attempt is reset.

Use local design, driver, and continuation audits and meaningful synthetic tests.
After execution, audit raw numerical bindings and all manuscript claims, retain
negative results, render and inspect every PDF page, pass private-repository CI,
and merge through a reviewed PR. Paid Pro review remains optional and requires
separate explicit approval under the human-controlled review policy.
